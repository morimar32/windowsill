"""XGBoost domain classifier training + inference pipeline.

Usage:
    python transform.py "medicine"          # single domain
    python transform.py                     # all domains
    python transform.py --threshold 0.5     # custom threshold
"""

import argparse
import os
import sys
import time

import numpy as np

import config
from lib import db, xgboost
from post_process_xgb import compute_idf_stats, compute_adjusted_score

_here = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_here, "v2.db")
MODELS_DIR = os.path.join(_here, "models")


def run_domain(domain, pos_wids, all_wids, X_global, raw_embeddings,
               emb_normalized, wid_to_row, group_map, word_texts,
               known_wids_by_domain, threshold, con):
    """Train model for one domain, run inference, persist results."""
    t0 = time.time()

    # --- Build training set ---
    # Negative sampling: 5:1 ratio
    neg_ratio = config.AUGMENT_NEG_RATIO
    pos_ids = [wid for wid in pos_wids if wid in wid_to_row]
    if len(pos_ids) < 10:
        print(f"  SKIP {domain}: only {len(pos_ids)} positive words with embeddings")
        return

    neg_pool = [wid for wid in all_wids if wid not in pos_wids and wid in wid_to_row]
    n_neg = min(len(neg_pool), len(pos_ids) * neg_ratio)
    rng = np.random.RandomState(42)
    neg_ids = list(rng.choice(neg_pool, size=n_neg, replace=False))

    train_wids = pos_ids + neg_ids
    y = np.array([1] * len(pos_ids) + [0] * len(neg_ids))

    # Global features for training subset
    train_rows = [wid_to_row[wid] for wid in train_wids]
    X_train_global = X_global[train_rows]

    # Domain-specific features for ALL words
    domain_feats_all = xgboost.compute_domain_features(
        pos_wids, raw_embeddings, emb_normalized, wid_to_row
    )

    # Domain-specific features for training subset
    domain_feats_train = domain_feats_all[train_rows]

    # Full training feature matrix: 775 + 2 = 777 cols
    X_train = np.hstack([X_train_global, domain_feats_train])

    # Group assignments for training set
    groups = np.array([group_map.get(wid, wid) for wid in train_wids])

    # --- Train ---
    result = xgboost.train_production_model(X_train, y, groups)
    if result is None:
        print(f"  SKIP {domain}: training failed")
        return

    clf = result["clf"]
    m = result["val_metrics"]

    # --- Save model ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{domain}.json")
    clf.save_model(model_path)

    # --- Inference on all words ---
    X_all = np.hstack([X_global, domain_feats_all])
    probs = clf.predict_proba(X_all)[:, 1]

    # Filter: score >= threshold AND not already known
    known = known_wids_by_domain.get(domain, set())
    inserts = []
    for i, wid in enumerate(all_wids):
        if probs[i] >= threshold and wid not in known:
            word = word_texts.get(wid, "")
            inserts.append((domain, word, wid, word, "xgboost", None, 1, float(probs[i])))

    # --- Persist: delete old xgboost rows, insert new ---
    con.execute(
        "DELETE FROM augmented_domains WHERE domain = ? AND source = 'xgboost'",
        (domain,)
    )
    if inserts:
        con.executemany(
            "INSERT OR IGNORE INTO augmented_domains "
            "(domain, word, word_id, matched_word, source, confidence, has_embedding, score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            inserts
        )
    con.commit()

    elapsed = time.time() - t0
    print(f"  {domain:<30} F1={m['f1']:.3f}  AUC-PR={m['auc_pr']:.3f}  "
          f"new={len(inserts):>5,}  ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="XGBoost domain training + inference")
    parser.add_argument("domain", nargs="?", default=None,
                        help="Domain name (omit for all domains)")
    parser.add_argument("--threshold", type=float, default=config.XGBOOST_SCORE_THRESHOLD,
                        help=f"Score threshold (default: {config.XGBOOST_SCORE_THRESHOLD})")
    args = parser.parse_args()

    con = db.get_connection(DB_PATH)
    db.ensure_score_column(con)

    # --- Load shared data ---
    print("Loading shared data...")
    t_start = time.time()

    means, stds = xgboost.load_dim_stats(con)
    emb_map = xgboost.load_embeddings(con)
    all_wids = sorted(emb_map.keys())
    group_map, n_groups, _ = xgboost.build_morphological_groups(con)
    total_dims_map = xgboost.load_total_dims(con)
    pos_flags_map = xgboost.load_pos_flags(con)
    word_meta_map = xgboost.load_word_meta(con)
    word_texts = xgboost.load_word_texts(con)

    domains, skipped, excl_info = xgboost.load_domain_word_ids(
        con, min_words=config.AUGMENT_MIN_DOMAIN_WORDS,
        exclude_sources={"xgboost"}, exclude_stop=True
    )
    print(f"  {len(emb_map):,} embeddings, {n_groups:,} morph groups, "
          f"{len(domains)} domains (skipped {skipped})")
    print(f"  Filtering: {excl_info}")

    # Known word_ids per domain (non-xgboost only) for INSERT OR IGNORE dedup
    known_rows = con.execute(
        "SELECT domain, word_id FROM augmented_domains "
        "WHERE word_id IS NOT NULL AND source != 'xgboost'"
    ).fetchall()
    known_wids_by_domain = {}
    for d, wid in known_rows:
        known_wids_by_domain.setdefault(d, set()).add(wid)

    # --- Build global features ---
    print("Building global feature matrix...")
    X_global, raw_embeddings, emb_normalized, wid_to_row = \
        xgboost.build_global_features(
            all_wids, emb_map, means, stds,
            total_dims_map, pos_flags_map, word_meta_map
        )
    print(f"  Global features: {X_global.shape[0]:,} x {X_global.shape[1]} columns")

    # Free emb_map — raw_embeddings matrix holds the data now
    del emb_map

    t_load = time.time() - t_start
    print(f"  Shared data loaded in {t_load:.1f}s\n")

    # --- Determine which domains to run ---
    if args.domain:
        if args.domain not in domains:
            print(f"ERROR: Domain '{args.domain}' not found or has < "
                  f"{config.AUGMENT_MIN_DOMAIN_WORDS} matched words.")
            available = sorted(domains.keys())
            print(f"Available domains ({len(available)}):")
            for d in available[:20]:
                print(f"  {d} ({len(domains[d]):,})")
            if len(available) > 20:
                print(f"  ... and {len(available) - 20} more")
            con.close()
            sys.exit(1)
        target_domains = {args.domain: domains[args.domain]}
    else:
        target_domains = domains

    print(f"Training {len(target_domains)} domain(s), threshold={args.threshold}")
    print(f"{'─' * 75}")

    # --- Run per domain ---
    t_train = time.time()
    for domain in sorted(target_domains):
        run_domain(
            domain, target_domains[domain], all_wids,
            X_global, raw_embeddings, emb_normalized, wid_to_row,
            group_map, word_texts, known_wids_by_domain,
            args.threshold, con
        )

    elapsed_total = time.time() - t_train
    print(f"{'─' * 75}")
    print(f"Done. {len(target_domains)} domain(s) in {elapsed_total:.1f}s")

    # Summary
    total_xgb = con.execute(
        "SELECT COUNT(*) FROM augmented_domains WHERE source = 'xgboost'"
    ).fetchone()[0]
    print(f"Total xgboost rows in DB: {total_xgb:,}")

    # --- IDF post-process (only when running all domains) ---
    if not args.domain:
        print(f"\nApplying IDF score adjustment (raw < 0.7)...")
        t_idf = time.time()
        word_domain_count, n_total, rows = compute_idf_stats(con)

        updates = []
        deletes = []
        for domain, word, wid, raw in rows:
            if raw is None or wid is None:
                deletes.append((domain, word))
                continue
            n_doms = word_domain_count.get(wid, 1)
            adj = compute_adjusted_score(raw, n_doms, n_total)
            if adj < args.threshold:
                deletes.append((domain, word))
            else:
                updates.append((adj, domain, word))

        if updates:
            con.executemany(
                "UPDATE augmented_domains SET score = ? "
                "WHERE domain = ? AND word = ?",
                updates
            )
        if deletes:
            con.executemany(
                "DELETE FROM augmented_domains "
                "WHERE domain = ? AND word = ?",
                deletes
            )
        con.commit()

        remaining = con.execute(
            "SELECT COUNT(*) FROM augmented_domains WHERE source = 'xgboost'"
        ).fetchone()[0]
        print(f"  IDF adjusted {len(updates):,} rows, pruned {len(deletes):,} "
              f"({time.time() - t_idf:.1f}s)")
        print(f"  Final xgboost rows: {remaining:,}")

        # --- Tag domainless words ---
        # Simple (non-compound) words with no domain association get a
        # 'domainless' pseudo-domain so downstream systems can identify them.
        print(f"\nTagging domainless words...")
        con.execute("DELETE FROM augmented_domains WHERE domain = 'domainless'")
        res = con.execute("""
            INSERT OR IGNORE INTO augmented_domains
                (domain, word, word_id, matched_word, source, confidence,
                 has_embedding, score)
            SELECT 'domainless', w.word, w.word_id, w.word, 'pipeline',
                   NULL, 1, NULL
            FROM words w
            WHERE w.embedding IS NOT NULL
              AND w.word NOT LIKE '% %' AND w.word NOT LIKE '%-%'
              AND w.word_id NOT IN (
                  SELECT DISTINCT word_id FROM augmented_domains
                  WHERE word_id IS NOT NULL
              )
        """)
        con.commit()
        n_domainless = con.execute(
            "SELECT COUNT(*) FROM augmented_domains WHERE domain = 'domainless'"
        ).fetchone()[0]
        print(f"  Tagged {n_domainless:,} simple words as 'domainless'")

    con.close()


if __name__ == "__main__":
    main()
