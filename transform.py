"""Transform pipeline: XGBoost training + post-processing + reef + archipelago + scoring.

Usage:
    python transform.py "medicine"          # single domain (XGBoost only)
    python transform.py                     # full pipeline (all domains)
    python transform.py --threshold 0.5     # custom threshold
"""

import argparse
import os
import sys
import time

import numpy as np

import config
from lib import db, xgboost
from lib import reef_pipeline, arch_pipeline, score_pipeline
from post_process_xgb import compute_idf_stats, compute_adjusted_score

_here = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_here, "v2.db")
MODELS_DIR = os.path.join(_here, "models")

TOTAL_STEPS = 9  # XGBoost, IDF, domain_name_cos, DNC floor, ubiquity, domainless, reef, archipelago, scoring


def run_domain(domain, pos_wids, all_wids, X_global, raw_embeddings,
               emb_normalized, wid_to_row, group_map, word_texts,
               known_wids_by_domain, threshold, con):
    """Train (or load) model for one domain, run inference, persist results."""
    from xgboost import XGBClassifier

    t0 = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{domain}.json")

    pos_ids = [wid for wid in pos_wids if wid in wid_to_row]
    if len(pos_ids) < 10:
        print(f"  SKIP {domain}: only {len(pos_ids)} positive words with embeddings")
        return

    # Domain-specific features for ALL words (needed for inference regardless)
    domain_feats_all = xgboost.compute_domain_features(
        pos_wids, raw_embeddings, emb_normalized, wid_to_row
    )

    # --- Train or load ---
    if os.path.exists(model_path):
        clf = XGBClassifier()
        clf.load_model(model_path)
        metrics_str = "cached"
    else:
        # Build training set
        neg_ratio = config.AUGMENT_NEG_RATIO
        neg_pool = [wid for wid in all_wids if wid not in pos_wids and wid in wid_to_row]
        n_neg = min(len(neg_pool), len(pos_ids) * neg_ratio)
        rng = np.random.RandomState(42)
        neg_ids = list(rng.choice(neg_pool, size=n_neg, replace=False))

        train_wids = pos_ids + neg_ids
        y = np.array([1] * len(pos_ids) + [0] * len(neg_ids))

        train_rows = [wid_to_row[wid] for wid in train_wids]
        X_train_global = X_global[train_rows]
        domain_feats_train = domain_feats_all[train_rows]
        X_train = np.hstack([X_train_global, domain_feats_train])

        groups = np.array([group_map.get(wid, wid) for wid in train_wids])

        result = xgboost.train_production_model(X_train, y, groups)
        if result is None:
            print(f"  SKIP {domain}: training failed")
            return

        clf = result["clf"]
        m = result["val_metrics"]
        clf.save_model(model_path)
        metrics_str = f"F1={m['f1']:.3f}  AUC-PR={m['auc_pr']:.3f}"

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
    print(f"  {domain:<30} {metrics_str:<30} "
          f"new={len(inserts):>5,}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Step 1: XGBoost training + inference
# ---------------------------------------------------------------------------

def step_xgboost(con, target_domains, all_wids, X_global, raw_embeddings,
                  emb_normalized, wid_to_row, group_map, word_texts,
                  known_wids_by_domain, threshold):
    """Train/load XGBoost models and run inference for all target domains."""
    n_cached = sum(1 for d in target_domains
                   if os.path.exists(os.path.join(MODELS_DIR, f"{d}.json")))
    n_to_train = len(target_domains) - n_cached
    print(f"Processing {len(target_domains)} domain(s): "
          f"{n_cached} cached, {n_to_train} to train  "
          f"(threshold={threshold})")
    print(f"{'─' * 75}")

    t_train = time.time()
    for domain in sorted(target_domains):
        run_domain(
            domain, target_domains[domain], all_wids,
            X_global, raw_embeddings, emb_normalized, wid_to_row,
            group_map, word_texts, known_wids_by_domain,
            threshold, con
        )

    elapsed_total = time.time() - t_train
    print(f"{'─' * 75}")
    print(f"Done. {len(target_domains)} domain(s) in {elapsed_total:.1f}s")

    total_xgb = con.execute(
        "SELECT COUNT(*) FROM augmented_domains WHERE source = 'xgboost'"
    ).fetchone()[0]
    print(f"Total xgboost rows in DB: {total_xgb:,}")


# ---------------------------------------------------------------------------
# Step 2: IDF post-process
# ---------------------------------------------------------------------------

def step_idf(con, threshold):
    """Apply IDF score adjustment to low-confidence XGBoost predictions."""
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
        if adj < threshold:
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


# ---------------------------------------------------------------------------
# Step 3: Domain-name cosine similarity
# ---------------------------------------------------------------------------

def step_domain_name_cos(con, emb_normalized, wid_to_row):
    """Compute cos(word_embedding, domain_name_embedding) for every word-domain pair."""
    import embedder

    print("\nComputing domain-name cosine similarities...")
    t0 = time.time()

    # 1. Get distinct domain names (excluding 'domainless')
    domain_names = [r[0] for r in con.execute(
        "SELECT DISTINCT domain FROM augmented_domains "
        "WHERE domain != 'domainless' AND word_id IS NOT NULL"
    ).fetchall()]
    print(f"  {len(domain_names)} domains to embed")

    # 2. Load embedding model and embed domain names
    model = embedder.load_model()
    texts = [config.EMBEDDING_PREFIX + name for name in domain_names]
    domain_embs = model.encode(texts, show_progress_bar=False)
    domain_embs = domain_embs[:, :config.MATRYOSHKA_DIM].astype(np.float32)

    # L2-normalize
    norms = np.linalg.norm(domain_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    domain_embs = domain_embs / norms

    del model
    print(f"  Domain embeddings: {domain_embs.shape}")

    # 3. For each domain, batch compute cosines and UPDATE
    total_updates = 0
    for i, domain in enumerate(domain_names):
        # Get word_ids for this domain
        rows = con.execute(
            "SELECT word_id FROM augmented_domains "
            "WHERE domain = ? AND word_id IS NOT NULL",
            (domain,)
        ).fetchall()
        wids = [r[0] for r in rows]

        # Filter to word_ids that have embeddings
        row_indices = []
        valid_wids = []
        for wid in wids:
            idx = wid_to_row.get(wid)
            if idx is not None:
                row_indices.append(idx)
                valid_wids.append(wid)

        if not valid_wids:
            continue

        # Vectorized cosine: emb_normalized[row_indices] @ domain_emb
        domain_emb = domain_embs[i]
        word_embs = emb_normalized[row_indices]
        cosines = word_embs @ domain_emb

        # Batch UPDATE
        updates = [(float(cos), domain, wid)
                    for cos, wid in zip(cosines, valid_wids)]
        con.executemany(
            "UPDATE augmented_domains SET domain_name_cos = ? "
            "WHERE domain = ? AND word_id = ?",
            updates
        )
        total_updates += len(updates)

    con.commit()
    elapsed = time.time() - t0
    print(f"  Updated {total_updates:,} rows ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Step 4: DNC floor pruning
# ---------------------------------------------------------------------------

def step_dnc_floor(con):
    """Delete xgboost entries with domain_name_cos below threshold."""
    floor = config.DOMAIN_NAME_COS_FLOOR
    print(f"\nPruning xgboost rows with domain_name_cos < {floor}...")
    t0 = time.time()
    deleted = con.execute("""
        DELETE FROM augmented_domains
        WHERE source = 'xgboost'
          AND domain_name_cos IS NOT NULL
          AND domain_name_cos < ?
    """, (floor,)).rowcount
    con.commit()
    remaining = con.execute(
        "SELECT COUNT(*) FROM augmented_domains WHERE source = 'xgboost'"
    ).fetchone()[0]
    print(f"  DNC floor ({floor}): deleted {deleted:,} xgboost rows")
    print(f"  Remaining xgboost rows: {remaining:,}")
    print(f"  ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Step 5: Ubiquity pruning
# ---------------------------------------------------------------------------

def step_ubiquity(con):
    """Prune/penalize XGBoost rows for words appearing in too many domains."""
    print(f"\nUbiquity pruning (words in {config.POLYSEMY_DOMAIN_THRESHOLD}+ domains)...")
    t_ub = time.time()

    # Temp table: word_ids appearing in too many domains (all sources)
    con.execute("DROP TABLE IF EXISTS _ub_wids")
    con.execute("""
        CREATE TEMP TABLE _ub_wids AS
        SELECT word_id FROM augmented_domains
        WHERE word_id IS NOT NULL
        GROUP BY word_id
        HAVING COUNT(DISTINCT domain) >= ?
    """, (config.POLYSEMY_DOMAIN_THRESHOLD,))
    n_ubiquitous = con.execute("SELECT COUNT(*) FROM _ub_wids").fetchone()[0]

    # Count before operations
    n_to_delete = con.execute("""
        SELECT COUNT(*) FROM augmented_domains
        WHERE source = 'xgboost' AND score < ?
          AND word_id IN (SELECT word_id FROM _ub_wids)
    """, (config.UBIQUITY_SCORE_FLOOR,)).fetchone()[0]

    n_to_penalize = con.execute("""
        SELECT COUNT(*) FROM augmented_domains
        WHERE source = 'xgboost'
          AND score >= ? AND score < ?
          AND word_id IN (SELECT word_id FROM _ub_wids)
    """, (config.UBIQUITY_SCORE_FLOOR, config.UBIQUITY_SCORE_CEILING)).fetchone()[0]

    # DELETE: xgboost rows with score < 0.80
    con.execute("""
        DELETE FROM augmented_domains
        WHERE source = 'xgboost' AND score < ?
          AND word_id IN (SELECT word_id FROM _ub_wids)
    """, (config.UBIQUITY_SCORE_FLOOR,))

    # PENALIZE: xgboost rows with 0.80 <= score < 0.95 -> score * 0.5
    con.execute("""
        UPDATE augmented_domains
        SET score = score * ?
        WHERE source = 'xgboost'
          AND score >= ? AND score < ?
          AND word_id IN (SELECT word_id FROM _ub_wids)
    """, (config.UBIQUITY_PENALTY,
          config.UBIQUITY_SCORE_FLOOR, config.UBIQUITY_SCORE_CEILING))

    con.execute("DROP TABLE _ub_wids")
    con.commit()

    remaining_ub = con.execute(
        "SELECT COUNT(*) FROM augmented_domains WHERE source = 'xgboost'"
    ).fetchone()[0]
    print(f"  Ubiquitous words: {n_ubiquitous:,}")
    print(f"  Deleted (score < {config.UBIQUITY_SCORE_FLOOR}): {n_to_delete:,}")
    print(f"  Penalized ({config.UBIQUITY_SCORE_FLOOR}–{config.UBIQUITY_SCORE_CEILING}"
          f" x{config.UBIQUITY_PENALTY}): {n_to_penalize:,}")
    print(f"  Remaining xgboost rows: {remaining_ub:,}")
    print(f"  ({time.time() - t_ub:.1f}s)")


# ---------------------------------------------------------------------------
# Step 6: Tag domainless words
# ---------------------------------------------------------------------------

def step_domainless(con):
    """Tag simple words with no domain association as 'domainless'."""
    print(f"\nTagging domainless words...")
    con.execute("DELETE FROM augmented_domains WHERE domain = 'domainless'")
    con.execute("""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Transform pipeline")
    parser.add_argument("domain", nargs="?", default=None,
                        help="Domain name (omit for full pipeline)")
    parser.add_argument("--threshold", type=float, default=config.XGBOOST_SCORE_THRESHOLD,
                        help=f"Score threshold (default: {config.XGBOOST_SCORE_THRESHOLD})")
    args = parser.parse_args()

    con = db.get_connection(DB_PATH)
    db.ensure_score_column(con)
    db.ensure_polysemy_column(con)
    db.ensure_domain_name_cos_column(con)

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
        exclude_sources={"xgboost"}, exclude_stop=True,
        exclude_polysemy=True
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

    del emb_map

    t_load = time.time() - t_start
    print(f"  Shared data loaded in {t_load:.1f}s\n")

    # --- Determine which domains to run ---
    if args.domain:
        if args.domain in config.XGBOOST_EXCLUDE_DOMAINS:
            print(f"ERROR: Domain '{args.domain}' is in XGBOOST_EXCLUDE_DOMAINS "
                  f"(stylistic/pragmatic category, curated seeds only).")
            con.close()
            sys.exit(1)
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
        excluded = {d for d in domains if d in config.XGBOOST_EXCLUDE_DOMAINS}
        target_domains = {d: wids for d, wids in domains.items()
                          if d not in config.XGBOOST_EXCLUDE_DOMAINS}
        if excluded:
            print(f"Excluding {len(excluded)} stylistic/pragmatic domains: "
                  f"{', '.join(sorted(excluded))}")

    # =====================================================================
    # Step 1: XGBoost training + inference
    # =====================================================================
    print(f"\n{'=' * 75}")
    print(f"[1/{TOTAL_STEPS}] XGBoost training + inference")
    print(f"{'=' * 75}")
    step_xgboost(con, target_domains, all_wids, X_global, raw_embeddings,
                  emb_normalized, wid_to_row, group_map, word_texts,
                  known_wids_by_domain, args.threshold)

    # Steps 2-9 only run for full pipeline (all domains)
    if args.domain:
        con.close()
        return

    # =====================================================================
    # Step 2: IDF post-process
    # =====================================================================
    print(f"\n{'=' * 75}")
    print(f"[2/{TOTAL_STEPS}] IDF score adjustment")
    print(f"{'=' * 75}")
    step_idf(con, args.threshold)

    # =====================================================================
    # Step 3: Domain-name cosine similarity
    # =====================================================================
    print(f"\n{'=' * 75}")
    print(f"[3/{TOTAL_STEPS}] Domain-name cosine similarity")
    print(f"{'=' * 75}")
    step_domain_name_cos(con, emb_normalized, wid_to_row)

    # =====================================================================
    # Step 4: DNC floor pruning
    # =====================================================================
    print(f"\n{'=' * 75}")
    print(f"[4/{TOTAL_STEPS}] DNC floor pruning")
    print(f"{'=' * 75}")
    step_dnc_floor(con)

    # =====================================================================
    # Step 5: Ubiquity pruning
    # =====================================================================
    print(f"\n{'=' * 75}")
    print(f"[5/{TOTAL_STEPS}] Ubiquity pruning")
    print(f"{'=' * 75}")
    step_ubiquity(con)

    # =====================================================================
    # Step 6: Tag domainless words
    # =====================================================================
    print(f"\n{'=' * 75}")
    print(f"[6/{TOTAL_STEPS}] Domainless tagging")
    print(f"{'=' * 75}")
    step_domainless(con)

    # Free XGBoost feature matrices before heavy clustering steps
    del X_global, raw_embeddings, emb_normalized

    # =====================================================================
    # Step 7: Reef clustering
    # =====================================================================
    print(f"\n{'=' * 75}")
    print(f"[7/{TOTAL_STEPS}] Reef clustering")
    print(f"{'=' * 75}")
    reef_pipeline.run_all(con)

    # =====================================================================
    # Step 8: Archipelago clustering
    # =====================================================================
    print(f"\n{'=' * 75}")
    print(f"[8/{TOTAL_STEPS}] Archipelago clustering")
    print(f"{'=' * 75}")
    arch_pipeline.run_all(con)

    # =====================================================================
    # Step 9: Scoring
    # =====================================================================
    print(f"\n{'=' * 75}")
    print(f"[9/{TOTAL_STEPS}] Scoring + verification")
    print(f"{'=' * 75}")
    score_pipeline.run_all(con)

    con.close()
    print(f"\nTransform pipeline complete.")


if __name__ == "__main__":
    main()
