"""Reef subdivision: cluster domain words into meaningful sub-groups.

Uses hybrid similarity (70% embedding cosine + 30% PMI co-membership)
→ kNN graph → Leiden community detection.

Usage:
    python reef.py                     # all domains
    python reef.py "medicine"          # single domain
    python reef.py --resolution 1.5    # custom Leiden resolution
    python reef.py --alpha 0.8         # more embedding weight
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

import config
from lib import db
from lib.db import pack_embedding, unpack_embedding
from lib.reef import compute_global_pmi, cluster_domain


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2.db")


def load_shared_data(con):
    """Load all shared data needed for reef clustering.

    Returns dict with: embeddings, word_ids, wid_to_row, word_texts,
    word_domains, all_domain_names, pmi_matrix, domain_index,
    domain_word_info.
    """
    print("Loading shared data...")
    t0 = time.time()

    # Load embeddings
    rows = con.execute(
        "SELECT word_id, embedding FROM words WHERE embedding IS NOT NULL"
    ).fetchall()
    emb_map = {}
    for wid, blob in rows:
        emb_map[wid] = unpack_embedding(blob).astype(np.float64)
    print(f"  {len(emb_map):,} embeddings loaded")

    # Load word texts
    word_texts = {}
    for wid, word in con.execute("SELECT word_id, word FROM words").fetchall():
        word_texts[wid] = word

    # Load all domain memberships (all sources) with scores
    ad_rows = con.execute("""
        SELECT domain, word_id, source, score
        FROM augmented_domains
        WHERE word_id IS NOT NULL AND domain != 'domainless'
    """).fetchall()

    # Build word_domains: {word_id: set of domains}
    word_domains = defaultdict(set)
    # Build per-domain word info: {domain: [(word_id, source, score), ...]}
    domain_word_info = defaultdict(list)
    all_domain_names = set()

    for domain, wid, source, score in ad_rows:
        word_domains[wid].add(domain)
        domain_word_info[domain].append((wid, source, score))
        all_domain_names.add(domain)

    all_domain_names = sorted(all_domain_names)
    print(f"  {len(all_domain_names)} domains, "
          f"{len(word_domains):,} words with domain memberships")

    # Compute global PMI matrix
    print("  Computing global PMI matrix...")
    pmi_matrix, domain_index = compute_global_pmi(
        dict(word_domains), all_domain_names
    )
    n_nonzero = (pmi_matrix > 0).sum() // 2
    print(f"  PMI matrix: {len(all_domain_names)}x{len(all_domain_names)}, "
          f"{n_nonzero:,} positive pairs")

    elapsed = time.time() - t0
    print(f"  Shared data loaded in {elapsed:.1f}s\n")

    return {
        "emb_map": emb_map,
        "word_texts": word_texts,
        "word_domains": dict(word_domains),
        "all_domain_names": all_domain_names,
        "pmi_matrix": pmi_matrix,
        "domain_index": domain_index,
        "domain_word_info": dict(domain_word_info),
    }


def select_core_and_non_core(domain, domain_word_info, emb_map,
                              score_threshold):
    """Split domain words into core (for clustering) and non-core (assigned after).

    Core = non-xgboost words OR xgboost with score >= threshold.
    Non-core = xgboost with score < threshold (but still has embedding).
    Words without embeddings are skipped entirely.

    Returns (core_wids, non_core_wids).
    """
    core_wids = []
    non_core_wids = []

    seen = set()
    for wid, source, score in domain_word_info.get(domain, []):
        if wid in seen or wid not in emb_map:
            continue
        seen.add(wid)

        if source == "xgboost":
            if score is not None and score >= score_threshold:
                core_wids.append(wid)
            else:
                non_core_wids.append(wid)
        else:
            # wordnet, claude_augmented, morphy, pipeline — always core
            core_wids.append(wid)

    return core_wids, non_core_wids


def persist_domain_reefs(con, domain, core_wids, non_core_wids,
                         result, word_texts):
    """Write reef assignments and stats to DB (idempotent: DELETE + INSERT)."""
    con.execute("DELETE FROM domain_reefs WHERE domain = ?", (domain,))
    con.execute("DELETE FROM domain_reef_stats WHERE domain = ?", (domain,))

    # Insert core word assignments
    core_inserts = []
    for i, wid in enumerate(core_wids):
        reef_id = int(result["core_labels"][i])
        sim = float(result["core_sims"][i]) if reef_id >= 0 else None
        core_inserts.append((
            domain, reef_id, wid, word_texts.get(wid, str(wid)),
            1, sim
        ))

    # Insert non-core word assignments
    nc_inserts = []
    for i, wid in enumerate(non_core_wids):
        reef_id = int(result["non_core_labels"][i])
        sim = float(result["non_core_sims"][i]) if reef_id >= 0 else None
        nc_inserts.append((
            domain, reef_id, wid, word_texts.get(wid, str(wid)),
            0, sim
        ))

    all_inserts = core_inserts + nc_inserts
    if all_inserts:
        con.executemany(
            "INSERT INTO domain_reefs "
            "(domain, reef_id, word_id, word, is_core, centroid_sim) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            all_inserts
        )

    # Insert reef stats
    for rs in result["reef_stats"]:
        centroid_blob = pack_embedding(rs["centroid"]) if rs["centroid"] is not None else None
        con.execute(
            "INSERT INTO domain_reef_stats "
            "(domain, reef_id, n_core, n_assigned, n_total, label, top_words, centroid) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (domain, rs["reef_id"], rs["n_core"], rs["n_assigned"],
             rs["n_total"], rs["label"], json.dumps(rs["top_words"]),
             centroid_blob)
        )

    con.commit()


def run_domain(domain, shared, con, alpha, resolution, min_community_size,
               score_threshold, min_domain_size, characteristic_n):
    """Cluster one domain's words into reefs."""
    t0 = time.time()

    emb_map = shared["emb_map"]
    word_texts = shared["word_texts"]
    word_domains = shared["word_domains"]
    pmi_matrix = shared["pmi_matrix"]
    domain_index = shared["domain_index"]
    domain_word_info = shared["domain_word_info"]

    # Split into core / non-core
    core_wids, non_core_wids = select_core_and_non_core(
        domain, domain_word_info, emb_map, score_threshold
    )

    if len(core_wids) < min_domain_size:
        # Tiny domain: all words → reef 0
        all_wids = core_wids + non_core_wids
        if not all_wids:
            return None

        con.execute("DELETE FROM domain_reefs WHERE domain = ?", (domain,))
        con.execute("DELETE FROM domain_reef_stats WHERE domain = ?", (domain,))

        inserts = []
        for wid in all_wids:
            inserts.append((domain, 0, wid, word_texts.get(wid, str(wid)), 1, None))
        con.executemany(
            "INSERT INTO domain_reefs "
            "(domain, reef_id, word_id, word, is_core, centroid_sim) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            inserts
        )

        # Stats for the single reef
        top = [word_texts.get(wid, str(wid)) for wid in all_wids[:characteristic_n]]
        embeddings = np.vstack([emb_map[wid] for wid in all_wids])
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        con.execute(
            "INSERT INTO domain_reef_stats "
            "(domain, reef_id, n_core, n_assigned, n_total, label, top_words, centroid) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (domain, 0, len(all_wids), 0, len(all_wids),
             "_".join(top[:3]), json.dumps(top), pack_embedding(centroid))
        )
        con.commit()

        elapsed = time.time() - t0
        print(f"  {domain:<30} reefs=1  core={len(all_wids)}  "
              f"assigned=0  noise=0  (tiny, {elapsed:.1f}s)")
        return {"n_reefs": 1, "n_core": len(all_wids),
                "n_assigned": 0, "n_noise": 0}

    # Build embedding matrices
    core_embeddings = np.vstack([emb_map[wid] for wid in core_wids])
    non_core_embeddings = (np.vstack([emb_map[wid] for wid in non_core_wids])
                           if non_core_wids else np.empty((0, core_embeddings.shape[1])))

    # Run clustering
    result = cluster_domain(
        core_embeddings=core_embeddings,
        core_word_ids=core_wids,
        core_word_domains=word_domains,
        non_core_embeddings=non_core_embeddings,
        non_core_word_ids=non_core_wids,
        parent_domain=domain,
        pmi_matrix=pmi_matrix,
        domain_index=domain_index,
        alpha=alpha,
        knn_k=config.REEF_KNN_K,
        resolution=resolution,
        min_community_size=min_community_size,
        characteristic_n=characteristic_n,
        word_texts=word_texts,
    )

    # Persist
    persist_domain_reefs(con, domain, core_wids, non_core_wids,
                         result, word_texts)

    # Stats
    n_reefs = len(result["reef_stats"])
    n_core = len(core_wids)
    n_assigned = int((result["non_core_labels"] >= 0).sum())
    n_noise_core = int((result["core_labels"] == -1).sum())
    n_noise_nc = int((result["non_core_labels"] == -1).sum())
    n_noise = n_noise_core + n_noise_nc
    mod = result["modularity"]

    elapsed = time.time() - t0
    print(f"  {domain:<30} reefs={n_reefs}  core={n_core}  "
          f"assigned={n_assigned}  noise={n_noise}  "
          f"modularity={mod:.2f}  ({elapsed:.1f}s)")

    return {"n_reefs": n_reefs, "n_core": n_core,
            "n_assigned": n_assigned, "n_noise": n_noise}


def main():
    parser = argparse.ArgumentParser(
        description="Reef subdivision: cluster domain words into sub-groups"
    )
    parser.add_argument("domain", nargs="?", default=None,
                        help="Domain name (omit for all domains)")
    parser.add_argument("--resolution", type=float,
                        default=config.REEF_LEIDEN_RESOLUTION,
                        help=f"Leiden resolution (default: {config.REEF_LEIDEN_RESOLUTION})")
    parser.add_argument("--alpha", type=float, default=config.REEF_ALPHA,
                        help=f"Hybrid weight (default: {config.REEF_ALPHA})")
    parser.add_argument("--score-threshold", type=float,
                        default=config.REEF_SCORE_THRESHOLD,
                        help=f"Min xgboost score for core (default: {config.REEF_SCORE_THRESHOLD})")
    args = parser.parse_args()

    con = db.get_connection(DB_PATH)
    db.create_reef_schema(con)

    # Load shared data
    shared = load_shared_data(con)

    # Determine target domains
    all_domains = sorted(shared["domain_word_info"].keys())

    if args.domain:
        if args.domain not in shared["domain_word_info"]:
            print(f"ERROR: Domain '{args.domain}' not found.")
            available = all_domains[:20]
            print(f"Available domains ({len(all_domains)}):")
            for d in available:
                n = len(shared["domain_word_info"][d])
                print(f"  {d} ({n:,} words)")
            if len(all_domains) > 20:
                print(f"  ... and {len(all_domains) - 20} more")
            con.close()
            sys.exit(1)
        target_domains = [args.domain]
    else:
        target_domains = all_domains

    print(f"Clustering {len(target_domains)} domain(s), "
          f"alpha={args.alpha}, resolution={args.resolution}, "
          f"score_threshold={args.score_threshold}")
    print(f"{'─' * 85}")

    # Run per domain
    t_start = time.time()
    stats = []
    for domain in target_domains:
        s = run_domain(
            domain, shared, con,
            alpha=args.alpha,
            resolution=args.resolution,
            min_community_size=config.REEF_MIN_COMMUNITY_SIZE,
            score_threshold=args.score_threshold,
            min_domain_size=config.REEF_MIN_DOMAIN_SIZE,
            characteristic_n=config.REEF_CHARACTERISTIC_WORDS_N,
        )
        if s:
            stats.append(s)

    elapsed_total = time.time() - t_start
    print(f"{'─' * 85}")

    # Summary
    if stats:
        total_reefs = sum(s["n_reefs"] for s in stats)
        avg_reefs = total_reefs / len(stats)
        total_noise = sum(s["n_noise"] for s in stats)
        reef_counts = [s["n_reefs"] for s in stats]

        print(f"Done. {len(stats)} domains in {elapsed_total:.1f}s")
        print(f"  Total reefs: {total_reefs:,}")
        print(f"  Average reefs/domain: {avg_reefs:.1f}")
        print(f"  Reef count distribution: "
              f"min={min(reef_counts)} median={sorted(reef_counts)[len(reef_counts)//2]} "
              f"max={max(reef_counts)}")
        print(f"  Total noise words: {total_noise:,}")

        # DB totals
        db_reefs = con.execute(
            "SELECT COUNT(DISTINCT domain || '|' || reef_id) "
            "FROM domain_reefs WHERE reef_id >= 0"
        ).fetchone()[0]
        db_words = con.execute(
            "SELECT COUNT(*) FROM domain_reefs"
        ).fetchone()[0]
        print(f"  DB totals: {db_reefs:,} reefs, {db_words:,} word assignments")
    else:
        print("No domains processed.")

    con.close()


if __name__ == "__main__":
    main()
