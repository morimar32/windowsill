"""Archipelago clustering: group domains into higher-level topic families.

Uses hybrid similarity (embedding cosine + PMI normalization)
→ kNN graph → Leiden community detection.

Usage:
    python archipelago.py                     # run clustering
    python archipelago.py --resolution 1.5    # more/smaller archipelagos
    python archipelago.py --alpha 0.8         # more embedding weight
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

import config
from lib import db
from lib.db import pack_embedding, unpack_embedding
from lib.reef import compute_global_pmi
from lib.archipelago import compute_domain_embeddings, cluster_archipelagos


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2.db")


def load_reef_data(con):
    """Load reef centroids and sizes from domain_reef_stats.

    Returns:
        reef_centroids_by_domain: dict {domain: [(reef_id, centroid_vec), ...]}
        reef_sizes_by_domain: dict {domain: [(reef_id, n_total), ...]}
    """
    rows = con.execute(
        "SELECT domain, reef_id, n_total, centroid "
        "FROM domain_reef_stats WHERE centroid IS NOT NULL"
    ).fetchall()

    centroids_by_domain = defaultdict(list)
    sizes_by_domain = defaultdict(list)

    for domain, reef_id, n_total, centroid_blob in rows:
        vec = unpack_embedding(centroid_blob).astype(np.float64)
        centroids_by_domain[domain].append((reef_id, vec))
        sizes_by_domain[domain].append((reef_id, n_total))

    return dict(centroids_by_domain), dict(sizes_by_domain)


def load_pmi_data(con):
    """Load word-domain memberships and compute global PMI matrix.

    Returns:
        pmi_matrix: ndarray (n_domains, n_domains)
        domain_index: dict {domain_name: index}
    """
    rows = con.execute("""
        SELECT domain, word_id
        FROM augmented_domains
        WHERE word_id IS NOT NULL AND domain != 'domainless'
    """).fetchall()

    word_domains = defaultdict(set)
    all_domains = set()
    for domain, wid in rows:
        word_domains[wid].add(domain)
        all_domains.add(domain)

    all_domains = sorted(all_domains)
    pmi_matrix, domain_index = compute_global_pmi(dict(word_domains), all_domains)
    return pmi_matrix, domain_index


def persist_archipelagos(con, domain_names, result):
    """Write archipelago assignments and stats to DB (idempotent)."""
    con.execute("DELETE FROM domain_archipelagos")
    con.execute("DELETE FROM domain_archipelago_stats")

    # Domain assignments
    inserts = []
    for i, domain in enumerate(domain_names):
        arch_id = int(result["labels"][i])
        sim = float(result["domain_sims"][i]) if arch_id >= 0 else None
        inserts.append((domain, arch_id, sim))

    con.executemany(
        "INSERT INTO domain_archipelagos (domain, archipelago_id, centroid_sim) "
        "VALUES (?, ?, ?)",
        inserts
    )

    # Archipelago stats
    for stats in result["arch_stats"]:
        centroid_blob = (pack_embedding(stats["centroid"])
                         if stats["centroid"] is not None else None)
        con.execute(
            "INSERT INTO domain_archipelago_stats "
            "(archipelago_id, n_domains, label, top_domains, centroid) "
            "VALUES (?, ?, ?, ?, ?)",
            (stats["archipelago_id"], stats["n_domains"], stats["label"],
             json.dumps(stats["top_domains"]), centroid_blob)
        )

    con.commit()


def main():
    parser = argparse.ArgumentParser(
        description="Archipelago clustering: group domains into topic families"
    )
    parser.add_argument("--resolution", type=float,
                        default=config.ARCH_LEIDEN_RESOLUTION,
                        help=f"Leiden resolution (default: {config.ARCH_LEIDEN_RESOLUTION})")
    parser.add_argument("--alpha", type=float, default=config.ARCH_ALPHA,
                        help=f"Hybrid weight (default: {config.ARCH_ALPHA})")
    parser.add_argument("--knn-k", type=int, default=config.ARCH_KNN_K,
                        help=f"kNN neighbors (default: {config.ARCH_KNN_K})")
    args = parser.parse_args()

    con = db.get_connection(DB_PATH)
    db.create_archipelago_schema(con)

    # Step 1: Load reef centroids + sizes
    print("Loading reef data...")
    t0 = time.time()
    reef_centroids, reef_sizes = load_reef_data(con)
    n_reefs = sum(len(v) for v in reef_centroids.values())
    print(f"  {len(reef_centroids)} domains, {n_reefs:,} reef centroids "
          f"({time.time() - t0:.1f}s)")

    # Step 2: Load/compute PMI matrix
    print("Computing PMI matrix...")
    t0 = time.time()
    pmi_matrix, pmi_domain_index = load_pmi_data(con)
    n_nonzero = (pmi_matrix > 0).sum() // 2
    print(f"  {pmi_matrix.shape[0]}x{pmi_matrix.shape[1]} matrix, "
          f"{n_nonzero:,} positive pairs ({time.time() - t0:.1f}s)")

    # Step 3: Compute domain embeddings
    print("Computing domain embeddings...")
    t0 = time.time()
    domain_embeddings, domain_names = compute_domain_embeddings(
        reef_centroids, reef_sizes
    )
    print(f"  {len(domain_names)} domain embeddings ({time.time() - t0:.1f}s)")

    # Step 4: Cluster
    print(f"Clustering (alpha={args.alpha}, k={args.knn_k}, "
          f"resolution={args.resolution})...")
    t0 = time.time()
    result = cluster_archipelagos(
        domain_embeddings=domain_embeddings,
        pmi_matrix=pmi_matrix,
        domain_names=domain_names,
        pmi_domain_index=pmi_domain_index,
        alpha=args.alpha,
        knn_k=args.knn_k,
        resolution=args.resolution,
        min_community_size=config.ARCH_MIN_COMMUNITY_SIZE,
        characteristic_n=config.ARCH_CHARACTERISTIC_DOMAINS_N,
    )
    print(f"  Clustering done ({time.time() - t0:.1f}s)")

    # Step 5: Persist
    print("Persisting to database...")
    persist_archipelagos(con, domain_names, result)

    # Step 6: Print per-archipelago results
    print(f"\n{'─' * 85}")
    print(f"{'ID':>4}  {'N':>4}  {'Label':<40}  Top domains")
    print(f"{'─' * 85}")

    for stats in sorted(result["arch_stats"], key=lambda s: s["n_domains"],
                        reverse=True):
        top = ", ".join(stats["top_domains"][:5])
        print(f"{stats['archipelago_id']:>4}  {stats['n_domains']:>4}  "
              f"{stats['label']:<40}  [{top}]")

    # Noise domains
    n_noise = int((result["labels"] == -1).sum())
    if n_noise > 0:
        noise_domains = [domain_names[i] for i in range(len(domain_names))
                         if result["labels"][i] == -1]
        print(f"  -1  {n_noise:>4}  {'(noise)':<40}  "
              f"[{', '.join(noise_domains[:5])}]")

    # Step 7: Summary
    n_archs = len(result["arch_stats"])
    sizes = [s["n_domains"] for s in result["arch_stats"]]
    print(f"\n{'─' * 85}")
    print(f"Archipelagos: {n_archs}")
    if sizes:
        print(f"  Size distribution: min={min(sizes)} "
              f"median={sorted(sizes)[len(sizes)//2]} max={max(sizes)}")
    print(f"  Noise domains: {n_noise}")
    print(f"  Modularity: {result['modularity']:.3f}")

    con.close()


if __name__ == "__main__":
    main()
