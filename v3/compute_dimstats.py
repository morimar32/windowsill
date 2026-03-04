"""Compute per-dimension embedding statistics for z-score features.

Populates DimStats table with mean, std, threshold, member_count,
and selectivity for each of the 768 embedding dimensions.

Used by train_town_xgboost.py for z-score normalization of embeddings.

No dependency on v2 database — computes from Words.embedding directly.

Pipeline position: runs AFTER load_wordnet_vocab.py + reembed_words.py
(embeddings must exist in Words table), BEFORE train_town_xgboost.py.

Usage:
    python v3/compute_dimstats.py --dry-run
    python v3/compute_dimstats.py
"""

import os
import struct
import sys
import time

import numpy as np

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_DB = os.path.join(_project, "v3/windowsill.db")

EMBEDDING_DIM = 768
THRESHOLD_SIGMA = 2.0  # threshold = mean + THRESHOLD_SIGMA * std


def unpack_embedding(blob):
    return np.array(struct.unpack(f"{EMBEDDING_DIM}f", blob), dtype=np.float64)


def main():
    import argparse
    import sqlite3

    parser = argparse.ArgumentParser(description="Compute DimStats from embeddings")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats without writing to database")
    args = parser.parse_args()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    # Load all embeddings
    print("Loading embeddings...")
    t0 = time.time()
    rows = con.execute(
        "SELECT word_id, embedding FROM Words WHERE embedding IS NOT NULL"
    ).fetchall()

    n_words = len(rows)
    if n_words == 0:
        print("ERROR: No words with embeddings found. Run reembed_words.py first.")
        con.close()
        sys.exit(1)

    print(f"  {n_words:,} words with embeddings ({time.time()-t0:.1f}s)")

    # Build matrix
    print("Building embedding matrix...")
    t0 = time.time()
    matrix = np.vstack([unpack_embedding(blob) for _, blob in rows])
    print(f"  Shape: {matrix.shape} ({time.time()-t0:.1f}s)")

    # Compute per-dimension stats
    print("Computing per-dimension statistics...")
    t0 = time.time()

    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    thresholds = means + THRESHOLD_SIGMA * stds

    # member_count: words above threshold per dimension
    member_counts = (matrix > thresholds).sum(axis=0)
    selectivities = member_counts / n_words

    print(f"  Computed stats for {EMBEDDING_DIM} dimensions ({time.time()-t0:.1f}s)")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  mean(mean):         {means.mean():.6f}")
    print(f"  mean(std):          {stds.mean():.6f}")
    print(f"  mean(threshold):    {thresholds.mean():.6f}")
    print(f"  mean(member_count): {member_counts.mean():.1f}")
    print(f"  mean(selectivity):  {selectivities.mean():.6f}")
    print(f"  min(std):           {stds.min():.6f}")
    print(f"  max(std):           {stds.max():.6f}")

    # Distribution of member_counts
    print(f"\n--- Member count distribution ---")
    for lo, hi in [(0, 1000), (1000, 2000), (2000, 3000),
                   (3000, 5000), (5000, 10000), (10000, n_words)]:
        n = int(((member_counts >= lo) & (member_counts < hi)).sum())
        print(f"  {lo:>6}-{hi:<6}: {n:>4} dimensions")

    if args.dry_run:
        print(f"\nDry run — would write {EMBEDDING_DIM} rows to DimStats.")
        con.close()
        return

    # Write to database
    print(f"\nWriting DimStats...")
    t0 = time.time()

    con.execute("DELETE FROM DimStats")

    dim_rows = []
    for d in range(EMBEDDING_DIM):
        dim_rows.append((
            d,
            float(means[d]),
            float(stds[d]),
            float(thresholds[d]),
            int(member_counts[d]),
            float(selectivities[d]),
        ))

    con.executemany("""
        INSERT INTO DimStats (dim_id, mean, std, threshold, member_count, selectivity)
        VALUES (?, ?, ?, ?, ?, ?)
    """, dim_rows)
    con.commit()

    # Verify
    count = con.execute("SELECT COUNT(*) FROM DimStats").fetchone()[0]
    print(f"  Wrote {count} rows ({time.time()-t0:.1f}s)")

    sample = con.execute(
        "SELECT dim_id, mean, std, threshold, member_count, selectivity FROM DimStats LIMIT 3"
    ).fetchall()
    for row in sample:
        print(f"  dim {row[0]}: mean={row[1]:.4f}, std={row[2]:.4f}, "
              f"threshold={row[3]:.4f}, members={row[4]}, sel={row[5]:.4f}")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
