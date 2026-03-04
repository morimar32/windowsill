"""Compute word-level statistics: specificity, IDF, island concentration.

Populates:
  - Words: reef_count, town_count, island_count, idf, specificity, cosine_sim
  - WordIslandStats: per (word, island) concentration and cosine

Pipeline position: runs AFTER clustering (cluster_reefs.py) and
IDF post-processing (post_process_xgb.py).

Usage:
    python v3/compute_word_stats.py
    python v3/compute_word_stats.py --dry-run
"""

import math
import os
import sys
import time

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_DB = os.path.join(_project, "v3/windowsill.db")

# Specificity categories based on reef count
# Higher = more specific (useful for discrimination)
SPECIFICITY_THRESHOLDS = [
    (1,   3),   # 1 reef: highly specific
    (3,   2),   # 2-3 reefs: very specific
    (5,   1),   # 4-5 reefs: specific
    (10,  0),   # 6-10 reefs: moderate
    (20, -1),   # 11-20 reefs: generic
]
SPECIFICITY_DEFAULT = -2  # 21+ reefs: very generic

IDF_FLOOR = 0.1


def compute_specificity(n_reefs):
    """Map reef count to specificity category."""
    for threshold, value in SPECIFICITY_THRESHOLDS:
        if n_reefs <= threshold:
            return value
    return SPECIFICITY_DEFAULT


def main():
    import sqlite3

    import argparse
    parser = argparse.ArgumentParser(description="Compute word-level stats")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    # -------------------------------------------------------------------
    # Step 0: Ensure columns exist
    # -------------------------------------------------------------------
    existing_cols = {row[1] for row in con.execute("PRAGMA table_info(Words)").fetchall()}
    for col, coltype in [("reef_count", "INTEGER DEFAULT 0"),
                         ("town_count", "INTEGER DEFAULT 0"),
                         ("island_count", "INTEGER DEFAULT 0")]:
        if col not in existing_cols:
            con.execute(f"ALTER TABLE Words ADD COLUMN {col} {coltype}")
            print(f"  Added Words.{col}")
    con.commit()

    # -------------------------------------------------------------------
    # Step 1: Compute global word stats (reef_count, town_count, island_count)
    # -------------------------------------------------------------------
    print("Step 1: Computing global word stats...")
    t0 = time.time()

    con.execute("DROP TABLE IF EXISTS _word_counts")
    con.execute("""
        CREATE TEMP TABLE _word_counts AS
        SELECT rw.word_id,
            COUNT(DISTINCT rw.reef_id) as reef_count,
            COUNT(DISTINCT r.town_id) as town_count,
            COUNT(DISTINCT t.island_id) as island_count,
            MAX(rw.cosine_sim) as best_cosine
        FROM ReefWords rw
        JOIN Reefs r USING(reef_id)
        JOIN Towns t USING(town_id)
        GROUP BY rw.word_id
    """)
    con.execute("CREATE INDEX _wc_idx ON _word_counts(word_id)")

    n_with_reefs = con.execute("SELECT COUNT(*) FROM _word_counts").fetchone()[0]
    total_reefs = con.execute("SELECT COUNT(*) FROM Reefs").fetchone()[0]
    print(f"  {n_with_reefs:,} words in reefs, {total_reefs:,} total reefs ({time.time()-t0:.1f}s)")

    if args.dry_run:
        # Show distribution preview
        for label, lo, hi in [("1 reef", 1, 1), ("2-3", 2, 3), ("4-5", 4, 5),
                               ("6-10", 6, 10), ("11-20", 11, 20), ("21+", 21, 9999)]:
            n = con.execute(
                "SELECT COUNT(*) FROM _word_counts WHERE reef_count BETWEEN ? AND ?",
                (lo, hi)
            ).fetchone()[0]
            print(f"    {label}: {n:,}")
        con.close()
        return

    # -------------------------------------------------------------------
    # Step 2: Update Words table
    # -------------------------------------------------------------------
    print("Step 2: Updating Words table...")
    t0 = time.time()

    # Batch fetch the computed stats
    rows = con.execute("""
        SELECT word_id, reef_count, town_count, island_count, best_cosine
        FROM _word_counts
    """).fetchall()

    # Compute IDF and specificity in Python, batch update
    updates = []
    for word_id, reef_count, town_count, island_count, best_cosine in rows:
        idf = max(math.log2(total_reefs / reef_count), IDF_FLOOR) if reef_count > 0 else 0.0
        spec = compute_specificity(reef_count)
        updates.append((reef_count, town_count, island_count, idf, spec, best_cosine, word_id))

    con.executemany("""
        UPDATE Words SET
            reef_count = ?, town_count = ?, island_count = ?,
            idf = ?, specificity = ?, cosine_sim = ?
        WHERE word_id = ?
    """, updates)

    # Words not in any reef get zeroes
    con.execute("""
        UPDATE Words SET reef_count = 0, town_count = 0, island_count = 0,
                         idf = 0, specificity = NULL, cosine_sim = NULL
        WHERE word_id NOT IN (SELECT word_id FROM _word_counts)
    """)
    con.commit()

    print(f"  Updated {len(updates):,} words ({time.time()-t0:.1f}s)")

    # -------------------------------------------------------------------
    # Step 3: Create and populate WordIslandStats
    # -------------------------------------------------------------------
    print("Step 3: Computing island concentration...")
    t0 = time.time()

    con.execute("""
        CREATE TABLE IF NOT EXISTS WordIslandStats (
            word_id         INTEGER NOT NULL REFERENCES Words(word_id),
            island_id       INTEGER NOT NULL REFERENCES Islands(island_id),
            reef_count      INTEGER NOT NULL,
            town_count      INTEGER NOT NULL,
            concentration   REAL    NOT NULL,
            avg_cosine      REAL,
            PRIMARY KEY (word_id, island_id)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_wis_island ON WordIslandStats(island_id)")
    con.execute("DELETE FROM WordIslandStats")

    # Compute per (word, island) stats
    con.execute("""
        INSERT INTO WordIslandStats (word_id, island_id, reef_count, town_count, concentration, avg_cosine)
        SELECT
            rw.word_id,
            t.island_id,
            COUNT(DISTINCT rw.reef_id) as reef_count,
            COUNT(DISTINCT r.town_id) as town_count,
            CAST(COUNT(DISTINCT rw.reef_id) AS REAL) / wc.reef_count as concentration,
            AVG(rw.cosine_sim) as avg_cosine
        FROM ReefWords rw
        JOIN Reefs r USING(reef_id)
        JOIN Towns t USING(town_id)
        JOIN _word_counts wc USING(word_id)
        GROUP BY rw.word_id, t.island_id
    """)
    con.commit()

    n_wis = con.execute("SELECT COUNT(*) FROM WordIslandStats").fetchone()[0]
    print(f"  {n_wis:,} (word, island) pairs ({time.time()-t0:.1f}s)")

    # -------------------------------------------------------------------
    # Step 4: Summary stats
    # -------------------------------------------------------------------
    print("\n--- Specificity distribution ---")
    for row in con.execute("""
        SELECT specificity, COUNT(*) as n,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM Words WHERE reef_count > 0), 1) as pct
        FROM Words
        WHERE specificity IS NOT NULL
        GROUP BY specificity
        ORDER BY specificity DESC
    """).fetchall():
        labels = {3: "highly specific (1 reef)", 2: "very specific (2-3)",
                  1: "specific (4-5)", 0: "moderate (6-10)",
                  -1: "generic (11-20)", -2: "very generic (21+)"}
        print(f"  {row[0]:>3}: {row[1]:>8,} words ({row[2]:>5}%)  {labels.get(row[0], '')}")

    print(f"\n--- IDF distribution ---")
    for row in con.execute("""
        SELECT
          CASE
            WHEN idf >= 11 THEN '11+ (1 reef)'
            WHEN idf >= 10 THEN '10-11'
            WHEN idf >= 8  THEN '8-10'
            WHEN idf >= 6  THEN '6-8'
            ELSE '<6'
          END as bucket,
          COUNT(*) as n
        FROM Words WHERE idf > 0
        GROUP BY bucket
        ORDER BY MIN(idf) DESC
    """).fetchall():
        print(f"  {row[0]:>15}: {row[1]:>8,}")

    print(f"\n--- Island concentration distribution ---")
    for row in con.execute("""
        SELECT
          CASE
            WHEN concentration >= 0.9 THEN '90-100%'
            WHEN concentration >= 0.7 THEN '70-90%'
            WHEN concentration >= 0.5 THEN '50-70%'
            WHEN concentration >= 0.3 THEN '30-50%'
            ELSE '<30%'
          END as bucket,
          COUNT(*) as n,
          ROUND(AVG(avg_cosine), 4) as mean_cosine
        FROM WordIslandStats
        GROUP BY bucket
        ORDER BY MIN(concentration) DESC
    """).fetchall():
        print(f"  {row[0]:>10}: {row[1]:>8,} pairs  (avg cosine: {row[2]})")

    print(f"\n--- Words with no reef assignments ---")
    n_no_reefs = con.execute("SELECT COUNT(*) FROM Words WHERE reef_count = 0").fetchone()[0]
    print(f"  {n_no_reefs:,} / {158060:,} ({n_no_reefs/158060*100:.1f}%)")

    con.execute("DROP TABLE IF EXISTS _word_counts")
    con.commit()
    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
