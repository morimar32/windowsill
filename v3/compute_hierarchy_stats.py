"""Compute hierarchy statistics: propagate stats bottom-up from words to archipelagos.

Populates:
  - ReefWords: pos, specificity, idf (from Words), island_idf, source_quality
  - Reefs: word_count, core_word_count, noun_frac, verb_frac, adj_frac, adv_frac, avg_specificity
  - Towns: reef_count, word_count, noun_frac, verb_frac, adj_frac, adv_frac, avg_specificity
  - Islands: town_count, reef_count, word_count, noun_frac, ..., avg_specificity
  - Archipelagos: island_count, town_count, reef_count, word_count

Pipeline position: step 17 (final), runs AFTER compute_word_stats.py.
Fully idempotent — unconditional overwrites on every run.

Usage:
    python v3/compute_hierarchy_stats.py
    python v3/compute_hierarchy_stats.py --dry-run
"""

import math
import os
import time

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_DB = os.path.join(_project, "v3/windowsill.db")


def main():
    import argparse
    import sqlite3

    parser = argparse.ArgumentParser(description="Compute hierarchy stats (bottom-up)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show current NULL counts without writing")
    args = parser.parse_args()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    if args.dry_run:
        print("--- Current NULL counts ---")
        checks = [
            ("ReefWords with NULL pos",        "SELECT COUNT(*) FROM ReefWords WHERE pos IS NULL"),
            ("ReefWords with NULL specificity", "SELECT COUNT(*) FROM ReefWords WHERE specificity IS NULL"),
            ("ReefWords with NULL idf",         "SELECT COUNT(*) FROM ReefWords WHERE idf IS NULL"),
            ("ReefWords with NULL island_idf",  "SELECT COUNT(*) FROM ReefWords WHERE island_idf IS NULL"),
            ("Reefs with NULL noun_frac",       "SELECT COUNT(*) FROM Reefs WHERE noun_frac IS NULL"),
            ("Towns with NULL noun_frac",       "SELECT COUNT(*) FROM Towns WHERE noun_frac IS NULL"),
            ("Islands with NULL noun_frac",     "SELECT COUNT(*) FROM Islands WHERE noun_frac IS NULL"),
            ("Archipelagos with 0 island_count","SELECT COUNT(*) FROM Archipelagos WHERE island_count = 0"),
        ]
        for label, query in checks:
            n = con.execute(query).fetchone()[0]
            print(f"  {label:40s} {n:,}")

        print(f"\n--- source_quality distribution ---")
        for row in con.execute("""
            SELECT source, is_core, COUNT(*), ROUND(AVG(source_quality), 2)
            FROM ReefWords GROUP BY source, is_core ORDER BY source, is_core DESC
        """).fetchall():
            print(f"  source={row[0]:20s}  is_core={row[1]}  count={row[2]:>8,}  avg_quality={row[3]}")

        con.close()
        return

    # -------------------------------------------------------------------
    # Step 1: ReefWords contextual stats (pos, specificity, idf from Words)
    # -------------------------------------------------------------------
    print("Step 1: Copying pos/specificity/idf from Words → ReefWords...")
    t0 = time.time()

    con.execute("""
        UPDATE ReefWords
        SET pos = w.pos, specificity = w.specificity, idf = w.idf
        FROM Words w
        WHERE ReefWords.word_id = w.word_id
    """)
    n_updated = con.execute("SELECT changes()").fetchone()[0]
    con.commit()
    print(f"  Updated {n_updated:,} rows ({time.time()-t0:.1f}s)")

    # -------------------------------------------------------------------
    # Step 1b: ReefWords island_idf
    # -------------------------------------------------------------------
    print("Step 1b: Computing island_idf for ReefWords...")
    t0 = time.time()

    # Ensure column exists (for DBs created before schema change)
    existing_cols = {row[1] for row in con.execute("PRAGMA table_info(ReefWords)").fetchall()}
    if "island_idf" not in existing_cols:
        con.execute("ALTER TABLE ReefWords ADD COLUMN island_idf REAL")
        print("  Added ReefWords.island_idf column")
        con.commit()

    # Per-(word, island) reef counts
    con.execute("DROP TABLE IF EXISTS _island_word_reefs")
    con.execute("""
        CREATE TEMP TABLE _island_word_reefs AS
        SELECT rw.word_id, t.island_id,
            COUNT(DISTINCT rw.reef_id) AS word_reefs_in_island
        FROM ReefWords rw
        JOIN Reefs r USING (reef_id)
        JOIN Towns t USING (town_id)
        GROUP BY rw.word_id, t.island_id
    """)

    # Per-island total reef counts
    con.execute("DROP TABLE IF EXISTS _island_reef_totals")
    con.execute("""
        CREATE TEMP TABLE _island_reef_totals AS
        SELECT t.island_id, COUNT(DISTINCT r.reef_id) AS total_reefs
        FROM Reefs r JOIN Towns t USING (town_id)
        GROUP BY t.island_id
    """)

    # Join ReefWords with island info to get (reef_id, word_id, total_reefs, word_reefs)
    rows = con.execute("""
        SELECT rw.reef_id, rw.word_id, irt.total_reefs, iwr.word_reefs_in_island
        FROM ReefWords rw
        JOIN Reefs r USING (reef_id)
        JOIN Towns t USING (town_id)
        JOIN _island_reef_totals irt ON t.island_id = irt.island_id
        JOIN _island_word_reefs iwr ON rw.word_id = iwr.word_id
                                    AND t.island_id = iwr.island_id
    """).fetchall()

    # Compute log2 in Python and batch-update
    updates = [
        (math.log2(total / word_reefs), reef_id, word_id)
        for reef_id, word_id, total, word_reefs in rows
    ]
    con.executemany(
        "UPDATE ReefWords SET island_idf = ? WHERE reef_id = ? AND word_id = ?",
        updates
    )
    con.commit()

    con.execute("DROP TABLE IF EXISTS _island_word_reefs")
    con.execute("DROP TABLE IF EXISTS _island_reef_totals")

    print(f"  Updated {len(updates):,} rows ({time.time()-t0:.1f}s)")

    # -------------------------------------------------------------------
    # Step 2: ReefWords source_quality
    # -------------------------------------------------------------------
    print("Step 2: Setting source_quality based on (source, is_core)...")
    t0 = time.time()

    # Curated sources (wordnet, claude_augmented, etc.) → 1.0
    con.execute("""
        UPDATE ReefWords SET source_quality = 1.0
        WHERE source != 'xgboost'
    """)
    n_curated = con.execute("SELECT changes()").fetchone()[0]

    # xgboost core → 0.9
    con.execute("""
        UPDATE ReefWords SET source_quality = 0.9
        WHERE source = 'xgboost' AND is_core = 1
    """)
    n_xgb_core = con.execute("SELECT changes()").fetchone()[0]

    # xgboost non-core → 0.7
    con.execute("""
        UPDATE ReefWords SET source_quality = 0.7
        WHERE source = 'xgboost' AND is_core = 0
    """)
    n_xgb_noncore = con.execute("SELECT changes()").fetchone()[0]

    con.commit()
    print(f"  curated→1.0: {n_curated:,}, xgboost core→0.9: {n_xgb_core:,}, xgboost non-core→0.7: {n_xgb_noncore:,} ({time.time()-t0:.1f}s)")

    # -------------------------------------------------------------------
    # Step 3: Reef aggregates
    # -------------------------------------------------------------------
    print("Step 3: Computing Reef aggregates...")
    t0 = time.time()

    con.execute("""
        UPDATE Reefs SET
            word_count      = sub.word_count,
            core_word_count = sub.core_word_count,
            noun_frac       = sub.noun_frac,
            verb_frac       = sub.verb_frac,
            adj_frac        = sub.adj_frac,
            adv_frac        = sub.adv_frac,
            avg_specificity = sub.avg_spec
        FROM (
            SELECT rw.reef_id,
                COUNT(*)       AS word_count,
                SUM(rw.is_core) AS core_word_count,
                CAST(SUM(CASE WHEN w.pos = 'noun' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS noun_frac,
                CAST(SUM(CASE WHEN w.pos = 'verb' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS verb_frac,
                CAST(SUM(CASE WHEN w.pos = 'adj'  THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS adj_frac,
                CAST(SUM(CASE WHEN w.pos = 'adv'  THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS adv_frac,
                AVG(w.specificity) AS avg_spec
            FROM ReefWords rw
            JOIN Words w USING (word_id)
            GROUP BY rw.reef_id
        ) sub
        WHERE Reefs.reef_id = sub.reef_id
    """)
    n_reefs = con.execute("SELECT changes()").fetchone()[0]
    con.commit()
    print(f"  Updated {n_reefs:,} reefs ({time.time()-t0:.1f}s)")

    # -------------------------------------------------------------------
    # Step 4: Town aggregates (unique words per town)
    # -------------------------------------------------------------------
    print("Step 4: Computing Town aggregates (deduped words)...")
    t0 = time.time()

    con.execute("""
        UPDATE Towns SET
            reef_count      = sub.reef_count,
            word_count      = sub.word_count,
            noun_frac       = sub.noun_frac,
            verb_frac       = sub.verb_frac,
            adj_frac        = sub.adj_frac,
            adv_frac        = sub.adv_frac,
            avg_specificity = sub.avg_spec
        FROM (
            SELECT town_id,
                COUNT(DISTINCT reef_id) AS reef_count,
                COUNT(*) AS word_count,
                CAST(SUM(CASE WHEN pos = 'noun' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS noun_frac,
                CAST(SUM(CASE WHEN pos = 'verb' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS verb_frac,
                CAST(SUM(CASE WHEN pos = 'adj'  THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS adj_frac,
                CAST(SUM(CASE WHEN pos = 'adv'  THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS adv_frac,
                AVG(specificity) AS avg_spec
            FROM (
                SELECT r.town_id, r.reef_id, rw.word_id, w.pos, w.specificity
                FROM ReefWords rw
                JOIN Reefs r USING (reef_id)
                JOIN Words w USING (word_id)
                GROUP BY r.town_id, rw.word_id
            )
            GROUP BY town_id
        ) sub
        WHERE Towns.town_id = sub.town_id
    """)
    n_towns = con.execute("SELECT changes()").fetchone()[0]
    con.commit()
    print(f"  Updated {n_towns:,} towns ({time.time()-t0:.1f}s)")

    # -------------------------------------------------------------------
    # Step 5: Island aggregates (unique words per island)
    # -------------------------------------------------------------------
    print("Step 5: Computing Island aggregates (deduped words)...")
    t0 = time.time()

    con.execute("""
        UPDATE Islands SET
            town_count      = sub.town_count,
            reef_count      = sub.reef_count,
            word_count      = sub.word_count,
            noun_frac       = sub.noun_frac,
            verb_frac       = sub.verb_frac,
            adj_frac        = sub.adj_frac,
            adv_frac        = sub.adv_frac,
            avg_specificity = sub.avg_spec
        FROM (
            SELECT island_id,
                COUNT(DISTINCT town_id) AS town_count,
                COUNT(DISTINCT reef_id) AS reef_count,
                COUNT(*) AS word_count,
                CAST(SUM(CASE WHEN pos = 'noun' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS noun_frac,
                CAST(SUM(CASE WHEN pos = 'verb' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS verb_frac,
                CAST(SUM(CASE WHEN pos = 'adj'  THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS adj_frac,
                CAST(SUM(CASE WHEN pos = 'adv'  THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS adv_frac,
                AVG(specificity) AS avg_spec
            FROM (
                SELECT t.island_id, r.town_id, r.reef_id, rw.word_id, w.pos, w.specificity
                FROM ReefWords rw
                JOIN Reefs r USING (reef_id)
                JOIN Towns t USING (town_id)
                JOIN Words w USING (word_id)
                GROUP BY t.island_id, rw.word_id
            )
            GROUP BY island_id
        ) sub
        WHERE Islands.island_id = sub.island_id
    """)
    n_islands = con.execute("SELECT changes()").fetchone()[0]
    con.commit()
    print(f"  Updated {n_islands:,} islands ({time.time()-t0:.1f}s)")

    # -------------------------------------------------------------------
    # Step 6: Archipelago counts
    # -------------------------------------------------------------------
    print("Step 6: Computing Archipelago counts...")
    t0 = time.time()

    con.execute("""
        UPDATE Archipelagos SET
            island_count = (
                SELECT COUNT(*) FROM Islands
                WHERE Islands.archipelago_id = Archipelagos.archipelago_id
            ),
            town_count = (
                SELECT COUNT(*) FROM Towns
                JOIN Islands USING (island_id)
                WHERE Islands.archipelago_id = Archipelagos.archipelago_id
            ),
            reef_count = (
                SELECT COUNT(*) FROM Reefs
                JOIN Towns USING (town_id)
                JOIN Islands USING (island_id)
                WHERE Islands.archipelago_id = Archipelagos.archipelago_id
            ),
            word_count = (
                SELECT COUNT(DISTINCT rw.word_id) FROM ReefWords rw
                JOIN Reefs USING (reef_id)
                JOIN Towns USING (town_id)
                JOIN Islands USING (island_id)
                WHERE Islands.archipelago_id = Archipelagos.archipelago_id
            )
    """)
    con.commit()
    print(f"  Done ({time.time()-t0:.1f}s)")

    # -------------------------------------------------------------------
    # Step 7: Summary report
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # source_quality distribution
    print("\n--- source_quality distribution ---")
    for row in con.execute("""
        SELECT source, is_core, COUNT(*), ROUND(AVG(source_quality), 2)
        FROM ReefWords GROUP BY source, is_core ORDER BY source, is_core DESC
    """).fetchall():
        print(f"  source={row[0]:20s}  is_core={row[1]}  count={row[2]:>8,}  avg_quality={row[3]}")

    # POS distribution (from ReefWords)
    print("\n--- POS distribution (ReefWords) ---")
    total_rw = con.execute("SELECT COUNT(*) FROM ReefWords").fetchone()[0]
    for row in con.execute("""
        SELECT pos, COUNT(*),
               ROUND(100.0 * COUNT(*) / ?, 1)
        FROM ReefWords GROUP BY pos ORDER BY COUNT(*) DESC
    """, (total_rw,)).fetchall():
        print(f"  {str(row[0]):>4s}: {row[1]:>8,} ({row[2]:>5}%)")

    # Reef size distribution
    print("\n--- Reef size distribution ---")
    for row in con.execute("""
        SELECT
          CASE
            WHEN word_count <= 10   THEN '1-10'
            WHEN word_count <= 50   THEN '11-50'
            WHEN word_count <= 100  THEN '51-100'
            WHEN word_count <= 500  THEN '101-500'
            ELSE '500+'
          END AS bucket,
          COUNT(*) AS n_reefs,
          SUM(word_count) AS total_words,
          ROUND(AVG(word_count), 1) AS avg_size
        FROM Reefs
        GROUP BY bucket
        ORDER BY MIN(word_count)
    """).fetchall():
        print(f"  {row[0]:>8s}: {row[1]:>5,} reefs, {row[2]:>8,} words (avg {row[3]})")

    # avg_specificity distribution at reef level
    print("\n--- Reef avg_specificity distribution ---")
    for row in con.execute("""
        SELECT
          CASE
            WHEN avg_specificity >= 2   THEN '2+'
            WHEN avg_specificity >= 1   THEN '1-2'
            WHEN avg_specificity >= 0   THEN '0-1'
            WHEN avg_specificity >= -1  THEN '-1-0'
            ELSE '<-1'
          END AS bucket,
          COUNT(*)
        FROM Reefs
        WHERE avg_specificity IS NOT NULL
        GROUP BY bucket
        ORDER BY MIN(avg_specificity) DESC
    """).fetchall():
        print(f"  {row[0]:>6s}: {row[1]:>5,} reefs")

    # Per-archipelago totals
    print("\n--- Archipelago totals ---")
    for row in con.execute("""
        SELECT name, island_count, town_count, reef_count, word_count
        FROM Archipelagos ORDER BY archipelago_id
    """).fetchall():
        print(f"  {row[0]:30s}  islands={row[1]:>3}  towns={row[2]:>4}  reefs={row[3]:>5}  words={row[4]:>7,}")

    # Verification: NULL checks
    print("\n--- Verification ---")
    checks = [
        ("ReefWords with NULL pos",        "SELECT COUNT(*) FROM ReefWords WHERE pos IS NULL"),
        ("ReefWords with NULL specificity", "SELECT COUNT(*) FROM ReefWords WHERE specificity IS NULL"),
        ("ReefWords with NULL idf",         "SELECT COUNT(*) FROM ReefWords WHERE idf IS NULL"),
        ("ReefWords with NULL island_idf",  "SELECT COUNT(*) FROM ReefWords WHERE island_idf IS NULL"),
        ("Reefs with NULL noun_frac",       "SELECT COUNT(*) FROM Reefs WHERE noun_frac IS NULL"),
        ("Towns with NULL noun_frac",       "SELECT COUNT(*) FROM Towns WHERE noun_frac IS NULL"),
        ("Islands with NULL noun_frac",     "SELECT COUNT(*) FROM Islands WHERE noun_frac IS NULL"),
        ("Archipelagos with 0 island_count","SELECT COUNT(*) FROM Archipelagos WHERE island_count = 0"),
    ]
    all_ok = True
    for label, query in checks:
        n = con.execute(query).fetchone()[0]
        status = "OK" if n == 0 else f"WARN: {n:,}"
        if n > 0:
            all_ok = False
        print(f"  {label:40s} {status}")

    if all_ok:
        print("\n  All checks passed.")
    else:
        print("\n  Some checks have warnings (expected for words without POS).")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
