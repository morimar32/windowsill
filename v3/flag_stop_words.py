"""
Flag stop words in the v3 Words table.

Two modes:

  --baseline   Flag lagoon STOP_WORDS (function words: "the", "i", "and", etc.)
               Runs early, right after load_wordnet_vocab.py.

  --ubiquity   Post-XGBoost ubiquity pruning: words appearing in 20+ towns
               get their AugmentedTowns rows deleted/penalized, and
               Words.is_stop is set additively.
               Runs after post_process_xgb.py, before clustering.

  --analyze    Dry-run showing counts and worst offenders (no writes).

Usage:
    python v3/flag_stop_words.py --baseline
    python v3/flag_stop_words.py --ubiquity
    python v3/flag_stop_words.py --analyze
"""

import argparse
import os
import sqlite3
import sys

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_DB = os.path.join(_project, "v3/windowsill.db")

# Import lagoon stop words
_lagoon_root = os.path.abspath(os.path.join(_project, "..", "lagoon", "src"))
if _lagoon_root not in sys.path:
    sys.path.insert(0, _lagoon_root)
from lagoon._stop_words import STOP_WORDS

# Ubiquity constants (matching v2 config.py)
UBIQUITY_TOWN_THRESHOLD = 20
UBIQUITY_SCORE_FLOOR = 0.80
UBIQUITY_SCORE_CEILING = 0.95
UBIQUITY_PENALTY = 0.5


def baseline(con):
    """Flag lagoon STOP_WORDS in Words.is_stop."""
    rows = con.execute("SELECT word_id, word FROM Words").fetchall()
    flagged = []
    for word_id, word in rows:
        if word.lower() in STOP_WORDS:
            flagged.append((word_id,))

    if flagged:
        con.executemany("UPDATE Words SET is_stop = 1 WHERE word_id = ?", flagged)
    con.commit()

    print(f"Baseline stop words:")
    print(f"  Lagoon STOP_WORDS list: {len(STOP_WORDS)} entries")
    print(f"  Matched in vocab: {len(flagged)}")

    # Show some examples
    if flagged:
        examples = con.execute("""
            SELECT word FROM Words WHERE is_stop = 1
            ORDER BY word LIMIT 20
        """).fetchall()
        print(f"  Examples: {', '.join(r[0] for r in examples)}")


def ubiquity(con):
    """Post-XGBoost ubiquity pruning: delete/penalize words in 20+ towns."""
    # Find ubiquitous words via AugmentedTowns
    con.execute("DROP TABLE IF EXISTS _ub_wids")
    con.execute("""
        CREATE TEMP TABLE _ub_wids AS
        SELECT word_id, COUNT(DISTINCT town_id) AS n_towns
        FROM AugmentedTowns
        WHERE word_id IS NOT NULL
        GROUP BY word_id
        HAVING COUNT(DISTINCT town_id) >= ?
    """, (UBIQUITY_TOWN_THRESHOLD,))

    n_ubiquitous = con.execute("SELECT COUNT(*) FROM _ub_wids").fetchone()[0]

    # Count before operations
    n_to_delete = con.execute("""
        SELECT COUNT(*) FROM AugmentedTowns
        WHERE source = 'xgboost' AND score < ?
          AND word_id IN (SELECT word_id FROM _ub_wids)
    """, (UBIQUITY_SCORE_FLOOR,)).fetchone()[0]

    n_to_penalize = con.execute("""
        SELECT COUNT(*) FROM AugmentedTowns
        WHERE source = 'xgboost'
          AND score >= ? AND score < ?
          AND word_id IN (SELECT word_id FROM _ub_wids)
    """, (UBIQUITY_SCORE_FLOOR, UBIQUITY_SCORE_CEILING)).fetchone()[0]

    n_untouched = con.execute("""
        SELECT COUNT(*) FROM AugmentedTowns
        WHERE source = 'xgboost' AND score >= ?
          AND word_id IN (SELECT word_id FROM _ub_wids)
    """, (UBIQUITY_SCORE_CEILING,)).fetchone()[0]

    # DELETE: xgboost rows with score < floor
    con.execute("""
        DELETE FROM AugmentedTowns
        WHERE source = 'xgboost' AND score < ?
          AND word_id IN (SELECT word_id FROM _ub_wids)
    """, (UBIQUITY_SCORE_FLOOR,))

    # PENALIZE: xgboost rows with floor <= score < ceiling -> score * penalty
    con.execute("""
        UPDATE AugmentedTowns
        SET score = score * ?
        WHERE source = 'xgboost'
          AND score >= ? AND score < ?
          AND word_id IN (SELECT word_id FROM _ub_wids)
    """, (UBIQUITY_PENALTY, UBIQUITY_SCORE_FLOOR, UBIQUITY_SCORE_CEILING))

    # Flag all ubiquitous words as stop (additive to baseline)
    con.execute("""
        UPDATE Words SET is_stop = 1
        WHERE word_id IN (SELECT word_id FROM _ub_wids)
    """)

    con.execute("DROP TABLE _ub_wids")
    con.commit()

    total_stop = con.execute(
        "SELECT COUNT(*) FROM Words WHERE is_stop = 1"
    ).fetchone()[0]

    print(f"Ubiquity pruning (words in {UBIQUITY_TOWN_THRESHOLD}+ towns):")
    print(f"  Ubiquitous words: {n_ubiquitous:,}")
    print(f"  Deleted (score < {UBIQUITY_SCORE_FLOOR}): {n_to_delete:,}")
    print(f"  Penalized ({UBIQUITY_SCORE_FLOOR}–{UBIQUITY_SCORE_CEILING}"
          f" x{UBIQUITY_PENALTY}): {n_to_penalize:,}")
    print(f"  Untouched (score >= {UBIQUITY_SCORE_CEILING}): {n_untouched:,}")
    print(f"  Total is_stop=1 words (baseline + ubiquity): {total_stop:,}")


def analyze(con):
    """Dry-run: show ubiquity stats without modifying anything."""
    # Current baseline count
    n_baseline = con.execute(
        "SELECT COUNT(*) FROM Words WHERE is_stop = 1"
    ).fetchone()[0]
    print(f"Current is_stop=1 (baseline): {n_baseline}")

    # Words in most towns via AugmentedTowns
    print(f"\nTop 30 most ubiquitous words (by town count in AugmentedTowns):")
    rows = con.execute("""
        SELECT w.word, COUNT(DISTINCT a.town_id) AS n_towns,
               AVG(a.score) AS avg_score, MIN(a.score) AS min_score
        FROM AugmentedTowns a
        JOIN Words w USING (word_id)
        GROUP BY a.word_id
        HAVING n_towns >= 10
        ORDER BY n_towns DESC, w.word
        LIMIT 30
    """).fetchall()

    if rows:
        print(f"  {'word':<20} {'towns':>6} {'avg_score':>10} {'min_score':>10}")
        print(f"  {'─'*20} {'─'*6} {'─'*10} {'─'*10}")
        for word, n_towns, avg_score, min_score in rows:
            marker = " *** STOP" if n_towns >= UBIQUITY_TOWN_THRESHOLD else ""
            print(f"  {word:<20} {n_towns:>6} {avg_score:>10.4f} {min_score:>10.4f}{marker}")
    else:
        print("  (no words in 10+ towns)")

    # What would happen with ubiquity pruning
    n_ubiquitous = con.execute("""
        SELECT COUNT(DISTINCT word_id) FROM AugmentedTowns
        WHERE word_id IS NOT NULL
        GROUP BY word_id
        HAVING COUNT(DISTINCT town_id) >= ?
    """, (UBIQUITY_TOWN_THRESHOLD,)).fetchall()
    n_ub = len(n_ubiquitous)

    # Build temp table for analysis
    con.execute("DROP TABLE IF EXISTS _ub_analysis")
    con.execute("""
        CREATE TEMP TABLE _ub_analysis AS
        SELECT word_id FROM AugmentedTowns
        WHERE word_id IS NOT NULL
        GROUP BY word_id
        HAVING COUNT(DISTINCT town_id) >= ?
    """, (UBIQUITY_TOWN_THRESHOLD,))

    would_delete = con.execute("""
        SELECT COUNT(*) FROM AugmentedTowns
        WHERE source = 'xgboost' AND score < ?
          AND word_id IN (SELECT word_id FROM _ub_analysis)
    """, (UBIQUITY_SCORE_FLOOR,)).fetchone()[0]

    would_penalize = con.execute("""
        SELECT COUNT(*) FROM AugmentedTowns
        WHERE source = 'xgboost'
          AND score >= ? AND score < ?
          AND word_id IN (SELECT word_id FROM _ub_analysis)
    """, (UBIQUITY_SCORE_FLOOR, UBIQUITY_SCORE_CEILING)).fetchone()[0]

    would_keep = con.execute("""
        SELECT COUNT(*) FROM AugmentedTowns
        WHERE source = 'xgboost' AND score >= ?
          AND word_id IN (SELECT word_id FROM _ub_analysis)
    """, (UBIQUITY_SCORE_CEILING,)).fetchone()[0]

    con.execute("DROP TABLE _ub_analysis")

    print(f"\nUbiquity pruning preview (threshold: {UBIQUITY_TOWN_THRESHOLD}+ towns):")
    print(f"  Ubiquitous words: {n_ub:,}")
    print(f"  Would delete (score < {UBIQUITY_SCORE_FLOOR}): {would_delete:,} rows")
    print(f"  Would penalize ({UBIQUITY_SCORE_FLOOR}–{UBIQUITY_SCORE_CEILING}): {would_penalize:,} rows")
    print(f"  Would keep (score >= {UBIQUITY_SCORE_CEILING}): {would_keep:,} rows")

    # Words in most reefs (post-clustering view, if available)
    has_reefs = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name = 'ReefWords'"
    ).fetchone()[0]
    if has_reefs:
        reef_count = con.execute("SELECT COUNT(*) FROM ReefWords").fetchone()[0]
        if reef_count > 0:
            print(f"\nTop 20 words by reef spread (post-clustering):")
            rows = con.execute("""
                SELECT w.word, COUNT(DISTINCT r.town_id) AS n_towns,
                       COUNT(DISTINCT t.island_id) AS n_islands
                FROM ReefWords rw
                JOIN Reefs r USING (reef_id)
                JOIN Towns t USING (town_id)
                JOIN Words w ON w.word_id = rw.word_id
                WHERE w.is_stop = 0
                GROUP BY rw.word_id
                ORDER BY n_islands DESC, n_towns DESC
                LIMIT 20
            """).fetchall()
            print(f"  {'word':<20} {'towns':>6} {'islands':>8}")
            print(f"  {'─'*20} {'─'*6} {'─'*8}")
            for word, n_towns, n_islands in rows:
                print(f"  {word:<20} {n_towns:>6} {n_islands:>8}")


def main():
    parser = argparse.ArgumentParser(description="Flag stop words in v3 Words table")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", action="store_true",
                       help="Flag lagoon STOP_WORDS (function words)")
    group.add_argument("--ubiquity", action="store_true",
                       help="Post-XGBoost ubiquity pruning")
    group.add_argument("--analyze", action="store_true",
                       help="Dry-run showing counts and worst offenders")
    args = parser.parse_args()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    if args.baseline:
        baseline(con)
    elif args.ubiquity:
        ubiquity(con)
    elif args.analyze:
        analyze(con)

    con.close()


if __name__ == "__main__":
    main()
