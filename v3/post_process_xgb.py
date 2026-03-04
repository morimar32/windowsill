"""Post-process XGBoost predictions: cross-town IDF cleanup.

Port of v2's post_process_xgb.py for the v3 4-tier hierarchy.

Only adjusts scores BELOW IDF_CUTOFF (0.7) using inverse town frequency to
penalize low-confidence words predicted into many towns. High-confidence
predictions are left untouched.

    if raw_score < 0.7:
        adjusted = raw_score * log2(N_towns / n_word_towns) / log2(N_towns)
    else:
        adjusted = raw_score

Where N_towns = total towns with XGBoost predictions.

Pipeline position: runs AFTER all islands complete XGBoost training,
BEFORE Leiden clustering (cluster_reefs.py).

Usage:
    python v3/post_process_xgb.py --analyze
    python v3/post_process_xgb.py --apply
    python v3/post_process_xgb.py --apply --cleanup-reefs
    python v3/post_process_xgb.py --apply --threshold 0.5
"""

import argparse
import math
import os
import sys
from collections import Counter

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

V3_DB = os.path.join(_project, "v3/windowsill.db")

IDF_CUTOFF = 0.7
DEFAULT_THRESHOLD = 0.4


def compute_idf_stats(con):
    """Compute per-word town frequency stats for XGBoost predictions.

    Returns:
        word_town_count: {word_id: n_towns_predicted}
        n_total_towns: total towns with XGBoost predictions
        rows: list of (town_id, word_id, score) for all XGBoost rows
    """
    n_total_towns = con.execute("""
        SELECT COUNT(DISTINCT town_id) FROM AugmentedTowns
        WHERE source = 'xgboost'
    """).fetchone()[0]

    wtc_rows = con.execute("""
        SELECT word_id, COUNT(DISTINCT town_id) as n_towns
        FROM AugmentedTowns
        WHERE source = 'xgboost'
        GROUP BY word_id
    """).fetchall()
    word_town_count = {wid: n for wid, n in wtc_rows}

    rows = con.execute("""
        SELECT town_id, word_id, score
        FROM AugmentedTowns
        WHERE source = 'xgboost'
    """).fetchall()

    return word_town_count, n_total_towns, rows


def compute_adjusted_score(raw_score, n_word_towns, n_total_towns):
    """IDF-adjusted score, only applied below IDF_CUTOFF.

    Words in 1 town get multiplier ~1.0.
    Words in many towns get multiplier approaching 0.
    """
    if raw_score >= IDF_CUTOFF:
        return raw_score
    if n_total_towns <= 1 or n_word_towns <= 0:
        return raw_score
    log_n = math.log2(n_total_towns)
    if log_n == 0:
        return raw_score
    idf = math.log2(n_total_towns / n_word_towns) / log_n
    return raw_score * idf


def analyze(con):
    """Print analysis of town frequency distribution and score adjustments."""
    word_town_count, n_total, rows = compute_idf_stats(con)

    print(f"Total XGBoost rows: {len(rows):,}")
    print(f"Total towns with predictions: {n_total}")
    print(f"Unique words predicted: {len(word_town_count):,}")

    # Town frequency distribution
    freq_dist = Counter(word_town_count.values())
    print(f"\n--- Town frequency distribution ---")
    print(f"  {'n_towns':>10}  {'n_words':>10}  {'cumulative':>10}")
    cumulative = 0
    for n_towns in sorted(freq_dist.keys()):
        n_words = freq_dist[n_towns]
        cumulative += n_words
        if n_towns <= 20 or n_towns % 10 == 0 or n_towns == max(freq_dist.keys()):
            print(f"  {n_towns:>10}  {n_words:>10,}  {cumulative:>10,}")

    # Score adjustment impact by bucket
    print(f"\n--- Score adjustment impact ---")
    print(f"  {'raw_bucket':>12}  {'count':>8}  {'avg_raw':>8}  {'avg_adj':>8}  "
          f"{'avg_ntowns':>10}  {'pruned':>8}")

    for lo in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        hi = lo + 0.1
        bucket_rows = [(tid, wid, s) for tid, wid, s in rows
                       if s is not None and lo <= s < hi]
        if not bucket_rows:
            continue
        raw_scores = [s for _, _, s in bucket_rows]
        adj_scores = [compute_adjusted_score(s, word_town_count.get(wid, 1), n_total)
                      for _, wid, s in bucket_rows]
        n_towns_list = [word_town_count.get(wid, 1) for _, wid, _ in bucket_rows]
        pruned = sum(1 for a in adj_scores if a < DEFAULT_THRESHOLD)

        print(f"  {lo:.1f}\u2013{hi:.1f}  {len(bucket_rows):>8,}  "
              f"{sum(raw_scores)/len(raw_scores):>8.3f}  "
              f"{sum(adj_scores)/len(adj_scores):>8.3f}  "
              f"{sum(n_towns_list)/len(n_towns_list):>10.1f}  "
              f"{pruned:>8,}")

    # Load word texts and town info for display
    word_texts = {}
    for wid in word_town_count:
        row = con.execute("SELECT word FROM Words WHERE word_id = ?", (wid,)).fetchone()
        if row:
            word_texts[wid] = row[0]

    town_info = {}
    for tid, name, island in con.execute("""
        SELECT t.town_id, t.name, i.name
        FROM Towns t JOIN Islands i USING(island_id)
    """).fetchall():
        town_info[tid] = (name, island)

    # Most penalized words
    adjustments = []
    for tid, wid, raw in rows:
        if raw is None or wid is None:
            continue
        n_towns = word_town_count.get(wid, 1)
        adj = compute_adjusted_score(raw, n_towns, n_total)
        drop = raw - adj
        adjustments.append((tid, wid, raw, adj, drop, n_towns))

    adjustments.sort(key=lambda x: x[4], reverse=True)
    print(f"\n--- Most penalized (biggest score drop) ---")
    print(f"  {'word':<30} {'raw':>6} {'adj':>6} {'drop':>6} {'n_towns':>7} {'town':<25} {'island'}")
    for tid, wid, raw, adj, drop, n_towns in adjustments[:30]:
        word = word_texts.get(wid, str(wid))
        town, island = town_info.get(tid, ("?", "?"))
        print(f"  {word:<30} {raw:>6.3f} {adj:>6.3f} {drop:>+6.3f} {n_towns:>7} {town:<25} {island}")

    # Summary
    total_before = len(rows)
    total_pruned = sum(1 for tid, wid, raw in rows
                       if raw is not None and wid is not None
                       and compute_adjusted_score(raw, word_town_count.get(wid, 1), n_total)
                       < DEFAULT_THRESHOLD)
    print(f"\n--- Summary ---")
    print(f"  Before: {total_before:,} XGBoost rows")
    print(f"  Would prune (adj < {DEFAULT_THRESHOLD}): {total_pruned:,} rows ({total_pruned/total_before*100:.1f}%)")
    print(f"  Would keep: {total_before - total_pruned:,} rows")


def apply_adjustments(con, threshold, cleanup_reefs=False):
    """Adjust scores, prune below threshold, optionally clean up ReefWords."""
    word_town_count, n_total, rows = compute_idf_stats(con)

    updates = []   # (adjusted_score, town_id, word_id)
    deletes = []   # (town_id, word_id)

    for tid, wid, raw in rows:
        if raw is None or wid is None:
            deletes.append((tid, wid))
            continue

        n_towns = word_town_count.get(wid, 1)
        adj = compute_adjusted_score(raw, n_towns, n_total)

        if adj < threshold:
            deletes.append((tid, wid))
        elif adj != raw:
            updates.append((adj, tid, wid))

    # Apply score updates
    if updates:
        con.executemany(
            "UPDATE AugmentedTowns SET score = ? WHERE town_id = ? AND word_id = ?",
            updates
        )

    # Prune below threshold
    if deletes:
        con.executemany(
            "DELETE FROM AugmentedTowns WHERE town_id = ? AND word_id = ?",
            deletes
        )
    con.commit()

    print(f"Applied IDF adjustments:")
    print(f"  Updated scores: {len(updates):,}")
    print(f"  Pruned rows: {len(deletes):,}")
    print(f"  Threshold: {threshold}")

    remaining = con.execute(
        "SELECT COUNT(*) FROM AugmentedTowns WHERE source = 'xgboost'"
    ).fetchone()[0]
    print(f"  Remaining XGBoost rows: {remaining:,}")

    if cleanup_reefs and deletes:
        cleanup_orphaned_reefwords(con, deletes)


def cleanup_orphaned_reefwords(con, pruned_pairs):
    """Remove ReefWords entries whose AugmentedTowns backing was pruned."""
    print(f"\nCleaning up orphaned ReefWords...")

    cur = con.cursor()

    # Temp table with pruned (town_id, word_id) pairs
    cur.execute("CREATE TEMP TABLE _pruned (town_id INTEGER, word_id INTEGER)")
    cur.executemany("INSERT INTO _pruned VALUES (?, ?)", pruned_pairs)
    cur.execute("CREATE INDEX _pruned_idx ON _pruned(town_id, word_id)")

    n_before = con.execute("SELECT COUNT(*) FROM ReefWords WHERE source = 'xgboost'").fetchone()[0]

    # Delete orphaned ReefWords: xgboost-sourced words whose town prediction was pruned
    cur.execute("""
        DELETE FROM ReefWords
        WHERE source = 'xgboost'
          AND EXISTS (
            SELECT 1 FROM _pruned p
            JOIN Reefs r ON r.town_id = p.town_id AND r.reef_id = ReefWords.reef_id
            WHERE p.word_id = ReefWords.word_id
          )
    """)
    n_deleted = cur.rowcount

    # Remove empty reefs
    cur.execute("""
        DELETE FROM Reefs WHERE reef_id NOT IN (
            SELECT DISTINCT reef_id FROM ReefWords
        )
    """)
    n_empty_reefs = cur.rowcount

    # Update reef stats
    cur.execute("""
        UPDATE Reefs SET
            word_count = (SELECT COUNT(*) FROM ReefWords WHERE ReefWords.reef_id = Reefs.reef_id),
            core_word_count = (SELECT COUNT(*) FROM ReefWords WHERE ReefWords.reef_id = Reefs.reef_id AND is_core = 1)
    """)

    # Update town stats
    cur.execute("""
        UPDATE Towns SET
            reef_count = (SELECT COUNT(*) FROM Reefs WHERE Reefs.town_id = Towns.town_id),
            word_count = (SELECT COUNT(*) FROM ReefWords rw JOIN Reefs r USING(reef_id) WHERE r.town_id = Towns.town_id)
    """)

    # Update island stats
    cur.execute("""
        UPDATE Islands SET
            reef_count = (SELECT COUNT(*) FROM Reefs r JOIN Towns t USING(town_id) WHERE t.island_id = Islands.island_id),
            word_count = (SELECT COUNT(DISTINCT rw.word_id) FROM ReefWords rw JOIN Reefs r USING(reef_id) JOIN Towns t USING(town_id) WHERE t.island_id = Islands.island_id)
    """)

    cur.execute("DROP TABLE _pruned")
    con.commit()

    n_after = con.execute("SELECT COUNT(*) FROM ReefWords WHERE source = 'xgboost'").fetchone()[0]
    n_total_reefs = con.execute("SELECT COUNT(*) FROM Reefs").fetchone()[0]
    n_tiny = con.execute("SELECT COUNT(*) FROM Reefs WHERE word_count <= 3").fetchone()[0]

    print(f"  ReefWords deleted: {n_deleted:,}")
    print(f"  Empty reefs removed: {n_empty_reefs}")
    print(f"  Tiny reefs (<=3 words): {n_tiny}")
    print(f"  XGBoost ReefWords: {n_before:,} \u2192 {n_after:,}")
    print(f"  Total reefs: {n_total_reefs:,}")


def main():
    import sqlite3

    parser = argparse.ArgumentParser(
        description="Post-process XGBoost predictions with cross-town IDF adjustment")
    parser.add_argument("--analyze", action="store_true",
                        help="Show analysis without modifying data")
    parser.add_argument("--apply", action="store_true",
                        help="Apply IDF adjustments and prune below threshold")
    parser.add_argument("--cleanup-reefs", action="store_true",
                        help="Also clean up orphaned ReefWords (use with --apply)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Pruning threshold (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    if not args.analyze and not args.apply:
        parser.print_help()
        print("\nSpecify --analyze or --apply")
        sys.exit(1)

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    if args.analyze:
        analyze(con)

    if args.apply:
        apply_adjustments(con, args.threshold, cleanup_reefs=args.cleanup_reefs)

    con.close()


if __name__ == "__main__":
    main()
