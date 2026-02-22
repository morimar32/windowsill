"""Post-process XGBoost predictions: domain-frequency TF-IDF cleanup.

Only adjusts scores BELOW 0.7 using inverse domain frequency (IDF) to penalize
low-confidence words that appear in many domains (generic/non-specific vocabulary).
High-confidence predictions (>= 0.7) are left untouched.

    if raw_score < 0.7:
        adjusted = raw_score * log2(N / n_domains_predicted) / log2(N)
    else:
        adjusted = raw_score

Where N = total domains with xgboost predictions.

Usage:
    python post_process_xgb.py --analyze          # just show stats, don't change anything
    python post_process_xgb.py --apply             # write adjusted scores + prune below threshold
    python post_process_xgb.py --apply --threshold 0.5
    python post_process_xgb.py --domain medicine   # analyze single domain
"""

import argparse
import math
import os
import sys

import config
from lib import db

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2.db")


def compute_idf_stats(con):
    """Compute per-word domain frequency stats for xgboost predictions.

    Returns:
        word_domain_count: {word_id: n_domains_predicted}
        n_total_domains: total number of domains with xgboost predictions
        rows: list of (domain, word, word_id, raw_score) for all xgboost rows
    """
    # Total domains with xgboost predictions
    n_total_domains = con.execute("""
        SELECT COUNT(DISTINCT domain) FROM augmented_domains
        WHERE source = 'xgboost'
    """).fetchone()[0]

    # Per-word: how many domains predicted this word?
    wdc_rows = con.execute("""
        SELECT word_id, COUNT(DISTINCT domain) as n_doms
        FROM augmented_domains
        WHERE source = 'xgboost' AND word_id IS NOT NULL
        GROUP BY word_id
    """).fetchall()
    word_domain_count = {wid: n for wid, n in wdc_rows}

    # All xgboost rows
    rows = con.execute("""
        SELECT domain, word, word_id, score
        FROM augmented_domains
        WHERE source = 'xgboost'
    """).fetchall()

    return word_domain_count, n_total_domains, rows


IDF_CUTOFF = 0.7  # Only apply IDF adjustment below this raw score


def compute_adjusted_score(raw_score, n_word_domains, n_total_domains):
    """TF-IDF-style adjusted score, only applied to low-confidence predictions.

    Scores >= IDF_CUTOFF are returned unchanged. Below that:
        adjusted = raw_score * log2(N / n_word_domains) / log2(N)

    Words in 1 domain get multiplier ~1.0.
    Words in many domains get multiplier approaching 0.
    """
    if raw_score >= IDF_CUTOFF:
        return raw_score
    if n_total_domains <= 1 or n_word_domains <= 0:
        return raw_score
    log_n = math.log2(n_total_domains)
    if log_n == 0:
        return raw_score
    idf = math.log2(n_total_domains / n_word_domains) / log_n
    return raw_score * idf


def analyze(con, domain_filter=None):
    """Print analysis of domain frequency distribution and score adjustments."""
    word_domain_count, n_total, rows = compute_idf_stats(con)

    print(f"Total xgboost rows: {len(rows):,}")
    print(f"Total domains with predictions: {n_total}")
    print(f"Unique words predicted: {len(word_domain_count):,}")

    # Domain frequency distribution
    from collections import Counter
    freq_dist = Counter(word_domain_count.values())
    print(f"\n--- Domain frequency distribution ---")
    print(f"  {'n_domains':>10}  {'n_words':>10}  {'cumulative':>10}")
    cumulative = 0
    for n_doms in sorted(freq_dist.keys()):
        n_words = freq_dist[n_doms]
        cumulative += n_words
        if n_doms <= 20 or n_doms % 10 == 0 or n_doms == max(freq_dist.keys()):
            print(f"  {n_doms:>10}  {n_words:>10,}  {cumulative:>10,}")

    # Score adjustment impact by bucket
    print(f"\n--- Score adjustment impact ---")
    print(f"  {'raw_bucket':>12}  {'count':>8}  {'avg_raw':>8}  {'avg_adj':>8}  "
          f"{'avg_ndoms':>10}  {'pruned':>8}")

    for lo in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        hi = lo + 0.1
        bucket_rows = [(d, w, wid, s) for d, w, wid, s in rows
                       if s is not None and lo <= s < hi]
        if not bucket_rows:
            continue
        raw_scores = [s for _, _, _, s in bucket_rows]
        adj_scores = [compute_adjusted_score(s, word_domain_count.get(wid, 1), n_total)
                      for _, _, wid, s in bucket_rows]
        n_doms_list = [word_domain_count.get(wid, 1) for _, _, wid, _ in bucket_rows]
        pruned = sum(1 for a in adj_scores if a < config.XGBOOST_SCORE_THRESHOLD)

        print(f"  {lo:.1f}–{hi:.1f}  {len(bucket_rows):>8,}  "
              f"{sum(raw_scores)/len(raw_scores):>8.3f}  "
              f"{sum(adj_scores)/len(adj_scores):>8.3f}  "
              f"{sum(n_doms_list)/len(n_doms_list):>10.1f}  "
              f"{pruned:>8,}")

    # Show most-penalized words (high raw score, big drop)
    adjustments = []
    for domain, word, wid, raw in rows:
        if raw is None or wid is None:
            continue
        n_doms = word_domain_count.get(wid, 1)
        adj = compute_adjusted_score(raw, n_doms, n_total)
        drop = raw - adj
        adjustments.append((domain, word, wid, raw, adj, drop, n_doms))

    adjustments.sort(key=lambda x: x[5], reverse=True)
    print(f"\n--- Most penalized (biggest score drop) ---")
    print(f"  {'word':<30} {'raw':>6} {'adj':>6} {'drop':>6} {'n_doms':>7} {'domain'}")
    for domain, word, wid, raw, adj, drop, n_doms in adjustments[:30]:
        print(f"  {word:<30} {raw:>6.3f} {adj:>6.3f} {drop:>+6.3f} {n_doms:>7} {domain}")

    # Show least-penalized borderline words (kept despite low raw score)
    borderline_kept = [(d, w, wid, r, a, n) for d, w, wid, r, a, _, n in adjustments
                       if 0.4 <= r < 0.6 and a >= config.XGBOOST_SCORE_THRESHOLD]
    borderline_kept.sort(key=lambda x: x[4], reverse=True)
    print(f"\n--- Borderline words KEPT after adjustment (raw 0.4–0.6, adj >= {config.XGBOOST_SCORE_THRESHOLD}) ---")
    print(f"  {'word':<30} {'raw':>6} {'adj':>6} {'n_doms':>7} {'domain'}")
    for domain, word, wid, raw, adj, n_doms in borderline_kept[:30]:
        print(f"  {word:<30} {raw:>6.3f} {adj:>6.3f} {n_doms:>7} {domain}")

    # Per-domain analysis
    if domain_filter:
        print(f"\n--- Domain detail: {domain_filter} ---")
        domain_rows = [(d, w, wid, r) for d, w, wid, r in rows if d == domain_filter]
        if not domain_rows:
            print(f"  No xgboost rows for '{domain_filter}'")
            return

        domain_adj = []
        for d, w, wid, raw in domain_rows:
            n_doms = word_domain_count.get(wid, 1)
            adj = compute_adjusted_score(raw, n_doms, n_total)
            domain_adj.append((w, raw, adj, n_doms))

        # Before/after counts at thresholds
        for t in [0.4, 0.5, 0.6, 0.7]:
            before = sum(1 for _, r, _, _ in domain_adj if r >= t)
            after = sum(1 for _, _, a, _ in domain_adj if a >= t)
            print(f"  threshold={t}: {before:,} raw → {after:,} adjusted ({before - after:,} pruned)")

        # Words that would be pruned
        pruned = [(w, r, a, n) for w, r, a, n in domain_adj
                  if a < config.XGBOOST_SCORE_THRESHOLD]
        pruned.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Words pruned (adj < {config.XGBOOST_SCORE_THRESHOLD}):")
        for w, raw, adj, n_doms in pruned[:30]:
            print(f"    {w:<30} raw={raw:.3f} adj={adj:.3f} n_doms={n_doms}")

        # Words kept
        kept = [(w, r, a, n) for w, r, a, n in domain_adj
                if a >= config.XGBOOST_SCORE_THRESHOLD]
        kept.sort(key=lambda x: x[2])
        print(f"\n  Lowest-scoring words kept (adj >= {config.XGBOOST_SCORE_THRESHOLD}):")
        for w, raw, adj, n_doms in kept[:20]:
            print(f"    {w:<30} raw={raw:.3f} adj={adj:.3f} n_doms={n_doms}")

    # Overall summary
    total_before = len(rows)
    total_after = sum(1 for d, w, wid, r in rows
                      if r is not None and wid is not None
                      and compute_adjusted_score(r, word_domain_count.get(wid, 1), n_total)
                      >= config.XGBOOST_SCORE_THRESHOLD)
    print(f"\n--- Summary ---")
    print(f"  Before: {total_before:,} xgboost rows")
    print(f"  After (adj >= {config.XGBOOST_SCORE_THRESHOLD}): {total_after:,} rows")
    print(f"  Pruned: {total_before - total_after:,} ({(total_before - total_after) / total_before * 100:.1f}%)")


def apply_adjustments(con, threshold):
    """Write adjusted scores and prune rows below threshold."""
    word_domain_count, n_total, rows = compute_idf_stats(con)

    n_updated = 0
    n_pruned = 0

    # Batch updates: compute all adjusted scores first
    updates = []  # (adjusted_score, domain, word)
    deletes = []  # (domain, word)

    for domain, word, wid, raw in rows:
        if raw is None or wid is None:
            deletes.append((domain, word))
            n_pruned += 1
            continue

        n_doms = word_domain_count.get(wid, 1)
        adj = compute_adjusted_score(raw, n_doms, n_total)

        if adj < threshold:
            deletes.append((domain, word))
            n_pruned += 1
        else:
            updates.append((adj, domain, word))
            n_updated += 1

    # Apply
    if updates:
        con.executemany(
            "UPDATE augmented_domains SET score = ? WHERE domain = ? AND word = ?",
            updates
        )
    if deletes:
        con.executemany(
            "DELETE FROM augmented_domains WHERE domain = ? AND word = ?",
            deletes
        )
    con.commit()

    print(f"Applied adjustments:")
    print(f"  Updated scores: {n_updated:,}")
    print(f"  Pruned rows: {n_pruned:,}")
    print(f"  Threshold: {threshold}")

    # Post-apply stats
    remaining = con.execute(
        "SELECT COUNT(*) FROM augmented_domains WHERE source = 'xgboost'"
    ).fetchone()[0]
    print(f"  Remaining xgboost rows: {remaining:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process XGBoost scores with IDF adjustment")
    parser.add_argument("--analyze", action="store_true",
                        help="Show analysis without modifying data")
    parser.add_argument("--apply", action="store_true",
                        help="Write adjusted scores and prune below threshold")
    parser.add_argument("--threshold", type=float,
                        default=config.XGBOOST_SCORE_THRESHOLD,
                        help=f"Pruning threshold (default: {config.XGBOOST_SCORE_THRESHOLD})")
    parser.add_argument("--domain", type=str, default=None,
                        help="Analyze a specific domain in detail")
    args = parser.parse_args()

    if not args.analyze and not args.apply:
        parser.print_help()
        print("\nSpecify --analyze or --apply")
        sys.exit(1)

    con = db.get_connection(DB_PATH)

    if args.analyze:
        analyze(con, domain_filter=args.domain)

    if args.apply:
        apply_adjustments(con, args.threshold)

    con.close()


if __name__ == "__main__":
    main()
