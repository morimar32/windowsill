"""Pre-compute domain & reef scores for all word-domain pairs.

Materializes domain_score and reef_score in domain_word_scores so the
export/load process can just read pre-computed values.

Usage:
    python score.py          # compute all scores
    python score.py --stats  # just print stats without recomputing
"""

import argparse
import os
import time
from collections import defaultdict

from lib import db
from lib.scoring import compute_scores

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2.db")

# 17 test queries: query words → list of keyword substrings that must appear
# in at least one top-10 domain name
TEST_QUERIES = {
    "Pearl Harbor attack Japan": ["military", "warfare", "war"],
    "DNA genetic mutation heredity": ["biolog", "organism", "genet"],
    "python snake reptile": ["fauna", "animal", "zoolog", "vertebr"],
    "apple fruit nutrition": ["plant", "botan", "food", "vegetat", "health"],
    "music opera symphony orchestra": ["music", "singing", "concert"],
    "French Revolution guillotine": ["histor", "politic", "government"],
    "python programming language": ["computer", "tech", "software"],
    "computer": ["computer", "tech"],
    "queen": ["monarch", "histor", "royal", "chess"],
    "earthquake": ["geolog", "tecton", "earth"],
    "mercury": ["astro", "chem", "planet", "mythol", "alchem"],
    "jupiter": ["astro", "planet", "celest", "mythol"],
    "commander fleet navy": ["military", "navy", "war"],
    "neuron synapse axon dendrite cortex brain neural hippocampus": ["neuro", "brain", "anatom", "biolog"],
    "cannon weapon military": ["military", "war"],
    "photosynthesis chloroplast": ["plant", "botan", "biolog", "cell"],
    "earthquake tectonic plates seismic": ["geolog", "tecton", "earth"],
}


def load_data(con):
    """Load augmented_domains and domain_reefs for scoring.

    Returns:
        ad_rows: list of (domain, word_id, source, xgb_score)
        reef_map: dict {(domain, word_id): (reef_id, centroid_sim)}
        n_total_domains: int
    """
    print("Loading data...")
    t0 = time.time()

    ad_rows = con.execute("""
        SELECT domain, word_id, source, score
        FROM augmented_domains
        WHERE word_id IS NOT NULL AND domain != 'domainless'
    """).fetchall()
    print(f"  {len(ad_rows):,} augmented_domains rows")

    reef_rows = con.execute("""
        SELECT domain, word_id, reef_id, centroid_sim
        FROM domain_reefs
    """).fetchall()
    reef_map = {(d, wid): (rid, csim) for d, wid, rid, csim in reef_rows}
    print(f"  {len(reef_map):,} domain_reefs entries")

    n_total_domains = con.execute("""
        SELECT COUNT(DISTINCT domain) FROM augmented_domains
        WHERE domain != 'domainless'
    """).fetchone()[0]
    print(f"  {n_total_domains} total domains")

    print(f"  Loaded in {time.time() - t0:.1f}s\n")
    return ad_rows, reef_map, n_total_domains


def compute_n_word_domains(ad_rows):
    """Count distinct domains per word_id."""
    counts = defaultdict(set)
    for domain, word_id, _source, _score in ad_rows:
        counts[word_id].add(domain)
    return {wid: len(domains) for wid, domains in counts.items()}


def deduplicate_pairs(ad_rows, reef_map, n_word_domains, n_total_domains):
    """Deduplicate (domain, word_id) pairs, keeping highest source_quality.

    Returns list of tuples ready for INSERT:
        (domain, word_id, reef_id, source, source_quality, centroid_sim,
         n_wd, idf, domain_score, reef_score)
    """
    from lib.scoring import resolve_source_quality

    # Group by (domain, word_id), keep best source_quality
    best = {}  # (domain, word_id) -> (source, xgb_score, source_quality)
    for domain, word_id, source, xgb_score in ad_rows:
        sq = resolve_source_quality(source, xgb_score)
        key = (domain, word_id)
        if key not in best or sq > best[key][2]:
            best[key] = (source, xgb_score, sq)

    # Compute scores
    rows = []
    n_no_reef = 0
    for (domain, word_id), (source, xgb_score, _sq) in best.items():
        reef_info = reef_map.get((domain, word_id))
        if reef_info:
            reef_id, centroid_sim = reef_info
            if centroid_sim is None:
                centroid_sim = 0.8
        else:
            reef_id = -1
            centroid_sim = 0.8
            n_no_reef += 1

        n_wd = n_word_domains.get(word_id, 1)
        domain_score, reef_score, idf, source_quality = compute_scores(
            source, xgb_score, centroid_sim, n_wd, n_total_domains
        )
        rows.append((domain, word_id, reef_id, source, source_quality,
                      centroid_sim, n_wd, idf, domain_score, reef_score))

    return rows, n_no_reef


def persist_scores(con, rows):
    """Write scores to domain_word_scores (idempotent: DELETE + INSERT)."""
    print(f"Writing {len(rows):,} scores...")
    t0 = time.time()

    con.execute("DELETE FROM domain_word_scores")
    con.executemany(
        "INSERT INTO domain_word_scores "
        "(domain, word_id, reef_id, source, source_quality, centroid_sim, "
        "n_word_domains, idf, domain_score, reef_score) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    con.commit()
    print(f"  Done in {time.time() - t0:.1f}s\n")


def print_stats(con):
    """Print score distribution and per-domain stats."""
    print("Score distribution:")
    for bucket, count in con.execute("""
        SELECT ROUND(domain_score, 1) as bucket, COUNT(*)
        FROM domain_word_scores
        GROUP BY bucket ORDER BY bucket
    """).fetchall():
        print(f"  {bucket:5.1f}: {count:,}")

    print("\nTop 20 domains by avg domain_score:")
    for domain, cnt, avg_s, max_s in con.execute("""
        SELECT domain, COUNT(*), ROUND(AVG(domain_score), 2),
               ROUND(MAX(domain_score), 2)
        FROM domain_word_scores
        GROUP BY domain ORDER BY AVG(domain_score) DESC LIMIT 20
    """).fetchall():
        print(f"  {domain:<35} n={cnt:<6} avg={avg_s:<6} max={max_s}")

    print("\nTop 20 word-domain pairs by domain_score:")
    for domain, wid, ds in con.execute("""
        SELECT domain, word_id, domain_score
        FROM domain_word_scores ORDER BY domain_score DESC LIMIT 20
    """).fetchall():
        word = con.execute(
            "SELECT word FROM words WHERE word_id = ?", (wid,)
        ).fetchone()
        word_text = word[0] if word else str(wid)
        print(f"  {word_text:<25} {domain:<30} score={ds:.3f}")

    total = con.execute("SELECT COUNT(*) FROM domain_word_scores").fetchone()[0]
    print(f"\nTotal scored pairs: {total:,}")


def run_verification(con):
    """Run 17 test queries against domain_word_scores, print pass/fail."""
    print("Verification: 17 test queries\n")

    # Build word→word_id lookup for test words
    all_test_words = set()
    for query in TEST_QUERIES:
        all_test_words.update(query.lower().split())

    word_to_id = {}
    for word in all_test_words:
        row = con.execute(
            "SELECT word_id FROM words WHERE word = ?", (word,)
        ).fetchone()
        if row:
            word_to_id[word] = row[0]

    n_pass = 0
    n_fail = 0

    for query_text, expected_kw in TEST_QUERIES.items():
        words = query_text.lower().split()
        wids = [word_to_id[w] for w in words if w in word_to_id]
        missing = [w for w in words if w not in word_to_id]

        if not wids:
            print(f"  [SKIP] \"{query_text}\" — no word_ids found")
            continue

        # Accumulate domain_score per domain
        domain_scores = defaultdict(float)
        placeholders = ",".join("?" * len(wids))
        for domain, ds in con.execute(f"""
            SELECT domain, SUM(domain_score)
            FROM domain_word_scores
            WHERE word_id IN ({placeholders})
            GROUP BY domain
            ORDER BY SUM(domain_score) DESC
        """, wids).fetchall():
            domain_scores[domain] = ds

        ranked = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        top10 = ranked[:10]

        # Check if any expected keyword appears in top 10
        found = None
        for domain, score in top10:
            for kw in expected_kw:
                if kw in domain.lower():
                    found = domain
                    break
            if found:
                break

        if found:
            rank = next(i for i, (d, _) in enumerate(ranked) if d == found) + 1
            top3 = ", ".join(f"{d}({s:.1f})" for d, s in ranked[:3])
            print(f"  [PASS] \"{query_text}\" → {found} @{rank}  [{top3}]")
            n_pass += 1
        else:
            top5 = ", ".join(f"{d}({s:.1f})" for d, s in ranked[:5])
            print(f"  [FAIL] \"{query_text}\" → top5: {top5}")
            n_fail += 1

    print(f"\n  {n_pass}/{n_pass + n_fail} passed")
    return n_pass, n_fail


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute domain & reef scores for word-domain pairs"
    )
    parser.add_argument("--stats", action="store_true",
                        help="Print stats only (no recomputation)")
    args = parser.parse_args()

    con = db.get_connection(DB_PATH)
    db.create_scoring_schema(con)

    if args.stats:
        print_stats(con)
        print()
        run_verification(con)
        con.close()
        return

    # Full computation
    ad_rows, reef_map, n_total_domains = load_data(con)
    n_word_domains = compute_n_word_domains(ad_rows)
    print(f"Word-domain counts: {len(n_word_domains):,} unique word_ids\n")

    rows, n_no_reef = deduplicate_pairs(
        ad_rows, reef_map, n_word_domains, n_total_domains
    )
    print(f"Deduplicated pairs: {len(rows):,} (no reef assignment: {n_no_reef:,})\n")

    persist_scores(con, rows)
    print_stats(con)
    print()
    run_verification(con)

    con.close()


if __name__ == "__main__":
    main()
