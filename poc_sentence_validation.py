"""POC: Sentence-level embedding validation of Lagoon scoring.

Embeds entire sentences/passages holistically with nomic-embed-text-v1.5 and
compares domain rankings against Lagoon's compositional (word-by-word BM25)
scoring. Two independent comparisons:

1. Centroid comparison: cos(sentence_embedding, domain_centroid)
2. Domain-name comparison: cos(sentence_embedding, domain_name_embedding)

Both are compared to Lagoon's score_raw() z-scores using Spearman rank
correlation and top-k overlap.

Usage:
    python poc_sentence_validation.py              # comparison mode
    python poc_sentence_validation.py --trace      # + word-level trace on worst cases
    python poc_sentence_validation.py --trace-all  # + word-level trace on ALL queries
    python poc_sentence_validation.py --trace -q "computer" -q "python snake reptile"
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

import config
import embedder
from lib import db

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2.db")


# ---------------------------------------------------------------------------
# 1. Load domain centroids from DB (weighted average of reef centroids)
# ---------------------------------------------------------------------------

def load_domain_centroids(db_path):
    """Compute size-weighted average of reef centroids per domain, L2-normalized.

    Replicates lib/archipelago.py:compute_domain_embeddings() logic inline.

    Returns dict[str, ndarray] mapping domain name -> 768-dim unit vector.
    """
    con = db.get_connection(db_path)
    rows = con.execute(
        "SELECT domain, reef_id, n_total, centroid FROM domain_reef_stats"
    ).fetchall()
    con.close()

    # Group by domain
    by_domain = {}
    for domain, reef_id, n_total, centroid_blob in rows:
        if centroid_blob is None:
            continue
        vec = db.unpack_embedding(centroid_blob).astype(np.float64)
        by_domain.setdefault(domain, []).append((vec, n_total))

    centroids = {}
    for domain, entries in by_domain.items():
        vecs = np.vstack([v for v, _ in entries])
        weights = np.array([w for _, w in entries], dtype=np.float64)
        weights /= weights.sum()
        centroid = (vecs * weights[:, np.newaxis]).sum(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        centroids[domain] = centroid.astype(np.float32)

    return centroids


# ---------------------------------------------------------------------------
# 2. Embed domain names
# ---------------------------------------------------------------------------

def embed_domain_names(model, domain_names):
    """Embed each domain name with classification prefix, L2-normalize.

    Returns dict[str, ndarray] mapping domain name -> 768-dim unit vector.
    """
    texts = [config.EMBEDDING_PREFIX + name for name in domain_names]
    raw = model.encode(texts, show_progress_bar=False)
    raw = raw[:, :config.MATRYOSHKA_DIM]

    # L2-normalize
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    raw = raw / norms

    return {name: raw[i] for i, name in enumerate(domain_names)}


# ---------------------------------------------------------------------------
# 3. Embed a text string
# ---------------------------------------------------------------------------

def embed_text(model, text):
    """Embed text with classification prefix, slice to 768 dims, L2-normalize.

    Returns ndarray of shape (768,).
    """
    full = model.encode([config.EMBEDDING_PREFIX + text], show_progress_bar=False)
    vec = full[0, :config.MATRYOSHKA_DIM].astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# 4. Compare rankings
# ---------------------------------------------------------------------------

def compare_rankings(text, sent_emb, centroid_matrix, centroid_domains,
                     name_matrix, name_domains, scorer, reef_name_to_idx):
    """Compare centroid/name cosine rankings with Lagoon z-score rankings.

    Returns dict with per-query comparison results.
    """
    # Centroid cosines
    centroid_cos = centroid_matrix @ sent_emb  # (n_domains,)
    centroid_ranking = sorted(
        zip(centroid_domains, centroid_cos), key=lambda x: x[1], reverse=True
    )

    # Domain-name cosines
    name_cos = name_matrix @ sent_emb  # (n_domains,)
    name_ranking = sorted(
        zip(name_domains, name_cos), key=lambda x: x[1], reverse=True
    )

    # Lagoon z-scores
    z_scores = scorer.score_raw(text)
    lagoon_ranking = []
    for domain, idx in reef_name_to_idx.items():
        lagoon_ranking.append((domain, z_scores[idx]))
    lagoon_ranking.sort(key=lambda x: x[1], reverse=True)

    # Build score dicts for Spearman correlation
    # Use the shared domain set (intersection of all three)
    shared = set(centroid_domains) & set(name_domains) & set(reef_name_to_idx.keys())

    centroid_scores = {d: s for d, s in zip(centroid_domains, centroid_cos) if d in shared}
    name_scores = {d: s for d, s in zip(name_domains, name_cos) if d in shared}
    lagoon_scores = {d: z_scores[reef_name_to_idx[d]] for d in shared}

    # Filter to domains with non-trivial signal in at least one system
    cos_median = np.median(list(centroid_scores.values()))
    active = [d for d in shared
              if lagoon_scores[d] > 0 or centroid_scores[d] > cos_median]

    if len(active) >= 10:
        domains_for_corr = active
    else:
        # Fall back to all shared domains
        domains_for_corr = sorted(shared)

    ordered = sorted(domains_for_corr)
    lagoon_vals = [lagoon_scores[d] for d in ordered]
    centroid_vals = [centroid_scores[d] for d in ordered]
    name_vals = [name_scores[d] for d in ordered]

    rho_centroid = spearmanr(lagoon_vals, centroid_vals).statistic
    rho_name = spearmanr(lagoon_vals, name_vals).statistic

    # Handle NaN (constant arrays)
    if np.isnan(rho_centroid):
        rho_centroid = 0.0
    if np.isnan(rho_name):
        rho_name = 0.0

    # Top-k overlap (Jaccard-style: |intersection| / k)
    lagoon_top10 = {d for d, _ in lagoon_ranking[:10]}
    lagoon_top20 = {d for d, _ in lagoon_ranking[:20]}
    centroid_top10 = {d for d, _ in centroid_ranking[:10]}
    centroid_top20 = {d for d, _ in centroid_ranking[:20]}
    name_top10 = {d for d, _ in name_ranking[:10]}
    name_top20 = {d for d, _ in name_ranking[:20]}

    return {
        "text": text,
        "lagoon_top5": lagoon_ranking[:5],
        "centroid_top5": centroid_ranking[:5],
        "name_top5": name_ranking[:5],
        "rho_centroid": rho_centroid,
        "rho_name": rho_name,
        "centroid_top10_overlap": len(lagoon_top10 & centroid_top10),
        "centroid_top20_overlap": len(lagoon_top20 & centroid_top20),
        "name_top10_overlap": len(lagoon_top10 & name_top10),
        "name_top20_overlap": len(lagoon_top20 & name_top20),
        "n_active": len(domains_for_corr),
    }


# ---------------------------------------------------------------------------
# 5. Word-level tracing
# ---------------------------------------------------------------------------

def build_word_id_to_string(db_path):
    """Build word_id -> word string lookup from v2.db."""
    con = db.get_connection(db_path)
    rows = con.execute("SELECT word_id, word FROM words").fetchall()
    con.close()
    return {wid: word for wid, word in rows}


def trace_lagoon_scoring(text, scorer, reef_name_to_idx, word_id_to_str,
                         top_n=5):
    """Trace word-level contributions to Lagoon's top reef scores.

    Shows exactly which words contribute how much weight to each top reef,
    plus the background model parameters, to distinguish data issues from
    bg model issues.
    """
    # Tokenize
    word_ids, unknown = scorer._tokenizer.process(text)
    if not word_ids:
        print("    [trace] No matched words")
        return

    # Map word_ids back to strings
    matched_words = {}
    for wid in word_ids:
        matched_words[wid] = word_id_to_str.get(wid, f"?id={wid}")

    print(f"    Matched: {', '.join(sorted(matched_words.values()))}")
    if unknown:
        print(f"    Unknown: {', '.join(unknown)}")

    # Accumulate per-reef, per-word contributions
    word_reefs = scorer._word_reefs
    reef_contributions = defaultdict(list)  # reef_id -> [(word, weight_q)]

    for wid in word_ids:
        word_str = matched_words[wid]
        for reef_id, weight_q, _sub in word_reefs[wid]:
            reef_contributions[reef_id].append((word_str, weight_q))

    # Get full z-scores for ranking
    z_scores = scorer.score_raw(text)
    ws = scorer._weight_scale

    # Also compute raw scores for bg decomposition
    scores_q, _ = scorer._accumulate_weights(word_ids)
    raw_scores = [sq / ws for sq in scores_q]

    # Build reef_id -> name lookup
    idx_to_name = {idx: name for name, idx in reef_name_to_idx.items()}

    # Sort reefs by z-score
    ranked = sorted(range(len(z_scores)), key=lambda i: z_scores[i], reverse=True)

    print(f"    {'Reef':<30} {'z':>6} {'raw':>6} {'bg_m':>6} {'bg_s':>6} | Word contributions")
    print(f"    {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*6}-+-{'-'*40}")

    for reef_id in ranked[:top_n]:
        name = idx_to_name.get(reef_id, f"?{reef_id}")
        z = z_scores[reef_id]
        raw = raw_scores[reef_id]
        bg_m = scorer._bg_mean[reef_id]
        bg_s = scorer._bg_std[reef_id]

        # Sort word contributions by weight descending
        contribs = sorted(reef_contributions.get(reef_id, []),
                          key=lambda x: x[1], reverse=True)
        contrib_str = ", ".join(
            f"{w}={wq/ws:.2f}" for w, wq in contribs
        )
        if not contrib_str:
            contrib_str = "(no direct contributions)"

        print(f"    {name:<30} {z:>6.1f} {raw:>6.2f} {bg_m:>6.2f} {bg_s:>6.2f} | {contrib_str}")

    # Also show centroid-expected reefs that Lagoon missed
    # (reefs in centroid top-5 but not in lagoon top-10)
    lagoon_top10 = set(ranked[:10])
    print()

    # Show reefs where a single word dominates (potential data issue indicator)
    suspicious = []
    for reef_id in ranked[:10]:
        contribs = reef_contributions.get(reef_id, [])
        if len(contribs) == 1:
            word_str, wq = contribs[0]
            name = idx_to_name.get(reef_id, f"?{reef_id}")
            suspicious.append((name, word_str, wq / ws, z_scores[reef_id]))

    if suspicious:
        print(f"    Single-word reefs in top-10 (potential data issues):")
        for name, word, score, z in suspicious:
            print(f"      {name:<30} sole contributor: '{word}' "
                  f"(score={score:.2f}, z={z:.1f})")


# ---------------------------------------------------------------------------
# 6. Test corpus
# ---------------------------------------------------------------------------

TEST_CORPUS = {
    "short_phrases": [
        "Pearl Harbor attack Japan",
        "DNA genetic mutation heredity",
        "python snake reptile",
        "apple fruit nutrition",
        "music opera symphony orchestra",
        "French Revolution guillotine",
        "python programming language",
        "computer",
        "queen",
        "earthquake",
        "mercury",
        "jupiter",
        "commander fleet navy",
        "neuron synapse axon dendrite cortex brain neural hippocampus",
        "cannon weapon military",
        "photosynthesis chloroplast",
        "earthquake tectonic plates seismic",
    ],
    "single_sentences": [
        "The mitochondria is the powerhouse of the cell.",
        "Bach composed the Well-Tempered Clavier in 1722.",
        "Tectonic plates collide along subduction zones.",
        "The Federal Reserve adjusts interest rates to control inflation.",
        "Enzymes catalyze biochemical reactions by lowering activation energy.",
        "Shakespeare wrote Hamlet around the year 1600.",
        "Photons exhibit wave-particle duality in quantum mechanics.",
        "The Roman Empire fell in 476 AD after centuries of decline.",
        "Neural networks use backpropagation to minimize the loss function.",
        "Coral reefs support more biodiversity per square meter than any other marine ecosystem.",
    ],
    "ambiguous": [
        "The bass played a deep note that reverberated through the concert hall.",
        "Mercury rises in the thermometer as the planet Mercury orbits the sun.",
        "The bank along the river eroded while the bank downtown closed.",
        "She set a new record for the javelin throw at the track meet.",
        "The crane lifted steel beams as a crane flew overhead.",
    ],
    "multi_topic": [
        "The military deployed sonar technology developed from marine biology research.",
        "Ancient Greek philosophy influenced modern political theory and democratic governance.",
        "Machine learning algorithms now assist radiologists in detecting tumors from medical imaging.",
        "Climate change affects agricultural yields, threatening global food security and economics.",
        "Renaissance art drew inspiration from classical mythology and advances in human anatomy.",
    ],
    "passages": [
        "Stars form when vast clouds of hydrogen gas collapse under their own gravity. "
        "As the core temperature rises to millions of degrees, nuclear fusion ignites. "
        "The outward radiation pressure balances gravitational collapse, creating a stable main-sequence star.",

        "The immune system has two main branches: innate and adaptive immunity. "
        "Innate immunity provides immediate but non-specific defense through barriers and phagocytes. "
        "Adaptive immunity develops targeted antibodies after exposure to specific pathogens.",

        "During the Industrial Revolution, steam engines transformed manufacturing and transportation. "
        "Factories replaced cottage industries, drawing workers into rapidly growing cities. "
        "This urbanization fundamentally altered social structures across Europe and North America.",

        "Volcanic eruptions release magma, ash, and gases from deep within the Earth's mantle. "
        "Pyroclastic flows can travel at hundreds of kilometers per hour. "
        "The 1815 eruption of Mount Tambora caused the 'Year Without a Summer' globally.",

        "Keynesian economics argues that government spending can stabilize the business cycle. "
        "During recessions, fiscal stimulus fills the gap left by reduced private demand. "
        "Critics counter that deficit spending risks inflation and crowds out private investment.",
    ],
}


# ---------------------------------------------------------------------------
# 7. Output formatting
# ---------------------------------------------------------------------------

def print_result(r):
    """Print per-query comparison result."""
    text = r["text"]
    if len(text) > 70:
        text = text[:67] + "..."
    print(f'\n"{text}"')

    def fmt_top5(ranking, fmt_score):
        return "  ".join(f"{d}({fmt_score(s)})" for d, s in ranking)

    print(f"  Lagoon top-5:    {fmt_top5(r['lagoon_top5'], lambda s: f'{s:.1f}')}")
    print(f"  Centroid top-5:  {fmt_top5(r['centroid_top5'], lambda s: f'{s:.2f}')}")
    print(f"  Name top-5:      {fmt_top5(r['name_top5'], lambda s: f'{s:.2f}')}")
    print(f"  Spearman rho:      centroid={r['rho_centroid']:.3f}  name={r['rho_name']:.3f}")
    print(f"  Top-10 overlap:  centroid={r['centroid_top10_overlap']}/10  "
          f"name={r['name_top10_overlap']}/10")


def print_summary(results_by_category):
    """Print summary table."""
    print("\n" + "=" * 80)
    print(f"{'Category':<20} | {'n':>3} | {'rho_centroid':>13} | {'rho_name':>13} | {'top10_overlap':>14}")
    print("-" * 80)

    all_results = []
    for category, results in results_by_category.items():
        n = len(results)
        rho_c = np.mean([r["rho_centroid"] for r in results])
        rho_n = np.mean([r["rho_name"] for r in results])
        ovl_c = np.mean([r["centroid_top10_overlap"] for r in results])
        ovl_n = np.mean([r["name_top10_overlap"] for r in results])
        print(f"{category:<20} | {n:>3} | {rho_c:>13.3f} | {rho_n:>13.3f} | "
              f"c={ovl_c:.1f} n={ovl_n:.1f}")
        all_results.extend(results)

    print("-" * 80)
    n = len(all_results)
    rho_c = np.mean([r["rho_centroid"] for r in all_results])
    rho_n = np.mean([r["rho_name"] for r in all_results])
    ovl_c = np.mean([r["centroid_top10_overlap"] for r in all_results])
    ovl_n = np.mean([r["name_top10_overlap"] for r in all_results])
    print(f"{'OVERALL':<20} | {n:>3} | {rho_c:>13.3f} | {rho_n:>13.3f} | "
          f"c={ovl_c:.1f} n={ovl_n:.1f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sentence-level embedding validation of Lagoon scoring"
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="Show word-level trace for worst-performing queries (top-10 overlap <= 3)",
    )
    parser.add_argument(
        "--trace-all", action="store_true",
        help="Show word-level trace for ALL queries",
    )
    parser.add_argument(
        "-q", "--query", action="append", dest="queries",
        help="Trace specific query text(s) only (implies --trace). Repeatable.",
    )
    parser.add_argument(
        "--trace-top", type=int, default=5,
        help="Number of top reefs to trace per query (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trace_mode = args.trace or args.trace_all or args.queries
    trace_queries = set(args.queries) if args.queries else None

    t_start = time.time()

    # Load embedding model
    print("Loading embedding model...")
    model = embedder.load_model()
    print(f"  Model loaded in {time.time() - t_start:.1f}s")

    # Load domain centroids from v2.db
    print("Loading domain centroids from v2.db...")
    t0 = time.time()
    domain_centroids = load_domain_centroids(DB_PATH)
    print(f"  {len(domain_centroids)} domain centroids loaded in {time.time() - t0:.1f}s")

    # Embed domain names
    print("Embedding domain names...")
    t0 = time.time()
    domain_names_list = sorted(domain_centroids.keys())
    domain_name_embs = embed_domain_names(model, domain_names_list)
    print(f"  {len(domain_name_embs)} domain names embedded in {time.time() - t0:.1f}s")

    # Load Lagoon scorer
    print("Loading Lagoon scorer...")
    t0 = time.time()
    import lagoon
    scorer = lagoon.load()
    print(f"  Lagoon loaded in {time.time() - t0:.1f}s")

    # Build domain -> reef_id mapping from scorer._reef_meta
    reef_name_to_idx = {}
    for rm in scorer._reef_meta:
        reef_name_to_idx[rm.name] = rm.reef_id

    # Build word_id -> string lookup for tracing
    word_id_to_str = None
    if trace_mode:
        print("Loading word_id -> string mapping...")
        word_id_to_str = build_word_id_to_string(DB_PATH)
        print(f"  {len(word_id_to_str)} words loaded")

    # Report overlap between windowsill domains and lagoon reefs
    ws_domains = set(domain_centroids.keys())
    lg_reefs = set(reef_name_to_idx.keys())
    shared = ws_domains & lg_reefs
    print(f"\n  Windowsill domains: {len(ws_domains)}")
    print(f"  Lagoon reefs:       {len(lg_reefs)}")
    print(f"  Shared (matched):   {len(shared)}")

    if len(shared) < len(ws_domains):
        missing = ws_domains - lg_reefs
        print(f"  Missing from Lagoon: {sorted(missing)[:10]}...")

    # Build matrices for fast dot products (shared domains only)
    shared_sorted = sorted(shared)
    centroid_matrix = np.vstack([domain_centroids[d] for d in shared_sorted])
    name_matrix = np.vstack([domain_name_embs[d] for d in shared_sorted])

    # If --query was used, filter the corpus to only matching queries
    if trace_queries:
        corpus = {"selected": list(trace_queries)}
    else:
        corpus = TEST_CORPUS

    print(f"\nRunning {sum(len(v) for v in corpus.values())} test inputs "
          f"across {len(corpus)} categories...\n")
    print("=" * 80)

    results_by_category = {}
    for category, texts in corpus.items():
        print(f"\n--- {category} ({len(texts)} inputs) ---")
        results = []
        for text in texts:
            sent_emb = embed_text(model, text)
            r = compare_rankings(
                text, sent_emb, centroid_matrix, shared_sorted,
                name_matrix, shared_sorted, scorer, reef_name_to_idx
            )
            print_result(r)

            # Decide whether to trace this query
            should_trace = False
            if args.trace_all or trace_queries:
                should_trace = True
            elif args.trace and r["centroid_top10_overlap"] <= 3:
                should_trace = True

            if should_trace and word_id_to_str is not None:
                trace_lagoon_scoring(
                    text, scorer, reef_name_to_idx, word_id_to_str,
                    top_n=args.trace_top,
                )

            results.append(r)
        results_by_category[category] = results

    print_summary(results_by_category)
    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
