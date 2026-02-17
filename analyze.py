"""
Generate a structured analysis report (ANALYSIS.md) from the windowsill database.

Queries the database read-only and produces a markdown document covering
hierarchy, valence, specificity, POS composition, universal word analytics,
reef quality, and other pipeline metrics.

Usage:
    python analyze.py [--db PATH] [--output PATH] [--stdout-only]
"""

import argparse
import os
from datetime import datetime

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

import config


# Archipelago colors for hierarchy chart (extend if more archipelagos emerge)
CHART_ARCH_COLORS = [
    (0.2, 0.4, 0.8),   # blue
    (0.9, 0.5, 0.1),   # orange
    (0.2, 0.7, 0.3),   # green
    (0.8, 0.2, 0.2),   # red
    (0.6, 0.3, 0.7),   # purple
    (0.1, 0.7, 0.7),   # teal
    (0.8, 0.6, 0.2),   # gold
    (0.5, 0.5, 0.5),   # gray
]


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def collect_metrics(con):
    """Run all analysis queries and return a nested dict of metrics."""
    m = {}

    # --- Hierarchy counts ---
    m["hierarchy"] = config.get_hierarchy_counts(con)

    # --- Noise dims per generation ---
    m["noise_dims"] = {}
    for gen in (0, 1, 2):
        n = con.execute(
            "SELECT COUNT(*) FROM dim_islands WHERE generation = ? AND island_id = -1",
            [gen],
        ).fetchone()[0]
        m["noise_dims"][gen] = n

    # --- Core pipeline metrics ---
    row = con.execute("""
        SELECT COUNT(*) AS total_memberships,
               ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT dim_id), 1) AS avg_members_per_dim,
               ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT word_id), 1) AS avg_dims_per_word
        FROM dim_memberships
    """).fetchone()
    m["pipeline"] = {
        "total_memberships": row[0],
        "avg_members_per_dim": row[1],
        "avg_dims_per_word": row[2],
    }

    # Word-reef affinity rows
    m["pipeline"]["wra_rows"] = con.execute(
        "SELECT COUNT(*) FROM word_reef_affinity"
    ).fetchone()[0]

    # Total words
    m["pipeline"]["total_words"] = con.execute(
        "SELECT COUNT(*) FROM words"
    ).fetchone()[0]

    # --- Reef coverage ---
    total_words = m["pipeline"]["total_words"]
    any_depth = con.execute("""
        SELECT COUNT(DISTINCT word_id) FROM word_reef_affinity
    """).fetchone()[0]
    depth_2 = con.execute(f"""
        SELECT COUNT(DISTINCT word_id)
        FROM word_reef_affinity
        WHERE n_dims >= {config.REEF_MIN_DEPTH}
    """).fetchone()[0]
    m["coverage"] = {
        "any_depth_count": any_depth,
        "any_depth_total": total_words,
        "any_depth_pct": round(100 * any_depth / total_words, 1) if total_words else 0,
        "depth2_count": depth_2,
        "depth2_total": total_words,
        "depth2_pct": round(100 * depth_2 / total_words, 1) if total_words else 0,
    }

    # --- Archipelago details (gen-0) ---
    m["archipelagos"] = con.execute("""
        SELECT s.island_id, s.island_name, s.n_dims, s.n_words,
               ROUND(s.valence, 3) AS valence,
               ROUND(s.avg_specificity, 3) AS avg_specificity,
               ROUND(s.noun_frac, 3) AS noun_frac,
               ROUND(s.verb_frac, 3) AS verb_frac,
               ROUND(s.adj_frac, 3) AS adj_frac,
               ROUND(s.adv_frac, 3) AS adv_frac
        FROM island_stats s
        WHERE s.generation = 0 AND s.island_id >= 0
        ORDER BY s.n_dims DESC
    """).fetchall()

    # --- Island details (gen-1) ---
    m["islands"] = con.execute("""
        SELECT s.island_id, s.island_name, s.parent_island_id, s.n_dims, s.n_words,
               ROUND(s.valence, 3) AS valence,
               ROUND(s.avg_specificity, 3) AS avg_specificity,
               ROUND(s.noun_frac, 3) AS noun_frac,
               ROUND(s.verb_frac, 3) AS verb_frac,
               ROUND(s.adj_frac, 3) AS adj_frac,
               ROUND(s.adv_frac, 3) AS adv_frac
        FROM island_stats s
        WHERE s.generation = 1 AND s.island_id >= 0
        ORDER BY s.parent_island_id, s.n_dims DESC
    """).fetchall()

    # --- Reef details (gen-2) ---
    m["reefs"] = con.execute("""
        SELECT s.island_id, s.island_name, s.parent_island_id, s.n_dims, s.n_words,
               ROUND(s.valence, 3) AS valence,
               ROUND(s.avg_specificity, 3) AS avg_specificity,
               ROUND(s.noun_frac, 3) AS noun_frac,
               ROUND(s.verb_frac, 3) AS verb_frac,
               ROUND(s.adj_frac, 3) AS adj_frac,
               ROUND(s.adv_frac, 3) AS adv_frac
        FROM island_stats s
        WHERE s.generation = 2 AND s.island_id >= 0
        ORDER BY s.parent_island_id, s.n_dims DESC
    """).fetchall()

    # --- Valence analysis ---
    pos_dims = con.execute(f"""
        SELECT COUNT(*) FROM dim_stats
        WHERE valence <= {config.POSITIVE_DIM_VALENCE_THRESHOLD}
    """).fetchone()[0]
    neg_dims = con.execute(f"""
        SELECT COUNT(*) FROM dim_stats
        WHERE valence >= {config.NEGATIVE_DIM_VALENCE_THRESHOLD}
    """).fetchone()[0]
    reef_val = con.execute("""
        SELECT ROUND(MIN(valence), 2), ROUND(MAX(valence), 2)
        FROM island_stats WHERE generation = 2 AND island_id >= 0
    """).fetchone()
    neg_pairs = con.execute("""
        SELECT n_pairs FROM computed_vectors WHERE name = 'negation'
    """).fetchone()

    m["valence"] = {
        "positive_pole_dims": pos_dims,
        "negative_pole_dims": neg_dims,
        "reef_valence_min": reef_val[0],
        "reef_valence_max": reef_val[1],
        "negation_pairs": neg_pairs[0] if neg_pairs else 0,
    }

    # Top 5 most positive and negative reefs
    m["valence"]["top_positive_reefs"] = con.execute("""
        SELECT island_id, island_name, ROUND(valence, 3) AS valence
        FROM island_stats WHERE generation = 2 AND island_id >= 0
        ORDER BY valence ASC LIMIT 5
    """).fetchall()
    m["valence"]["top_negative_reefs"] = con.execute("""
        SELECT island_id, island_name, ROUND(valence, 3) AS valence
        FROM island_stats WHERE generation = 2 AND island_id >= 0
        ORDER BY valence DESC LIMIT 5
    """).fetchall()

    # --- Specificity bands ---
    m["specificity_bands"] = con.execute("""
        SELECT specificity, COUNT(*) AS n_words
        FROM words
        GROUP BY specificity
        ORDER BY specificity
    """).fetchall()

    # --- POS composition (corpus-level averages) ---
    m["pos_corpus"] = con.execute("""
        SELECT ROUND(AVG(noun_frac), 3) AS avg_noun,
               ROUND(AVG(verb_frac), 3) AS avg_verb,
               ROUND(AVG(adj_frac), 3) AS avg_adj,
               ROUND(AVG(adv_frac), 3) AS avg_adv
        FROM dim_stats
    """).fetchone()

    # Most verb-heavy reefs
    m["pos_top_verb_reefs"] = con.execute("""
        SELECT island_id, island_name, ROUND(verb_frac, 3) AS verb_frac, n_dims
        FROM island_stats WHERE generation = 2 AND island_id >= 0
        ORDER BY verb_frac DESC LIMIT 5
    """).fetchall()

    # Most adj-heavy reefs
    m["pos_top_adj_reefs"] = con.execute("""
        SELECT island_id, island_name, ROUND(adj_frac, 3) AS adj_frac, n_dims
        FROM island_stats WHERE generation = 2 AND island_id >= 0
        ORDER BY adj_frac DESC LIMIT 5
    """).fetchall()

    # --- Universal word analytics ---
    m["universal"] = {}
    m["universal"]["count"] = con.execute(
        "SELECT COUNT(*) FROM words WHERE specificity < 0"
    ).fetchone()[0]
    m["universal"]["abstract_dims"] = con.execute(f"""
        SELECT COUNT(*) FROM dim_stats
        WHERE universal_pct >= {config.ABSTRACT_DIM_THRESHOLD}
    """).fetchone()[0]
    m["universal"]["concrete_dims"] = con.execute(f"""
        SELECT COUNT(*) FROM dim_stats
        WHERE universal_pct <= {config.CONCRETE_DIM_THRESHOLD}
    """).fetchone()[0]
    m["universal"]["domain_generals"] = con.execute(f"""
        SELECT COUNT(*) FROM words
        WHERE specificity < 0 AND arch_concentration >= {config.DOMAIN_GENERAL_THRESHOLD}
    """).fetchone()[0]
    m["universal"]["polysemy_inflated"] = con.execute(f"""
        SELECT COUNT(*) FROM words
        WHERE polysemy_inflated = TRUE
    """).fetchone()[0]

    # --- Senses and compounds ---
    m["senses"] = {}
    m["senses"]["total"] = con.execute(
        "SELECT COUNT(*) FROM word_senses"
    ).fetchone()[0]
    m["senses"]["domain_anchored"] = con.execute(
        "SELECT COUNT(*) FROM word_senses WHERE is_domain_anchored = TRUE"
    ).fetchone()[0]
    m["compounds"] = {}
    m["compounds"]["total"] = con.execute(
        "SELECT COUNT(DISTINCT compound_word_id) FROM word_components"
    ).fetchone()[0]
    m["compounds"]["compositional"] = con.execute("""
        SELECT COUNT(*) FROM compositionality
        WHERE is_compositional = TRUE
    """).fetchone()[0]

    # --- Reef edges ---
    m["reef_edges"] = {}
    m["reef_edges"]["count"] = con.execute(
        "SELECT COUNT(*) FROM reef_edges"
    ).fetchone()[0]
    edge_range = con.execute("""
        SELECT ROUND(MIN(containment), 3), ROUND(MAX(containment), 3)
        FROM reef_edges
    """).fetchone()
    m["reef_edges"]["containment_min"] = edge_range[0]
    m["reef_edges"]["containment_max"] = edge_range[1]

    # --- Word variants ---
    m["variants"] = {}
    variant_rows = con.execute("""
        SELECT source, COUNT(*) FROM word_variants GROUP BY source
    """).fetchall()
    for source, cnt in variant_rows:
        m["variants"][source] = cnt
    m["variants"]["total"] = con.execute(
        "SELECT COUNT(*) FROM word_variants"
    ).fetchone()[0]

    # --- Reef IDF range ---
    idf_range = con.execute("""
        SELECT ROUND(MIN(reef_idf), 2), ROUND(MAX(reef_idf), 2)
        FROM words WHERE reef_idf IS NOT NULL
    """).fetchone()
    m["reef_idf"] = {"min": idf_range[0], "max": idf_range[1]}

    # --- Reef quality summary (aggregates only) ---
    # Exclusive word ratio
    ewr = con.execute("""
        WITH reef_words AS (
            SELECT DISTINCT dm.reef_id, dm.word_id
            FROM dim_memberships dm
            JOIN words w ON dm.word_id = w.word_id
            WHERE dm.reef_id IS NOT NULL AND w.word NOT LIKE '%% %%'
        ),
        reef_parent AS (
            SELECT island_id AS reef_id, parent_island_id
            FROM island_stats WHERE generation = 2 AND island_id >= 0
        ),
        word_sibling_spread AS (
            SELECT rw.word_id, rp.parent_island_id, COUNT(DISTINCT rw.reef_id) AS n_reefs
            FROM reef_words rw
            JOIN reef_parent rp ON rw.reef_id = rp.reef_id
            GROUP BY rw.word_id, rp.parent_island_id
        ),
        per_reef AS (
            SELECT rw.reef_id,
                   COUNT(DISTINCT rw.word_id) AS total_words,
                   COUNT(DISTINCT rw.word_id) FILTER (WHERE wss.n_reefs = 1) AS exclusive_words
            FROM reef_words rw
            JOIN reef_parent rp ON rw.reef_id = rp.reef_id
            JOIN word_sibling_spread wss
                ON rw.word_id = wss.word_id AND rp.parent_island_id = wss.parent_island_id
            GROUP BY rw.reef_id
        )
        SELECT ROUND(AVG(exclusive_words * 100.0 / total_words), 1) AS mean_pct,
               ROUND(MEDIAN(exclusive_words * 100.0 / total_words), 1) AS median_pct
        FROM per_reef
    """).fetchone()
    m["quality"] = {
        "exclusive_word_ratio_mean": ewr[0],
        "exclusive_word_ratio_median": ewr[1],
    }

    # Internal Jaccard
    ij = con.execute("""
        WITH reef_dims AS (
            SELECT di.dim_id, di.island_id AS reef_id, ist.parent_island_id
            FROM dim_islands di
            JOIN island_stats ist ON di.island_id = ist.island_id AND ist.generation = 2
            WHERE di.generation = 2 AND di.island_id >= 0
        ),
        all_pairs AS (
            SELECT dim_id_a, dim_id_b, jaccard FROM dim_jaccard
            UNION ALL
            SELECT dim_id_b, dim_id_a, jaccard FROM dim_jaccard
        ),
        internal AS (
            SELECT rd1.reef_id, AVG(ap.jaccard) AS avg_internal
            FROM all_pairs ap
            JOIN reef_dims rd1 ON ap.dim_id_a = rd1.dim_id
            JOIN reef_dims rd2 ON ap.dim_id_b = rd2.dim_id
            WHERE rd1.reef_id = rd2.reef_id
            GROUP BY rd1.reef_id
        )
        SELECT ROUND(AVG(avg_internal), 4) AS mean_ij,
               ROUND(MEDIAN(avg_internal), 4) AS median_ij
        FROM internal
    """).fetchone()
    m["quality"]["internal_jaccard_mean"] = ij[0]
    m["quality"]["internal_jaccard_median"] = ij[1]

    return m


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def format_analysis_md(metrics, db_path, db_size_str):
    """Format the metrics dict into a markdown string."""
    m = metrics
    lines = []

    def ln(s=""):
        lines.append(s)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ln("# Windowsill Database Analysis")
    ln()
    ln(f"> Generated: {now}")
    ln(f"> Database: `{db_path}` ({db_size_str})")
    ln(f"> Key parameters: ZSCORE_THRESHOLD={config.ZSCORE_THRESHOLD}, "
       f"REEF_MIN_DEPTH={config.REEF_MIN_DEPTH}, "
       f"ISLAND_MIN_DIMS_FOR_SUBDIVISION={config.ISLAND_MIN_DIMS_FOR_SUBDIVISION}")
    ln()

    # --- Pipeline Metrics ---
    ln("## Pipeline Metrics")
    ln()
    ln("| Metric | Value |")
    ln("|--------|-------|")
    ln(f"| Total memberships | {m['pipeline']['total_memberships']:,} |")
    ln(f"| Avg members/dim | {m['pipeline']['avg_members_per_dim']:,} |")
    ln(f"| Avg dims/word | {m['pipeline']['avg_dims_per_word']} |")
    ln(f"| Word-reef affinity rows | {m['pipeline']['wra_rows']:,} |")
    ln(f"| Reef coverage (any depth) | {m['coverage']['any_depth_pct']}% ({m['coverage']['any_depth_count']:,} / {m['coverage']['any_depth_total']:,}) |")
    ln(f"| Reef coverage (depth >= {config.REEF_MIN_DEPTH}) | {m['coverage']['depth2_pct']}% ({m['coverage']['depth2_count']:,} / {m['coverage']['depth2_total']:,}) |")
    ln(f"| Reef IDF range | [{m['reef_idf']['min']}, {m['reef_idf']['max']}] |")
    ln(f"| Database file size | {db_size_str} |")
    ln()

    # --- Hierarchy ---
    ln("## Hierarchy")
    ln()
    hc = m["hierarchy"]
    ln(f"- **{hc['n_archs']}** archipelagos (gen-0)")
    ln(f"- **{hc['n_islands']}** islands (gen-1)")
    ln(f"- **{hc['n_reefs']}** reefs (gen-2)")
    ln(f"- Noise dims: gen-0={m['noise_dims'][0]}, gen-1={m['noise_dims'][1]}, gen-2={m['noise_dims'][2]}")
    ln()

    # --- Archipelagos ---
    ln("### Gen-0 Archipelagos")
    ln()
    ln("| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |")
    ln("|----|------|------|-------|---------|-------------|------|------|-----|-----|")
    for row in m["archipelagos"]:
        ln(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:,} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} | {row[9]} |")
    ln()

    # --- Islands (grouped by parent) ---
    ln("### Gen-1 Islands")
    ln()
    # Group by parent
    arch_names = {r[0]: r[1] for r in m["archipelagos"]}
    current_parent = None
    for row in m["islands"]:
        parent_id = row[2]
        if parent_id != current_parent:
            current_parent = parent_id
            parent_name = arch_names.get(parent_id, f"Archipelago {parent_id}")
            ln(f"#### {parent_name} (archipelago {parent_id})")
            ln()
            ln("| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |")
            ln("|----|------|------|-------|---------|-------------|------|------|-----|-----|")
        ln(f"| {row[0]} | {row[1]} | {row[3]} | {row[4]:,} | {row[5]} | {row[6]} | {row[7]} | {row[8]} | {row[9]} | {row[10]} |")
    ln()

    # --- Reefs (grouped by parent island, compact) ---
    ln("### Gen-2 Reefs")
    ln()
    island_names = {r[0]: r[1] for r in m["islands"]}
    current_parent = None
    for row in m["reefs"]:
        parent_id = row[2]
        if parent_id != current_parent:
            current_parent = parent_id
            parent_name = island_names.get(parent_id, f"Island {parent_id}")
            ln(f"#### {parent_name} (island {parent_id})")
            ln()
            ln("| ID | Name | Dims | Words | Valence | Spec |")
            ln("|----|------|------|-------|---------|------|")
        ln(f"| {row[0]} | {row[1]} | {row[3]} | {row[4]:,} | {row[5]} | {row[6]} |")
    ln()

    # --- Valence ---
    ln("## Valence Analysis")
    ln()
    v = m["valence"]
    ln("| Metric | Value |")
    ln("|--------|-------|")
    ln(f"| Positive-pole dims (valence <= {config.POSITIVE_DIM_VALENCE_THRESHOLD}) | {v['positive_pole_dims']} |")
    ln(f"| Negative-pole dims (valence >= {config.NEGATIVE_DIM_VALENCE_THRESHOLD}) | {v['negative_pole_dims']} |")
    ln(f"| Reef valence range | [{v['reef_valence_min']}, {v['reef_valence_max']}] |")
    ln(f"| Negation vector pairs | {v['negation_pairs']} |")
    ln()
    ln("**Top 5 positive-pole reefs** (most negative valence = negation decreases activation):")
    ln()
    ln("| Reef ID | Name | Valence |")
    ln("|---------|------|---------|")
    for r in v["top_positive_reefs"]:
        ln(f"| {r[0]} | {r[1]} | {r[2]} |")
    ln()
    ln("**Top 5 negative-pole reefs** (most positive valence = negation increases activation):")
    ln()
    ln("| Reef ID | Name | Valence |")
    ln("|---------|------|---------|")
    for r in v["top_negative_reefs"]:
        ln(f"| {r[0]} | {r[1]} | {r[2]} |")
    ln()

    # --- Specificity Bands ---
    ln("## Specificity Bands")
    ln()
    ln("| Band | Words |")
    ln("|------|-------|")
    for band, count in m["specificity_bands"]:
        ln(f"| {band} | {count:,} |")
    ln()

    # --- POS Composition ---
    ln("## POS Composition")
    ln()
    pc = m["pos_corpus"]
    ln("**Corpus-level averages (across all dims):**")
    ln()
    ln("| POS | Avg Fraction |")
    ln("|-----|-------------|")
    ln(f"| Noun | {pc[0]} |")
    ln(f"| Verb | {pc[1]} |")
    ln(f"| Adj  | {pc[2]} |")
    ln(f"| Adv  | {pc[3]} |")
    ln()
    ln("**Most verb-heavy reefs:**")
    ln()
    ln("| Reef ID | Name | Verb Frac | Dims |")
    ln("|---------|------|-----------|------|")
    for r in m["pos_top_verb_reefs"]:
        ln(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    ln()
    ln("**Most adjective-heavy reefs:**")
    ln()
    ln("| Reef ID | Name | Adj Frac | Dims |")
    ln("|---------|------|----------|------|")
    for r in m["pos_top_adj_reefs"]:
        ln(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    ln()

    # --- Universal Word Analytics ---
    ln("## Universal Word Analytics")
    ln()
    u = m["universal"]
    ln("| Metric | Value |")
    ln("|--------|-------|")
    ln(f"| Universal words (specificity < 0) | {u['count']:,} |")
    ln(f"| Abstract dims (universal_pct >= {config.ABSTRACT_DIM_THRESHOLD}) | {u['abstract_dims']} |")
    ln(f"| Concrete dims (universal_pct <= {config.CONCRETE_DIM_THRESHOLD}) | {u['concrete_dims']} |")
    ln(f"| Domain generals (arch_concentration >= {config.DOMAIN_GENERAL_THRESHOLD}) | {u['domain_generals']} |")
    ln(f"| Polysemy-inflated (sense_spread >= {config.SENSE_SPREAD_INFLATED_THRESHOLD}) | {u['polysemy_inflated']} |")
    ln()

    # --- Senses and Compounds ---
    ln("## Senses and Compounds")
    ln()
    ln("| Metric | Value |")
    ln("|--------|-------|")
    ln(f"| Total senses | {m['senses']['total']:,} |")
    ln(f"| Domain-anchored senses | {m['senses']['domain_anchored']:,} |")
    ln(f"| Total compounds | {m['compounds']['total']:,} |")
    ln(f"| Compositional | {m['compounds']['compositional']:,} |")
    ln()

    # --- Reef Quality Summary ---
    ln("## Reef Quality Summary")
    ln()
    q = m["quality"]
    ln("| Metric | Mean | Median |")
    ln("|--------|------|--------|")
    ln(f"| Exclusive word ratio (%) | {q['exclusive_word_ratio_mean']} | {q['exclusive_word_ratio_median']} |")
    ln(f"| Internal Jaccard | {q['internal_jaccard_mean']} | {q['internal_jaccard_median']} |")
    ln()

    # --- Reef Edges ---
    ln("## Reef Edges")
    ln()
    re = m["reef_edges"]
    ln("| Metric | Value |")
    ln("|--------|-------|")
    ln(f"| Total reef edges | {re['count']:,} |")
    ln(f"| Containment range | [{re['containment_min']}, {re['containment_max']}] |")
    ln()

    # --- Word Variants ---
    ln("## Word Variants")
    ln()
    wv = m["variants"]
    ln("| Source | Count |")
    ln("|--------|-------|")
    for source in sorted(k for k in wv if k != "total"):
        ln(f"| {source} | {wv[source]:,} |")
    ln(f"| **Total** | **{wv['total']:,}** |")
    ln()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------

def print_stdout_summary(metrics):
    """Print a concise terminal summary."""
    m = metrics
    hc = m["hierarchy"]
    p = m["pipeline"]
    c = m["coverage"]
    v = m["valence"]
    q = m["quality"]

    print("=" * 60)
    print("  WINDOWSILL DATABASE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Hierarchy: {hc['n_archs']} archipelagos, {hc['n_islands']} islands, {hc['n_reefs']} reefs")
    print(f"  Noise dims: gen-0={m['noise_dims'][0]}, gen-1={m['noise_dims'][1]}, gen-2={m['noise_dims'][2]}")
    print(f"  Memberships: {p['total_memberships']:,} total")
    print(f"    Avg members/dim: {p['avg_members_per_dim']:,}  |  Avg dims/word: {p['avg_dims_per_word']}")
    print(f"  Word-reef affinity: {p['wra_rows']:,} rows")
    print(f"  Coverage: any={c['any_depth_pct']}% ({c['any_depth_count']:,}/{c['any_depth_total']:,})")
    print(f"           depth>={config.REEF_MIN_DEPTH}: {c['depth2_pct']}% ({c['depth2_count']:,}/{c['depth2_total']:,})")
    print(f"  Valence: +pole={v['positive_pole_dims']} dims, -pole={v['negative_pole_dims']} dims")
    print(f"    Reef range: [{v['reef_valence_min']}, {v['reef_valence_max']}]")
    print(f"    Negation pairs: {v['negation_pairs']}")
    print(f"  Universal words: {m['universal']['count']:,}")
    print(f"    Abstract dims: {m['universal']['abstract_dims']}  |  Concrete dims: {m['universal']['concrete_dims']}")
    print(f"    Domain generals: {m['universal']['domain_generals']}  |  Polysemy-inflated: {m['universal']['polysemy_inflated']}")
    print(f"  Senses: {m['senses']['total']:,} total, {m['senses']['domain_anchored']:,} domain-anchored")
    print(f"  Compounds: {m['compounds']['total']:,} total, {m['compounds']['compositional']:,} compositional")
    print(f"  Reef quality: exclusive word ratio mean={q['exclusive_word_ratio_mean']}% median={q['exclusive_word_ratio_median']}%")
    print(f"    Internal Jaccard mean={q['internal_jaccard_mean']} median={q['internal_jaccard_median']}")
    print(f"  Reef edges: {m['reef_edges']['count']:,} (containment [{m['reef_edges']['containment_min']}, {m['reef_edges']['containment_max']}])")
    print(f"  Variants: {m['variants']['total']:,} total")
    print(f"  Reef IDF: [{m['reef_idf']['min']}, {m['reef_idf']['max']}]")
    print("=" * 60)


# ---------------------------------------------------------------------------
# File size helper
# ---------------------------------------------------------------------------

def format_file_size(path):
    """Return a human-readable file size string."""
    if not os.path.exists(path):
        return "file not found"
    size_bytes = os.path.getsize(path)
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.2f} GB"
    elif size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} bytes"


# ---------------------------------------------------------------------------
# Hierarchy chart
# ---------------------------------------------------------------------------

def generate_hierarchy_chart(metrics, output_path="great_chart.png"):
    """Generate the full hierarchy chart: islands decomposed into reefs."""
    df_arch = pd.DataFrame(
        metrics["archipelagos"],
        columns=["island_id", "island_name", "n_dims", "n_words", "valence",
                 "avg_specificity", "noun_frac", "verb_frac", "adj_frac", "adv_frac"],
    )
    df_islands = pd.DataFrame(
        metrics["islands"],
        columns=["island_id", "island_name", "parent_island_id", "n_dims", "n_words",
                 "valence", "avg_specificity", "noun_frac", "verb_frac", "adj_frac", "adv_frac"],
    )
    df_reefs = pd.DataFrame(
        metrics["reefs"],
        columns=["island_id", "island_name", "parent_island_id", "n_dims", "n_words",
                 "valence", "avg_specificity", "noun_frac", "verb_frac", "adj_frac", "adv_frac"],
    )

    n_arch = len(df_arch)
    arch_colors = CHART_ARCH_COLORS[:n_arch]

    fig, ax = plt.subplots(figsize=(16, max(10, len(df_islands) * 0.35)))

    y = 0
    yticks = []
    ylabels = []
    y_arch_boundaries = []

    for arch_idx in range(n_arch):
        arch_row = df_arch.iloc[arch_idx]
        arch_id = int(arch_row["island_id"])
        y_start = y

        islands = df_islands[df_islands["parent_island_id"] == arch_id].sort_values(
            "n_dims", ascending=False
        )

        for _, island_row in islands.iterrows():
            island_id = int(island_row["island_id"])
            reefs = df_reefs[df_reefs["parent_island_id"] == island_id].sort_values(
                "n_dims", ascending=False
            )

            base_color = arch_colors[arch_idx]

            if len(reefs) > 0:
                n_children = len(reefs)
                shades = [
                    tuple(c * (0.3 + 0.7 * i / max(n_children - 1, 1)) for c in base_color)
                    for i in range(n_children)
                ]
                cumulative = 0
                for j, (_, reef_row) in enumerate(reefs.iterrows()):
                    ax.barh(y, reef_row["n_dims"], left=cumulative, color=shades[j],
                            edgecolor="white", linewidth=0.3, height=0.7)
                    if reef_row["n_dims"] >= 4:
                        ax.text(cumulative + reef_row["n_dims"] / 2, y,
                                str(int(reef_row["n_dims"])),
                                ha="center", va="center", fontsize=5,
                                color="white", fontweight="bold")
                    cumulative += reef_row["n_dims"]

                noise = int(island_row["n_dims"]) - int(reefs["n_dims"].sum())
                if noise > 0:
                    ax.barh(y, noise, left=cumulative, color="#dddddd",
                            edgecolor="white", linewidth=0.3, height=0.7)
            else:
                ax.barh(y, island_row["n_dims"], color=base_color, alpha=0.4,
                        edgecolor="white", linewidth=0.3, height=0.7)

            island_name = (
                island_row["island_name"]
                if pd.notna(island_row["island_name"])
                else f"island-{island_id}"
            )
            yticks.append(y)
            ylabels.append(f"  {island_name[:35]} ({int(island_row['n_dims'])}d)")
            y += 1

        y_arch_boundaries.append((y_start, y - 1, arch_row["island_name"]))
        y += 0.5

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=5)
    ax.set_xlabel("Number of dimensions")
    ax.set_title("Full Hierarchy: Islands decomposed into Reefs (gray = noise)")
    ax.invert_yaxis()

    for y_start, y_end, name in y_arch_boundaries:
        mid = (y_start + y_end) / 2
        ax.text(ax.get_xlim()[1] * 1.02, mid, name, fontsize=8, fontweight="bold",
                va="center", ha="left", color="#333333")

    patches = [
        mpatches.Patch(color=arch_colors[i], label=str(df_arch.iloc[i]["island_name"])[:40])
        for i in range(n_arch)
    ]
    patches.append(mpatches.Patch(color="#dddddd", label="Noise (unassigned)"))
    ax.legend(handles=patches, loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hierarchy chart to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate windowsill database analysis report")
    parser.add_argument("--db", default=config.DB_PATH, help="Path to DuckDB database")
    parser.add_argument("--output", default="ANALYSIS.md", help="Output markdown file path")
    parser.add_argument("--stdout-only", action="store_true", help="Print summary to stdout only, skip file generation")
    args = parser.parse_args()

    db_path = args.db
    db_size_str = format_file_size(db_path)
    print(f"Connecting to {db_path} ({db_size_str})...")

    con = duckdb.connect(db_path, read_only=True)

    try:
        print("Collecting metrics...")
        metrics = collect_metrics(con)

        print_stdout_summary(metrics)

        if not args.stdout_only:
            md = format_analysis_md(metrics, db_path, db_size_str)
            with open(args.output, "w") as f:
                f.write(md)
            print(f"\nWrote {args.output}")

            generate_hierarchy_chart(metrics)
    finally:
        con.close()


if __name__ == "__main__":
    main()
