import pandas as pd

import config


def create_artificial_dimensions(con):
    """Inject one artificial dimension per qualifying WordNet domain.

    Domain vocabulary is identified via the [domain:X] tag in word_senses glosses.
    Each qualifying domain (>= MIN_ARTIFICIAL_DIM_WORDS distinct words) gets a
    contiguous dim_id starting at MATRYOSHKA_DIM (768).

    Artificial dims participate in the Jaccard matrix and Leiden clustering just
    like natural embedding dimensions, allowing domain vocabulary to organically
    cluster into reefs.
    """
    print("  Creating artificial dimensions from domain-anchored senses...")

    # Clean up any previous artificial dims (idempotent re-run)
    con.execute("DELETE FROM dim_memberships WHERE dim_id >= ?", [config.MATRYOSHKA_DIM])
    con.execute("DELETE FROM dim_stats WHERE is_artificial = TRUE")

    # Extract domains and their word_ids from gloss tags
    domain_words = con.execute("""
        SELECT REGEXP_EXTRACT(gloss, '\\[domain:([^\\]]+)\\]', 1) AS domain,
               word_id
        FROM word_senses
        WHERE is_domain_anchored = TRUE
          AND gloss LIKE '[domain:%'
    """).fetchall()

    if not domain_words:
        print("    No domain-anchored senses found")
        return

    # Group by domain -> set of distinct word_ids
    from collections import defaultdict
    domain_map = defaultdict(set)
    for domain, word_id in domain_words:
        if domain:
            domain_map[domain].add(word_id)

    # Filter to domains meeting minimum size
    min_words = config.MIN_ARTIFICIAL_DIM_WORDS
    qualifying = {d: wids for d, wids in domain_map.items() if len(wids) >= min_words}

    if not qualifying:
        print(f"    No domains with >= {min_words} words")
        return

    # Sort by domain name for deterministic dim_id assignment
    sorted_domains = sorted(qualifying.keys())

    # Preload word total_dims for dim_weight computation
    total_words = con.execute("SELECT COUNT(*) FROM words").fetchone()[0]

    # Build dim_stats and dim_memberships rows
    stats_rows = []
    membership_rows = []
    base_dim_id = config.MATRYOSHKA_DIM

    for i, domain in enumerate(sorted_domains):
        dim_id = base_dim_id + i
        word_ids = qualifying[domain]
        n_members = len(word_ids)

        # universal_pct: fraction of total vocabulary in this dim
        universal_pct = n_members / total_words if total_words > 0 else 0.0

        # dim_weight: inverse of universal_pct (rare dims weight more), capped
        dim_weight = 1.0 / universal_pct if universal_pct > 0 else 1.0

        stats_rows.append({
            "dim_id": dim_id,
            "mean": None,
            "std": None,
            "min_val": None,
            "max_val": None,
            "median": None,
            "skewness": None,
            "kurtosis": None,
            "threshold": config.ZSCORE_THRESHOLD,
            "threshold_method": "artificial",
            "n_members": n_members,
            "selectivity": n_members / total_words if total_words > 0 else 0.0,
            "is_artificial": True,
            "domain": domain,
            "universal_pct": universal_pct,
            "dim_weight": dim_weight,
        })

        for word_id in word_ids:
            membership_rows.append((dim_id, word_id, None, config.ARTIFICIAL_DIM_ZSCORE))

    # Insert dim_stats
    stats_df = pd.DataFrame(stats_rows)
    # Only insert columns that exist in the table
    existing_cols = [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'dim_stats'").fetchall()]
    stats_cols = [c for c in stats_df.columns if c in existing_cols]
    stats_df = stats_df[stats_cols]
    cols = ", ".join(stats_cols)
    con.execute(f"INSERT INTO dim_stats ({cols}) SELECT * FROM stats_df")

    # Insert dim_memberships
    mem_df = pd.DataFrame(membership_rows, columns=["dim_id", "word_id", "value", "z_score"])
    con.execute("INSERT INTO dim_memberships (dim_id, word_id, value, z_score) SELECT * FROM mem_df")

    max_dim_id = base_dim_id + len(sorted_domains) - 1
    total_memberships = len(membership_rows)
    print(f"    Created {len(sorted_domains)} artificial dimensions (dim_ids {base_dim_id}-{max_dim_id})")
    print(f"    {total_memberships:,} artificial dim memberships")

    # Print top domains by size
    top_domains = sorted(qualifying.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for domain, wids in top_domains:
        print(f"      {domain}: {len(wids)} words")
