import os

import numpy as np
from tqdm import tqdm

import config
import database


def update_word_dim_counts(con):
    con.execute("""
        UPDATE words SET total_dims = (
            SELECT COUNT(*) FROM dim_memberships dm
            WHERE dm.word_id = words.word_id
        )
    """)
    result = con.execute("SELECT AVG(total_dims), MIN(total_dims), MAX(total_dims) FROM words").fetchone()
    print(f"  Word dim counts updated: avg={result[0]:.1f}, min={result[1]}, max={result[2]}")


def compute_word_specificity(con):
    """Assign specificity bands based on sigma distance from mean total_dims.

    Positive = specific/unique (fewer dims), negative = general/universal (more dims).
    """
    stats = con.execute("SELECT AVG(total_dims), STDDEV(total_dims) FROM words").fetchone()
    mean, std = stats[0], stats[1]
    print(f"  Specificity: mean={mean:.1f}, std={std:.1f}")

    # fewer dims than normal -> positive (specific)
    # more dims than normal -> negative (universal)
    con.execute("""
        UPDATE words SET specificity = CASE
            WHEN total_dims <= ? THEN 2
            WHEN total_dims <= ? THEN 1
            WHEN total_dims >= ? THEN -2
            WHEN total_dims >= ? THEN -1
            ELSE 0
        END
    """, [mean - 2 * std, mean - std, mean + 2 * std, mean + std])

    bands = con.execute("""
        SELECT specificity, COUNT(*), MIN(total_dims), MAX(total_dims)
        FROM words GROUP BY specificity ORDER BY specificity DESC
    """).fetchall()
    for band, cnt, lo, hi in bands:
        label = {2: "highly specific", 1: "specific", 0: "typical",
                 -1: "universal", -2: "highly universal"}[band]
        print(f"    {band:+d} {label:<18s} {cnt:>7,} words  (dims {lo}-{hi})")


def create_views(con):
    con.execute("""
        CREATE OR REPLACE VIEW v_unique_words AS
        SELECT w.word_id, w.word, w.total_dims, w.specificity
        FROM words w
        WHERE w.specificity > 0
        ORDER BY w.total_dims ASC
    """)

    con.execute("""
        CREATE OR REPLACE VIEW v_universal_words AS
        SELECT w.word_id, w.word, w.total_dims, w.specificity
        FROM words w
        WHERE w.specificity < 0
        ORDER BY w.total_dims DESC
    """)

    con.execute("""
        CREATE OR REPLACE VIEW v_selective_dims AS
        SELECT ds.*
        FROM dim_stats ds
        WHERE ds.selectivity < 0.05
        ORDER BY ds.selectivity ASC
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW v_abstract_dims AS
        SELECT ds.* FROM dim_stats ds
        WHERE ds.universal_pct >= {config.ABSTRACT_DIM_THRESHOLD}
        ORDER BY ds.universal_pct DESC
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW v_concrete_dims AS
        SELECT ds.* FROM dim_stats ds
        WHERE ds.universal_pct IS NOT NULL AND ds.universal_pct <= {config.CONCRETE_DIM_THRESHOLD}
        ORDER BY ds.universal_pct ASC
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW v_domain_generals AS
        SELECT w.word_id, w.word, w.total_dims, w.specificity,
               w.arch_concentration, w.sense_spread, w.polysemy_inflated
        FROM words w
        WHERE w.specificity < 0 AND w.arch_concentration >= {config.DOMAIN_GENERAL_THRESHOLD}
        ORDER BY w.arch_concentration DESC
    """)

    print("  Views created: v_unique_words, v_universal_words, v_selective_dims, "
          "v_abstract_dims, v_concrete_dims, v_domain_generals")


def materialize_word_pair_overlap(con, threshold=None):
    if threshold is None:
        threshold = config.PAIR_OVERLAP_THRESHOLD

    print(f"  Materializing word pair overlap (threshold={threshold})...")

    # Constrain DuckDB memory to 50% of system RAM to prevent OOM/freeze.
    # The unconstrained self-join on dim_memberships produces billions of
    # intermediate rows that will otherwise consume all available memory.
    try:
        total_mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        mem_limit_gb = max(total_mem * 0.5 / (1024 ** 3), 2)
        con.execute(f"SET memory_limit='{mem_limit_gb:.0f}GB'")
        print(f"  DuckDB memory limit set to {mem_limit_gb:.0f}GB")
    except Exception:
        pass

    con.execute("DELETE FROM word_pair_overlap")

    # Process in chunks of word_ids to bound intermediate join size.
    # The full self-join produces ~75 billion intermediate rows for 146K words;
    # chunking the left side to 500 words at a time keeps each iteration's
    # intermediate result ~300x smaller.
    word_ids = [r[0] for r in con.execute(
        "SELECT DISTINCT word_id FROM dim_memberships ORDER BY word_id"
    ).fetchall()]

    CHUNK_SIZE = 500
    n_chunks = (len(word_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in tqdm(range(0, len(word_ids), CHUNK_SIZE),
                  total=n_chunks, desc="Pair overlap"):
        chunk = word_ids[i:i + CHUNK_SIZE]
        min_id, max_id = chunk[0], chunk[-1]

        con.execute(f"""
            INSERT INTO word_pair_overlap
            SELECT a.word_id AS word_id_a, b.word_id AS word_id_b,
                   COUNT(*) AS shared_dims
            FROM dim_memberships a
            JOIN dim_memberships b
                ON a.dim_id = b.dim_id AND a.word_id < b.word_id
            WHERE a.word_id >= ? AND a.word_id <= ?
            GROUP BY a.word_id, b.word_id
            HAVING COUNT(*) >= {threshold}
        """, [min_id, max_id])

    count = con.execute("SELECT COUNT(*) FROM word_pair_overlap").fetchone()[0]
    print(f"  Word pair overlap materialized: {count:,} pairs")
    return count


def print_summary(con):
    word_count = con.execute("SELECT COUNT(*) FROM words").fetchone()[0]
    dim_count = con.execute("SELECT COUNT(*) FROM dim_stats").fetchone()[0]
    membership_count = con.execute("SELECT COUNT(*) FROM dim_memberships").fetchone()[0]
    avg_members = con.execute("SELECT AVG(n_members) FROM dim_stats").fetchone()[0]
    avg_selectivity = con.execute("SELECT AVG(selectivity) FROM dim_stats").fetchone()[0]

    avg_dims = con.execute("SELECT AVG(total_dims) FROM words").fetchone()[0]
    min_dims = con.execute("SELECT MIN(total_dims) FROM words").fetchone()[0]
    max_dims = con.execute("SELECT MAX(total_dims) FROM words").fetchone()[0]

    pair_count = 0
    try:
        pair_count = con.execute("SELECT COUNT(*) FROM word_pair_overlap").fetchone()[0]
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("VECTOR DISTILLERY SUMMARY")
    print("=" * 60)
    print(f"Words:              {word_count:,}")
    print(f"Dimensions:         {dim_count}")
    print(f"Total memberships:  {membership_count:,}")
    print(f"Avg members/dim:    {avg_members:.1f}" if avg_members else "")
    print(f"Avg selectivity:    {avg_selectivity:.4f}" if avg_selectivity else "")
    print(f"Avg dims/word:      {avg_dims:.1f}" if avg_dims else "")
    print(f"Min dims/word:      {min_dims}" if min_dims is not None else "")
    print(f"Max dims/word:      {max_dims}" if max_dims is not None else "")
    print(f"Word pairs:         {pair_count:,}")
    print("=" * 60)


def backfill_pos_and_category(con):
    from word_list import classify_word

    rows = con.execute("SELECT word_id, word FROM words ORDER BY word_id").fetchall()
    print(f"  Classifying {len(rows)} words...")

    word_pos_rows = []
    batch_size = 1000
    updates = []

    for word_id, word in tqdm(rows, desc="Classifying"):
        lemma_name = word.replace(" ", "_")
        pos, category, all_pos, pos_counts = classify_word(word, lemma_name)
        word_count = len(word.split())
        updates.append((pos, category, word_count, word_id))

        for p in all_pos:
            word_pos_rows.append((word_id, p, pos_counts[p]))

        if len(updates) >= batch_size:
            _flush_pos_updates(con, updates)
            updates = []

    if updates:
        _flush_pos_updates(con, updates)

    con.execute("DELETE FROM word_pos")
    # Insert in chunks to avoid memory issues
    chunk_size = 50000
    for i in range(0, len(word_pos_rows), chunk_size):
        database.insert_word_pos(con, word_pos_rows[i:i + chunk_size])

    stats = con.execute(
        "SELECT category, count(*) FROM words GROUP BY category ORDER BY count(*) DESC"
    ).fetchall()
    pos_set = con.execute("SELECT count(*) FROM words WHERE pos IS NOT NULL").fetchone()[0]
    print(f"  POS set for {pos_set} words (unambiguous)")
    for cat, cnt in stats:
        print(f"    {cat}: {cnt:,}")


def _flush_pos_updates(con, updates):
    import pandas as pd
    df = pd.DataFrame(updates, columns=["pos", "category", "word_count", "word_id"])
    con.execute("""
        UPDATE words SET
            pos = df.pos,
            category = df.category,
            word_count = df.word_count
        FROM df WHERE words.word_id = df.word_id
    """)


def populate_word_components(con):
    rows = con.execute(
        "SELECT word_id, word FROM words WHERE word_count > 1"
    ).fetchall()
    word_map = con.execute("SELECT word, word_id FROM words").fetchall()
    word_to_id = {r[0]: r[1] for r in word_map}

    print(f"  Decomposing {len(rows)} multi-word entries...")
    con.execute("DELETE FROM word_components")

    components = []
    for compound_id, compound_word in rows:
        parts = compound_word.split()
        for pos, part in enumerate(parts):
            comp_id = word_to_id.get(part)
            components.append((compound_id, comp_id, part, pos))

    database.insert_word_components(con, components)
    print(f"  Inserted {len(components)} component entries")


def compute_pos_enrichment(con):
    # Base rates for unambiguous words
    base = con.execute("""
        SELECT pos, count(*) as cnt
        FROM words WHERE pos IS NOT NULL
        GROUP BY pos
    """).fetchall()
    total = sum(r[1] for r in base)
    base_rates = {r[0]: r[1] / total for r in base}

    print(f"  Base POS rates: {', '.join(f'{k}={v:.3f}' for k, v in sorted(base_rates.items()))}")

    dim_ids = [r[0] for r in con.execute("SELECT dim_id FROM dim_stats ORDER BY dim_id").fetchall()]

    updates = []
    for dim_id in tqdm(dim_ids, desc="POS enrichment"):
        pos_dist = con.execute("""
            SELECT w.pos, count(*) FROM dim_memberships dm
            JOIN words w ON w.word_id = dm.word_id
            WHERE dm.dim_id = ? AND w.pos IS NOT NULL
            GROUP BY w.pos
        """, [dim_id]).fetchall()

        dim_total = sum(r[1] for r in pos_dist)
        if dim_total == 0:
            continue

        dim_rates = {r[0]: r[1] / dim_total for r in pos_dist}
        noun_pct = dim_rates.get("noun", 0.0)
        verb_enr = (dim_rates.get("verb", 0.0) / base_rates.get("verb", 0.001))
        adj_enr = (dim_rates.get("adj", 0.0) / base_rates.get("adj", 0.001))
        adv_enr = (dim_rates.get("adv", 0.0) / base_rates.get("adv", 0.001))

        updates.append((verb_enr, adj_enr, adv_enr, noun_pct, dim_id))

    con.executemany("""
        UPDATE dim_stats SET
            verb_enrichment = ?, adj_enrichment = ?, adv_enrichment = ?, noun_pct = ?
        WHERE dim_id = ?
    """, updates)
    print(f"  Updated POS enrichment for {len(updates)} dimensions")


def compute_dimension_abstractness(con):
    """Compute universal_pct and dim_weight for each dimension."""
    con.execute("""
        UPDATE dim_stats SET universal_pct = sub.upct,
            dim_weight = -log2(GREATEST(sub.upct, 0.01))
        FROM (SELECT dm.dim_id, COUNT(*) FILTER (WHERE w.specificity < 0)::DOUBLE / COUNT(*) as upct
              FROM dim_memberships dm JOIN words w ON dm.word_id = w.word_id GROUP BY dm.dim_id) sub
        WHERE dim_stats.dim_id = sub.dim_id
    """)

    stats = con.execute("""
        SELECT MIN(universal_pct), MAX(universal_pct), MIN(dim_weight), MAX(dim_weight)
        FROM dim_stats WHERE universal_pct IS NOT NULL
    """).fetchone()
    count = con.execute("SELECT COUNT(*) FROM dim_stats WHERE universal_pct IS NOT NULL").fetchone()[0]
    print(f"  Dimension abstractness: {count} dims, universal_pct range [{stats[0]:.3f}, {stats[1]:.3f}], "
          f"dim_weight range [{stats[2]:.2f}, {stats[3]:.2f}]")


def compute_dimension_specificity(con):
    """Compute avg_specificity for each dimension: mean specificity of member words."""
    con.execute("""
        UPDATE dim_stats SET avg_specificity = sub.avg_spec
        FROM (SELECT dm.dim_id, AVG(w.specificity) as avg_spec
              FROM dim_memberships dm JOIN words w ON dm.word_id = w.word_id
              GROUP BY dm.dim_id) sub
        WHERE dim_stats.dim_id = sub.dim_id
    """)

    stats = con.execute("""
        SELECT MIN(avg_specificity), MAX(avg_specificity), AVG(avg_specificity)
        FROM dim_stats WHERE avg_specificity IS NOT NULL
    """).fetchone()
    count = con.execute("SELECT COUNT(*) FROM dim_stats WHERE avg_specificity IS NOT NULL").fetchone()[0]
    print(f"  Dimension specificity: {count} dims, avg_specificity range [{stats[0]:.3f}, {stats[1]:.3f}], "
          f"mean {stats[2]:.3f}")


def compute_sense_spread(con):
    """Compute sense_spread and polysemy_inflated for words with multiple senses."""
    # Check if word_senses table exists and has data
    try:
        sense_count = con.execute("SELECT COUNT(*) FROM word_senses").fetchone()[0]
    except Exception:
        print("  Sense spread: skipped (word_senses table not found, run phase 5b first)")
        return

    if sense_count == 0:
        print("  Sense spread: skipped (word_senses table is empty, run phase 5b first)")
        return

    con.execute("""
        UPDATE words SET sense_spread = sub.spread
        FROM (
            SELECT ws.word_id, MAX(ws.total_dims) - MIN(ws.total_dims) as spread
            FROM word_senses ws
            GROUP BY ws.word_id
            HAVING COUNT(*) >= 2
        ) sub
        WHERE words.word_id = sub.word_id
    """)

    con.execute(f"""
        UPDATE words SET polysemy_inflated = TRUE
        WHERE specificity < 0 AND sense_spread >= {config.SENSE_SPREAD_INFLATED_THRESHOLD}
    """)

    stats = con.execute("""
        SELECT COUNT(*), AVG(sense_spread)
        FROM words WHERE sense_spread IS NOT NULL
    """).fetchone()
    inflated = con.execute("SELECT COUNT(*) FROM words WHERE polysemy_inflated = TRUE").fetchone()[0]
    print(f"  Sense spread: {stats[0]} words with spread, avg={stats[1]:.1f}, {inflated} polysemy-inflated")


def compute_reef_idf(con):
    """Compute reef IDF for each word based on how many reefs it appears in."""
    con.execute("""
        UPDATE words SET reef_idf = sub.idf
        FROM (
            SELECT word_id,
                   LN((207 - COUNT(*) + 0.5) / (COUNT(*) + 0.5) + 1) AS idf
            FROM word_reef_affinity
            GROUP BY word_id
        ) sub
        WHERE words.word_id = sub.word_id
    """)

    stats = con.execute("""
        SELECT COUNT(*), MIN(reef_idf), MAX(reef_idf)
        FROM words WHERE reef_idf IS NOT NULL
    """).fetchone()
    print(f"  Reef IDF: {stats[0]:,} words, range [{stats[1]:.2f}, {stats[2]:.2f}]")


def compute_arch_concentration(con):
    """Compute arch_concentration for universal words (needs island data)."""
    con.execute("""
        UPDATE words SET arch_concentration = sub.concentration
        FROM (
            SELECT dm.word_id, MAX(arch_count)::DOUBLE / SUM(arch_count) as concentration
            FROM (
                SELECT dm.word_id, di.island_id, COUNT(*) as arch_count
                FROM dim_memberships dm
                JOIN dim_islands di ON dm.dim_id = di.dim_id
                WHERE di.generation = 0 AND di.island_id >= 0
                GROUP BY dm.word_id, di.island_id
            ) dm
            GROUP BY dm.word_id HAVING SUM(arch_count) >= 2
        ) sub
        JOIN words w ON w.word_id = sub.word_id
        WHERE words.word_id = sub.word_id AND w.specificity < 0
    """)

    stats = con.execute("""
        SELECT COUNT(*), MIN(arch_concentration), MAX(arch_concentration)
        FROM words WHERE arch_concentration IS NOT NULL
    """).fetchone()
    generals = con.execute(f"""
        SELECT COUNT(*) FROM words
        WHERE arch_concentration >= {config.DOMAIN_GENERAL_THRESHOLD}
    """).fetchone()[0]
    print(f"  Arch concentration: {stats[0]} universal words, range [{stats[1]:.3f}, {stats[2]:.3f}], "
          f"{generals} domain generals (>= {config.DOMAIN_GENERAL_THRESHOLD})")


def score_compound_contamination(con):
    # Find all words that are components of compounds
    component_words = con.execute("""
        SELECT DISTINCT wc.component_word_id
        FROM word_components wc
        WHERE wc.component_word_id IS NOT NULL
    """).fetchall()
    component_ids = [r[0] for r in component_words]
    print(f"  Scoring compound contamination for {len(component_ids)} component words...")

    # Load embeddings for all words we need
    embedding_map = {}
    needed_ids = set(component_ids)

    # Also need compound embeddings
    compounds_for = con.execute("""
        SELECT DISTINCT wc.compound_word_id, wc.component_word_id
        FROM word_components wc
        WHERE wc.component_word_id IS NOT NULL
    """).fetchall()
    for cid, _ in compounds_for:
        needed_ids.add(cid)

    if not needed_ids:
        print("  No components to score")
        return

    id_list = list(needed_ids)
    for i in range(0, len(id_list), 5000):
        chunk = id_list[i:i + 5000]
        placeholders = ",".join(["?"] * len(chunk))
        emb_rows = con.execute(f"""
            SELECT word_id, embedding FROM words WHERE word_id IN ({placeholders})
        """, chunk).fetchall()
        for wid, emb in emb_rows:
            embedding_map[wid] = np.array(emb, dtype=np.float32)

    # Load thresholds
    thresholds = {}
    for row in con.execute("SELECT dim_id, threshold FROM dim_stats").fetchall():
        thresholds[row[0]] = row[1]

    # Reset compound_support
    con.execute("UPDATE dim_memberships SET compound_support = 0")

    updates = []
    for comp_id in tqdm(component_ids, desc="Contamination"):
        if comp_id not in embedding_map:
            continue
        comp_emb = embedding_map[comp_id]

        # Get this word's dimension memberships
        comp_dims = con.execute(
            "SELECT dim_id FROM dim_memberships WHERE word_id = ?", [comp_id]
        ).fetchall()
        comp_dim_set = {r[0] for r in comp_dims}
        if not comp_dim_set:
            continue

        # Get compounds containing this word
        compound_ids = con.execute(
            "SELECT DISTINCT compound_word_id FROM word_components WHERE component_word_id = ?",
            [comp_id]
        ).fetchall()
        compound_ids = [r[0] for r in compound_ids]

        for dim_id in comp_dim_set:
            thresh = thresholds.get(dim_id)
            if thresh is None:
                continue

            support = 0
            for cmpd_id in compound_ids:
                # Is compound also in this dimension?
                in_dim = con.execute(
                    "SELECT 1 FROM dim_memberships WHERE dim_id = ? AND word_id = ?",
                    [dim_id, cmpd_id]
                ).fetchone()
                if in_dim is None:
                    continue

                if cmpd_id in embedding_map:
                    residual = embedding_map[cmpd_id] - comp_emb
                    if residual[dim_id] >= thresh:
                        support += 1

            if support > 0:
                updates.append((support, dim_id, comp_id))

    if updates:
        con.executemany(
            "UPDATE dim_memberships SET compound_support = ? WHERE dim_id = ? AND word_id = ?",
            updates
        )
    print(f"  Updated compound_support for {len(updates)} memberships")


def compute_compositionality(con):
    # Get compound words with their embeddings
    compounds = con.execute("""
        SELECT w.word_id, w.word, w.embedding
        FROM words w
        WHERE w.category IN ('compound', 'phrasal_verb')
    """).fetchall()

    print(f"  Computing compositionality for {len(compounds)} compounds...")

    # Preload all component relationships
    all_components = con.execute("""
        SELECT wc.compound_word_id, wc.component_word_id
        FROM word_components wc
        WHERE wc.component_word_id IS NOT NULL
    """).fetchall()
    compound_to_components = {}
    for cid, comp_id in all_components:
        compound_to_components.setdefault(cid, []).append(comp_id)

    # Load embeddings for components
    comp_ids_needed = set()
    for comps in compound_to_components.values():
        comp_ids_needed.update(comps)

    comp_embeddings = {}
    if comp_ids_needed:
        id_list = list(comp_ids_needed)
        for i in range(0, len(id_list), 5000):
            chunk = id_list[i:i + 5000]
            placeholders = ",".join(["?"] * len(chunk))
            rows = con.execute(f"""
                SELECT word_id, embedding FROM words WHERE word_id IN ({placeholders})
            """, chunk).fetchall()
            for wid, emb in rows:
                comp_embeddings[wid] = np.array(emb, dtype=np.float32)

    # Preload dimension memberships for all relevant words
    all_word_ids = set(r[0] for r in compounds) | comp_ids_needed
    word_dims = {}
    if all_word_ids:
        id_list = list(all_word_ids)
        for i in range(0, len(id_list), 5000):
            chunk = id_list[i:i + 5000]
            placeholders = ",".join(["?"] * len(chunk))
            rows = con.execute(f"""
                SELECT word_id, dim_id FROM dim_memberships WHERE word_id IN ({placeholders})
            """, chunk).fetchall()
            for wid, did in rows:
                word_dims.setdefault(wid, set()).add(did)

    # Create compositionality results table
    con.execute("""
        CREATE OR REPLACE TABLE compositionality (
            word_id INTEGER PRIMARY KEY,
            word TEXT,
            jaccard DOUBLE,
            is_compositional BOOLEAN,
            compound_dims INTEGER,
            component_union_dims INTEGER,
            shared_dims INTEGER,
            emergent_dims INTEGER
        )
    """)

    results = []
    for word_id, word, embedding in tqdm(compounds, desc="Compositionality"):
        comp_list = compound_to_components.get(word_id, [])
        if not comp_list:
            continue

        # Union of component dimension sets
        component_dim_union = set()
        for cid in comp_list:
            component_dim_union |= word_dims.get(cid, set())

        compound_dim_set = word_dims.get(word_id, set())

        shared = compound_dim_set & component_dim_union
        union = compound_dim_set | component_dim_union
        jaccard = len(shared) / len(union) if union else 0.0
        is_compositional = jaccard >= config.COMPOSITIONALITY_THRESHOLD
        emergent = len(compound_dim_set - component_dim_union)

        results.append((
            word_id, word, jaccard, is_compositional,
            len(compound_dim_set), len(component_dim_union),
            len(shared), emergent
        ))

    if results:
        import pandas as pd
        df = pd.DataFrame(results, columns=[
            "word_id", "word", "jaccard", "is_compositional",
            "compound_dims", "component_union_dims", "shared_dims", "emergent_dims"
        ])
        con.execute("INSERT INTO compositionality SELECT * FROM df")

    compositional = sum(1 for r in results if r[3])
    idiomatic = len(results) - compositional
    print(f"  Compositionality: {compositional} compositional, {idiomatic} idiomatic (threshold={config.COMPOSITIONALITY_THRESHOLD})")
