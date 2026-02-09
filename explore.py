import numpy as np

import config


def _resolve_word_id(con, word):
    row = con.execute(
        "SELECT word_id, word FROM words WHERE LOWER(word) = LOWER(?)", [word]
    ).fetchone()
    if row is None:
        return None, None
    return row[0], row[1]


def _dim_island_label(con, dim_id):
    """Return a compact island hierarchy label for a dimension, e.g. 'reef: skin disease pathology'."""
    rows = con.execute("""
        SELECT di.generation, s.island_name, s.island_id
        FROM dim_islands di
        JOIN island_stats s ON di.island_id = s.island_id AND di.generation = s.generation
        WHERE di.dim_id = ? AND di.island_id >= 0
        ORDER BY di.generation DESC
    """, [dim_id]).fetchall()
    if not rows:
        return None
    # Return the most specific (highest generation) name
    gen, name, iid = rows[0]
    gen_label = {0: "archipelago", 1: "island", 2: "reef"}[gen]
    return f"{gen_label}: {name}" if name else f"{gen_label} {iid}"


def _has_pair_overlap_table(con):
    try:
        count = con.execute("SELECT COUNT(*) FROM word_pair_overlap").fetchone()[0]
        return count > 0
    except Exception:
        return False


def what_is(con, word):
    word_id, canonical = _resolve_word_id(con, word)
    if word_id is None:
        print(f"Word '{word}' not found.")
        return

    total = con.execute(
        "SELECT total_dims FROM words WHERE word_id = ?", [word_id]
    ).fetchone()[0]

    rows = con.execute("""
        SELECT dm.dim_id, dm.value, dm.z_score, ds.threshold_method, ds.n_members
        FROM dim_memberships dm
        JOIN dim_stats ds ON dm.dim_id = ds.dim_id
        WHERE dm.word_id = ?
        ORDER BY dm.z_score DESC
    """, [word_id]).fetchall()

    # Get archipelago profile summary
    profile = con.execute("""
        SELECT archipelago, archipelago_ext, reef_0, reef_1, reef_2, reef_3, reef_4, reef_5
        FROM words WHERE word_id = ?
    """, [word_id]).fetchone()
    arch, arch_ext, r0, r1, r2, r3, r4, r5 = profile
    has_encoding = arch != 0 or arch_ext != 0 or r0 != 0 or r1 != 0 or r2 != 0 or r3 != 0 or r4 != 0 or r5 != 0

    print(f"\n'{canonical}' — member of {total} dimensions")

    if has_encoding:
        reef_count = con.execute("""
            SELECT bit_count(?::BIGINT) + bit_count(?::BIGINT) + bit_count(?::BIGINT) +
                   bit_count(?::BIGINT) + bit_count(?::BIGINT) + bit_count(?::BIGINT)
        """, [r0, r1, r2, r3, r4, r5]).fetchone()[0]
        gen0_count = con.execute(
            "SELECT COUNT(*) FROM island_stats WHERE generation = 0 AND island_id >= 0"
        ).fetchone()[0]
        island_count = con.execute(
            "SELECT bit_count(?::BIGINT >> ?) + bit_count(?::BIGINT)", [arch, gen0_count, arch_ext]
        ).fetchone()[0]
        print(f"  Clusters: {island_count} islands, {reef_count} reefs")

    print(f"\n{'Dim':>5}  {'Value':>8}  {'Z-score':>8}  {'Method':>7}  {'Members':>7}")
    print("-" * 45)
    for r in rows:
        print(f"{r[0]:>5}  {r[1]:>8.4f}  {r[2]:>8.2f}  {r[3]:>7}  {r[4]:>7}")


def words_like(con, word, top_n=20):
    word_id, canonical = _resolve_word_id(con, word)
    if word_id is None:
        print(f"Word '{word}' not found.")
        return

    if _has_pair_overlap_table(con):
        rows = con.execute("""
            SELECT
                CASE WHEN wpo.word_id_a = ? THEN wpo.word_id_b ELSE wpo.word_id_a END AS other_id,
                w.word,
                wpo.shared_dims
            FROM word_pair_overlap wpo
            JOIN words w ON w.word_id = CASE WHEN wpo.word_id_a = ? THEN wpo.word_id_b ELSE wpo.word_id_a END
            WHERE wpo.word_id_a = ? OR wpo.word_id_b = ?
            ORDER BY wpo.shared_dims DESC
            LIMIT ?
        """, [word_id, word_id, word_id, word_id, top_n]).fetchall()
    else:
        rows = con.execute("""
            SELECT dm2.word_id, w.word, COUNT(*) AS shared
            FROM dim_memberships dm1
            JOIN dim_memberships dm2 ON dm1.dim_id = dm2.dim_id AND dm1.word_id != dm2.word_id
            JOIN words w ON w.word_id = dm2.word_id
            WHERE dm1.word_id = ?
            GROUP BY dm2.word_id, w.word
            ORDER BY shared DESC
            LIMIT ?
        """, [word_id, top_n]).fetchall()

    print(f"\nWords most similar to '{canonical}':")
    print(f"{'Word':<30}  {'Shared dims':>11}")
    print("-" * 43)
    for r in rows:
        print(f"{r[1]:<30}  {r[2]:>11}")


def dimension_members(con, dim_id, top_n=50):
    ds = con.execute("SELECT * FROM dim_stats WHERE dim_id = ?", [dim_id]).fetchone()
    if ds is None:
        print(f"Dimension {dim_id} not found.")
        return

    cols = [d[0] for d in con.description]
    stats = dict(zip(cols, ds))

    rows = con.execute("""
        SELECT dm.word_id, w.word, dm.value, dm.z_score
        FROM dim_memberships dm
        JOIN words w ON w.word_id = dm.word_id
        WHERE dm.dim_id = ?
        ORDER BY dm.value DESC
        LIMIT ?
    """, [dim_id, top_n]).fetchall()

    method = stats["threshold_method"]
    print(f"\nDimension {dim_id} — {method} threshold={stats['threshold']:.4f}")
    print(f"  Members: {stats['n_members']}, Selectivity: {stats['selectivity']:.4f}")
    island_label = _dim_island_label(con, dim_id)
    if island_label:
        print(f"  Cluster: {island_label}")
    print(f"\n{'Word':<30}  {'Value':>8}  {'Z-score':>8}")
    print("-" * 50)
    for r in rows:
        print(f"{r[1]:<30}  {r[2]:>8.4f}  {r[3]:>8.2f}")


def compare_words(con, word_a, word_b):
    id_a, name_a = _resolve_word_id(con, word_a)
    id_b, name_b = _resolve_word_id(con, word_b)
    if id_a is None:
        print(f"Word '{word_a}' not found.")
        return
    if id_b is None:
        print(f"Word '{word_b}' not found.")
        return

    dims_a = set(r[0] for r in con.execute(
        "SELECT dim_id FROM dim_memberships WHERE word_id = ?", [id_a]
    ).fetchall())
    dims_b = set(r[0] for r in con.execute(
        "SELECT dim_id FROM dim_memberships WHERE word_id = ?", [id_b]
    ).fetchall())

    shared = dims_a & dims_b
    only_a = dims_a - dims_b
    only_b = dims_b - dims_a
    union = dims_a | dims_b
    jaccard = len(shared) / len(union) if union else 0.0

    print(f"\nComparing '{name_a}' vs '{name_b}':")
    print(f"  Dims for '{name_a}': {len(dims_a)}")
    print(f"  Dims for '{name_b}': {len(dims_b)}")
    print(f"  Shared dims:        {len(shared)}")
    print(f"  Unique to '{name_a}': {len(only_a)}")
    print(f"  Unique to '{name_b}': {len(only_b)}")
    print(f"  Jaccard similarity: {jaccard:.4f}")

    if shared:
        print(f"\n  Shared dimensions (top 20): {sorted(shared)[:20]}")
    if only_a:
        print(f"  Unique to '{name_a}' (top 20): {sorted(only_a)[:20]}")
    if only_b:
        print(f"  Unique to '{name_b}' (top 20): {sorted(only_b)[:20]}")


def disambiguate(con, word):
    word_id, canonical = _resolve_word_id(con, word)
    if word_id is None:
        print(f"Word '{word}' not found.")
        return

    dims = con.execute("""
        SELECT dm.dim_id, dm.value, dm.z_score
        FROM dim_memberships dm
        WHERE dm.word_id = ?
        ORDER BY dm.z_score DESC
    """, [word_id]).fetchall()

    if len(dims) < 2:
        print(f"'{canonical}' has too few dimensions ({len(dims)}) to disambiguate.")
        return

    dim_ids = [d[0] for d in dims]
    n = len(dim_ids)

    co_matrix = np.zeros((n, n))
    for i, d1 in enumerate(dim_ids):
        members_i = set(r[0] for r in con.execute(
            "SELECT word_id FROM dim_memberships WHERE dim_id = ?", [d1]
        ).fetchall())
        for j in range(i + 1, n):
            d2 = dim_ids[j]
            members_j = set(r[0] for r in con.execute(
                "SELECT word_id FROM dim_memberships WHERE dim_id = ?", [d2]
            ).fetchall())
            overlap = len(members_i & members_j)
            union = len(members_i | members_j)
            sim = overlap / union if union > 0 else 0
            co_matrix[i, j] = sim
            co_matrix[j, i] = sim

    np.fill_diagonal(co_matrix, 1.0)
    distance = 1.0 - co_matrix

    from sklearn.cluster import AgglomerativeClustering
    n_clusters = min(max(2, n // 5), 5)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance)

    print(f"\n'{canonical}' — potential senses ({n_clusters} clusters from {n} dims):")
    for cluster_id in range(n_clusters):
        cluster_dims = [dim_ids[i] for i in range(n) if labels[i] == cluster_id]
        if not cluster_dims:
            continue

        sample_words = set()
        for d in cluster_dims[:3]:
            rows = con.execute("""
                SELECT w.word FROM dim_memberships dm
                JOIN words w ON w.word_id = dm.word_id
                WHERE dm.dim_id = ? AND dm.word_id != ?
                ORDER BY dm.z_score DESC LIMIT 5
            """, [d, word_id]).fetchall()
            for r in rows:
                sample_words.add(r[0])

        print(f"\n  Sense {cluster_id + 1} ({len(cluster_dims)} dims): {cluster_dims[:10]}")
        print(f"    Related words: {', '.join(list(sample_words)[:10])}")


def find_bridges(con, word_a, word_b, top_n=10):
    id_a, name_a = _resolve_word_id(con, word_a)
    id_b, name_b = _resolve_word_id(con, word_b)
    if id_a is None:
        print(f"Word '{word_a}' not found.")
        return
    if id_b is None:
        print(f"Word '{word_b}' not found.")
        return

    dims_a = set(r[0] for r in con.execute(
        "SELECT dim_id FROM dim_memberships WHERE word_id = ?", [id_a]
    ).fetchall())
    dims_b = set(r[0] for r in con.execute(
        "SELECT dim_id FROM dim_memberships WHERE word_id = ?", [id_b]
    ).fetchall())

    only_a = dims_a - dims_b
    only_b = dims_b - dims_a

    if not only_a or not only_b:
        print("No unique dimensions to bridge between.")
        return

    only_a_list = list(only_a)
    only_b_list = list(only_b)

    placeholders_a = ",".join(["?"] * len(only_a_list))
    placeholders_b = ",".join(["?"] * len(only_b_list))

    rows = con.execute(f"""
        SELECT dm_a.word_id, w.word,
               COUNT(DISTINCT dm_a.dim_id) AS from_a,
               COUNT(DISTINCT dm_b.dim_id) AS from_b,
               COUNT(DISTINCT dm_a.dim_id) + COUNT(DISTINCT dm_b.dim_id) AS total
        FROM dim_memberships dm_a
        JOIN dim_memberships dm_b ON dm_a.word_id = dm_b.word_id
        JOIN words w ON w.word_id = dm_a.word_id
        WHERE dm_a.dim_id IN ({placeholders_a})
          AND dm_b.dim_id IN ({placeholders_b})
          AND dm_a.word_id != ? AND dm_a.word_id != ?
        GROUP BY dm_a.word_id, w.word
        ORDER BY total DESC
        LIMIT ?
    """, only_a_list + only_b_list + [id_a, id_b, top_n]).fetchall()

    print(f"\nBridge words between '{name_a}' and '{name_b}':")
    print(f"{'Word':<30}  {'From A':>6}  {'From B':>6}  {'Total':>5}")
    print("-" * 55)
    for r in rows:
        print(f"{r[1]:<30}  {r[2]:>6}  {r[3]:>6}  {r[4]:>5}")


def dim_info(con, dim_id):
    ds = con.execute("SELECT * FROM dim_stats WHERE dim_id = ?", [dim_id]).fetchone()
    if ds is None:
        print(f"Dimension {dim_id} not found.")
        return

    cols = [d[0] for d in con.description]
    stats = dict(zip(cols, ds))

    print(f"\nDimension {dim_id}:")
    for k, v in stats.items():
        if k == "dim_id":
            continue
        if isinstance(v, float):
            print(f"  {k:>20}: {v:.6f}")
        else:
            print(f"  {k:>20}: {v}")

    # Show island hierarchy for this dimension
    island_rows = con.execute("""
        SELECT di.generation, s.island_id, s.island_name, s.n_dims, s.n_words
        FROM dim_islands di
        JOIN island_stats s ON di.island_id = s.island_id AND di.generation = s.generation
        WHERE di.dim_id = ? AND di.island_id >= 0
        ORDER BY di.generation
    """, [dim_id]).fetchall()
    if island_rows:
        print(f"\n  Island hierarchy:")
        for gen, iid, name, ndims, nwords in island_rows:
            gen_label = {0: "archipelago", 1: "island", 2: "reef"}[gen]
            name_str = name or f"(unnamed {iid})"
            print(f"    {gen_label:>12}: [{iid}] {name_str} ({ndims} dims, {nwords:,} words)")

    print("\n  Top 20 members:")
    dimension_members(con, dim_id, top_n=20)


def search_words(con, pattern):
    rows = con.execute("""
        SELECT w.word_id, w.word, w.total_dims
        FROM words w
        WHERE w.word LIKE ?
        ORDER BY w.word
        LIMIT 50
    """, [pattern]).fetchall()

    print(f"\nWords matching '{pattern}':")
    print(f"{'ID':>7}  {'Word':<30}  {'Dims':>5}")
    print("-" * 45)
    for r in rows:
        print(f"{r[0]:>7}  {r[1]:<30}  {r[2]:>5}")
    if len(rows) == 50:
        print("  (limited to 50 results)")


def senses(con, word):
    word_id, canonical = _resolve_word_id(con, word)
    if word_id is None:
        print(f"Word '{word}' not found.")
        return

    rows = con.execute("""
        SELECT ws.sense_id, ws.pos, ws.synset_name, ws.gloss, ws.total_dims
        FROM word_senses ws
        WHERE ws.word_id = ?
        ORDER BY ws.synset_name
    """, [word_id]).fetchall()

    if not rows:
        pos = con.execute("SELECT pos FROM words WHERE word_id = ?", [word_id]).fetchone()
        if pos and pos[0]:
            print(f"'{canonical}' is unambiguous (POS: {pos[0]}), no separate senses stored.")
        else:
            print(f"No senses found for '{canonical}'. Run phase 5b first.")
        return

    print(f"\n'{canonical}' — {len(rows)} senses:")
    for sense_id, pos, synset, gloss, total_dims in rows:
        print(f"\n  [{synset}] ({pos}) — {total_dims} dims")
        print(f"    {gloss}")

        # Show top dims for this sense
        dims = con.execute("""
            SELECT sdm.dim_id, sdm.value, sdm.z_score
            FROM sense_dim_memberships sdm
            WHERE sdm.sense_id = ?
            ORDER BY sdm.z_score DESC
            LIMIT 10
        """, [sense_id]).fetchall()

        if dims:
            dim_strs = [f"d{d[0]}({d[2]:.1f})" for d in dims]
            print(f"    Top dims: {', '.join(dim_strs)}")


def compositionality(con, word):
    word_id, canonical = _resolve_word_id(con, word)
    if word_id is None:
        print(f"Word '{word}' not found.")
        return

    try:
        row = con.execute("""
            SELECT jaccard, is_compositional, compound_dims, component_union_dims,
                   shared_dims, emergent_dims
            FROM compositionality WHERE word_id = ?
        """, [word_id]).fetchone()
    except Exception:
        print("Compositionality table not found. Run phase 6b first.")
        return

    if row is None:
        cat = con.execute("SELECT category FROM words WHERE word_id = ?", [word_id]).fetchone()
        if cat and cat[0] == "single":
            print(f"'{canonical}' is a single word, not a compound.")
        else:
            print(f"No compositionality data for '{canonical}'.")
        return

    jaccard, is_comp, c_dims, cu_dims, shared, emergent = row
    label = "COMPOSITIONAL" if is_comp else "IDIOMATIC"

    print(f"\n'{canonical}' — {label}")
    print(f"  Jaccard similarity:     {jaccard:.4f}")
    print(f"  Compound dims:          {c_dims}")
    print(f"  Component union dims:   {cu_dims}")
    print(f"  Shared dims:            {shared}")
    print(f"  Emergent dims:          {emergent}")

    # Show components
    components = con.execute("""
        SELECT wc.component_text, wc.component_word_id, w.total_dims
        FROM word_components wc
        LEFT JOIN words w ON w.word_id = wc.component_word_id
        WHERE wc.compound_word_id = ?
        ORDER BY wc.position
    """, [word_id]).fetchall()

    if components:
        print(f"\n  Components:")
        for text, comp_id, comp_dims in components:
            if comp_id:
                print(f"    '{text}' (id={comp_id}, {comp_dims} dims)")
            else:
                print(f"    '{text}' (not in vocabulary)")

    # Show emergent dimensions (in compound but not in any component)
    if emergent > 0:
        compound_dims_set = set(r[0] for r in con.execute(
            "SELECT dim_id FROM dim_memberships WHERE word_id = ?", [word_id]
        ).fetchall())

        comp_ids = [c[1] for c in components if c[1] is not None]
        comp_dims_union = set()
        for cid in comp_ids:
            comp_dims_union.update(r[0] for r in con.execute(
                "SELECT dim_id FROM dim_memberships WHERE word_id = ?", [cid]
            ).fetchall())

        emergent_dims = sorted(compound_dims_set - comp_dims_union)
        print(f"\n  Emergent dimensions: {emergent_dims[:20]}")


def contamination(con, word):
    word_id, canonical = _resolve_word_id(con, word)
    if word_id is None:
        print(f"Word '{word}' not found.")
        return

    rows = con.execute("""
        SELECT dm.dim_id, dm.value, dm.z_score, dm.compound_support, ds.n_members
        FROM dim_memberships dm
        JOIN dim_stats ds ON dm.dim_id = ds.dim_id
        WHERE dm.word_id = ? AND dm.compound_support > 0
        ORDER BY dm.compound_support DESC, dm.z_score DESC
    """, [word_id]).fetchall()

    total_dims = con.execute(
        "SELECT total_dims FROM words WHERE word_id = ?", [word_id]
    ).fetchone()[0]

    print(f"\n'{canonical}' — {total_dims} total dims, {len(rows)} with compound support")

    if not rows:
        print("  No compound contamination detected.")
        return

    print(f"\n{'Dim':>5}  {'Value':>8}  {'Z-score':>8}  {'Support':>8}  {'Dim members':>11}")
    print("-" * 50)
    for dim_id, value, z_score, support, n_members in rows:
        print(f"{dim_id:>5}  {value:>8.4f}  {z_score:>8.2f}  {support:>8}  {n_members:>11}")

    # Show which compounds contribute
    print(f"\n  Compounds containing '{canonical}':")
    compounds = con.execute("""
        SELECT w.word, w.total_dims
        FROM word_components wc
        JOIN words w ON w.word_id = wc.compound_word_id
        WHERE wc.component_word_id = ?
        ORDER BY w.word
    """, [word_id]).fetchall()
    for cword, cdims in compounds:
        print(f"    '{cword}' ({cdims} dims)")


def pos_dims(con, pos):
    pos = pos.lower()
    col_map = {"verb": "verb_enrichment", "adj": "adj_enrichment", "adv": "adv_enrichment"}
    col = col_map.get(pos)

    if col is None:
        print(f"POS must be one of: verb, adj, adv (nouns are the majority baseline)")
        return

    try:
        rows = con.execute(f"""
            SELECT dim_id, {col}, noun_pct, n_members, selectivity
            FROM dim_stats
            WHERE {col} IS NOT NULL
            ORDER BY {col} DESC
            LIMIT 30
        """).fetchall()
    except Exception:
        print("POS enrichment not computed. Run phase 6b first.")
        return

    print(f"\nDimensions most enriched for '{pos}':")
    print(f"{'Dim':>5}  {'Enrichment':>11}  {'Noun%':>6}  {'Members':>7}  {'Selectivity':>11}")
    print("-" * 50)
    for dim_id, enr, noun_pct, n_members, sel in rows:
        print(f"{dim_id:>5}  {enr:>11.2f}  {noun_pct:>6.1%}  {n_members:>7}  {sel:>11.4f}")


def _build_archipelago_tree(rows):
    """Build nested tree from raw SQL rows: arch -> island -> reef -> dims."""
    from collections import OrderedDict

    tree = OrderedDict()
    for row in rows:
        (dim_id, value, z_score, threshold, threshold_method, selectivity,
         arch_id, arch_name, island_id, island_name, reef_id, reef_name) = row

        # Normalize noise/null to None keys
        a_key = None if (arch_id is None or arch_id < 0) else arch_id
        i_key = None if (island_id is None or island_id < 0) else island_id
        r_key = None if (reef_id is None or reef_id < 0) else reef_id

        if a_key not in tree:
            tree[a_key] = {"name": arch_name, "islands": OrderedDict()}
        arch_node = tree[a_key]

        if i_key not in arch_node["islands"]:
            arch_node["islands"][i_key] = {"name": island_name, "reefs": OrderedDict()}
        island_node = arch_node["islands"][i_key]

        if r_key not in island_node["reefs"]:
            island_node["reefs"][r_key] = {"name": reef_name, "dims": []}
        reef_node = island_node["reefs"][r_key]

        margin = value - threshold if (value is not None and threshold is not None) else None
        sel_pct = selectivity * 100 if selectivity is not None else None

        reef_node["dims"].append({
            "dim_id": dim_id, "value": value, "z_score": z_score,
            "threshold": threshold, "margin": margin,
            "sel_pct": sel_pct,
        })

    return tree


def _count_tree_nodes(tree):
    """Count non-None archipelagos, islands, and reefs in the tree."""
    n_arch = sum(1 for k in tree if k is not None)
    n_islands = 0
    n_reefs = 0
    for arch_node in tree.values():
        for i_key, island_node in arch_node["islands"].items():
            if i_key is not None:
                n_islands += 1
            for r_key in island_node["reefs"]:
                if r_key is not None:
                    n_reefs += 1
    return n_arch, n_islands, n_reefs


def _prune_tree_by_depth(tree, qualifying):
    """Remove tree nodes where word-structure depth is below REEF_MIN_DEPTH.

    qualifying: set of (island_id, generation) tuples that pass the depth filter.
    Named reefs are filtered by reef-level depth, unassigned-to-reef dims by
    island-level depth, and unassigned-to-island dims by archipelago-level depth.
    """
    pruned_dims = 0

    for a_key in list(tree.keys()):
        arch_node = tree[a_key]

        for i_key in list(arch_node["islands"].keys()):
            island_node = arch_node["islands"][i_key]

            for r_key in list(island_node["reefs"].keys()):
                should_prune = False
                if r_key is not None:
                    should_prune = (r_key, 2) not in qualifying
                elif i_key is not None:
                    should_prune = (i_key, 1) not in qualifying
                elif a_key is not None:
                    should_prune = (a_key, 0) not in qualifying

                if should_prune:
                    pruned_dims += len(island_node["reefs"][r_key]["dims"])
                    del island_node["reefs"][r_key]

            if not island_node["reefs"]:
                del arch_node["islands"][i_key]

        if not arch_node["islands"]:
            del tree[a_key]

    return pruned_dims


def _render_archipelago_tree(tree, canonical, n_arch, n_islands, n_reefs):
    """Print the nested archipelago tree with dimension-level detail."""
    # ANSI: bold + color for hierarchy names, reset after
    _ARCH  = "\033[1;36m"  # bold cyan
    _ISLE  = "\033[1;32m"  # bold green
    _REEF  = "\033[1;35m"  # bold magenta
    _RST   = "\033[0m"

    print(f"\n'{canonical}' — archipelago profile ({n_arch} archipelagos, {n_islands} islands, {n_reefs} reefs)")
    print()

    for a_key, arch_node in tree.items():
        if a_key is not None:
            a_label = arch_node["name"] if arch_node["name"] else "(unnamed)"
            print(f"{_ARCH}{a_label}{_RST} [arch {a_key}]")
        else:
            print("(unassigned to archipelago)")

        for i_key, island_node in arch_node["islands"].items():
            if i_key is not None:
                i_label = island_node["name"] if island_node["name"] else "(unnamed)"
                print(f"  {_ISLE}{i_label}{_RST} [island {i_key}]")
            else:
                print("  (unassigned to island)")

            for r_key, reef_node in island_node["reefs"].items():
                if r_key is not None:
                    r_label = reef_node["name"] if reef_node["name"] else "(unnamed)"
                    print(f"    {_REEF}{r_label}{_RST} [reef {r_key}]")
                else:
                    print("    (unassigned to reef)")

                for d in reef_node["dims"]:
                    val_s = f"val={d['value']:.4f}" if d["value"] is not None else "val=?"
                    z_s = f"z={d['z_score']:.2f}" if d["z_score"] is not None else "z=?"
                    thr_s = f"thr={d['threshold']:.4f}" if d["threshold"] is not None else "thr=?"
                    if d["margin"] is not None:
                        margin_s = f"\u0394={d['margin']:+.4f}"
                    else:
                        margin_s = "\u0394=?"
                    sel_s = f"sel={d['sel_pct']:.1f}%" if d["sel_pct"] is not None else "sel=?"
                    print(f"      dim {d['dim_id']}  {val_s}  {z_s}  {thr_s}  {margin_s}  {sel_s}")


def archipelago(con, word):
    word_id, canonical = _resolve_word_id(con, word)
    if word_id is None:
        print(f"Word '{word}' not found.")
        return

    row = con.execute("""
        SELECT archipelago, archipelago_ext, reef_0, reef_1, reef_2, reef_3, reef_4, reef_5
        FROM words WHERE word_id = ?
    """, [word_id]).fetchone()

    arch, arch_ext, r0, r1, r2, r3, r4, r5 = row
    if arch == 0 and arch_ext == 0 and r0 == 0 and r1 == 0 and r2 == 0 and r3 == 0 and r4 == 0 and r5 == 0:
        print(f"'{canonical}' has no archipelago encoding. Run phase 9b first.")
        return

    # Compute word's depth at each (island_id, generation)
    min_depth = config.REEF_MIN_DEPTH
    depth_rows = con.execute("""
        SELECT di.island_id, di.generation, COUNT(*) as depth
        FROM dim_memberships dm
        JOIN dim_islands di ON dm.dim_id = di.dim_id
        WHERE dm.word_id = ? AND di.island_id >= 0
        GROUP BY di.island_id, di.generation
    """, [word_id]).fetchall()
    qualifying = {(iid, gen) for iid, gen, d in depth_rows if d >= min_depth}

    rows = con.execute("""
        SELECT
            dm.dim_id, dm.value, dm.z_score,
            ds.threshold, ds.threshold_method, ds.selectivity,
            di0.island_id AS arch_id,   s0.island_name AS arch_name,
            di1.island_id AS island_id, s1.island_name AS island_name,
            di2.island_id AS reef_id,   s2.island_name AS reef_name
        FROM dim_memberships dm
        JOIN dim_stats ds ON dm.dim_id = ds.dim_id
        LEFT JOIN dim_islands di0 ON dm.dim_id = di0.dim_id AND di0.generation = 0
        LEFT JOIN island_stats s0  ON di0.island_id = s0.island_id AND s0.generation = 0
        LEFT JOIN dim_islands di1 ON dm.dim_id = di1.dim_id AND di1.generation = 1
        LEFT JOIN island_stats s1  ON di1.island_id = s1.island_id AND s1.generation = 1
        LEFT JOIN dim_islands di2 ON dm.dim_id = di2.dim_id AND di2.generation = 2
        LEFT JOIN island_stats s2  ON di2.island_id = s2.island_id AND s2.generation = 2
        WHERE dm.word_id = ?
        ORDER BY
            CASE WHEN di0.island_id IS NULL OR di0.island_id < 0 THEN 1 ELSE 0 END,
            di0.island_id,
            CASE WHEN di1.island_id IS NULL OR di1.island_id < 0 THEN 1 ELSE 0 END,
            di1.island_id,
            CASE WHEN di2.island_id IS NULL OR di2.island_id < 0 THEN 1 ELSE 0 END,
            di2.island_id,
            dm.value DESC
    """, [word_id]).fetchall()

    tree = _build_archipelago_tree(rows)
    pruned_dims = _prune_tree_by_depth(tree, qualifying)
    n_arch, n_islands, n_reefs = _count_tree_nodes(tree)
    _render_archipelago_tree(tree, canonical, n_arch, n_islands, n_reefs)
    if pruned_dims > 0:
        total_dims = con.execute(
            "SELECT total_dims FROM words WHERE word_id = ?", [word_id]
        ).fetchone()[0]
        print(f"\n  ({pruned_dims}/{total_dims} dims below depth {min_depth} not shown)")


def relationship(con, word_a, word_b):
    id_a, name_a = _resolve_word_id(con, word_a)
    id_b, name_b = _resolve_word_id(con, word_b)
    if id_a is None:
        print(f"Word '{word_a}' not found.")
        return
    if id_b is None:
        print(f"Word '{word_b}' not found.")
        return

    row = con.execute("""
        SELECT
            a.archipelago, a.archipelago_ext, a.reef_0, a.reef_1, a.reef_2, a.reef_3, a.reef_4, a.reef_5,
            b.archipelago, b.archipelago_ext, b.reef_0, b.reef_1, b.reef_2, b.reef_3, b.reef_4, b.reef_5
        FROM words a, words b
        WHERE a.word_id = ? AND b.word_id = ?
    """, [id_a, id_b]).fetchone()

    a_arch, a_arch_ext, a_r0, a_r1, a_r2, a_r3, a_r4, a_r5 = row[0:8]
    b_arch, b_arch_ext, b_r0, b_r1, b_r2, b_r3, b_r4, b_r5 = row[8:16]

    if a_arch == 0 and a_arch_ext == 0 and b_arch == 0 and b_arch_ext == 0:
        print("Neither word has archipelago encoding. Run phase 9b first.")
        return

    gen0_count = con.execute(
        "SELECT COUNT(*) FROM island_stats WHERE generation = 0 AND island_id >= 0"
    ).fetchone()[0]
    gen0_mask = (1 << gen0_count) - 1

    # Classify relationship
    reef_overlap = (a_r0 & b_r0) | (a_r1 & b_r1) | (a_r2 & b_r2) | (a_r3 & b_r3) | (a_r4 & b_r4) | (a_r5 & b_r5)
    island_overlap = ((a_arch & b_arch) >> gen0_count) | (a_arch_ext & b_arch_ext)  # gen-1 bits
    arch_overlap = a_arch & b_arch & gen0_mask  # gen-0 bits

    if reef_overlap != 0:
        classification = "same reef (closest)"
    elif island_overlap != 0:
        classification = "reef neighbors (siblings)"
    elif arch_overlap != 0:
        classification = "island neighbors (cousins)"
    else:
        classification = "different archipelagos (distant)"

    # Shared structure counts
    counts = con.execute("""
        SELECT
            bit_count((?::BIGINT & ?::BIGINT) & ?::BIGINT) as shared_arch,
            bit_count((?::BIGINT & ?::BIGINT) >> ?) + bit_count(?::BIGINT & ?::BIGINT) as shared_islands,
            bit_count(?::BIGINT & ?::BIGINT) + bit_count(?::BIGINT & ?::BIGINT) +
            bit_count(?::BIGINT & ?::BIGINT) + bit_count(?::BIGINT & ?::BIGINT) +
            bit_count(?::BIGINT & ?::BIGINT) + bit_count(?::BIGINT & ?::BIGINT) as shared_reefs
    """, [a_arch, b_arch, gen0_mask,
          a_arch, b_arch, gen0_count, a_arch_ext, b_arch_ext,
          a_r0, b_r0, a_r1, b_r1, a_r2, b_r2, a_r3, b_r3, a_r4, b_r4, a_r5, b_r5]).fetchone()
    shared_arch, shared_islands, shared_reefs = counts

    print(f"\n'{name_a}' <-> '{name_b}': {classification}")
    print(f"  Shared archipelagos: {shared_arch}")
    print(f"  Shared islands:      {shared_islands}")
    print(f"  Shared reefs:        {shared_reefs}")

    # Show names of shared structures
    if shared_arch > 0:
        shared_arch_names = con.execute("""
            SELECT s.island_id, s.island_name
            FROM island_stats s
            WHERE s.generation = 0 AND s.island_id >= 0
              AND s.arch_column = 'archipelago'
              AND (?::BIGINT & ?::BIGINT & (1::BIGINT << s.arch_bit)) != 0
            ORDER BY s.island_id
        """, [a_arch, b_arch]).fetchall()
        if shared_arch_names:
            names = [n or f"archipelago {i}" for i, n in shared_arch_names]
            print(f"  Shared archipelago names: {', '.join(names)}")

    if shared_islands > 0:
        shared_island_names = con.execute("""
            SELECT s.island_id, s.island_name
            FROM island_stats s
            WHERE s.generation = 1 AND s.island_id >= 0
              AND s.arch_column IN ('archipelago', 'archipelago_ext')
              AND (CASE WHEN s.arch_column = 'archipelago'
                   THEN (?::BIGINT & ?::BIGINT & (1::BIGINT << s.arch_bit)) != 0
                   ELSE (?::BIGINT & ?::BIGINT & (1::BIGINT << s.arch_bit)) != 0
                   END)
            ORDER BY s.island_id
        """, [a_arch, b_arch, a_arch_ext, b_arch_ext]).fetchall()
        if shared_island_names:
            names = [n or f"island {i}" for i, n in shared_island_names]
            print(f"  Shared island names: {', '.join(names)}")

    if shared_reefs > 0:
        shared_reef_names = []
        for col_name, a_val, b_val in [("reef_0", a_r0, b_r0), ("reef_1", a_r1, b_r1),
                                         ("reef_2", a_r2, b_r2), ("reef_3", a_r3, b_r3),
                                         ("reef_4", a_r4, b_r4), ("reef_5", a_r5, b_r5)]:
            overlap = a_val & b_val
            if overlap == 0:
                continue
            reefs = con.execute("""
                SELECT s.island_id, s.island_name
                FROM island_stats s
                WHERE s.generation = 2 AND s.island_id >= 0
                  AND s.arch_column = ?
                  AND (?::BIGINT & (CASE WHEN s.arch_bit = 63 THEN (-9223372036854775808)::BIGINT ELSE 1::BIGINT << s.arch_bit END)) != 0
                ORDER BY s.island_id
            """, [col_name, overlap]).fetchall()
            for i, n in reefs:
                shared_reef_names.append(n or f"reef {i}")
        if shared_reef_names:
            print(f"  Shared reef names: {', '.join(shared_reef_names)}")
