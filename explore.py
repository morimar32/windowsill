import numpy as np


def _resolve_word_id(con, word):
    row = con.execute(
        "SELECT word_id, word FROM words WHERE LOWER(word) = LOWER(?)", [word]
    ).fetchone()
    if row is None:
        return None, None
    return row[0], row[1]


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
        SELECT dm.dim_id, dm.value, dm.z_score, ds.threshold_method, ds.is_bimodal, ds.n_members
        FROM dim_memberships dm
        JOIN dim_stats ds ON dm.dim_id = ds.dim_id
        WHERE dm.word_id = ?
        ORDER BY dm.z_score DESC
    """, [word_id]).fetchall()

    print(f"\n'{canonical}' — member of {total} dimensions")
    print(f"{'Dim':>5}  {'Value':>8}  {'Z-score':>8}  {'Method':>7}  {'Bimodal':>7}  {'Members':>7}")
    print("-" * 55)
    for r in rows:
        print(f"{r[0]:>5}  {r[1]:>8.4f}  {r[2]:>8.2f}  {r[3]:>7}  {'yes' if r[4] else 'no':>7}  {r[5]:>7}")


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
    bimodal = "bimodal" if stats["is_bimodal"] else "unimodal"
    print(f"\nDimension {dim_id} — {bimodal}, {method} threshold={stats['threshold']:.4f}")
    print(f"  Members: {stats['n_members']}, Selectivity: {stats['selectivity']:.4f}")
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


def archipelago(con, word):
    word_id, canonical = _resolve_word_id(con, word)
    if word_id is None:
        print(f"Word '{word}' not found.")
        return

    row = con.execute("""
        SELECT archipelago, reef_0, reef_1, reef_2, reef_3
        FROM words WHERE word_id = ?
    """, [word_id]).fetchone()

    arch, r0, r1, r2, r3 = row
    if arch == 0 and r0 == 0 and r1 == 0 and r2 == 0 and r3 == 0:
        print(f"'{canonical}' has no archipelago encoding. Run phase 9b first.")
        return

    counts = con.execute("""
        SELECT
            bit_count(?::BIGINT & 15) as n_arch,
            bit_count(?::BIGINT >> 4) as n_islands,
            bit_count(?::BIGINT) + bit_count(?::BIGINT) +
            bit_count(?::BIGINT) + bit_count(?::BIGINT) as n_reefs
    """, [arch, arch, r0, r1, r2, r3]).fetchone()
    n_arch, n_islands, n_reefs = counts

    print(f"\n'{canonical}' — archipelago profile")
    print(f"  Archipelagos (gen-0): {n_arch}")
    print(f"  Islands (gen-1):      {n_islands}")
    print(f"  Reefs (gen-2):        {n_reefs}")

    # List specific island names using bit positions
    # Gen-0 archipelagos
    gen0_names = con.execute("""
        SELECT s.island_id, s.island_name
        FROM island_stats s
        WHERE s.generation = 0 AND s.island_id >= 0
          AND s.arch_column = 'archipelago'
          AND (?::BIGINT & (1::BIGINT << s.arch_bit)) != 0
        ORDER BY s.island_id
    """, [arch]).fetchall()
    if gen0_names:
        print(f"\n  Archipelagos:")
        for iid, name in gen0_names:
            name_str = name if name else f"(island {iid})"
            print(f"    [{iid}] {name_str}")

    # Gen-1 islands
    gen1_names = con.execute("""
        SELECT s.island_id, s.island_name, s.parent_island_id
        FROM island_stats s
        WHERE s.generation = 1 AND s.island_id >= 0
          AND s.arch_column = 'archipelago'
          AND (?::BIGINT & (1::BIGINT << s.arch_bit)) != 0
        ORDER BY s.parent_island_id, s.island_id
    """, [arch]).fetchall()
    if gen1_names:
        print(f"\n  Islands:")
        for iid, name, parent in gen1_names:
            name_str = name if name else f"(island {iid})"
            print(f"    [{iid}] {name_str} (parent: {parent})")

    # Gen-2 reefs per archipelago
    reef_cols = [("reef_0", r0), ("reef_1", r1), ("reef_2", r2), ("reef_3", r3)]
    any_reefs = False
    for col_name, col_val in reef_cols:
        if col_val == 0:
            continue
        reef_names = con.execute("""
            SELECT s.island_id, s.island_name, s.parent_island_id
            FROM island_stats s
            WHERE s.generation = 2 AND s.island_id >= 0
              AND s.arch_column = ?
              AND (?::BIGINT & (1::BIGINT << s.arch_bit)) != 0
            ORDER BY s.island_id
        """, [col_name, col_val]).fetchall()
        if reef_names:
            if not any_reefs:
                print(f"\n  Reefs:")
                any_reefs = True
            for iid, name, parent in reef_names:
                name_str = name if name else f"(reef {iid})"
                print(f"    [{iid}] {name_str} (parent island: {parent})")


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
            a.archipelago, a.reef_0, a.reef_1, a.reef_2, a.reef_3,
            b.archipelago, b.reef_0, b.reef_1, b.reef_2, b.reef_3
        FROM words a, words b
        WHERE a.word_id = ? AND b.word_id = ?
    """, [id_a, id_b]).fetchone()

    a_arch, a_r0, a_r1, a_r2, a_r3 = row[0], row[1], row[2], row[3], row[4]
    b_arch, b_r0, b_r1, b_r2, b_r3 = row[5], row[6], row[7], row[8], row[9]

    if a_arch == 0 and b_arch == 0:
        print("Neither word has archipelago encoding. Run phase 9b first.")
        return

    # Classify relationship
    reef_overlap = (a_r0 & b_r0) | (a_r1 & b_r1) | (a_r2 & b_r2) | (a_r3 & b_r3)
    island_overlap = a_arch & b_arch & ~15  # bits 4+ (gen-1)
    arch_overlap = a_arch & b_arch & 15  # bits 0-3 (gen-0)

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
            bit_count((?::BIGINT & ?::BIGINT) & 15::BIGINT) as shared_arch,
            bit_count((?::BIGINT & ?::BIGINT) >> 4) as shared_islands,
            bit_count(?::BIGINT & ?::BIGINT) + bit_count(?::BIGINT & ?::BIGINT) +
            bit_count(?::BIGINT & ?::BIGINT) + bit_count(?::BIGINT & ?::BIGINT) as shared_reefs
    """, [a_arch, b_arch, a_arch, b_arch,
          a_r0, b_r0, a_r1, b_r1, a_r2, b_r2, a_r3, b_r3]).fetchone()
    shared_arch, shared_islands, shared_reefs = counts

    print(f"\n'{name_a}' <-> '{name_b}': {classification}")
    print(f"  Shared archipelagos: {shared_arch}")
    print(f"  Shared islands:      {shared_islands}")
    print(f"  Shared reefs:        {shared_reefs}")
