"""Reef subdivision quality metrics — split by concrete vs diffuse reef type."""
import duckdb
import config

con = duckdb.connect(config.DB_PATH, read_only=True)

SW = "w.word NOT LIKE '% %'"

# =============================================================================
# First: Understand the concrete/diffuse split
# =============================================================================
print("=" * 80)
print("REEF TYPE CENSUS")
print("=" * 80)

census = con.execute("""
    WITH reef_spec AS (
        SELECT di.island_id AS reef_id,
               AVG(ds.avg_specificity) AS reef_avg_spec,
               COUNT(*) AS n_dims
        FROM dim_islands di
        JOIN dim_stats ds ON di.dim_id = ds.dim_id
        WHERE di.generation = 2 AND di.island_id >= 0
        GROUP BY di.island_id
    ),
    reef_words AS (
        SELECT reef_id, COUNT(*) AS total_memberships, COUNT(DISTINCT word_id) AS distinct_words
        FROM dim_memberships
        WHERE reef_id IS NOT NULL
        GROUP BY reef_id
    ),
    reef_words_single AS (
        SELECT dm.reef_id, COUNT(DISTINCT dm.word_id) AS distinct_single_words
        FROM dim_memberships dm
        JOIN words w ON dm.word_id = w.word_id
        WHERE dm.reef_id IS NOT NULL AND w.word NOT LIKE '%% %%'
        GROUP BY dm.reef_id
    )
    SELECT rs.reef_id, rs.reef_avg_spec, rs.n_dims,
           rw.total_memberships, rw.distinct_words, rws.distinct_single_words,
           ist.island_name, ist.parent_island_id
    FROM reef_spec rs
    JOIN reef_words rw ON rs.reef_id = rw.reef_id
    JOIN reef_words_single rws ON rs.reef_id = rws.reef_id
    JOIN island_stats ist ON rs.reef_id = ist.island_id AND ist.generation = 2
    ORDER BY rs.reef_avg_spec DESC
""").fetchdf()

concrete = census[census['reef_avg_spec'] >= 0]
diffuse = census[census['reef_avg_spec'] < 0]

print(f"\nConcrete reefs (avg_spec >= 0): {len(concrete)}")
print(f"  Total memberships: {concrete['total_memberships'].sum():,}")
print(f"  Total distinct words: {concrete['distinct_words'].sum():,}")
print(f"  Total distinct single-token words: {concrete['distinct_single_words'].sum():,}")
print(f"  n_dims range: {concrete['n_dims'].min()}-{concrete['n_dims'].max()}")
print(f"  avg_specificity range: {concrete['reef_avg_spec'].min():.3f} to {concrete['reef_avg_spec'].max():.3f}")

print(f"\nDiffuse reefs (avg_spec < 0): {len(diffuse)}")
print(f"  Total memberships: {diffuse['total_memberships'].sum():,}")
print(f"  Total distinct words: {diffuse['distinct_words'].sum():,}")
print(f"  Total distinct single-token words: {diffuse['distinct_single_words'].sum():,}")
print(f"  n_dims range: {diffuse['n_dims'].min()}-{diffuse['n_dims'].max()}")
print(f"  avg_specificity range: {diffuse['reef_avg_spec'].min():.3f} to {diffuse['reef_avg_spec'].max():.3f}")

print(f"\nAll {len(concrete)} concrete reefs:")
for _, row in concrete.iterrows():
    print(f"  reef {int(row['reef_id']):>3d} | {row['island_name'] or '(unnamed)':40s} | "
          f"spec={row['reef_avg_spec']:.3f} | {int(row['n_dims'])} dims | "
          f"{int(row['distinct_words']):,} words | {int(row['total_memberships']):,} memberships | "
          f"parent={int(row['parent_island_id'])}")

# Also show the near-concrete (top of diffuse) for context
near_concrete = diffuse.nlargest(10, 'reef_avg_spec')
print(f"\nTop 10 near-concrete diffuse reefs (just below threshold):")
for _, row in near_concrete.iterrows():
    print(f"  reef {int(row['reef_id']):>3d} | {row['island_name'] or '(unnamed)':40s} | "
          f"spec={row['reef_avg_spec']:.3f} | {int(row['n_dims'])} dims | "
          f"{int(row['distinct_words']):,} words")


# =============================================================================
# Helper: classify reef type
# =============================================================================
reef_type_map = {}
for _, row in census.iterrows():
    reef_type_map[row['reef_id']] = 'concrete' if row['reef_avg_spec'] >= 0 else 'diffuse'
reef_spec_map = {row['reef_id']: row['reef_avg_spec'] for _, row in census.iterrows()}
reef_ndims_map = {row['reef_id']: row['n_dims'] for _, row in census.iterrows()}


# =============================================================================
# Metric 1: Exclusive Word Ratio — by type
# =============================================================================
print("\n\n" + "=" * 80)
print("METRIC 1: Exclusive Word Ratio — by reef type")
print("=" * 80)

exclusive_df = con.execute(f"""
    WITH reef_words AS (
        SELECT DISTINCT dm.reef_id, dm.word_id
        FROM dim_memberships dm
        JOIN words w ON dm.word_id = w.word_id
        WHERE dm.reef_id IS NOT NULL AND {SW}
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
    )
    SELECT rw.reef_id,
           rp.parent_island_id,
           COUNT(DISTINCT rw.word_id) AS total_words,
           COUNT(DISTINCT rw.word_id) FILTER (WHERE wss.n_reefs = 1) AS exclusive_words,
           ROUND(COUNT(DISTINCT rw.word_id) FILTER (WHERE wss.n_reefs = 1) * 100.0
                 / COUNT(DISTINCT rw.word_id), 1) AS exclusive_pct
    FROM reef_words rw
    JOIN reef_parent rp ON rw.reef_id = rp.reef_id
    JOIN word_sibling_spread wss
        ON rw.word_id = wss.word_id AND rp.parent_island_id = wss.parent_island_id
    GROUP BY rw.reef_id, rp.parent_island_id
    ORDER BY exclusive_pct
""").fetchdf()

exclusive_df['type'] = exclusive_df['reef_id'].map(reef_type_map)

for reef_type in ['concrete', 'diffuse']:
    subset = exclusive_df[exclusive_df['type'] == reef_type]
    print(f"\n  {reef_type.upper()} ({len(subset)} reefs):")
    print(f"    exclusive_pct: min={subset['exclusive_pct'].min():.1f}%, "
          f"median={subset['exclusive_pct'].median():.1f}%, "
          f"mean={subset['exclusive_pct'].mean():.1f}%, "
          f"max={subset['exclusive_pct'].max():.1f}%")
    print(f"    exclusive_words: min={subset['exclusive_words'].min():,}, "
          f"median={subset['exclusive_words'].median():,.0f}, "
          f"max={subset['exclusive_words'].max():,}")
    print(f"    total_words: min={subset['total_words'].min():,}, "
          f"median={subset['total_words'].median():,.0f}, "
          f"max={subset['total_words'].max():,}")

    # Per-reef detail for concrete (only 8)
    if reef_type == 'concrete':
        print(f"\n    Per-reef detail:")
        for _, row in subset.sort_values('exclusive_pct').iterrows():
            name = census.loc[census['reef_id'] == row['reef_id'], 'island_name'].values[0] or '?'
            ndims = reef_ndims_map.get(row['reef_id'], '?')
            print(f"      reef {int(row['reef_id']):>3d} ({ndims} dims) {name:40s}: "
                  f"{row['exclusive_pct']:>5.1f}% ({int(row['exclusive_words']):,} / {int(row['total_words']):,})")


# =============================================================================
# Metric 2: Internal/External Jaccard — by type
# =============================================================================
print("\n\n" + "=" * 80)
print("METRIC 2: Jaccard Silhouette — by reef type")
print("=" * 80)

jaccard_df = con.execute("""
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
    classified AS (
        SELECT rd1.reef_id,
               ap.jaccard,
               CASE WHEN rd1.reef_id = rd2.reef_id THEN 'internal' ELSE 'external' END AS pair_type
        FROM all_pairs ap
        JOIN reef_dims rd1 ON ap.dim_id_a = rd1.dim_id
        JOIN reef_dims rd2 ON ap.dim_id_b = rd2.dim_id
        WHERE rd1.parent_island_id = rd2.parent_island_id
    )
    SELECT reef_id,
           AVG(jaccard) FILTER (WHERE pair_type = 'internal') AS avg_internal,
           AVG(jaccard) FILTER (WHERE pair_type = 'external') AS avg_external,
           COUNT(*) FILTER (WHERE pair_type = 'internal') AS n_internal_edges,
           COUNT(*) FILTER (WHERE pair_type = 'external') AS n_external_edges,
           ROUND(AVG(jaccard) FILTER (WHERE pair_type = 'internal') /
               NULLIF(AVG(jaccard) FILTER (WHERE pair_type = 'external'), 0), 3) AS ratio
    FROM classified
    GROUP BY reef_id
    HAVING avg_internal IS NOT NULL AND avg_external IS NOT NULL
    ORDER BY ratio
""").fetchdf()

jaccard_df['type'] = jaccard_df['reef_id'].map(reef_type_map)

for reef_type in ['concrete', 'diffuse']:
    subset = jaccard_df[jaccard_df['type'] == reef_type]
    print(f"\n  {reef_type.upper()} ({len(subset)} reefs):")
    print(f"    avg_internal:  min={subset['avg_internal'].min():.4f}, "
          f"median={subset['avg_internal'].median():.4f}, "
          f"mean={subset['avg_internal'].mean():.4f}, "
          f"max={subset['avg_internal'].max():.4f}")
    print(f"    avg_external:  min={subset['avg_external'].min():.4f}, "
          f"median={subset['avg_external'].median():.4f}, "
          f"mean={subset['avg_external'].mean():.4f}, "
          f"max={subset['avg_external'].max():.4f}")
    print(f"    ratio:         min={subset['ratio'].min():.3f}, "
          f"median={subset['ratio'].median():.3f}, "
          f"mean={subset['ratio'].mean():.3f}, "
          f"max={subset['ratio'].max():.3f}")

    if reef_type == 'concrete':
        print(f"\n    Per-reef detail:")
        for _, row in subset.sort_values('ratio').iterrows():
            name = census.loc[census['reef_id'] == row['reef_id'], 'island_name'].values[0] or '?'
            ndims = reef_ndims_map.get(row['reef_id'], '?')
            print(f"      reef {int(row['reef_id']):>3d} ({ndims} dims) {name:40s}: "
                  f"ratio={row['ratio']:.3f} (int={row['avg_internal']:.4f}, ext={row['avg_external']:.4f}, "
                  f"edges: {int(row['n_internal_edges'])} int / {int(row['n_external_edges'])} ext)")

    # Distribution
    print(f"\n    Distribution of ratio:")
    for lo, hi, label in [(0, 1.0, "  < 1.0  (OVER-SPLIT)"),
                           (1.0, 1.5, "  1.0-1.5 (weak)"),
                           (1.5, 2.0, "  1.5-2.0 (moderate)"),
                           (2.0, 3.0, "  2.0-3.0 (good)"),
                           (3.0, 100.0, "  3.0+    (excellent)")]:
        n = ((subset['ratio'] >= lo) & (subset['ratio'] < hi)).sum()
        print(f"      {label}: {n} reefs")


# =============================================================================
# Metric 3: Word Depth — by type
# =============================================================================
print("\n\n" + "=" * 80)
print("METRIC 3: Word Depth — by reef type")
print("=" * 80)

depth_df = con.execute(f"""
    WITH word_depths AS (
        SELECT dm.reef_id, dm.word_id, COUNT(*) AS depth
        FROM dim_memberships dm
        JOIN words w ON dm.word_id = w.word_id
        WHERE dm.reef_id IS NOT NULL AND {SW}
        GROUP BY dm.reef_id, dm.word_id
    )
    SELECT reef_id,
           COUNT(*) AS n_words,
           ROUND(AVG(depth), 3) AS avg_depth,
           MEDIAN(depth) AS median_depth,
           MAX(depth) AS max_depth,
           ROUND(COUNT(*) FILTER (WHERE depth = 1) * 100.0 / COUNT(*), 1) AS pct_depth_1,
           COUNT(*) FILTER (WHERE depth >= 2) AS n_depth_2plus,
           ROUND(COUNT(*) FILTER (WHERE depth >= 2) * 100.0 / COUNT(*), 1) AS pct_depth_2plus,
           COUNT(*) FILTER (WHERE depth >= 3) AS n_depth_3plus,
           ROUND(COUNT(*) FILTER (WHERE depth >= 3) * 100.0 / COUNT(*), 2) AS pct_depth_3plus
    FROM word_depths
    GROUP BY reef_id
    ORDER BY avg_depth DESC
""").fetchdf()

depth_df['type'] = depth_df['reef_id'].map(reef_type_map)
depth_df['n_dims'] = depth_df['reef_id'].map(reef_ndims_map)

for reef_type in ['concrete', 'diffuse']:
    subset = depth_df[depth_df['type'] == reef_type]
    print(f"\n  {reef_type.upper()} ({len(subset)} reefs):")
    print(f"    avg_depth:     min={subset['avg_depth'].min():.3f}, "
          f"median={subset['avg_depth'].median():.3f}, "
          f"mean={subset['avg_depth'].mean():.3f}, "
          f"max={subset['avg_depth'].max():.3f}")
    print(f"    max_depth:     min={subset['max_depth'].min()}, "
          f"median={subset['max_depth'].median():.0f}, "
          f"max={subset['max_depth'].max()}")
    print(f"    pct_depth_1:   min={subset['pct_depth_1'].min():.1f}%, "
          f"median={subset['pct_depth_1'].median():.1f}%, "
          f"mean={subset['pct_depth_1'].mean():.1f}%")
    print(f"    pct_depth_2+:  min={subset['pct_depth_2plus'].min():.1f}%, "
          f"median={subset['pct_depth_2plus'].median():.1f}%, "
          f"mean={subset['pct_depth_2plus'].mean():.1f}%, "
          f"max={subset['pct_depth_2plus'].max():.1f}%")
    print(f"    n_depth_2+:    min={subset['n_depth_2plus'].min():,}, "
          f"median={subset['n_depth_2plus'].median():,.0f}, "
          f"max={subset['n_depth_2plus'].max():,}")
    print(f"    pct_depth_3+:  min={subset['pct_depth_3plus'].min():.2f}%, "
          f"median={subset['pct_depth_3plus'].median():.2f}%, "
          f"mean={subset['pct_depth_3plus'].mean():.2f}%, "
          f"max={subset['pct_depth_3plus'].max():.2f}%")
    print(f"    n_depth_3+:    min={subset['n_depth_3plus'].min():,}, "
          f"median={subset['n_depth_3plus'].median():,.0f}, "
          f"max={subset['n_depth_3plus'].max():,}")

    if reef_type == 'concrete':
        print(f"\n    Per-reef detail:")
        for _, row in subset.sort_values('avg_depth', ascending=False).iterrows():
            name = census.loc[census['reef_id'] == row['reef_id'], 'island_name'].values[0] or '?'
            print(f"      reef {int(row['reef_id']):>3d} ({int(row['n_dims'])} dims) {name:40s}: "
                  f"avg_depth={row['avg_depth']:.3f}, max={int(row['max_depth'])}, "
                  f"depth2+: {int(row['n_depth_2plus']):,} ({row['pct_depth_2plus']:.1f}%), "
                  f"depth3+: {int(row['n_depth_3plus']):,} ({row['pct_depth_3plus']:.2f}%)")

# Depth 2+ words in concrete reefs — how many are there in absolute terms?
print("\n\n  CONCRETE REEF DEPTH 2+ WORDS — absolute counts:")
concrete_reef_ids = list(concrete['reef_id'].astype(int))
for reef_id in concrete_reef_ids:
    name = census.loc[census['reef_id'] == reef_id, 'island_name'].values[0] or '?'
    ndims = reef_ndims_map.get(reef_id, '?')
    deep_words = con.execute(f"""
        WITH word_depths AS (
            SELECT dm.word_id, w.word, COUNT(*) AS depth
            FROM dim_memberships dm
            JOIN words w ON dm.word_id = w.word_id
            WHERE dm.reef_id = {reef_id} AND {SW}
            GROUP BY dm.word_id, w.word
            HAVING COUNT(*) >= 2
        )
        SELECT word, depth FROM word_depths ORDER BY depth DESC LIMIT 15
    """).fetchall()
    total_deep = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT dm.word_id
            FROM dim_memberships dm
            JOIN words w ON dm.word_id = w.word_id
            WHERE dm.reef_id = {reef_id} AND {SW}
            GROUP BY dm.word_id
            HAVING COUNT(*) >= 2
        )
    """).fetchone()[0]
    total = con.execute(f"""
        SELECT COUNT(DISTINCT dm.word_id)
        FROM dim_memberships dm
        JOIN words w ON dm.word_id = w.word_id
        WHERE dm.reef_id = {reef_id} AND {SW}
    """).fetchone()[0]
    print(f"\n  reef {reef_id} ({ndims} dims) {name}")
    print(f"    {total_deep:,} words at depth 2+ out of {total:,} total ({100*total_deep/total:.1f}%)")
    if deep_words:
        print(f"    Top depth-2+ words: {', '.join(f'{w}({d})' for w, d in deep_words[:15])}")


# =============================================================================
# Metric 4: PMI Lift — by type
# =============================================================================
print("\n\n" + "=" * 80)
print("METRIC 4: PMI Lift — by reef type")
print("=" * 80)

pmi_df = con.execute("""
    WITH ranked_pmi AS (
        SELECT island_id, generation, pmi,
               ROW_NUMBER() OVER (PARTITION BY island_id, generation ORDER BY pmi DESC) AS rn
        FROM island_characteristic_words
    ),
    parent_pmi AS (
        SELECT island_id, AVG(pmi) AS avg_top_pmi
        FROM ranked_pmi
        WHERE generation = 1 AND rn <= 10
        GROUP BY island_id
    ),
    child_pmi AS (
        SELECT rp.island_id AS reef_id, AVG(rp.pmi) AS avg_top_pmi
        FROM ranked_pmi rp
        WHERE rp.generation = 2 AND rp.rn <= 10
        GROUP BY rp.island_id
    )
    SELECT cp.reef_id,
           ist.parent_island_id,
           ROUND(cp.avg_top_pmi, 3) AS child_pmi,
           ROUND(pp.avg_top_pmi, 3) AS parent_pmi,
           ROUND(cp.avg_top_pmi / NULLIF(pp.avg_top_pmi, 0), 3) AS pmi_lift
    FROM child_pmi cp
    JOIN island_stats ist ON cp.reef_id = ist.island_id AND ist.generation = 2
    JOIN parent_pmi pp ON ist.parent_island_id = pp.island_id
    ORDER BY pmi_lift
""").fetchdf()

pmi_df['type'] = pmi_df['reef_id'].map(reef_type_map)

for reef_type in ['concrete', 'diffuse']:
    subset = pmi_df[pmi_df['type'] == reef_type]
    print(f"\n  {reef_type.upper()} ({len(subset)} reefs):")
    print(f"    child_pmi:  min={subset['child_pmi'].min():.3f}, "
          f"median={subset['child_pmi'].median():.3f}, "
          f"mean={subset['child_pmi'].mean():.3f}, "
          f"max={subset['child_pmi'].max():.3f}")
    print(f"    parent_pmi: min={subset['parent_pmi'].min():.3f}, "
          f"median={subset['parent_pmi'].median():.3f}, "
          f"mean={subset['parent_pmi'].mean():.3f}, "
          f"max={subset['parent_pmi'].max():.3f}")
    print(f"    pmi_lift:   min={subset['pmi_lift'].min():.3f}, "
          f"median={subset['pmi_lift'].median():.3f}, "
          f"mean={subset['pmi_lift'].mean():.3f}, "
          f"max={subset['pmi_lift'].max():.3f}")

    if reef_type == 'concrete':
        print(f"\n    Per-reef detail:")
        for _, row in subset.sort_values('pmi_lift', ascending=False).iterrows():
            name = census.loc[census['reef_id'] == row['reef_id'], 'island_name'].values[0] or '?'
            ndims = reef_ndims_map.get(row['reef_id'], '?')
            print(f"      reef {int(row['reef_id']):>3d} ({ndims} dims) {name:40s}: "
                  f"lift={row['pmi_lift']:.3f} (child={row['child_pmi']:.3f}, parent={row['parent_pmi']:.3f})")


# =============================================================================
# SYNTHESIS: Composite readiness score for gen-3 subdivision
# =============================================================================
print("\n\n" + "=" * 80)
print("SYNTHESIS: Gen-3 subdivision readiness")
print("=" * 80)

# For each reef, combine metrics into a subdivision readiness assessment
merged = exclusive_df[['reef_id', 'exclusive_pct', 'total_words']].merge(
    jaccard_df[['reef_id', 'ratio', 'n_internal_edges']].rename(columns={'ratio': 'jaccard_ratio'}),
    on='reef_id', how='inner'
).merge(
    depth_df[['reef_id', 'avg_depth', 'n_depth_2plus', 'pct_depth_2plus', 'n_dims', 'max_depth']],
    on='reef_id', how='inner'
).merge(
    pmi_df[['reef_id', 'pmi_lift']],
    on='reef_id', how='inner'
)
merged['type'] = merged['reef_id'].map(reef_type_map)
merged['reef_avg_spec'] = merged['reef_id'].map(reef_spec_map)

# Criteria for "could benefit from gen-3":
# - n_dims >= 4 (enough dims to actually split)
# - n_depth_2plus >= 100 (meaningful number of deep words to separate)
# - max_depth >= 3 (some words go deep enough to create structure)
# - n_internal_edges >= 4 (enough graph structure for Leiden)
candidates = merged[
    (merged['n_dims'] >= 4) &
    (merged['n_depth_2plus'] >= 100) &
    (merged['max_depth'] >= 3)
].sort_values('n_dims', ascending=False)

print(f"\n  Gen-3 candidates (n_dims >= 4, depth2+ >= 100, max_depth >= 3):")
print(f"  Found {len(candidates)} candidates out of {len(merged)} reefs\n")

if len(candidates) > 0:
    for _, row in candidates.iterrows():
        name = census.loc[census['reef_id'] == row['reef_id'], 'island_name'].values[0] or '?'
        print(f"    reef {int(row['reef_id']):>3d} | {row['type']:>8s} | spec={row['reef_avg_spec']:.3f} | "
              f"{int(row['n_dims'])} dims | {int(row['n_depth_2plus']):,} depth2+ | "
              f"max_d={int(row['max_depth'])} | jacc_ratio={row['jaccard_ratio']:.2f} | "
              f"excl={row['exclusive_pct']:.0f}% | lift={row['pmi_lift']:.2f} | {name}")
else:
    print("    None found.")

# Also show how many reefs have >= 4 dims at all
print(f"\n  Context: reefs with >= 4 dims: {(merged['n_dims'] >= 4).sum()}")
print(f"  Context: reefs with >= 5 dims: {(merged['n_dims'] >= 5).sum()}")
print(f"  Context: reefs with >= 3 dims: {(merged['n_dims'] >= 3).sum()}")

# n_dims distribution
print(f"\n  n_dims distribution across all reefs:")
for nd in sorted(merged['n_dims'].unique()):
    n = (merged['n_dims'] == nd).sum()
    nc = ((merged['n_dims'] == nd) & (merged['type'] == 'concrete')).sum()
    nd_str = f"    {int(nd)} dims: {n} reefs"
    if nc > 0:
        nd_str += f" ({nc} concrete)"
    print(nd_str)

con.close()
print("\nDone.")
