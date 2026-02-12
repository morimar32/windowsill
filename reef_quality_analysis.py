"""Reef subdivision quality metrics — evaluating whether gen-2 reefs are well-formed."""
import duckdb
import config

con = duckdb.connect(config.DB_PATH, read_only=True)

# Filter for single-token words only (consistent with island detection)
SW = "w.word NOT LIKE '% %'"

# =============================================================================
# Metric 1: Exclusive Word Ratio
# For each gen-2 reef, what fraction of its words are exclusive (not in any sibling)?
# =============================================================================
print("=" * 80)
print("METRIC 1: Exclusive Word Ratio (per gen-2 reef within parent island)")
print("=" * 80)

exclusive_df = con.execute(f"""
    WITH reef_words AS (
        -- All (reef, word) pairs for single-token words
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
        -- For each word + parent island, how many sibling reefs contain it?
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

print(f"\nSummary across {len(exclusive_df)} reefs:")
print(f"  exclusive_pct: min={exclusive_df['exclusive_pct'].min():.1f}%, "
      f"median={exclusive_df['exclusive_pct'].median():.1f}%, "
      f"mean={exclusive_df['exclusive_pct'].mean():.1f}%, "
      f"max={exclusive_df['exclusive_pct'].max():.1f}%")
print(f"  exclusive_words: min={exclusive_df['exclusive_words'].min()}, "
      f"median={exclusive_df['exclusive_words'].median():.0f}, "
      f"max={exclusive_df['exclusive_words'].max()}")

# Distribution buckets
print("\n  Distribution of exclusive_pct:")
for lo, hi, label in [(0, 5, "  0-5%  (very low)"),
                       (5, 10, "  5-10% (low)"),
                       (10, 20, " 10-20% (moderate)"),
                       (20, 40, " 20-40% (healthy)"),
                       (40, 100.1, " 40%+   (very distinct)")]:
    n = ((exclusive_df['exclusive_pct'] >= lo) & (exclusive_df['exclusive_pct'] < hi)).sum()
    print(f"    {label}: {n} reefs")

# Bottom 10
print("\n  Bottom 10 (lowest exclusivity — most redundant with siblings):")
bottom = exclusive_df.head(10)
for _, row in bottom.iterrows():
    print(f"    reef {int(row['reef_id']):>3d} (parent {int(row['parent_island_id']):>2d}): "
          f"{row['exclusive_pct']:>5.1f}% exclusive "
          f"({int(row['exclusive_words']):,} / {int(row['total_words']):,})")

# Top 10
print("\n  Top 10 (highest exclusivity — most distinct from siblings):")
top = exclusive_df.tail(10).iloc[::-1]
for _, row in top.iterrows():
    print(f"    reef {int(row['reef_id']):>3d} (parent {int(row['parent_island_id']):>2d}): "
          f"{row['exclusive_pct']:>5.1f}% exclusive "
          f"({int(row['exclusive_words']):,} / {int(row['total_words']):,})")


# =============================================================================
# Metric 2: Internal / External Jaccard Ratio (Jaccard Silhouette)
# =============================================================================
print("\n" + "=" * 80)
print("METRIC 2: Internal/External Jaccard Ratio (Jaccard Silhouette)")
print("=" * 80)

jaccard_df = con.execute("""
    WITH reef_dims AS (
        SELECT di.dim_id, di.island_id AS reef_id, ist.parent_island_id
        FROM dim_islands di
        JOIN island_stats ist ON di.island_id = ist.island_id AND ist.generation = 2
        WHERE di.generation = 2 AND di.island_id >= 0
    ),
    -- Symmetrize so every reef sees all its edges
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
               NULLIF(AVG(jaccard) FILTER (WHERE pair_type = 'external'), 0), 2) AS ratio
    FROM classified
    GROUP BY reef_id
    HAVING avg_internal IS NOT NULL AND avg_external IS NOT NULL
    ORDER BY ratio
""").fetchdf()

print(f"\nSummary across {len(jaccard_df)} reefs (with both internal and external edges):")
print(f"  avg_internal:  min={jaccard_df['avg_internal'].min():.4f}, "
      f"median={jaccard_df['avg_internal'].median():.4f}, "
      f"mean={jaccard_df['avg_internal'].mean():.4f}, "
      f"max={jaccard_df['avg_internal'].max():.4f}")
print(f"  avg_external:  min={jaccard_df['avg_external'].min():.4f}, "
      f"median={jaccard_df['avg_external'].median():.4f}, "
      f"mean={jaccard_df['avg_external'].mean():.4f}, "
      f"max={jaccard_df['avg_external'].max():.4f}")
print(f"  ratio (int/ext): min={jaccard_df['ratio'].min():.2f}, "
      f"median={jaccard_df['ratio'].median():.2f}, "
      f"mean={jaccard_df['ratio'].mean():.2f}, "
      f"max={jaccard_df['ratio'].max():.2f}")

print("\n  Distribution of internal/external ratio:")
for lo, hi, label in [(0, 1.0, "  < 1.0  (OVER-SPLIT: external > internal)"),
                       (1.0, 1.5, "  1.0-1.5 (weak separation)"),
                       (1.5, 2.0, "  1.5-2.0 (moderate separation)"),
                       (2.0, 3.0, "  2.0-3.0 (good separation)"),
                       (3.0, 100.0, "  3.0+    (excellent separation)")]:
    n = ((jaccard_df['ratio'] >= lo) & (jaccard_df['ratio'] < hi)).sum()
    print(f"    {label}: {n} reefs")

# Bottom 10
print("\n  Bottom 10 (weakest separation — worst splits):")
for _, row in jaccard_df.head(10).iterrows():
    print(f"    reef {int(row['reef_id']):>3d}: ratio={row['ratio']:.2f} "
          f"(int={row['avg_internal']:.4f}, ext={row['avg_external']:.4f}, "
          f"edges: {int(row['n_internal_edges'])} int / {int(row['n_external_edges'])} ext)")

# Top 10
print("\n  Top 10 (strongest separation — best splits):")
for _, row in jaccard_df.tail(10).iloc[::-1].iterrows():
    print(f"    reef {int(row['reef_id']):>3d}: ratio={row['ratio']:.2f} "
          f"(int={row['avg_internal']:.4f}, ext={row['avg_external']:.4f}, "
          f"edges: {int(row['n_internal_edges'])} int / {int(row['n_external_edges'])} ext)")


# =============================================================================
# Metric 3: Word Depth Distribution
# =============================================================================
print("\n" + "=" * 80)
print("METRIC 3: Word Depth Distribution (per gen-2 reef)")
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
           ROUND(AVG(depth), 2) AS avg_depth,
           MEDIAN(depth) AS median_depth,
           MAX(depth) AS max_depth,
           ROUND(COUNT(*) FILTER (WHERE depth = 1) * 100.0 / COUNT(*), 1) AS pct_depth_1,
           ROUND(COUNT(*) FILTER (WHERE depth >= 2) * 100.0 / COUNT(*), 1) AS pct_depth_2plus,
           COUNT(*) FILTER (WHERE depth >= 3) AS n_depth_3plus
    FROM word_depths
    GROUP BY reef_id
    ORDER BY median_depth, avg_depth
""").fetchdf()

# Also get n_dims per reef for context
reef_ndims = con.execute("""
    SELECT island_id AS reef_id, n_dims
    FROM island_stats WHERE generation = 2 AND island_id >= 0
""").fetchdf()
depth_df = depth_df.merge(reef_ndims, on='reef_id', how='left')

print(f"\nSummary across {len(depth_df)} reefs:")
print(f"  median_depth: min={depth_df['median_depth'].min():.1f}, "
      f"median={depth_df['median_depth'].median():.1f}, "
      f"mean={depth_df['median_depth'].mean():.1f}, "
      f"max={depth_df['median_depth'].max():.1f}")
print(f"  avg_depth:    min={depth_df['avg_depth'].min():.2f}, "
      f"median={depth_df['avg_depth'].median():.2f}, "
      f"mean={depth_df['avg_depth'].mean():.2f}, "
      f"max={depth_df['avg_depth'].max():.2f}")
print(f"  pct_depth_1:  min={depth_df['pct_depth_1'].min():.1f}%, "
      f"median={depth_df['pct_depth_1'].median():.1f}%, "
      f"mean={depth_df['pct_depth_1'].mean():.1f}%, "
      f"max={depth_df['pct_depth_1'].max():.1f}%")

print("\n  Distribution of median_depth:")
for lo, hi, label in [(1.0, 1.01, "  = 1.0 (NO depth structure — floor hit)"),
                       (1.01, 1.5, "  1.0-1.5 (minimal depth)"),
                       (1.5, 2.0, "  1.5-2.0 (some structure)"),
                       (2.0, 3.0, "  2.0-3.0 (healthy)"),
                       (3.0, 100.0, "  3.0+    (deep structure)")]:
    n = ((depth_df['median_depth'] >= lo) & (depth_df['median_depth'] < hi)).sum()
    print(f"    {label}: {n} reefs")

# Depth vs n_dims relationship
print("\n  Depth by reef size (n_dims):")
for ndim_lo, ndim_hi in [(2, 3), (3, 5), (5, 8), (8, 15), (15, 100)]:
    subset = depth_df[(depth_df['n_dims'] >= ndim_lo) & (depth_df['n_dims'] < ndim_hi)]
    if len(subset) > 0:
        print(f"    {ndim_lo}-{ndim_hi-1} dims ({len(subset)} reefs): "
              f"median_depth avg={subset['median_depth'].mean():.2f}, "
              f"pct_depth_1 avg={subset['pct_depth_1'].mean():.1f}%")

# Bottom 10 (shallowest)
print("\n  Bottom 10 (shallowest reefs):")
for _, row in depth_df.head(10).iterrows():
    print(f"    reef {int(row['reef_id']):>3d} ({int(row['n_dims'])} dims): "
          f"median={row['median_depth']:.1f}, avg={row['avg_depth']:.2f}, "
          f"depth=1: {row['pct_depth_1']:.0f}%, max_depth={int(row['max_depth'])}")


# =============================================================================
# Metric 4: PMI Lift (child gen-2 vs parent gen-1)
# =============================================================================
print("\n" + "=" * 80)
print("METRIC 4: PMI Lift (gen-2 reef top-10 PMI vs gen-1 parent top-10 PMI)")
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
           ROUND(cp.avg_top_pmi / NULLIF(pp.avg_top_pmi, 0), 2) AS pmi_lift
    FROM child_pmi cp
    JOIN island_stats ist ON cp.reef_id = ist.island_id AND ist.generation = 2
    JOIN parent_pmi pp ON ist.parent_island_id = pp.island_id
    ORDER BY pmi_lift
""").fetchdf()

print(f"\nSummary across {len(pmi_df)} reefs:")
print(f"  child_pmi:  min={pmi_df['child_pmi'].min():.3f}, "
      f"median={pmi_df['child_pmi'].median():.3f}, "
      f"mean={pmi_df['child_pmi'].mean():.3f}, "
      f"max={pmi_df['child_pmi'].max():.3f}")
print(f"  parent_pmi: min={pmi_df['parent_pmi'].min():.3f}, "
      f"median={pmi_df['parent_pmi'].median():.3f}, "
      f"mean={pmi_df['parent_pmi'].mean():.3f}, "
      f"max={pmi_df['parent_pmi'].max():.3f}")
print(f"  pmi_lift:   min={pmi_df['pmi_lift'].min():.2f}, "
      f"median={pmi_df['pmi_lift'].median():.2f}, "
      f"mean={pmi_df['pmi_lift'].mean():.2f}, "
      f"max={pmi_df['pmi_lift'].max():.2f}")

print("\n  Distribution of PMI lift:")
for lo, hi, label in [(0, 0.8, "  < 0.8  (WORSE than parent — over-split?)"),
                       (0.8, 1.0, "  0.8-1.0 (no improvement)"),
                       (1.0, 1.2, "  1.0-1.2 (slight improvement)"),
                       (1.2, 1.5, "  1.2-1.5 (clear improvement)"),
                       (1.5, 2.0, "  1.5-2.0 (strong improvement)"),
                       (2.0, 100.0, "  2.0+    (major improvement)")]:
    n = ((pmi_df['pmi_lift'] >= lo) & (pmi_df['pmi_lift'] < hi)).sum()
    print(f"    {label}: {n} reefs")

print(f"\n  Reefs with lift > 1.0: {(pmi_df['pmi_lift'] > 1.0).sum()} / {len(pmi_df)} "
      f"({100 * (pmi_df['pmi_lift'] > 1.0).sum() / len(pmi_df):.0f}%)")

# Bottom 10
print("\n  Bottom 10 (lowest PMI lift — subdivision didn't help specificity):")
for _, row in pmi_df.head(10).iterrows():
    print(f"    reef {int(row['reef_id']):>3d} (parent {int(row['parent_island_id']):>2d}): "
          f"lift={row['pmi_lift']:.2f} (child={row['child_pmi']:.3f}, parent={row['parent_pmi']:.3f})")

# Top 10
print("\n  Top 10 (highest PMI lift — subdivision greatly improved specificity):")
for _, row in pmi_df.tail(10).iloc[::-1].iterrows():
    print(f"    reef {int(row['reef_id']):>3d} (parent {int(row['parent_island_id']):>2d}): "
          f"lift={row['pmi_lift']:.2f} (child={row['child_pmi']:.3f}, parent={row['parent_pmi']:.3f})")


# =============================================================================
# Cross-metric correlation: do the metrics agree?
# =============================================================================
print("\n" + "=" * 80)
print("CROSS-METRIC CORRELATION")
print("=" * 80)

# Merge all metrics
merged = exclusive_df[['reef_id', 'exclusive_pct', 'total_words']].merge(
    jaccard_df[['reef_id', 'ratio']].rename(columns={'ratio': 'jaccard_ratio'}),
    on='reef_id', how='inner'
).merge(
    depth_df[['reef_id', 'median_depth', 'avg_depth', 'pct_depth_1', 'n_dims']],
    on='reef_id', how='inner'
).merge(
    pmi_df[['reef_id', 'pmi_lift']],
    on='reef_id', how='inner'
)

print(f"\n{len(merged)} reefs with all 4 metrics available\n")

# Correlations
cols = ['exclusive_pct', 'jaccard_ratio', 'median_depth', 'pmi_lift', 'total_words', 'n_dims']
print("Pearson correlations:")
print(f"  {'':>18s}", end="")
for c in cols:
    print(f"  {c:>14s}", end="")
print()
for c1 in cols:
    print(f"  {c1:>18s}", end="")
    for c2 in cols:
        r = merged[c1].corr(merged[c2])
        print(f"  {r:>14.3f}", end="")
    print()

# Also look at avg_specificity to connect back to the bimodality
print("\n\nMetrics by reef avg_specificity (connecting to bimodality):")
reef_spec = con.execute("""
    SELECT di.island_id AS reef_id,
           AVG(ds.avg_specificity) AS reef_avg_spec
    FROM dim_islands di
    JOIN dim_stats ds ON di.dim_id = ds.dim_id
    WHERE di.generation = 2 AND di.island_id >= 0
    GROUP BY di.island_id
""").fetchdf()
merged2 = merged.merge(reef_spec, on='reef_id', how='inner')

concrete = merged2[merged2['reef_avg_spec'] >= 0]
diffuse = merged2[merged2['reef_avg_spec'] < 0]

print(f"\n  Concrete reefs (avg_spec >= 0): {len(concrete)}")
print(f"    exclusive_pct:  mean={concrete['exclusive_pct'].mean():.1f}%, median={concrete['exclusive_pct'].median():.1f}%")
print(f"    jaccard_ratio:  mean={concrete['jaccard_ratio'].mean():.2f}, median={concrete['jaccard_ratio'].median():.2f}")
print(f"    median_depth:   mean={concrete['median_depth'].mean():.2f}, median={concrete['median_depth'].median():.2f}")
print(f"    pmi_lift:       mean={concrete['pmi_lift'].mean():.2f}, median={concrete['pmi_lift'].median():.2f}")
print(f"    total_words:    mean={concrete['total_words'].mean():.0f}, median={concrete['total_words'].median():.0f}")

print(f"\n  Diffuse reefs (avg_spec < 0): {len(diffuse)}")
print(f"    exclusive_pct:  mean={diffuse['exclusive_pct'].mean():.1f}%, median={diffuse['exclusive_pct'].median():.1f}%")
print(f"    jaccard_ratio:  mean={diffuse['jaccard_ratio'].mean():.2f}, median={diffuse['jaccard_ratio'].median():.2f}")
print(f"    median_depth:   mean={diffuse['median_depth'].mean():.2f}, median={diffuse['median_depth'].median():.2f}")
print(f"    pmi_lift:       mean={diffuse['pmi_lift'].mean():.2f}, median={diffuse['pmi_lift'].median():.2f}")
print(f"    total_words:    mean={diffuse['total_words'].mean():.0f}, median={diffuse['total_words'].median():.0f}")

con.close()
print("\nDone.")
