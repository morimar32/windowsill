"""Split vs Regroup analysis for large reefs.

For each reef with 4+ dims, answers:
  1. CLIQUE — all dims equally similar, no natural cut. Leave it alone.
  2. SUBDIVIDE — internal structure has sub-clusters (gen-3 makes sense)
  3. REGROUP — some dims have stronger affinity to sibling reefs than their own

The key measurements:
  A. Internal homogeneity: Are all pairwise Jaccards within the reef similar (clique)
     or is there a gap (sub-clusters)?
  B. Dim loyalty: For each dim, is its avg Jaccard to reef-mates higher than
     its avg Jaccard to any sibling reef?
  C. Word separation: If we hypothetically split this reef's dims at the
     internal Jaccard gap, do the resulting word sets actually differ?
"""
import duckdb
import numpy as np
from collections import defaultdict
import config

con = duckdb.connect(config.DB_PATH, read_only=True)
SW = "w.word NOT LIKE '% %'"

# =============================================================================
# Load the candidate reefs (4+ dims)
# =============================================================================
candidates = con.execute("""
    SELECT ist.island_id AS reef_id, ist.n_dims, ist.parent_island_id,
           ist.island_name
    FROM island_stats ist
    WHERE ist.generation = 2 AND ist.island_id >= 0 AND ist.n_dims >= 4
    ORDER BY ist.n_dims DESC, ist.island_id
""").fetchall()

print(f"Analyzing {len(candidates)} reefs with 4+ dims\n")

# Load all dim-to-reef assignments for gen-2
dim_reef = {}
reef_dims = defaultdict(list)
dim_to_parent = {}
for row in con.execute("""
    SELECT di.dim_id, di.island_id AS reef_id, ist.parent_island_id
    FROM dim_islands di
    JOIN island_stats ist ON di.island_id = ist.island_id AND ist.generation = 2
    WHERE di.generation = 2 AND di.island_id >= 0
""").fetchall():
    dim_reef[row[0]] = row[1]
    reef_dims[row[1]].append(row[0])
    dim_to_parent[row[0]] = row[2]

# Load ALL Jaccard values into memory (only ~295K pairs, small enough)
jaccard_lookup = {}
for a, b, j in con.execute("SELECT dim_id_a, dim_id_b, jaccard FROM dim_jaccard").fetchall():
    jaccard_lookup[(a, b)] = j
    jaccard_lookup[(b, a)] = j

def get_jaccard(d1, d2):
    if d1 == d2:
        return 1.0
    return jaccard_lookup.get((d1, d2), jaccard_lookup.get((d2, d1), 0.0))

# Get parent island -> child reefs mapping
parent_to_reefs = defaultdict(list)
for row in con.execute("""
    SELECT island_id, parent_island_id FROM island_stats
    WHERE generation = 2 AND island_id >= 0
""").fetchall():
    parent_to_reefs[row[1]].append(row[0])


# =============================================================================
# Analysis per reef
# =============================================================================
results = []

for reef_id, n_dims, parent_id, reef_name in candidates:
    dims = reef_dims[reef_id]
    sibling_reef_ids = [r for r in parent_to_reefs[parent_id] if r != reef_id]

    # --- A. Internal homogeneity ---
    # All pairwise Jaccards within the reef
    internal_jaccards = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            internal_jaccards.append(get_jaccard(dims[i], dims[j]))

    int_arr = np.array(internal_jaccards)
    int_mean = int_arr.mean()
    int_std = int_arr.std()
    int_min = int_arr.min()
    int_max = int_arr.max()
    int_range = int_max - int_min
    int_cv = int_std / int_mean if int_mean > 0 else 0  # coefficient of variation

    # Gap analysis: sort pairwise Jaccards and find the biggest gap
    # A large gap suggests natural sub-clusters
    sorted_jaccards = np.sort(int_arr)
    if len(sorted_jaccards) >= 2:
        gaps = np.diff(sorted_jaccards)
        max_gap = gaps.max()
        max_gap_pos = gaps.argmax()
        # The gap divides pairs into "low Jaccard" (cross-cluster) and "high Jaccard" (within-cluster)
        gap_threshold = sorted_jaccards[max_gap_pos]
        n_below_gap = max_gap_pos + 1
        n_above_gap = len(sorted_jaccards) - n_below_gap
    else:
        max_gap = 0
        gap_threshold = 0
        n_below_gap = 0
        n_above_gap = 0

    # --- B. Dim loyalty analysis ---
    # For each dim, compute avg Jaccard to own reef vs to each sibling reef
    dim_loyalty = []
    misplaced_dims = []

    for d in dims:
        # Affinity to own reef (other dims in same reef)
        own_jaccards = [get_jaccard(d, d2) for d2 in dims if d2 != d]
        own_affinity = np.mean(own_jaccards) if own_jaccards else 0

        # Affinity to each sibling reef
        best_sibling_id = None
        best_sibling_affinity = 0
        sibling_affinities = {}

        for sib_id in sibling_reef_ids:
            sib_dims = reef_dims[sib_id]
            sib_jaccards = [get_jaccard(d, sd) for sd in sib_dims]
            sib_aff = np.mean(sib_jaccards) if sib_jaccards else 0
            sibling_affinities[sib_id] = sib_aff
            if sib_aff > best_sibling_affinity:
                best_sibling_affinity = sib_aff
                best_sibling_id = sib_id

        loyalty_ratio = own_affinity / best_sibling_affinity if best_sibling_affinity > 0 else float('inf')

        dim_loyalty.append({
            'dim_id': d,
            'own_affinity': own_affinity,
            'best_sibling_affinity': best_sibling_affinity,
            'best_sibling_id': best_sibling_id,
            'loyalty_ratio': loyalty_ratio,
        })

        if loyalty_ratio < 1.0:
            misplaced_dims.append(d)

    avg_loyalty = np.mean([dl['loyalty_ratio'] for dl in dim_loyalty])
    min_loyalty = min(dl['loyalty_ratio'] for dl in dim_loyalty)
    n_misplaced = len(misplaced_dims)

    # --- C. Hypothetical split word separation ---
    # If we split dims at the internal Jaccard gap, how different are the word sets?
    # Build a simple 2-way partition: for each dim, compute avg Jaccard to all others,
    # then split into "tighter cluster" vs "looser" using spectral-ish approach
    # Simpler: use the Jaccard gap. Dims that are mutually highly connected go together.
    # Build adjacency and do a greedy bipartition based on internal Jaccard.

    # Simple approach: pick the dim pair with lowest internal Jaccard as the "cut"
    # and assign each dim to whichever side it's more similar to
    if n_dims >= 4 and len(internal_jaccards) >= 3:
        # Build a dim-dim similarity matrix
        dim_sim = np.zeros((n_dims, n_dims))
        for i in range(n_dims):
            for j in range(n_dims):
                if i != j:
                    dim_sim[i, j] = get_jaccard(dims[i], dims[j])

        # Find the weakest-linked pair
        min_pair = np.unravel_index(np.argmin(dim_sim + np.eye(n_dims) * 999), dim_sim.shape)

        # Seed two groups from the weakest pair
        group_a = {min_pair[0]}
        group_b = {min_pair[1]}
        remaining = set(range(n_dims)) - group_a - group_b

        # Assign remaining dims to whichever group they're more similar to
        for idx in sorted(remaining, key=lambda x: -max(dim_sim[x, list(group_a)].mean(),
                                                         dim_sim[x, list(group_b)].mean())):
            aff_a = np.mean([dim_sim[idx, g] for g in group_a])
            aff_b = np.mean([dim_sim[idx, g] for g in group_b])
            if aff_a >= aff_b:
                group_a.add(idx)
            else:
                group_b.add(idx)

        dims_a = [dims[i] for i in group_a]
        dims_b = [dims[i] for i in group_b]

        # Compute word overlap between the two hypothetical sub-reefs
        placeholders_a = ",".join(str(d) for d in dims_a)
        placeholders_b = ",".join(str(d) for d in dims_b)

        words_a = set(r[0] for r in con.execute(f"""
            SELECT DISTINCT dm.word_id FROM dim_memberships dm
            JOIN words w ON dm.word_id = w.word_id
            WHERE dm.dim_id IN ({placeholders_a}) AND {SW}
        """).fetchall())

        words_b = set(r[0] for r in con.execute(f"""
            SELECT DISTINCT dm.word_id FROM dim_memberships dm
            JOIN words w ON dm.word_id = w.word_id
            WHERE dm.dim_id IN ({placeholders_b}) AND {SW}
        """).fetchall())

        overlap = len(words_a & words_b)
        union = len(words_a | words_b)
        word_jaccard = overlap / union if union > 0 else 1.0
        exclusive_a = len(words_a - words_b)
        exclusive_b = len(words_b - words_a)
        split_exclusive_pct = (exclusive_a + exclusive_b) / (2 * union) * 100 if union > 0 else 0

        # Internal Jaccard within each hypothetical sub-group
        intra_a = []
        for i in group_a:
            for j in group_a:
                if i < j:
                    intra_a.append(dim_sim[i, j])
        intra_b = []
        for i in group_b:
            for j in group_b:
                if i < j:
                    intra_b.append(dim_sim[i, j])
        cross = []
        for i in group_a:
            for j in group_b:
                cross.append(dim_sim[i, j])

        intra_mean = np.mean(intra_a + intra_b) if (intra_a or intra_b) else 0
        cross_mean = np.mean(cross) if cross else 0
        split_ratio = intra_mean / cross_mean if cross_mean > 0 else float('inf')
    else:
        word_jaccard = None
        exclusive_a = exclusive_b = 0
        split_exclusive_pct = 0
        split_ratio = 0
        dims_a = dims_b = []
        group_a = group_b = set()

    # --- Classification ---
    # CLIQUE: low CV, high min loyalty, no gap
    # SUBDIVIDE: high CV or big gap, all dims loyal, word sets diverge
    # REGROUP: misplaced dims (loyalty < 1.0)
    if n_misplaced >= 2:
        verdict = "REGROUP"
    elif n_misplaced == 1:
        verdict = "REGROUP-MILD"
    elif int_cv > 0.3 and split_ratio and split_ratio > 1.3 and word_jaccard and word_jaccard < 0.7:
        verdict = "SUBDIVIDE"
    elif int_cv > 0.2 and split_ratio and split_ratio > 1.2:
        verdict = "SUBDIVIDE-MAYBE"
    else:
        verdict = "CLIQUE"

    results.append({
        'reef_id': reef_id,
        'n_dims': n_dims,
        'parent_id': parent_id,
        'reef_name': reef_name,
        'int_mean': int_mean,
        'int_min': int_min,
        'int_max': int_max,
        'int_range': int_range,
        'int_cv': int_cv,
        'max_gap': max_gap,
        'avg_loyalty': avg_loyalty,
        'min_loyalty': min_loyalty,
        'n_misplaced': n_misplaced,
        'misplaced_dims': misplaced_dims,
        'word_jaccard': word_jaccard,
        'split_exclusive_pct': split_exclusive_pct,
        'split_ratio': split_ratio,
        'split_sizes': (len(group_a), len(group_b)),
        'verdict': verdict,
    })


# =============================================================================
# Report
# =============================================================================
print("=" * 120)
print("REEF-BY-REEF ANALYSIS: Split vs Regroup vs Clique")
print("=" * 120)

# Sort by verdict then reef size
verdict_order = {"REGROUP": 0, "REGROUP-MILD": 1, "SUBDIVIDE": 2, "SUBDIVIDE-MAYBE": 3, "CLIQUE": 4}
results.sort(key=lambda r: (verdict_order.get(r['verdict'], 5), -r['n_dims']))

# Summary counts
verdicts = defaultdict(int)
for r in results:
    verdicts[r['verdict']] += 1

print(f"\nVerdict distribution across {len(results)} reefs with 4+ dims:")
for v in ["REGROUP", "REGROUP-MILD", "SUBDIVIDE", "SUBDIVIDE-MAYBE", "CLIQUE"]:
    print(f"  {v:>16s}: {verdicts.get(v, 0)}")

print(f"\n{'Reef':>5s} {'Dims':>4s} {'Verdict':>16s} │ {'IntCV':>6s} {'Gap':>6s} {'MinLoy':>6s} {'#Misp':>5s} │ "
      f"{'SplitJ':>6s} {'SplRat':>6s} {'ExclPct':>7s} {'Sizes':>7s} │ Name")
print("─" * 120)

for r in results:
    wj_str = f"{r['word_jaccard']:.3f}" if r['word_jaccard'] is not None else "  N/A"
    sr_str = f"{r['split_ratio']:.2f}" if r['split_ratio'] else "  N/A"
    sp_str = f"{r['split_exclusive_pct']:.1f}%" if r['split_exclusive_pct'] else "  N/A"
    sz_str = f"{r['split_sizes'][0]}+{r['split_sizes'][1]}" if r['split_sizes'] != (0, 0) else "N/A"

    print(f"{r['reef_id']:>5d} {r['n_dims']:>4d} {r['verdict']:>16s} │ "
          f"{r['int_cv']:.3f}  {r['max_gap']:.4f} {r['min_loyalty']:.3f} {r['n_misplaced']:>5d} │ "
          f"{wj_str} {sr_str:>6s} {sp_str:>7s} {sz_str:>7s} │ {r['reef_name'] or '?'}")

# =============================================================================
# Deep dives on interesting cases
# =============================================================================
print("\n\n" + "=" * 120)
print("DEEP DIVES")
print("=" * 120)

for r in results:
    if r['verdict'] in ('REGROUP', 'SUBDIVIDE') or \
       (r['verdict'] == 'REGROUP-MILD' and r['n_dims'] >= 5):
        print(f"\n  ── Reef {r['reef_id']} ({r['n_dims']} dims): {r['reef_name']} → {r['verdict']} ──")

        dims = reef_dims[r['reef_id']]

        # Print the internal Jaccard matrix
        print(f"\n  Internal Jaccard matrix:")
        print(f"  {'dim':>6s}", end="")
        for d in dims:
            print(f" {d:>7d}", end="")
        print()
        for i, d1 in enumerate(dims):
            print(f"  {d1:>6d}", end="")
            for j, d2 in enumerate(dims):
                if i == j:
                    print(f"    --- ", end="")
                else:
                    j_val = get_jaccard(d1, d2)
                    print(f" {j_val:.4f}", end="")
            print()

        # Dim loyalty details
        print(f"\n  Dim loyalty (own_affinity / best_sibling_affinity):")
        for dl in sorted([dl for dl in results if dl['reef_id'] == r['reef_id']], key=lambda x: x['reef_id']):
            pass  # we need the dim_loyalty list, let me recompute

        # Recompute dim loyalty for display
        for d in dims:
            own_j = [get_jaccard(d, d2) for d2 in dims if d2 != d]
            own_aff = np.mean(own_j)

            best_sib_id = None
            best_sib_aff = 0
            sibling_reef_ids = [rid for rid in parent_to_reefs[r['parent_id']] if rid != r['reef_id']]
            for sib_id in sibling_reef_ids:
                sib_dims = reef_dims[sib_id]
                sib_aff = np.mean([get_jaccard(d, sd) for sd in sib_dims]) if sib_dims else 0
                if sib_aff > best_sib_aff:
                    best_sib_aff = sib_aff
                    best_sib_id = sib_id

            loyalty = own_aff / best_sib_aff if best_sib_aff > 0 else float('inf')
            flag = " ← MISPLACED" if loyalty < 1.0 else ""
            sib_name = ""
            if best_sib_id is not None:
                sib_name_row = con.execute(
                    "SELECT island_name FROM island_stats WHERE island_id = ? AND generation = 2",
                    [best_sib_id]).fetchone()
                sib_name = sib_name_row[0] if sib_name_row and sib_name_row[0] else f"reef {best_sib_id}"

            print(f"    dim {d:>3d}: own={own_aff:.4f}, best_sib={best_sib_aff:.4f} (→ {sib_name}), "
                  f"loyalty={loyalty:.3f}{flag}")

        # If SUBDIVIDE, show word separation
        if r['verdict'] in ('SUBDIVIDE', 'SUBDIVIDE-MAYBE') and r['word_jaccard'] is not None:
            print(f"\n  Hypothetical split: {r['split_sizes'][0]}+{r['split_sizes'][1]} dims")
            print(f"    Word Jaccard between halves: {r['word_jaccard']:.3f}")
            print(f"    Split exclusive %: {r['split_exclusive_pct']:.1f}%")
            print(f"    Intra/cross Jaccard ratio: {r['split_ratio']:.2f}")

        if r['n_misplaced'] > 0:
            print(f"\n  Misplaced dims: {r['misplaced_dims']}")
            for md in r['misplaced_dims']:
                # Which sibling would this dim go to?
                own_j = [get_jaccard(md, d2) for d2 in dims if d2 != md]
                own_aff = np.mean(own_j)
                sibling_reef_ids = [rid for rid in parent_to_reefs[r['parent_id']] if rid != r['reef_id']]
                for sib_id in sibling_reef_ids:
                    sib_dims = reef_dims[sib_id]
                    sib_aff = np.mean([get_jaccard(md, sd) for sd in sib_dims]) if sib_dims else 0
                    if sib_aff > own_aff:
                        sib_name_row = con.execute(
                            "SELECT island_name FROM island_stats WHERE island_id = ? AND generation = 2",
                            [sib_id]).fetchone()
                        sib_name = sib_name_row[0] if sib_name_row and sib_name_row[0] else f"reef {sib_id}"
                        print(f"    dim {md}: own_reef_aff={own_aff:.4f}, "
                              f"better fit → reef {sib_id} ({sib_name}) aff={sib_aff:.4f}")


# =============================================================================
# Overall summary
# =============================================================================
print("\n\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)

subdivide = [r for r in results if r['verdict'] in ('SUBDIVIDE', 'SUBDIVIDE-MAYBE')]
regroup = [r for r in results if r['verdict'].startswith('REGROUP')]
clique = [r for r in results if r['verdict'] == 'CLIQUE']

print(f"\n  CLIQUE ({len(clique)} reefs): No action needed. Internal structure is homogeneous.")
if clique:
    print(f"    Avg internal CV: {np.mean([r['int_cv'] for r in clique]):.3f}")
    print(f"    Avg min loyalty: {np.mean([r['min_loyalty'] for r in clique]):.3f}")

print(f"\n  SUBDIVIDE / SUBDIVIDE-MAYBE ({len(subdivide)} reefs): Internal sub-clusters exist.")
if subdivide:
    print(f"    Avg internal CV: {np.mean([r['int_cv'] for r in subdivide]):.3f}")
    print(f"    Avg split word Jaccard: {np.mean([r['word_jaccard'] for r in subdivide if r['word_jaccard'] is not None]):.3f}")
    print(f"    Avg split ratio: {np.mean([r['split_ratio'] for r in subdivide if r['split_ratio']]):.2f}")
    for r in subdivide:
        print(f"      reef {r['reef_id']:>3d} ({r['n_dims']} dims, CV={r['int_cv']:.3f}): {r['reef_name']}")

print(f"\n  REGROUP / REGROUP-MILD ({len(regroup)} reefs): Some dims better fit a sibling reef.")
if regroup:
    total_misplaced = sum(r['n_misplaced'] for r in regroup)
    print(f"    Total misplaced dims: {total_misplaced}")
    for r in regroup:
        print(f"      reef {r['reef_id']:>3d} ({r['n_dims']} dims, {r['n_misplaced']} misplaced): {r['reef_name']}")

con.close()
print("\nDone.")
