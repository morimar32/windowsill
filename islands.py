import math

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import config

# Filter for single-token words only (no spaces) — excludes multi-word disambiguators
_SINGLE_WORD_FILTER = "word NOT LIKE '%% %%'"


def compute_jaccard_matrix(con):
    """Step 1: Compute pairwise Jaccard similarity between all 768 dimensions."""
    print("  Computing Jaccard similarity matrix...")

    # Load (dim_id, word_id) pairs for single-token words only
    rows = con.execute(f"""
        SELECT m.dim_id, m.word_id
        FROM dim_memberships m
        INNER JOIN words w ON m.word_id = w.word_id
        WHERE w.{_SINGLE_WORD_FILTER}
    """).fetchall()
    dim_ids = np.array([r[0] for r in rows], dtype=np.int32)
    word_ids = np.array([r[1] for r in rows], dtype=np.int32)
    print(f"    Loaded {len(rows):,} memberships (single-token words only)")

    # Build sparse binary matrix M (768 x N_words)
    n_dims = config.MATRYOSHKA_DIM
    max_word_id = int(word_ids.max()) + 1
    data = np.ones(len(rows), dtype=np.float32)
    M = csr_matrix((data, (dim_ids, word_ids)), shape=(n_dims, max_word_id))

    # Intersection matrix: I = M @ M.T (768 x 768)
    I = (M @ M.T).toarray()
    counts = np.diag(I).astype(np.float64)

    # Total single-token word count for hypergeometric z-score
    N = con.execute(f"SELECT COUNT(*) FROM words WHERE {_SINGLE_WORD_FILTER}").fetchone()[0]

    # Compute Jaccard and hypergeometric z-score for all pairs (i < j) with intersection > 0
    con.execute("DELETE FROM dim_jaccard")
    jaccard_rows = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            inter = int(I[i, j])
            if inter == 0:
                continue
            union = int(counts[i] + counts[j] - inter)
            jacc = inter / union if union > 0 else 0.0
            n_i, n_j = counts[i], counts[j]
            expected = n_i * n_j / N
            variance = n_i * n_j * (N - n_i) * (N - n_j) / (N**2 * (N - 1))
            z = (inter - expected) / math.sqrt(variance) if variance > 0 else 0.0
            jaccard_rows.append((i, j, inter, union, jacc, expected, z))

    df = pd.DataFrame(jaccard_rows, columns=[
        "dim_id_a", "dim_id_b", "intersection_size", "union_size", "jaccard",
        "expected_intersection", "z_score",
    ])
    con.execute("INSERT INTO dim_jaccard SELECT * FROM df")
    print(f"    Stored {len(jaccard_rows):,} dimension pairs with intersection > 0")

    # Summary stats
    jaccards = df["jaccard"]
    zscores = df["z_score"]
    print(f"    Jaccard range: {jaccards.min():.4f} - {jaccards.max():.4f}, "
          f"mean: {jaccards.mean():.4f}, median: {jaccards.median():.4f}")
    print(f"    Z-score range: {zscores.min():.2f} - {zscores.max():.2f}, "
          f"mean: {zscores.mean():.2f}, median: {zscores.median():.2f}")
    above_threshold = (zscores >= config.ISLAND_JACCARD_ZSCORE).sum()
    print(f"    Pairs above z-score threshold ({config.ISLAND_JACCARD_ZSCORE}): {above_threshold:,}")


def detect_islands(con):
    """Step 2: Leiden community detection on the Jaccard graph."""
    print("  Running Leiden community detection...")

    # Build graph with edges above z-score threshold
    edges = con.execute(f"""
        SELECT dim_id_a, dim_id_b, jaccard
        FROM dim_jaccard
        WHERE z_score >= {config.ISLAND_JACCARD_ZSCORE}
    """).fetchall()

    n_dims = config.MATRYOSHKA_DIM
    g = ig.Graph(n=n_dims)
    if edges:
        edge_list = [(r[0], r[1]) for r in edges]
        weights = [r[2] for r in edges]
        g.add_edges(edge_list)
        g.es["weight"] = weights

    print(f"    Graph: {g.vcount()} vertices, {g.ecount()} edges")

    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight" if g.ecount() > 0 else None,
        resolution_parameter=config.ISLAND_LEIDEN_RESOLUTION,
        seed=42,
    )

    # Assign island IDs, filtering small communities to noise (-1)
    min_size = config.ISLAND_MIN_COMMUNITY_SIZE
    community_map = {}  # community_index -> island_id
    next_island_id = 0
    for idx, community in enumerate(partition):
        if len(community) >= min_size:
            community_map[idx] = next_island_id
            next_island_id += 1
        else:
            community_map[idx] = -1

    # Build dim -> island_id mapping
    con.execute("DELETE FROM dim_islands WHERE generation = 0")
    island_rows = []
    for idx, community in enumerate(partition):
        island_id = community_map[idx]
        for dim_id in community:
            island_rows.append((dim_id, island_id, 0, None))

    df = pd.DataFrame(island_rows, columns=["dim_id", "island_id", "generation", "parent_island_id"])
    con.execute("INSERT INTO dim_islands SELECT * FROM df")

    n_islands = sum(1 for v in community_map.values() if v >= 0)
    n_noise = sum(1 for r in island_rows if r[1] == -1)
    print(f"    Found {n_islands} islands, {n_noise} noise dimensions (island_id = -1)")
    print(f"    Modularity: {partition.modularity:.4f}")


def detect_sub_islands(con, parent_generation=0):
    """Subdivide islands of parent_generation into the next generation via Leiden."""
    child_generation = parent_generation + 1
    print(f"  Detecting sub-islands (generation {child_generation} from gen {parent_generation})...")

    min_dims = config.ISLAND_MIN_DIMS_FOR_SUBDIVISION
    resolution = config.ISLAND_SUB_LEIDEN_RESOLUTION
    min_size = config.ISLAND_MIN_COMMUNITY_SIZE

    # Clear child generation data for idempotency
    con.execute("DELETE FROM dim_islands WHERE generation = ?", [child_generation])
    con.execute("DELETE FROM island_stats WHERE generation = ?", [child_generation])
    con.execute("DELETE FROM island_characteristic_words WHERE generation = ?", [child_generation])

    # Get parent islands above the min-dims threshold
    parent_islands = con.execute(f"""
        SELECT island_id, COUNT(*) as n_dims
        FROM dim_islands
        WHERE generation = ? AND island_id >= 0
        GROUP BY island_id
        HAVING n_dims >= {min_dims}
        ORDER BY island_id
    """, [parent_generation]).fetchall()

    if not parent_islands:
        print("    No islands large enough to subdivide")
        return

    print(f"    {len(parent_islands)} islands eligible for subdivision (>= {min_dims} dims)")

    next_sub_island_id = 0
    total_sub_islands = 0
    total_noise = 0

    for parent_id, parent_n_dims in parent_islands:
        # Get dim_ids in this parent island
        dims = con.execute("""
            SELECT dim_id FROM dim_islands
            WHERE island_id = ? AND generation = ?
        """, [parent_id, parent_generation]).fetchall()
        dim_ids = [d[0] for d in dims]

        # Map global dim_ids -> local vertex indices
        dim_to_local = {d: i for i, d in enumerate(dim_ids)}
        n_local = len(dim_ids)

        # Query edges where BOTH dims are in this island AND z_score >= threshold
        placeholders = ",".join(str(d) for d in dim_ids)
        edges = con.execute(f"""
            SELECT dim_id_a, dim_id_b, jaccard
            FROM dim_jaccard
            WHERE dim_id_a IN ({placeholders})
              AND dim_id_b IN ({placeholders})
              AND z_score >= {config.ISLAND_JACCARD_ZSCORE}
        """).fetchall()

        # Build local igraph
        g = ig.Graph(n=n_local)
        if edges:
            edge_list = [(dim_to_local[r[0]], dim_to_local[r[1]]) for r in edges]
            weights = [r[2] for r in edges]
            g.add_edges(edge_list)
            g.es["weight"] = weights

        # Run Leiden with higher resolution
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight" if g.ecount() > 0 else None,
            resolution_parameter=resolution,
            seed=42,
        )

        # Assign sub-island IDs (globally sequential), noise = -1
        community_map = {}
        for idx, community in enumerate(partition):
            if len(community) >= min_size:
                community_map[idx] = next_sub_island_id
                next_sub_island_id += 1
            else:
                community_map[idx] = -1

        # Build rows for dim_islands
        sub_rows = []
        for idx, community in enumerate(partition):
            sub_id = community_map[idx]
            for local_v in community:
                global_dim = dim_ids[local_v]
                sub_rows.append((global_dim, sub_id, child_generation, parent_id))

        if sub_rows:
            df = pd.DataFrame(sub_rows, columns=["dim_id", "island_id", "generation", "parent_island_id"])
            con.execute("INSERT INTO dim_islands SELECT * FROM df")

        n_subs = sum(1 for v in community_map.values() if v >= 0)
        n_noise = sum(1 for r in sub_rows if r[1] == -1)
        total_sub_islands += n_subs
        total_noise += n_noise
        print(f"    Parent {parent_id} ({parent_n_dims} dims) -> {n_subs} sub-islands, {n_noise} noise")

    print(f"    Total: {total_sub_islands} sub-islands, {total_noise} noise dims across {len(parent_islands)} parents")


def compute_island_stats(con, generation=0):
    """Compute per-island summary statistics for the given generation."""
    print(f"  Computing island statistics (gen {generation})...")

    islands = con.execute("""
        SELECT DISTINCT island_id FROM dim_islands
        WHERE generation = ? AND island_id >= 0
        ORDER BY island_id
    """, [generation]).fetchall()

    con.execute("DELETE FROM island_stats WHERE generation = ?", [generation])
    stats_rows = []

    for (island_id,) in islands:
        # Get dims in this island
        dims = con.execute("""
            SELECT dim_id FROM dim_islands
            WHERE island_id = ? AND generation = ?
        """, [island_id, generation]).fetchall()
        dim_ids = [d[0] for d in dims]
        n_dims = len(dim_ids)

        # Count unique single-token words across all dims in island
        placeholders = ",".join(str(d) for d in dim_ids)
        n_words = con.execute(f"""
            SELECT COUNT(DISTINCT m.word_id) FROM dim_memberships m
            INNER JOIN words w ON m.word_id = w.word_id
            WHERE m.dim_id IN ({placeholders}) AND w.{_SINGLE_WORD_FILTER}
        """).fetchone()[0]

        # Word depth distribution (single-token only)
        word_depths = con.execute(f"""
            SELECT COUNT(*) as cnt
            FROM dim_memberships m
            INNER JOIN words w ON m.word_id = w.word_id
            WHERE m.dim_id IN ({placeholders}) AND w.{_SINGLE_WORD_FILTER}
            GROUP BY m.word_id
        """).fetchall()
        depths = [r[0] for r in word_depths]

        # Core words: >= max(2, ceil(n_dims * 0.10)) island dims
        core_threshold = max(2, math.ceil(n_dims * 0.10))
        n_core_words = sum(1 for d in depths if d >= core_threshold)

        # Median depth
        median_depth = float(np.median(depths)) if depths else None

        # Internal Jaccard stats
        if n_dims >= 2:
            jacc_stats = con.execute(f"""
                SELECT AVG(jaccard), MAX(jaccard), MIN(jaccard)
                FROM dim_jaccard
                WHERE dim_id_a IN ({placeholders}) AND dim_id_b IN ({placeholders})
            """).fetchone()
            avg_jacc, max_jacc, min_jacc = jacc_stats
        else:
            avg_jacc, max_jacc, min_jacc = None, None, None

        # Look up parent_island_id for gen > 0
        parent_island_id = None
        if generation > 0:
            parent_row = con.execute("""
                SELECT DISTINCT parent_island_id FROM dim_islands
                WHERE island_id = ? AND generation = ?
            """, [island_id, generation]).fetchone()
            if parent_row:
                parent_island_id = parent_row[0]

        stats_rows.append((
            island_id, generation, n_dims, n_words,
            avg_jacc, max_jacc, min_jacc,
            None, parent_island_id, None,
            n_core_words, median_depth,
        ))

    df = pd.DataFrame(stats_rows, columns=[
        "island_id", "generation", "n_dims", "n_words",
        "avg_internal_jaccard", "max_internal_jaccard", "min_internal_jaccard",
        "modularity_contribution", "parent_island_id", "island_name",
        "n_core_words", "median_word_depth",
    ])
    con.execute("INSERT INTO island_stats SELECT * FROM df")
    print(f"    Computed stats for {len(stats_rows)} islands")


def compute_characteristic_words(con, generation=0):
    """Compute PMI-ranked characteristic words per island for the given generation."""
    print(f"  Computing characteristic words (PMI, gen {generation})...")

    n_total_dims = config.MATRYOSHKA_DIM
    top_n = config.ISLAND_CHARACTERISTIC_WORDS_N

    islands = con.execute("""
        SELECT island_id FROM island_stats
        WHERE generation = ?
        ORDER BY island_id
    """, [generation]).fetchall()

    # Preload word corpus frequencies: total_dims / 768 for each word (single-token only)
    word_freq = con.execute(f"""
        SELECT word_id, word, total_dims FROM v_unique_words
        WHERE total_dims > 0 AND {_SINGLE_WORD_FILTER}
    """).fetchall()
    word_corpus_freq = {r[0]: r[2] / n_total_dims for r in word_freq}
    word_text = {r[0]: r[1] for r in word_freq}

    con.execute("DELETE FROM island_characteristic_words WHERE generation = ?", [generation])
    total_words_stored = 0

    for (island_id,) in islands:
        # Get dims in this island
        dims = con.execute("""
            SELECT dim_id FROM dim_islands
            WHERE island_id = ? AND generation = ?
        """, [island_id, generation]).fetchall()
        dim_ids = [d[0] for d in dims]
        n_island_dims = len(dim_ids)
        if n_island_dims == 0:
            continue

        # Count how many island dims each single-token word appears in (>= 2 dims)
        placeholders = ",".join(str(d) for d in dim_ids)
        word_counts = con.execute(f"""
            SELECT m.word_id, COUNT(*) as cnt
            FROM dim_memberships m
            INNER JOIN words w ON m.word_id = w.word_id
            WHERE m.dim_id IN ({placeholders}) AND w.{_SINGLE_WORD_FILTER}
            GROUP BY m.word_id
            HAVING cnt >= 2
        """).fetchall()

        # Compute PMI for each word
        pmi_rows = []
        for wid, cnt in word_counts:
            island_freq = cnt / n_island_dims
            corpus_freq = word_corpus_freq.get(wid, 0)
            if corpus_freq <= 0:
                continue
            pmi = math.log2(island_freq / corpus_freq)
            word = word_text.get(wid, "")
            pmi_rows.append((island_id, generation, wid, word, pmi, island_freq, corpus_freq, cnt))

        # Sort by PMI descending, keep top N
        pmi_rows.sort(key=lambda r: r[4], reverse=True)
        pmi_rows = pmi_rows[:top_n]

        if pmi_rows:
            df = pd.DataFrame(pmi_rows, columns=[
                "island_id", "generation", "word_id", "word",
                "pmi", "island_freq", "corpus_freq", "n_dims_in_island",
            ])
            con.execute("INSERT INTO island_characteristic_words SELECT * FROM df")
            total_words_stored += len(pmi_rows)

    print(f"    Stored {total_words_stored:,} characteristic words across {len(islands)} islands")


def set_island_name(con, island_id, generation, name):
    """Set a human-readable name for an island. For ad-hoc use (notebook, one-liner)."""
    exists = con.execute(
        "SELECT COUNT(*) FROM island_stats WHERE island_id = ? AND generation = ?",
        [island_id, generation],
    ).fetchone()[0]
    if not exists:
        print(f"Island {island_id} gen {generation} not found.")
        return
    con.execute(
        "UPDATE island_stats SET island_name = ? WHERE island_id = ? AND generation = ?",
        [name, island_id, generation],
    )
    print(f"  Named island {island_id} gen {generation}: {name}")


def print_archipelago_summary(con):
    """Print hierarchical summary of gen-0 islands and their gen-1 sub-islands."""
    print("\n  === Archipelago Summary ===")

    gen0_stats = con.execute("""
        SELECT island_id, n_dims, n_words, avg_internal_jaccard, island_name,
               n_core_words, median_word_depth
        FROM island_stats WHERE generation = 0
        ORDER BY n_dims DESC
    """).fetchall()

    if not gen0_stats:
        print("  No islands detected.")
        return

    print(f"  {'Island':>7s}  {'Name':<25s}  {'Dims':>5s}  {'Core':>6s}  {'Med.Dp':>6s}  {'Avg Jacc':>9s}  Top words")
    print(f"  {'------':>7s}  {'----':<25s}  {'----':>5s}  {'----':>6s}  {'------':>6s}  {'--------':>9s}  ---------")

    total_gen0_dims = 0
    total_sub_islands = 0

    for island_id, n_dims, n_words, avg_jacc, island_name, n_core_words, median_depth in gen0_stats:
        total_gen0_dims += n_dims

        # Top words for gen-0 island
        top_words = con.execute("""
            SELECT word FROM island_characteristic_words
            WHERE island_id = ? AND generation = 0
            ORDER BY pmi DESC LIMIT 5
        """, [island_id]).fetchall()
        word_str = ", ".join(w[0] for w in top_words)
        avg_jacc_str = f"{avg_jacc:.4f}" if avg_jacc is not None else "    N/A"
        name_str = island_name if island_name else "(unnamed)"
        core_str = f"{n_core_words:>6,d}" if n_core_words is not None else "   N/A"
        depth_str = f"{median_depth:>6.1f}" if median_depth is not None else "   N/A"

        print(f"  {island_id:>7d}  {name_str:<25s}  {n_dims:>5d}  {core_str}  {depth_str}  {avg_jacc_str:>9s}  {word_str}")

        # Gen-1 sub-islands for this parent
        sub_stats = con.execute("""
            SELECT island_id, n_dims, n_words, avg_internal_jaccard, island_name,
                   n_core_words, median_word_depth
            FROM island_stats
            WHERE generation = 1 AND parent_island_id = ?
            ORDER BY n_dims DESC
        """, [island_id]).fetchall()

        for sub_id, sub_n_dims, sub_n_words, sub_avg_jacc, sub_name, sub_core, sub_med_depth in sub_stats:
            total_sub_islands += 1
            sub_top = con.execute("""
                SELECT word FROM island_characteristic_words
                WHERE island_id = ? AND generation = 1
                ORDER BY pmi DESC LIMIT 5
            """, [sub_id]).fetchall()
            sub_word_str = ", ".join(w[0] for w in sub_top)
            sub_jacc_str = f"{sub_avg_jacc:.4f}" if sub_avg_jacc is not None else "    N/A"
            sub_name_str = sub_name if sub_name else ""
            sub_core_str = f"{sub_core:>6,d}" if sub_core is not None else "   N/A"
            sub_depth_str = f"{sub_med_depth:>6.1f}" if sub_med_depth is not None else "   N/A"

            print(f"    {sub_id:>5d}  {sub_name_str:<25s}  {sub_n_dims:>5d}  {sub_core_str}  {sub_depth_str}  {sub_jacc_str:>9s}  {sub_word_str}")

    noise_gen0 = con.execute("""
        SELECT COUNT(*) FROM dim_islands WHERE island_id = -1 AND generation = 0
    """).fetchone()[0]
    noise_gen1 = con.execute("""
        SELECT COUNT(*) FROM dim_islands WHERE island_id = -1 AND generation = 1
    """).fetchone()[0]

    print(f"\n  Gen 0: {len(gen0_stats)} islands covering {total_gen0_dims} dims, {noise_gen0} noise dims")
    print(f"  Gen 1: {total_sub_islands} sub-islands, {noise_gen1} noise dims")


def compute_archipelago_encoding(con):
    """Encode the island hierarchy as bitmask columns on the words table."""
    print("  Computing archipelago encoding...")

    # === Phase A: Assign bit positions to island_stats ===
    print("    Phase A: Assigning bit positions...")

    # Clear existing bit positions
    con.execute("UPDATE island_stats SET arch_column = NULL, arch_bit = NULL")

    # Gen-0: arch_column='archipelago', arch_bit=island_id (0-3)
    con.execute("""
        UPDATE island_stats
        SET arch_column = 'archipelago', arch_bit = island_id
        WHERE generation = 0 AND island_id >= 0
    """)
    gen0_count = con.execute(
        "SELECT COUNT(*) FROM island_stats WHERE generation = 0 AND island_id >= 0"
    ).fetchone()[0]
    print(f"      Gen-0: {gen0_count} archipelagos (bits 0-{gen0_count - 1})")

    # Gen-1: arch_column='archipelago', arch_bit=4+island_id
    con.execute("""
        UPDATE island_stats
        SET arch_column = 'archipelago', arch_bit = 4 + island_id
        WHERE generation = 1 AND island_id >= 0
    """)
    gen1_count = con.execute(
        "SELECT COUNT(*) FROM island_stats WHERE generation = 1 AND island_id >= 0"
    ).fetchone()[0]
    print(f"      Gen-1: {gen1_count} islands (bits 4-{4 + gen1_count - 1})")

    # Gen-2: group by gen-0 grandparent, assign reef_N columns
    # Find each gen-2 island's grandparent via gen-1 parent -> gen-0 grandparent
    gen2_islands = con.execute("""
        SELECT s2.island_id, s2.parent_island_id as gen1_parent,
               s1.parent_island_id as gen0_grandparent
        FROM island_stats s2
        JOIN island_stats s1 ON s2.parent_island_id = s1.island_id AND s1.generation = 1
        WHERE s2.generation = 2 AND s2.island_id >= 0
        ORDER BY gen0_grandparent, s2.island_id
    """).fetchall()

    # Group by grandparent and assign bit positions within each group
    from collections import defaultdict
    grandparent_groups = defaultdict(list)
    for island_id, gen1_parent, gen0_grandparent in gen2_islands:
        grandparent_groups[gen0_grandparent].append(island_id)

    for gp_id, island_ids in sorted(grandparent_groups.items()):
        base_id = min(island_ids)
        for island_id in island_ids:
            bit = island_id - base_id
            con.execute("""
                UPDATE island_stats
                SET arch_column = ?, arch_bit = ?
                WHERE island_id = ? AND generation = 2
            """, [f"reef_{gp_id}", bit, island_id])
        print(f"      Gen-2 reef_{gp_id}: {len(island_ids)} reefs (bits 0-{len(island_ids) - 1})")

    total_assigned = con.execute(
        "SELECT COUNT(*) FROM island_stats WHERE arch_column IS NOT NULL"
    ).fetchone()[0]
    print(f"      Total: {total_assigned} bit positions assigned")

    # === Phase B: Bulk-compute bitmasks on words ===
    print("    Phase B: Computing word bitmasks...")

    con.execute("""
        UPDATE words SET archipelago = 0, reef_0 = 0, reef_1 = 0, reef_2 = 0, reef_3 = 0
    """)

    con.execute("""
        WITH bit_positions AS (
            SELECT m.word_id, s.arch_column, s.arch_bit
            FROM dim_memberships m
            JOIN dim_islands di ON m.dim_id = di.dim_id
            JOIN island_stats s ON di.island_id = s.island_id AND di.generation = s.generation
            WHERE di.island_id >= 0 AND s.arch_column IS NOT NULL
        ),
        word_masks AS (
            SELECT word_id,
                BIT_OR(CASE WHEN arch_column = 'archipelago' THEN 1::BIGINT << arch_bit END) as archipelago,
                BIT_OR(CASE WHEN arch_column = 'reef_0' THEN 1::BIGINT << arch_bit END) as reef_0,
                BIT_OR(CASE WHEN arch_column = 'reef_1' THEN 1::BIGINT << arch_bit END) as reef_1,
                BIT_OR(CASE WHEN arch_column = 'reef_2' THEN 1::BIGINT << arch_bit END) as reef_2,
                BIT_OR(CASE WHEN arch_column = 'reef_3' THEN 1::BIGINT << arch_bit END) as reef_3
            FROM bit_positions
            GROUP BY word_id
        )
        UPDATE words SET
            archipelago = COALESCE(wm.archipelago, 0),
            reef_0 = COALESCE(wm.reef_0, 0),
            reef_1 = COALESCE(wm.reef_1, 0),
            reef_2 = COALESCE(wm.reef_2, 0),
            reef_3 = COALESCE(wm.reef_3, 0)
        FROM word_masks wm WHERE words.word_id = wm.word_id
    """)

    # === Phase C: Verification and reporting ===
    print("    Phase C: Verification...")

    counts = con.execute("""
        SELECT
            COUNT(*) FILTER (WHERE archipelago != 0) as n_arch,
            COUNT(*) FILTER (WHERE reef_0 != 0) as n_r0,
            COUNT(*) FILTER (WHERE reef_1 != 0) as n_r1,
            COUNT(*) FILTER (WHERE reef_2 != 0) as n_r2,
            COUNT(*) FILTER (WHERE reef_3 != 0) as n_r3
        FROM words
    """).fetchone()
    print(f"      Words with non-zero archipelago: {counts[0]:,}")
    print(f"      Words with non-zero reef_0: {counts[1]:,}")
    print(f"      Words with non-zero reef_1: {counts[2]:,}")
    print(f"      Words with non-zero reef_2: {counts[3]:,}")
    print(f"      Words with non-zero reef_3: {counts[4]:,}")

    # Spot-check: top-5 words by total_dims
    spot = con.execute("""
        SELECT w.word, w.total_dims,
            bit_count(w.archipelago & 15) as n_arch,
            bit_count(w.archipelago >> 4) as n_islands,
            bit_count(w.reef_0) + bit_count(w.reef_1) +
            bit_count(w.reef_2) + bit_count(w.reef_3) as n_reefs,
            (SELECT COUNT(DISTINCT di.island_id)
             FROM dim_memberships m
             JOIN dim_islands di ON m.dim_id = di.dim_id
             WHERE m.word_id = w.word_id AND di.generation = 0 AND di.island_id >= 0
            ) as actual_arch
        FROM words w
        WHERE w.archipelago != 0
        ORDER BY w.total_dims DESC
        LIMIT 5
    """).fetchall()
    print(f"      Spot-check (top-5 by total_dims):")
    for word, td, na, ni, nr, actual in spot:
        print(f"        {word}: {td} dims, {na} arch(actual={actual}), {ni} islands, {nr} reefs")

    print("  Archipelago encoding complete")


def run_island_detection(con):
    """Entry point: run all island detection steps and print summary."""
    # Gen 0
    compute_jaccard_matrix(con)
    detect_islands(con)
    compute_island_stats(con, generation=0)
    compute_characteristic_words(con, generation=0)
    # Gen 1
    detect_sub_islands(con)
    compute_island_stats(con, generation=1)
    compute_characteristic_words(con, generation=1)
    # Gen 2
    detect_sub_islands(con, parent_generation=1)
    compute_island_stats(con, generation=2)
    compute_characteristic_words(con, generation=2)
    # Encode bitmasks
    compute_archipelago_encoding(con)
    # Summary
    print_archipelago_summary(con)
