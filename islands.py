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
            None, None,
        ))

    df = pd.DataFrame(stats_rows, columns=[
        "island_id", "generation", "n_dims", "n_words",
        "avg_internal_jaccard", "max_internal_jaccard", "min_internal_jaccard",
        "modularity_contribution", "parent_island_id", "island_name",
        "n_core_words", "median_word_depth",
        "arch_column", "arch_bit",
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

    # Gen-0: low bits of archipelago column
    con.execute("""
        UPDATE island_stats
        SET arch_column = 'archipelago', arch_bit = island_id
        WHERE generation = 0 AND island_id >= 0
    """)
    gen0_count = con.execute(
        "SELECT COUNT(*) FROM island_stats WHERE generation = 0 AND island_id >= 0"
    ).fetchone()[0]
    print(f"      Gen-0: {gen0_count} archipelagos (archipelago bits 0-{gen0_count - 1})")

    # Gen-1: split evenly across archipelago (after gen-0 bits) and archipelago_ext
    gen1_count = con.execute(
        "SELECT COUNT(*) FROM island_stats WHERE generation = 1 AND island_id >= 0"
    ).fetchone()[0]
    gen1_half = (gen1_count + 1) // 2

    # First half in archipelago, offset by gen0_count
    con.execute("""
        UPDATE island_stats
        SET arch_column = 'archipelago', arch_bit = ? + island_id
        WHERE generation = 1 AND island_id >= 0 AND island_id < ?
    """, [gen0_count, gen1_half])

    # Second half in archipelago_ext
    con.execute("""
        UPDATE island_stats
        SET arch_column = 'archipelago_ext', arch_bit = island_id - ?
        WHERE generation = 1 AND island_id >= 0 AND island_id >= ?
    """, [gen1_half, gen1_half])

    gen1_first = min(gen1_half, gen1_count)
    gen1_second = gen1_count - gen1_first
    max_bit_arch = gen0_count + gen1_first - 1
    if max_bit_arch >= 64:
        raise ValueError(f"Archipelago column overflow: need bit {max_bit_arch}, max is 63")
    if gen1_second > 64:
        raise ValueError(f"Archipelago_ext column overflow: need {gen1_second} bits, max is 64")
    print(f"      Gen-1: {gen1_count} islands ({gen1_first} in archipelago bits {gen0_count}-{max_bit_arch}"
          + (f", {gen1_second} in archipelago_ext bits 0-{gen1_second - 1})" if gen1_second > 0 else ")"))

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

    # Group by grandparent and assign bit positions within each group.
    # Each BIGINT column holds at most 64 bits; overflow groups spill into the next column.
    from collections import defaultdict
    grandparent_groups = defaultdict(list)
    for island_id, gen1_parent, gen0_grandparent in gen2_islands:
        grandparent_groups[gen0_grandparent].append(island_id)

    reef_columns = ["reef_0", "reef_1", "reef_2", "reef_3", "reef_4", "reef_5"]
    col_idx = 0
    for gp_id, island_ids in sorted(grandparent_groups.items()):
        for chunk_start in range(0, len(island_ids), 64):
            if col_idx >= len(reef_columns):
                raise ValueError(
                    f"Reef column overflow: need more than {len(reef_columns)} columns "
                    f"({sum(len(v) for v in grandparent_groups.values())} total reefs)"
                )
            chunk = island_ids[chunk_start:chunk_start + 64]
            col_name = reef_columns[col_idx]
            for i, island_id in enumerate(chunk):
                con.execute("""
                    UPDATE island_stats
                    SET arch_column = ?, arch_bit = ?
                    WHERE island_id = ? AND generation = 2
                """, [col_name, i, island_id])
            print(f"      Gen-2 {col_name}: {len(chunk)} reefs for arch {gp_id} (bits 0-{len(chunk) - 1})")
            col_idx += 1

    total_assigned = con.execute(
        "SELECT COUNT(*) FROM island_stats WHERE arch_column IS NOT NULL"
    ).fetchone()[0]
    print(f"      Total: {total_assigned} bit positions assigned")

    # === Phase B: Bulk-compute bitmasks on words ===
    min_depth = config.REEF_MIN_DEPTH
    print(f"    Phase B: Computing word bitmasks (min depth = {min_depth})...")

    con.execute("""
        UPDATE words SET archipelago = 0, archipelago_ext = 0,
            reef_0 = 0, reef_1 = 0, reef_2 = 0, reef_3 = 0, reef_4 = 0, reef_5 = 0
    """)

    # DuckDB's signed BIGINT overflows on 1::BIGINT << 63, so use a safe expression
    # that handles the sign bit explicitly (bit 63 = MIN_BIGINT = -9223372036854775808)
    SAFE_BIT = "(CASE WHEN arch_bit = 63 THEN (-9223372036854775808)::BIGINT ELSE 1::BIGINT << arch_bit END)"

    cols = ["archipelago", "archipelago_ext", "reef_0", "reef_1", "reef_2", "reef_3", "reef_4", "reef_5"]
    bit_or_exprs = ",\n                ".join(
        f"BIT_OR(CASE WHEN arch_column = '{c}' THEN {SAFE_BIT} END) as {c}" for c in cols
    )
    coalesce_sets = ",\n            ".join(f"{c} = COALESCE(wm.{c}, 0)" for c in cols)

    con.execute(f"""
        WITH word_island_depth AS (
            -- For each word + island (any generation), count activated dims
            SELECT m.word_id, di.island_id, di.generation
            FROM dim_memberships m
            JOIN dim_islands di ON m.dim_id = di.dim_id
            WHERE di.island_id >= 0
            GROUP BY m.word_id, di.island_id, di.generation
            HAVING COUNT(*) >= {min_depth}
        ),
        bit_positions AS (
            SELECT m.word_id, s.arch_column, s.arch_bit
            FROM dim_memberships m
            JOIN dim_islands di ON m.dim_id = di.dim_id
            JOIN island_stats s ON di.island_id = s.island_id AND di.generation = s.generation
            JOIN word_island_depth wid
                ON m.word_id = wid.word_id
                AND di.island_id = wid.island_id
                AND di.generation = wid.generation
            WHERE s.arch_column IS NOT NULL
        ),
        word_masks AS (
            SELECT word_id,
                {bit_or_exprs}
            FROM bit_positions
            GROUP BY word_id
        )
        UPDATE words SET
            {coalesce_sets}
        FROM word_masks wm WHERE words.word_id = wm.word_id
    """)

    # === Phase C: Verification and reporting ===
    print("    Phase C: Verification...")

    counts = con.execute("""
        SELECT
            COUNT(*) FILTER (WHERE archipelago != 0) as n_arch,
            COUNT(*) FILTER (WHERE archipelago_ext != 0) as n_arch_ext,
            COUNT(*) FILTER (WHERE reef_0 != 0) as n_r0,
            COUNT(*) FILTER (WHERE reef_1 != 0) as n_r1,
            COUNT(*) FILTER (WHERE reef_2 != 0) as n_r2,
            COUNT(*) FILTER (WHERE reef_3 != 0) as n_r3,
            COUNT(*) FILTER (WHERE reef_4 != 0) as n_r4,
            COUNT(*) FILTER (WHERE reef_5 != 0) as n_r5
        FROM words
    """).fetchone()
    print(f"      Words with non-zero archipelago: {counts[0]:,}")
    print(f"      Words with non-zero archipelago_ext: {counts[1]:,}")
    for i in range(6):
        print(f"      Words with non-zero reef_{i}: {counts[2 + i]:,}")

    # Spot-check: top-5 words by total_dims
    gen0_mask = (1 << gen0_count) - 1
    spot = con.execute(f"""
        SELECT w.word, w.total_dims,
            bit_count(w.archipelago & {gen0_mask}) as n_arch,
            bit_count(w.archipelago >> {gen0_count}) + bit_count(w.archipelago_ext) as n_islands,
            bit_count(w.reef_0) + bit_count(w.reef_1) + bit_count(w.reef_2) +
            bit_count(w.reef_3) + bit_count(w.reef_4) + bit_count(w.reef_5) as n_reefs,
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


def generate_island_names(con):
    """Generate descriptive names bottom-up: reefs -> islands -> archipelagos.

    Reefs (gen-2) are named from their exclusive + PMI words.
    Islands (gen-1) are named by synthesizing their child reef names.
    Archipelagos (gen-0) are named by synthesizing their child island names.
    """
    import anthropic
    import time

    client = anthropic.Anthropic()

    # Preload word_id -> word text mapping (single words only)
    word_lookup = {}
    rows = con.execute(
        f"SELECT word_id, word FROM words WHERE {_SINGLE_WORD_FILTER}"
    ).fetchall()
    for wid, word in rows:
        word_lookup[wid] = word
    print(f"  Loaded {len(word_lookup):,} single-word lookups")

    # Step 1: Name gen-2 reefs (most specific — exclusive + PMI words)
    print("\n  === Step 1: Naming gen-2 reefs (bottom level) ===")
    n_reefs = _name_reefs(con, client, word_lookup)

    # Step 2: Name gen-1 islands (synthesize from child reef names)
    print("\n  === Step 2: Naming gen-1 islands (from reef composition) ===")
    n_islands = _name_from_children(con, client, parent_generation=0, child_generation=1)

    # Step 3: Name gen-0 archipelagos (synthesize from child island names)
    print("\n  === Step 3: Naming gen-0 archipelagos (from island composition) ===")
    n_archs = _name_from_children(con, client, parent_generation=None, child_generation=0)

    print(f"\n  Naming complete: {n_archs} archipelagos + {n_islands} islands + {n_reefs} reefs")
    _print_naming_summary(con)


def _name_reefs(con, client, word_lookup):
    """Name gen-2 reefs using exclusive words + PMI, batched by parent island."""
    import time

    # Get all gen-1 islands that have child reefs
    parents = con.execute("""
        SELECT s.island_id, s.island_name, s.n_dims, s.n_words
        FROM island_stats s
        WHERE s.generation = 1 AND s.island_id >= 0
        AND EXISTS (
            SELECT 1 FROM island_stats c
            WHERE c.generation = 2 AND c.parent_island_id = s.island_id AND c.island_id >= 0
        )
        ORDER BY s.island_id
    """).fetchall()

    total_named = 0

    for parent_id, parent_name, parent_ndims, parent_nwords in parents:
        # Get child reefs
        children = con.execute("""
            SELECT island_id, n_dims, n_words, n_core_words
            FROM island_stats
            WHERE generation = 2 AND parent_island_id = ? AND island_id >= 0
            ORDER BY island_id
        """, [parent_id]).fetchall()

        if not children:
            continue

        child_ids = [c[0] for c in children]

        # Compute exclusive words for all sibling reefs at once
        exclusive_map, word_dims_map = _compute_exclusive_words(
            con, parent_id, child_generation=2
        )

        # Get PMI words for each reef
        pmi_map = {}
        for child_id in child_ids:
            pmi_words = con.execute("""
                SELECT word, pmi FROM island_characteristic_words
                WHERE island_id = ? AND generation = 2
                ORDER BY pmi DESC LIMIT 20
            """, [child_id]).fetchall()
            pmi_map[child_id] = [w for w, _ in pmi_words]

        # Build per-reef data
        child_data = []
        for child_id, n_dims, n_words, n_core in children:
            exc_word_ids = exclusive_map.get(child_id, set())
            reef_dims = word_dims_map.get(child_id, {})

            exc_ranked = sorted(
                [(word_lookup[wid], reef_dims.get(wid, 0))
                 for wid in exc_word_ids if wid in word_lookup],
                key=lambda x: x[1], reverse=True
            )[:20]

            child_data.append({
                "id": child_id,
                "n_dims": n_dims,
                "n_words": n_words,
                "n_core": n_core or 0,
                "n_exclusive_total": len(exc_word_ids),
                "exclusive": [w for w, _ in exc_ranked],
                "pmi": pmi_map.get(child_id, []),
            })

        prompt = _build_reef_naming_prompt(child_data)

        parent_label = parent_name or f"island {parent_id}"
        print(f"\n    {parent_label} -> {len(children)} reefs...")

        names = _call_claude_for_names(client, prompt, child_ids)

        for child_id, name in names.items():
            con.execute(
                "UPDATE island_stats SET island_name = ? WHERE island_id = ? AND generation = 2",
                [name, child_id],
            )
            print(f"      [{child_id}] {name}")
            total_named += 1

        time.sleep(0.5)

    return total_named


def _name_from_children(con, client, parent_generation, child_generation):
    """Name entities at child_generation by synthesizing their children's names.

    For gen-1 islands: looks at child reef names to synthesize.
    For gen-0 archipelagos: looks at child island names to synthesize.
    """
    import time

    grandchild_generation = child_generation + 1
    gen_label = {0: "archipelago", 1: "island"}[child_generation]
    child_label = {1: "reef", 2: "reef"}  # what the children of children are called

    if parent_generation is not None:
        # Gen-1 islands: batch by parent archipelago
        parents = con.execute("""
            SELECT s.island_id, s.island_name
            FROM island_stats s
            WHERE s.generation = ? AND s.island_id >= 0
            AND EXISTS (
                SELECT 1 FROM island_stats c
                WHERE c.generation = ? AND c.parent_island_id = s.island_id AND c.island_id >= 0
            )
            ORDER BY s.island_id
        """, [parent_generation, child_generation]).fetchall()
    else:
        # Gen-0 archipelagos: no parent, process all together
        parents = [(None, None)]

    total_named = 0

    for parent_id, parent_name in parents:
        # Get the entities we need to name
        if parent_id is not None:
            entities = con.execute("""
                SELECT island_id, n_dims, n_words
                FROM island_stats
                WHERE generation = ? AND parent_island_id = ? AND island_id >= 0
                ORDER BY island_id
            """, [child_generation, parent_id]).fetchall()
        else:
            entities = con.execute("""
                SELECT island_id, n_dims, n_words
                FROM island_stats
                WHERE generation = ? AND island_id >= 0
                ORDER BY island_id
            """, [child_generation]).fetchall()

        if not entities:
            continue

        entity_ids = [e[0] for e in entities]

        # For each entity, get its children's names (or PMI words as fallback)
        entity_data = []
        for entity_id, n_dims, n_words in entities:
            children_names = con.execute("""
                SELECT island_id, island_name, n_dims, n_words
                FROM island_stats
                WHERE generation = ? AND parent_island_id = ? AND island_id >= 0
                ORDER BY n_dims DESC
            """, [grandchild_generation, entity_id]).fetchall()

            # Fallback: PMI words for entities without children
            pmi_words = []
            if not children_names:
                pmi_rows = con.execute("""
                    SELECT word FROM island_characteristic_words
                    WHERE island_id = ? AND generation = ?
                    ORDER BY pmi DESC LIMIT 20
                """, [entity_id, child_generation]).fetchall()
                pmi_words = [w for (w,) in pmi_rows]

            entity_data.append({
                "id": entity_id,
                "n_dims": n_dims,
                "n_words": n_words,
                "children": [
                    {"id": cid, "name": cname or f"(unnamed {cid})", "n_dims": cd, "n_words": cw}
                    for cid, cname, cd, cw in children_names
                ],
                "pmi_words": pmi_words,
            })

        prompt = _build_synthesis_naming_prompt(
            gen_label, entity_data, parent_name
        )

        batch_label = parent_name or f"all {gen_label}s"
        n_entities = len(entities)
        print(f"\n    {batch_label} -> {n_entities} {gen_label}s...")

        names = _call_claude_for_names(client, prompt, entity_ids)

        for entity_id, name in names.items():
            con.execute(
                "UPDATE island_stats SET island_name = ? WHERE island_id = ? AND generation = ?",
                [name, entity_id, child_generation],
            )
            print(f"      [{entity_id}] {name}")
            total_named += 1

        time.sleep(0.5)

    return total_named


def _compute_exclusive_words(con, parent_id, child_generation):
    """Compute words exclusive to each child among siblings of the same parent.

    Returns:
        exclusive_map: {child_id: set of exclusive word_ids}
        word_dims_map: {child_id: {word_id: n_dims_in_child}}
    """
    from collections import defaultdict

    rows = con.execute(f"""
        SELECT di.island_id, dm.word_id, COUNT(*) as n_dims
        FROM dim_islands di
        JOIN dim_memberships dm ON dm.dim_id = di.dim_id
        JOIN words w ON dm.word_id = w.word_id
        WHERE di.generation = ? AND di.parent_island_id = ? AND di.island_id >= 0
        AND w.{_SINGLE_WORD_FILTER}
        GROUP BY di.island_id, dm.word_id
    """, [child_generation, parent_id]).fetchall()

    child_words = defaultdict(set)
    word_dims_map = defaultdict(dict)
    for child_id, word_id, n_dims in rows:
        child_words[child_id].add(word_id)
        word_dims_map[child_id][word_id] = n_dims

    all_child_ids = list(child_words.keys())
    exclusive_map = {}
    for child_id in all_child_ids:
        sibling_words = set()
        for other_id in all_child_ids:
            if other_id != child_id:
                sibling_words |= child_words[other_id]
        exclusive_map[child_id] = child_words[child_id] - sibling_words

    return exclusive_map, word_dims_map


def _build_reef_naming_prompt(child_data):
    """Build prompt for naming reefs from their exclusive + PMI words."""
    sections = []
    for cd in child_data:
        exc_str = ", ".join(cd["exclusive"]) if cd["exclusive"] else "(none — high overlap with siblings)"
        pmi_str = ", ".join(cd["pmi"]) if cd["pmi"] else "(none)"
        exc_count = cd["n_exclusive_total"]

        section = (
            f"### Reef {cd['id']} "
            f"({cd['n_dims']} dims, {cd['n_words']:,} words, {cd['n_core']} core words)\n"
            f"Exclusive words ({exc_count:,} total, top 20):\n"
            f"  {exc_str}\n\n"
            f"Top PMI words (may overlap with siblings):\n"
            f"  {pmi_str}"
        )
        sections.append(section)

    siblings_text = "\n\n".join(sections)
    example_id = str(child_data[0]["id"])

    return f"""You are naming semantic clusters discovered in a word embedding space. Each reef is a tight cluster of embedding dimensions that share similar word activation patterns.

## Task
Name each of the following {len(child_data)} sibling reefs. Each name should be:
- 2-4 words, lowercase
- Descriptive of the semantic theme these words share
- Distinct from sibling names (these reefs are all under the same parent island, so contrast matters)
- The exclusive words are words that appear ONLY in this reef and not in any sibling reef — they are the strongest signal for what makes this reef unique
- The PMI words show what's most statistically overrepresented in the reef overall
- Prefer readable, informative names over obscure technical terms

## Reef data

{siblings_text}

## Response format
Return ONLY a JSON object mapping reef IDs to names. No markdown, no commentary.
Example: {{"{example_id}": "example name"}}"""


def _build_synthesis_naming_prompt(gen_label, entity_data, parent_name=None):
    """Build prompt for naming entities by synthesizing their children's names."""
    child_label = "reef" if gen_label == "island" else "island"

    sections = []
    for ed in entity_data:
        if ed["children"]:
            children_list = "\n".join(
                f"    - {c['name']} ({c['n_dims']} dims, {c['n_words']:,} words)"
                for c in ed["children"]
            )
            section = (
                f"### {gen_label.title()} {ed['id']} "
                f"({ed['n_dims']} dims, {ed['n_words']:,} words)\n"
                f"  Constituent {child_label}s:\n{children_list}"
            )
        else:
            # Fallback: show PMI words for childless entities
            pmi_str = ", ".join(ed.get("pmi_words", [])) or "(no data)"
            section = (
                f"### {gen_label.title()} {ed['id']} "
                f"({ed['n_dims']} dims, {ed['n_words']:,} words)\n"
                f"  No sub-clusters. Top characteristic words:\n"
                f"    {pmi_str}"
            )
        sections.append(section)

    siblings_text = "\n\n".join(sections)
    example_id = str(entity_data[0]["id"])

    context = ""
    if parent_name:
        context = f"\nParent: \"{parent_name}\"\n"

    return f"""You are naming semantic clusters in a word embedding space. Each {gen_label} is composed of smaller {child_label}s listed below (or, for smaller clusters, characterized by their top words). Your job is to find the overarching theme.
{context}
## Task
Name each of the following {len(entity_data)} {gen_label}s. Each name should be:
- 2-4 words, lowercase
- For {gen_label}s with sub-clusters: a broader theme that encompasses its children
- For {gen_label}s with only word lists: a theme that captures what the words share
- Distinct from sibling {gen_label} names
- More general than any single child name, but specific enough to be informative

## {gen_label.title()} data

{siblings_text}

## Response format
Return ONLY a JSON object mapping {gen_label} IDs to names. No markdown, no commentary.
Example: {{"{example_id}": "example name"}}"""


def _call_claude_for_names(client, prompt, expected_ids):
    """Call Claude API and parse the JSON response into {id: name} mapping."""
    import json

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    try:
        names = json.loads(text)
    except json.JSONDecodeError:
        print(f"      WARNING: Failed to parse response: {text[:200]}")
        return {}

    result = {}
    for key, name in names.items():
        kid = int(key)
        if kid in expected_ids:
            result[kid] = name
        else:
            print(f"      WARNING: Unexpected ID {kid} in response")

    missing = set(expected_ids) - set(result.keys())
    if missing:
        print(f"      WARNING: Missing names for IDs: {missing}")

    return result


def _print_naming_summary(con):
    """Print the full hierarchy with names."""
    print("\n  === Island & Reef Naming Summary ===\n")

    gen0 = con.execute("""
        SELECT island_id, island_name, n_dims
        FROM island_stats WHERE generation = 0 AND island_id >= 0
        ORDER BY island_id
    """).fetchall()

    for arch_id, arch_name, arch_dims in gen0:
        print(f"  [{arch_id}] {arch_name or '(unnamed)'} ({arch_dims} dims)")

        gen1 = con.execute("""
            SELECT island_id, island_name, n_dims, n_words
            FROM island_stats WHERE generation = 1 AND parent_island_id = ? AND island_id >= 0
            ORDER BY island_id
        """, [arch_id]).fetchall()

        for isl_id, isl_name, isl_dims, isl_words in gen1:
            print(f"    [{isl_id}] {isl_name or '(unnamed)'} ({isl_dims} dims, {isl_words:,} words)")

            gen2 = con.execute("""
                SELECT island_id, island_name, n_dims, n_words
                FROM island_stats WHERE generation = 2 AND parent_island_id = ? AND island_id >= 0
                ORDER BY island_id
            """, [isl_id]).fetchall()

            for reef_id, reef_name, reef_dims, reef_words in gen2:
                print(f"      [{reef_id}] {reef_name or '(unnamed)'} ({reef_dims} dims, {reef_words:,} words)")

        print()


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
