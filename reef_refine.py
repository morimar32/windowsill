"""Reef refinement (Phase 10): iterative dim loyalty analysis and reassignment.

After Leiden community detection (Phase 9), some dimensions end up in a reef
where they have higher Jaccard affinity to a sibling reef than to their own.
This module detects those misplaced dims and reassigns them, iterating until
convergence (or a safety-valve max iteration count).
"""
import numpy as np
from collections import defaultdict

import config
import islands


def _load_jaccard_lookup(con):
    """Load all Jaccard values from dim_jaccard into a symmetric dict."""
    lookup = {}
    for a, b, j in con.execute("SELECT dim_id_a, dim_id_b, jaccard FROM dim_jaccard").fetchall():
        lookup[(a, b)] = j
        lookup[(b, a)] = j
    return lookup


def _get_jaccard(d1, d2, lookup):
    """Get Jaccard between two dims (1.0 for self-pairs, 0.0 if missing)."""
    if d1 == d2:
        return 1.0
    return lookup.get((d1, d2), 0.0)


def _compute_dim_loyalty(dims, reef_id, sibling_reef_dims, jaccard_lookup):
    """Compute loyalty metrics for each dim in a reef.

    Returns list of dicts with: dim_id, own_affinity, best_sibling_id,
    best_sibling_affinity, loyalty_ratio.
    """
    results = []
    for d in dims:
        # Affinity to own reef (other dims in same reef)
        own_jaccards = [_get_jaccard(d, d2, jaccard_lookup) for d2 in dims if d2 != d]
        own_affinity = np.mean(own_jaccards) if own_jaccards else 0.0

        # Affinity to each sibling reef
        best_sibling_id = None
        best_sibling_affinity = 0.0

        for sib_id, sib_dims in sibling_reef_dims.items():
            if not sib_dims:
                continue
            sib_jaccards = [_get_jaccard(d, sd, jaccard_lookup) for sd in sib_dims]
            sib_aff = np.mean(sib_jaccards)
            if sib_aff > best_sibling_affinity:
                best_sibling_affinity = sib_aff
                best_sibling_id = sib_id

        loyalty_ratio = own_affinity / best_sibling_affinity if best_sibling_affinity > 0 else float('inf')

        results.append({
            'dim_id': d,
            'own_affinity': own_affinity,
            'best_sibling_id': best_sibling_id,
            'best_sibling_affinity': best_sibling_affinity,
            'loyalty_ratio': loyalty_ratio,
        })

    return results


def run_reef_refinement(con):
    """Iteratively refine reef boundaries by reassigning misplaced dims."""
    min_dims = config.REEF_REFINE_MIN_DIMS
    loyalty_threshold = config.REEF_REFINE_LOYALTY_THRESHOLD
    max_iterations = config.REEF_REFINE_MAX_ITERATIONS

    # Load Jaccard lookup once (reused across iterations)
    print("  Loading Jaccard similarity data...")
    jaccard_lookup = _load_jaccard_lookup(con)
    print(f"    {len(jaccard_lookup) // 2:,} dimension pairs loaded")

    total_moves = 0

    for iteration in range(1, max_iterations + 1):
        print(f"\n  --- Iteration {iteration} ---")

        # Load current reef assignments
        reef_dims = defaultdict(list)
        dim_to_reef = {}
        reef_parent = {}
        for row in con.execute("""
            SELECT di.dim_id, di.island_id AS reef_id, di.parent_island_id
            FROM dim_islands di
            WHERE di.generation = 2 AND di.island_id >= 0
        """).fetchall():
            dim_id, reef_id, parent_id = row
            reef_dims[reef_id].append(dim_id)
            dim_to_reef[dim_id] = reef_id
            reef_parent[reef_id] = parent_id

        # Build parent -> child reefs mapping
        parent_to_reefs = defaultdict(list)
        for reef_id, parent_id in reef_parent.items():
            parent_to_reefs[parent_id].append(reef_id)

        # Load reef names for reporting
        reef_names = {}
        for row in con.execute("""
            SELECT island_id, island_name FROM island_stats
            WHERE generation = 2 AND island_id >= 0
        """).fetchall():
            reef_names[row[0]] = row[1] or f"reef {row[0]}"

        # Analyze each reef with enough dims
        moves = []  # (dim_id, from_reef, to_reef)

        for reef_id, dims in reef_dims.items():
            if len(dims) < min_dims:
                continue

            parent_id = reef_parent[reef_id]
            sibling_ids = [r for r in parent_to_reefs[parent_id] if r != reef_id]
            if not sibling_ids:
                continue

            # Build sibling dims map
            sibling_reef_dims = {sid: reef_dims[sid] for sid in sibling_ids}

            # Compute loyalty for each dim
            loyalty_results = _compute_dim_loyalty(dims, reef_id, sibling_reef_dims, jaccard_lookup)

            for lr in loyalty_results:
                if lr['loyalty_ratio'] < loyalty_threshold and lr['best_sibling_id'] is not None:
                    moves.append((lr['dim_id'], reef_id, lr['best_sibling_id']))

        if not moves:
            print("  No misplaced dims found — converged!")
            break

        # Apply moves
        affected_reefs = set()
        for dim_id, from_reef, to_reef in moves:
            con.execute("""
                UPDATE dim_islands SET island_id = ?
                WHERE dim_id = ? AND generation = 2
            """, [to_reef, dim_id])
            affected_reefs.add(from_reef)
            affected_reefs.add(to_reef)
            total_moves += 1

            from_name = reef_names.get(from_reef, f"reef {from_reef}")
            to_name = reef_names.get(to_reef, f"reef {to_reef}")
            print(f"    dim {dim_id}: {from_name} [{from_reef}] → {to_name} [{to_reef}]")

        print(f"  Moved {len(moves)} dims across {len(affected_reefs)} reefs")

        # Recompute stats and characteristic words for all gen-2 reefs
        print("  Recomputing reef stats and characteristic words...")
        islands.compute_island_stats(con, generation=2)
        islands.compute_characteristic_words(con, generation=2)

    else:
        print(f"\n  Reached max iterations ({max_iterations}) — stopping")

    # Post-convergence: refresh denormalized columns and affinity scores
    print("\n  Refreshing denormalized columns and affinity scores...")
    islands.backfill_membership_islands(con)
    islands.compute_word_reef_affinity(con)

    # Refresh reef valence and POS composition after reassignment
    import post_process
    post_process.compute_reef_valence(con)
    post_process.compute_hierarchy_pos_composition(con)
    post_process.compute_hierarchy_specificity(con)
    post_process.compute_reef_edges(con)

    print(f"\n  Reef refinement complete: {total_moves} total dim moves")
