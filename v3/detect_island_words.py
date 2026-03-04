"""
Detect island-level words and triage capital seeds.

Two-phase seed cleanup using cosine-std signal:

  Phase 1 — Capital seed triage (adaptive threshold per island):
    For each capital town, computes cosine-std of every seed against
    NON-CAPITAL town centroids only.
    - GENERIC (std < NC p25): seed is non-discriminative across sibling
      towns → promote to IslandWords.
    - MISPLACED (best_nc_cos > island_cos + margin): seed clearly belongs
      to a specific non-capital town → move it there.
    - GENUINE: keep in capital.

  Phase 2 — Cross-town island-level detection (fixed threshold):
    For ALL remaining seeds (capital and non-capital), computes cosine-std
    across ALL town centroids.  Words with low std that are in the capital
    or in 2+ towns are non-discriminative → promote to IslandWords.

Run this BEFORE XGBoost training so classifiers only see town-specific seeds.

Usage:
    python v3/detect_island_words.py                    # preview (dry-run)
    python v3/detect_island_words.py --apply            # modify the database
    python v3/detect_island_words.py --island Sport     # single island
"""

import argparse
import os
import struct
import sys
import time
from collections import defaultdict

import numpy as np

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project)

V3_DB = os.path.join(_project, "v3/windowsill.db")

PHASE2_STD_THRESHOLD = 0.010  # fixed threshold for cross-town detection
MIN_TOWNS_FOR_DETECTION = 3   # need ≥3 towns to compute meaningful std
MISPLACED_MARGIN = 0.03       # best_nc_cos must exceed island_cos by this much


def unpack_embedding(blob):
    return np.array(struct.unpack(f"{768}f", blob), dtype=np.float64)


def norm_vec(vec):
    """Return L2-normalized copy, or None if zero-norm."""
    n = np.linalg.norm(vec)
    if n == 0:
        return None
    return vec / n


def build_centroid(wids, emb_map):
    """Build L2-normalized centroid from word_id set."""
    vecs = []
    for w in wids:
        if w in emb_map:
            v = norm_vec(emb_map[w])
            if v is not None:
                vecs.append(v)
    if not vecs:
        return None
    c = np.mean(vecs, axis=0)
    n = np.linalg.norm(c)
    if n == 0:
        return None
    return c / n


def load_island_seeds(con, island_id):
    """Load all seed data for an island. Returns (towns, town_seed_wids,
    seed_words, seed_town_map, capital_ids)."""
    towns = con.execute(
        "SELECT town_id, name, is_capital FROM Towns WHERE island_id = ? ORDER BY town_id",
        (island_id,)
    ).fetchall()

    town_ids = [t[0] for t in towns]
    capital_ids = {t[0] for t in towns if t[2]}

    town_seed_wids = defaultdict(set)
    seed_words = {}
    seed_town_map = defaultdict(set)

    placeholders = ",".join("?" * len(town_ids))
    rows = con.execute(f"""
        SELECT town_id, word, word_id FROM SeedWords
        WHERE town_id IN ({placeholders}) AND word_id IS NOT NULL
    """, town_ids).fetchall()

    for tid, word, wid in rows:
        town_seed_wids[tid].add(wid)
        seed_words[wid] = word
        seed_town_map[wid].add(tid)

    return towns, town_seed_wids, seed_words, seed_town_map, capital_ids


def phase1_capital_triage(island_id, island_name, towns, town_seed_wids,
                          seed_words, seed_town_map, capital_ids, emb_map):
    """Phase 1: triage capital seeds using adaptive cosine-std threshold.

    Returns (generic, misplaced, genuine, stats) where:
      generic:   [(wid, word, std, avg_cos)]
      misplaced: [(wid, word, std, dest_tid, dest_name, best_cos, island_cos)]
      genuine:   [(wid, word, std, avg_cos)]
      stats:     dict with nc_p25, nc_p50, etc.
    """
    town_names = {t[0]: t[1] for t in towns}
    nc_tids = [t[0] for t in towns if not t[2]]

    if not capital_ids or len(nc_tids) < 2:
        return [], [], [], {}

    cap_tid = list(capital_ids)[0]  # assume single capital per island

    # Build non-capital centroids from their seeds
    nc_centroids = {}
    for tid in nc_tids:
        c = build_centroid(town_seed_wids[tid], emb_map)
        if c is not None:
            nc_centroids[tid] = c

    if len(nc_centroids) < 2:
        return [], [], [], {}

    centroid_tids = sorted(nc_centroids.keys())
    centroid_matrix = np.vstack([nc_centroids[tid] for tid in centroid_tids])

    # Build island-wide centroid from ALL seeds (all towns)
    all_wids = set()
    for wids in town_seed_wids.values():
        all_wids.update(wids)
    island_centroid = build_centroid(all_wids, emb_map)
    if island_centroid is None:
        return [], [], [], {}

    # Compute NC-p25: the 25th percentile of cosine-std across non-capital seeds.
    # This is our adaptive "generic" threshold — if a capital seed has lower std
    # than 75% of non-capital seeds, it's less discriminative than a typical
    # non-capital seed and should be promoted to island level.
    nc_seed_stds = []
    for tid in nc_tids:
        for wid in town_seed_wids[tid]:
            v = norm_vec(emb_map[wid]) if wid in emb_map else None
            if v is None:
                continue
            sims = centroid_matrix @ v
            nc_seed_stds.append(float(np.std(sims)))

    if not nc_seed_stds:
        return [], [], [], {}

    nc_p25 = float(np.percentile(nc_seed_stds, 25))
    nc_p50 = float(np.percentile(nc_seed_stds, 50))

    # Triage each capital seed
    generic = []
    misplaced = []
    genuine = []

    for wid in town_seed_wids.get(cap_tid, set()):
        word = seed_words.get(wid, "?")
        v = norm_vec(emb_map[wid]) if wid in emb_map else None
        if v is None:
            continue

        sims = centroid_matrix @ v
        std = float(np.std(sims))
        avg = float(np.mean(sims))
        best_idx = int(np.argmax(sims))
        best_tid = centroid_tids[best_idx]
        best_cos = float(sims[best_idx])
        island_cos = float(island_centroid @ v)

        if std < nc_p25:
            generic.append((wid, word, std, avg))
        elif best_cos > island_cos + MISPLACED_MARGIN:
            misplaced.append((wid, word, std, best_tid,
                              town_names[best_tid], best_cos, island_cos))
        else:
            genuine.append((wid, word, std, avg))

    stats = {
        "nc_p25": nc_p25,
        "nc_p50": nc_p50,
        "cap_tid": cap_tid,
        "cap_name": town_names.get(cap_tid, "?"),
        "n_nc_centroids": len(nc_centroids),
    }
    return generic, misplaced, genuine, stats


def phase2_crosstown_detection(island_id, island_name, towns, town_seed_wids,
                               seed_words, seed_town_map, capital_ids,
                               emb_map, threshold):
    """Phase 2: detect island-level words across all towns using fixed threshold.

    Returns detected: [(wid, word, std, avg_cos)]
    """
    town_centroids = {}
    for t in towns:
        tid = t[0]
        c = build_centroid(town_seed_wids[tid], emb_map)
        if c is not None:
            town_centroids[tid] = c

    if len(town_centroids) < MIN_TOWNS_FOR_DETECTION:
        return []

    centroid_tids = sorted(town_centroids.keys())
    centroid_matrix = np.vstack([town_centroids[tid] for tid in centroid_tids])

    detected = []
    for wid, word in seed_words.items():
        v = norm_vec(emb_map[wid]) if wid in emb_map else None
        if v is None:
            continue

        sims = centroid_matrix @ v
        std = float(np.std(sims))
        avg = float(np.mean(sims))

        if std < threshold:
            # Skip if in exactly 1 non-capital town (deliberately placed)
            tids = seed_town_map[wid]
            noncap_tids = tids - capital_ids
            if len(noncap_tids) == 1 and not (tids & capital_ids):
                continue
            detected.append((wid, word, std, avg))

    return detected


def main():
    import sqlite3

    parser = argparse.ArgumentParser(
        description="Detect island-level words and triage capital seeds")
    parser.add_argument("--apply", action="store_true",
                        help="Modify the database (default: dry-run)")
    parser.add_argument("--island", type=str, default=None,
                        help="Process single island (default: all)")
    parser.add_argument("--phase2-threshold", type=float,
                        default=PHASE2_STD_THRESHOLD,
                        help=f"Phase 2 cosine-std threshold (default: {PHASE2_STD_THRESHOLD})")
    parser.add_argument("--misplaced-margin", type=float,
                        default=MISPLACED_MARGIN,
                        help=f"Margin for misplaced detection (default: {MISPLACED_MARGIN})")
    args = parser.parse_args()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    # Load embeddings
    print("Loading embeddings...")
    t0 = time.time()
    emb_map = {}
    for wid, blob in con.execute(
        "SELECT word_id, embedding FROM Words WHERE embedding IS NOT NULL"
    ).fetchall():
        emb_map[wid] = unpack_embedding(blob)
    print(f"  {len(emb_map):,} embeddings ({time.time()-t0:.1f}s)")

    # Get islands
    if args.island:
        islands = con.execute(
            "SELECT island_id, name FROM Islands WHERE name = ?",
            (args.island,)
        ).fetchall()
    else:
        islands = con.execute(
            "SELECT island_id, name FROM Islands ORDER BY island_id"
        ).fetchall()

    # ── Phase 1: Capital Seed Triage ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Capital Seed Triage")
    print("=" * 60)

    p1_total_generic = 0
    p1_total_misplaced = 0
    p1_total_genuine = 0
    p1_actions = []  # (island_id, generic, misplaced) for apply

    for island_id, island_name in islands:
        data = load_island_seeds(con, island_id)
        towns, town_seed_wids, seed_words, seed_town_map, capital_ids = data

        if len(towns) < MIN_TOWNS_FOR_DETECTION or not capital_ids:
            continue

        generic, misplaced, genuine, stats = phase1_capital_triage(
            island_id, island_name, towns, town_seed_wids,
            seed_words, seed_town_map, capital_ids, emb_map
        )

        if not generic and not misplaced:
            continue

        total = len(generic) + len(misplaced) + len(genuine)
        nc_seed_avg = 0
        nc_tids = [t[0] for t in towns if not t[2]]
        if nc_tids:
            nc_seed_avg = sum(len(town_seed_wids[t]) for t in nc_tids) / len(nc_tids)
        ratio_before = total / nc_seed_avg if nc_seed_avg > 0 else 0
        ratio_after = len(genuine) / nc_seed_avg if nc_seed_avg > 0 else 0

        print(f"\n  {island_name} > {stats['cap_name']}  "
              f"(threshold: NC-p25={stats['nc_p25']:.4f})")
        print(f"    GENERIC:   {len(generic):>4d}  "
              f"MISPLACED: {len(misplaced):>4d}  "
              f"GENUINE: {len(genuine):>4d}  "
              f"(cap/nc: {ratio_before:.1f}x -> {ratio_after:.1f}x)")

        if generic:
            generic.sort(key=lambda x: x[2])
            samples = ", ".join(f"'{g[1]}'" for g in generic[:5])
            print(f"    generic sample: {samples}")
        if misplaced:
            misplaced.sort(key=lambda x: -(x[5] - x[6]))
            for _, word, std, _, dest, best_cos, isl_cos in misplaced[:5]:
                print(f"    misplaced: '{word}' -> {dest} "
                      f"(nc={best_cos:.3f} > isl={isl_cos:.3f})")

        p1_total_generic += len(generic)
        p1_total_misplaced += len(misplaced)
        p1_total_genuine += len(genuine)
        p1_actions.append((island_id, generic, misplaced))

    print(f"\n  Phase 1 total: {p1_total_generic:,} generic, "
          f"{p1_total_misplaced:,} misplaced, "
          f"{p1_total_genuine:,} genuine")

    # ── Phase 2: Cross-town Detection ─────────────────────────────────────
    # NOTE: in dry-run, phase 2 runs on the CURRENT seed state (before phase 1
    # changes).  In apply mode, phase 2 runs AFTER phase 1 mutations, so the
    # capital seeds are already cleaned up.

    print("\n" + "=" * 60)
    print("PHASE 2: Cross-town Island-level Detection")
    print("=" * 60)

    p2_total = 0
    p2_actions = []

    for island_id, island_name in islands:
        data = load_island_seeds(con, island_id)
        towns, town_seed_wids, seed_words, seed_town_map, capital_ids = data

        if len(towns) < MIN_TOWNS_FOR_DETECTION:
            continue

        detected = phase2_crosstown_detection(
            island_id, island_name, towns, town_seed_wids,
            seed_words, seed_town_map, capital_ids,
            emb_map, args.phase2_threshold
        )

        if not detected:
            continue

        n_seeds = len(seed_words)
        pct = 100 * len(detected) / n_seeds if n_seeds > 0 else 0
        print(f"  {island_name}: {len(detected)}/{n_seeds} "
              f"island-level ({pct:.1f}%)")

        detected.sort(key=lambda x: x[2])
        for wid, word, std, avg in detected[:3]:
            print(f"    '{word}' std={std:.4f}")

        p2_total += len(detected)
        p2_actions.append((island_id, detected, seed_town_map, capital_ids))

    print(f"\n  Phase 2 total: {p2_total:,} island-level words")
    print(f"  Phase 2 threshold: cosine_std < {args.phase2_threshold}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Phase 1 — capital generic:   {p1_total_generic:>5,d} -> IslandWords")
    print(f"  Phase 1 — capital misplaced: {p1_total_misplaced:>5,d} -> move to NC town")
    print(f"  Phase 2 — cross-town generic:{p2_total:>5,d} -> IslandWords")

    if not args.apply:
        print("\nDry run — no changes made. Use --apply to modify the database.")
        con.close()
        return

    # ── Apply Phase 1 ─────────────────────────────────────────────────────
    print("\nApplying Phase 1...")
    cur = con.cursor()
    cur.execute("BEGIN TRANSACTION")

    n_p1_island = 0
    n_p1_moved = 0
    n_p1_removed = 0

    for island_id, generic, misplaced in p1_actions:
        cap_tid = con.execute(
            "SELECT town_id FROM Towns WHERE island_id = ? AND is_capital = 1",
            (island_id,)
        ).fetchone()[0]

        # Generic → IslandWords, remove from capital
        for wid, word, std, avg in generic:
            try:
                cur.execute("""
                    INSERT INTO IslandWords (island_id, word_id, word, source,
                                             cosine_std, avg_cosine)
                    VALUES (?, ?, ?, 'capital_triage_generic', ?, ?)
                """, (island_id, wid, word, std, avg))
                n_p1_island += 1
            except Exception:
                pass  # duplicate

            cur.execute(
                "DELETE FROM SeedWords WHERE town_id = ? AND word_id = ?",
                (cap_tid, wid)
            )
            n_p1_removed += 1

        # Misplaced → move to destination town
        for wid, word, std, dest_tid, dest_name, best_cos, isl_cos in misplaced:
            # Remove from capital
            cur.execute(
                "DELETE FROM SeedWords WHERE town_id = ? AND word_id = ?",
                (cap_tid, wid)
            )

            # Insert into destination (skip if already there)
            existing = cur.execute(
                "SELECT 1 FROM SeedWords WHERE town_id = ? AND word = ?",
                (dest_tid, word)
            ).fetchone()
            if not existing:
                # Get source/confidence from the capital entry we just deleted
                cur.execute("""
                    INSERT INTO SeedWords (town_id, word, word_id, source, confidence)
                    VALUES (?, ?, ?, 'capital_triage', 'core')
                """, (dest_tid, word, wid))
                n_p1_moved += 1

    cur.execute("COMMIT")
    print(f"  Phase 1: {n_p1_island:,} -> IslandWords, "
          f"{n_p1_moved:,} moved to NC towns, "
          f"{n_p1_removed:,} removed from capitals")

    # ── Apply Phase 2 (re-compute after phase 1 mutations) ───────────────
    print("\nApplying Phase 2 (re-computing with updated seeds)...")
    cur.execute("BEGIN TRANSACTION")

    n_p2_island = 0
    n_p2_removed = 0

    for island_id, island_name in islands:
        data = load_island_seeds(con, island_id)
        towns, town_seed_wids, seed_words, seed_town_map, capital_ids = data

        if len(towns) < MIN_TOWNS_FOR_DETECTION:
            continue

        detected = phase2_crosstown_detection(
            island_id, island_name, towns, town_seed_wids,
            seed_words, seed_town_map, capital_ids,
            emb_map, args.phase2_threshold
        )

        for wid, word, std, avg in detected:
            try:
                cur.execute("""
                    INSERT INTO IslandWords (island_id, word_id, word, source,
                                             cosine_std, avg_cosine)
                    VALUES (?, ?, ?, 'seed_cosine_std', ?, ?)
                """, (island_id, wid, word, std, avg))
                n_p2_island += 1
            except Exception:
                pass  # duplicate (may already exist from phase 1)

            for tid in seed_town_map[wid]:
                cur.execute(
                    "DELETE FROM SeedWords WHERE town_id = ? AND word_id = ?",
                    (tid, wid)
                )
                n_p2_removed += 1

    cur.execute("COMMIT")
    print(f"  Phase 2: {n_p2_island:,} -> IslandWords, "
          f"{n_p2_removed:,} removed from SeedWords")

    # ── Verify ────────────────────────────────────────────────────────────
    iw_count = con.execute("SELECT COUNT(*) FROM IslandWords").fetchone()[0]
    sw_count = con.execute("SELECT COUNT(*) FROM SeedWords").fetchone()[0]
    print(f"\n  IslandWords total: {iw_count:,}")
    print(f"  SeedWords remaining: {sw_count:,}")

    con.close()


if __name__ == "__main__":
    main()
