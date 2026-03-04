"""Populate export tables: ReefWordExports, TownWordExports, IslandWordExports.

Applies the "A2 + singleton rescue" promotion rules to classify each word
into exactly one export level, then computes weights with per-group min-max
normalization to u8 [0, 255], followed by island-exclusivity scaling.

Weight formula (reef/town levels):
  raw_score = global_idf * source_quality * effective_sim
  effective_sim = (1 - a1 - a2) * centroid_sim + a1 * group_name_cos + a2 * island_name_cos
  a1 = 0.2 (GROUP_NAME_COS_ALPHA), a2 = 0.3 (ISLAND_NAME_COS_ALPHA)

Weight formula (island level):
  raw_score = island_idf * source_quality * effective_sim
  effective_sim = (1 - a) * centroid_sim + a * island_name_cos
  a = 0.5 (ISLAND_ONLY_ALPHA)

  Reef/town: per-group min-max normalization to [0, 255].
  Island: hybrid normalization (80% global + 20% per-island min-max) to [0, 255],
  then scaled by exclusivity_factor.

  - global_idf:    log2(total_reefs / word_reef_count) — reef-level rarity
  - island_idf:    log2(N_islands / n_islands_word) — island-level rarity
  - centroid_sim:   cosine to reef/town/island centroid embedding
  - group_name_cos: cosine to reef/town name embedding (local topic)
  - island_name_cos: cosine to island name embedding (broad domain)
  - source_quality: 1.0 (curated), 0.9 (xgboost core), 0.7 (xgboost non-core)

Promotion rules (A2 + singleton rescue):
  spec >= 0, 1 town in island         → reef   (specific, single-town)
  spec >= 1, 2+ towns                 → town   (specific, multi-town)
  spec == 0, 2+ towns                 → island (moderate, multi-town)
  spec == -1, 1 reef in island        → reef   (singleton rescue)
  spec == -1, 2+ reefs in island      → island (generic + spread)
  spec <= -2                          → island (very generic)

Pipeline position: step 18, runs AFTER compute_hierarchy_stats.py.
Fully idempotent — clears and repopulates all three export tables.

Usage:
    python v3/populate_exports.py
    python v3/populate_exports.py --dry-run
"""

import os
import struct
import sys
import time
from collections import defaultdict

import numpy as np

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project)

V3_DB = os.path.join(_project, "v3/windowsill.db")
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_PREFIX = "clustering: "
EMBEDDING_DIM = 768
GROUP_NAME_COS_ALPHA = 0.2   # weight of reef/town name cosine
ISLAND_NAME_COS_ALPHA = 0.3  # weight of island name cosine (for reef/town exports)
ISLAND_ONLY_ALPHA = 0.5      # weight of island name cosine (for island-level exports)
SOURCE_QUALITY_FLOOR = 0.9   # floor for source_quality (softens xgboost non-core penalty)
EMBED_BATCH_SIZE = 256


def unpack_embedding(blob):
    return np.array(struct.unpack(f"{EMBEDDING_DIM}f", blob), dtype=np.float32)


def cosine_sim(a, b):
    """Cosine similarity between two vectors. Returns 0.0 if either is zero-norm."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def minmax_normalize(scores):
    """Per-group min-max normalize to u8 [0, 255]."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi > lo:
        return [round(255 * (s - lo) / (hi - lo)) for s in scores]
    else:
        return [128] * len(scores)


def compute_island_idf(total_islands, n_islands):
    """Island-level IDF: log2(N_islands / n_islands_for_word).

    Much better differentiation than reef-level IDF (which compresses to 8-11
    for nearly all words).  Island IDF ranges from ~2.6 (7 islands) to ~5.5
    (1 island) — a 2x ratio vs the 1.3x ratio of reef IDF.
    """
    import math
    return math.log2(total_islands / max(1, n_islands))


def exclusivity_factor(total_islands, n_islands):
    """Post-normalization scaling: gently penalises words in many islands.

    Uses cube-root dampening to avoid over-penalising moderately spread
    words.  Values:  1 island → 1.0;  2 → 0.79;  4 → 0.63;  7 → 0.52.
    """
    return 1.0 / (max(1, n_islands) ** 0.33)


def main():
    import argparse
    import sqlite3

    parser = argparse.ArgumentParser(description="Populate export tables (step 18)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify words and show distribution, don't write")
    args = parser.parse_args()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    # ===================================================================
    # Step 1: Embed hierarchy names
    # ===================================================================
    print("Step 1: Embedding hierarchy names...")
    t0 = time.time()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    # Gather names
    reefs = con.execute("""
        SELECT r.reef_id, r.name, t.name
        FROM Reefs r JOIN Towns t USING (town_id)
    """).fetchall()
    towns = con.execute("SELECT town_id, name FROM Towns").fetchall()
    islands = con.execute("SELECT island_id, name FROM Islands").fetchall()

    # Embed reef names (fall back to town name if reef name is NULL)
    reef_texts = [f"{EMBEDDING_PREFIX}{name if name else town_name}"
                  for _, name, town_name in reefs]
    reef_embs_arr = model.encode(reef_texts, show_progress_bar=False,
                                 batch_size=EMBED_BATCH_SIZE)
    reef_name_embs = {reefs[i][0]: reef_embs_arr[i] for i in range(len(reefs))}

    # Embed town names
    town_texts = [f"{EMBEDDING_PREFIX}{name}" for _, name in towns]
    town_embs_arr = model.encode(town_texts, show_progress_bar=False,
                                 batch_size=EMBED_BATCH_SIZE)
    town_name_embs = {towns[i][0]: town_embs_arr[i] for i in range(len(towns))}

    # Embed island names
    island_texts = [f"{EMBEDDING_PREFIX}{name}" for _, name in islands]
    island_embs_arr = model.encode(island_texts, show_progress_bar=False,
                                   batch_size=EMBED_BATCH_SIZE)
    island_name_embs = {islands[i][0]: island_embs_arr[i] for i in range(len(islands))}

    print(f"  Embedded {len(reef_name_embs)} reefs, {len(town_name_embs)} towns, "
          f"{len(island_name_embs)} islands ({time.time()-t0:.1f}s)")

    # Build hierarchy lookups: reef→island, town→island
    town_to_island = {}
    for town_id, island_id in con.execute("SELECT town_id, island_id FROM Towns").fetchall():
        town_to_island[town_id] = island_id

    reef_to_town = {}
    for reef_id, town_id in con.execute("SELECT reef_id, town_id FROM Reefs").fetchall():
        reef_to_town[reef_id] = town_id

    # ===================================================================
    # Step 2: Load word embeddings
    # ===================================================================
    print("Step 2: Loading word embeddings...")
    t0 = time.time()

    word_embs = {}
    for word_id, blob in con.execute(
        "SELECT word_id, embedding FROM Words WHERE embedding IS NOT NULL"
    ).fetchall():
        word_embs[word_id] = unpack_embedding(blob)

    print(f"  Loaded {len(word_embs):,} word embeddings ({time.time()-t0:.1f}s)")

    # ===================================================================
    # Step 3: Classify export level
    # ===================================================================
    print("Step 3: Classifying export level per word...")
    t0 = time.time()

    # Global word stats
    word_stats = {}
    for row in con.execute("""
        SELECT rw.word_id, w.specificity,
            COUNT(DISTINCT t.island_id) AS n_islands,
            COUNT(DISTINCT r.town_id) AS n_towns,
            COUNT(DISTINCT rw.reef_id) AS n_reefs
        FROM ReefWords rw
        JOIN Reefs r USING (reef_id)
        JOIN Towns t USING (town_id)
        JOIN Words w USING (word_id)
        GROUP BY rw.word_id
    """).fetchall():
        word_stats[row[0]] = {
            "specificity": row[1],
            "n_islands": row[2],
            "n_towns": row[3],
            "n_reefs": row[4],
        }

    # Per-(word, island) reef counts for singleton rescue
    word_island_reefs = defaultdict(dict)  # {word_id: {island_id: reef_count}}
    for word_id, island_id, reefs_in_island in con.execute("""
        SELECT rw.word_id, t.island_id,
            COUNT(DISTINCT rw.reef_id) AS reefs_in_island
        FROM ReefWords rw
        JOIN Reefs r USING (reef_id)
        JOIN Towns t USING (town_id)
        GROUP BY rw.word_id, t.island_id
    """).fetchall():
        word_island_reefs[word_id][island_id] = reefs_in_island

    # Classify
    word_level = {}  # word_id -> 'reef' | 'town' | 'island'
    for word_id, stats in word_stats.items():
        spec = stats["specificity"]
        n_towns = stats["n_towns"]

        if spec is None:
            # No specificity (word not in any reef shouldn't reach here, but guard)
            word_level[word_id] = "island"
        elif spec >= 1 and n_towns >= 2:
            word_level[word_id] = "town"
        elif spec >= 0 and n_towns == 1:
            word_level[word_id] = "reef"
        elif spec == 0 and n_towns >= 2:
            word_level[word_id] = "island"
        elif spec == -1:
            # Singleton rescue: check if any island has only 1 reef for this word
            island_reef_counts = word_island_reefs.get(word_id, {})
            max_reefs_in_any_island = max(island_reef_counts.values()) if island_reef_counts else 0
            if max_reefs_in_any_island == 1:
                word_level[word_id] = "reef"
            else:
                word_level[word_id] = "island"
        else:  # spec <= -2
            word_level[word_id] = "island"

    # Count distribution
    level_counts = defaultdict(int)
    for lvl in word_level.values():
        level_counts[lvl] += 1

    print(f"  Classification: reef={level_counts['reef']:,}, "
          f"town={level_counts['town']:,}, island={level_counts['island']:,} "
          f"({time.time()-t0:.1f}s)")

    if args.dry_run:
        # Show more detail
        print("\n--- Specificity × level breakdown ---")
        spec_level = defaultdict(lambda: defaultdict(int))
        for word_id, lvl in word_level.items():
            spec = word_stats[word_id]["specificity"]
            spec_level[spec][lvl] += 1
        for spec in sorted(spec_level.keys(), reverse=True):
            parts = ", ".join(f"{lvl}={n:,}" for lvl, n in sorted(spec_level[spec].items()))
            print(f"  spec={spec:>3}: {parts}")

        con.close()
        print("\nDry run — no changes made.")
        return

    # ===================================================================
    # Step 4: Populate ReefWordExports
    # ===================================================================
    print("Step 4: Populating ReefWordExports...")
    t0 = time.time()

    con.execute("DELETE FROM ReefWordExports")

    total_islands = len(islands)

    # Pre-load global IDF for all words
    global_idf = {}
    for wid, idf in con.execute("SELECT word_id, idf FROM Words WHERE idf IS NOT NULL").fetchall():
        global_idf[wid] = idf

    # Fetch all reef-level word data
    reef_word_rows = con.execute("""
        SELECT rw.reef_id, rw.word_id, rw.cosine_sim,
               rw.source_quality, rw.specificity, r.town_id
        FROM ReefWords rw
        JOIN Reefs r USING (reef_id)
        WHERE rw.cosine_sim IS NOT NULL
    """).fetchall()

    # Group by reef, compute scores, normalize
    # effective_sim = (1 - a1 - a2) * centroid_sim + a1 * reef_name_cos + a2 * island_name_cos
    a1 = GROUP_NAME_COS_ALPHA
    a2 = ISLAND_NAME_COS_ALPHA
    reef_exports = defaultdict(list)  # reef_id -> [(word_id, idf_val, csim, ncos, esim, spec, sq, raw)]
    for reef_id, word_id, centroid_sim, source_quality, spec, town_id in reef_word_rows:
        if word_level.get(word_id) != "reef":
            continue
        if word_id not in word_embs:
            continue

        island_id = town_to_island.get(town_id)
        reef_ncos = cosine_sim(word_embs[word_id], reef_name_embs.get(reef_id, np.zeros(EMBEDDING_DIM)))
        island_ncos = cosine_sim(word_embs[word_id], island_name_embs.get(island_id, np.zeros(EMBEDDING_DIM)))
        effective_sim = (1 - a1 - a2) * centroid_sim + a1 * reef_ncos + a2 * island_ncos
        gidf = global_idf.get(word_id, 0.0)
        sq = max(source_quality if source_quality is not None else 1.0, SOURCE_QUALITY_FLOOR)
        raw_score = gidf * sq * effective_sim

        # Store island_ncos as name_cos column (most useful for diagnostics)
        reef_exports[reef_id].append(
            (word_id, gidf, centroid_sim, island_ncos, effective_sim, spec, sq, raw_score)
        )

    # Per-reef min-max normalization (no exclusivity scaling for reef-level)
    insert_rows = []
    for reef_id, entries in reef_exports.items():
        raw_scores = [e[7] for e in entries]
        weights = minmax_normalize(raw_scores)
        for (word_id, idf, csim, ncos, esim, spec, sq, _raw), w in zip(entries, weights):
            insert_rows.append((reef_id, word_id, idf, csim, ncos, esim, spec, sq, w))

    con.executemany("""
        INSERT INTO ReefWordExports
            (reef_id, word_id, idf, centroid_sim, name_cos, effective_sim,
             specificity, source_quality, export_weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, insert_rows)
    con.commit()

    print(f"  Inserted {len(insert_rows):,} reef export rows ({time.time()-t0:.1f}s)")

    # ===================================================================
    # Step 5: Populate TownWordExports
    # ===================================================================
    print("Step 5: Populating TownWordExports...")
    t0 = time.time()

    con.execute("DELETE FROM TownWordExports")

    # For town-level words, aggregate across reefs within each town
    town_word_rows = con.execute("""
        SELECT r.town_id, rw.word_id, rw.cosine_sim,
               rw.source_quality, rw.specificity
        FROM ReefWords rw
        JOIN Reefs r USING (reef_id)
        WHERE rw.cosine_sim IS NOT NULL
    """).fetchall()

    # Group by (town_id, word_id), take MAX of cosine_sim and source_quality
    town_word_agg = defaultdict(lambda: {"max_csim": -1, "max_sq": 0.0, "spec": None})
    for town_id, word_id, csim, sq, spec in town_word_rows:
        if word_level.get(word_id) != "town":
            continue
        key = (town_id, word_id)
        agg = town_word_agg[key]
        if csim is not None and csim > agg["max_csim"]:
            agg["max_csim"] = csim
        if sq is not None and sq > agg["max_sq"]:
            agg["max_sq"] = sq
        if spec is not None:
            agg["spec"] = spec

    # Compute scores, group by town for normalization
    # Three-signal blend: town_name_cos (a1) + island_name_cos (a2) + centroid (remainder)
    town_exports = defaultdict(list)  # town_id -> [(word_id, idf_val, csim, ncos, esim, spec, sq, raw)]
    for (town_id, word_id), agg in town_word_agg.items():
        if word_id not in word_embs:
            continue
        centroid_sim = agg["max_csim"]
        island_id = town_to_island.get(town_id)
        town_ncos = cosine_sim(word_embs[word_id], town_name_embs.get(town_id, np.zeros(EMBEDDING_DIM)))
        island_ncos = cosine_sim(word_embs[word_id], island_name_embs.get(island_id, np.zeros(EMBEDDING_DIM)))
        effective_sim = (1 - a1 - a2) * centroid_sim + a1 * town_ncos + a2 * island_ncos
        gidf = global_idf.get(word_id, 0.0)
        sq = max(agg["max_sq"], SOURCE_QUALITY_FLOOR)
        raw_score = gidf * sq * effective_sim

        # Store island_ncos as name_cos column (most useful for diagnostics)
        town_exports[town_id].append(
            (word_id, gidf, centroid_sim, island_ncos, effective_sim, agg["spec"], sq, raw_score)
        )

    # Per-town min-max normalization (no exclusivity scaling for town-level)
    insert_rows = []
    for town_id, entries in town_exports.items():
        raw_scores = [e[7] for e in entries]
        weights = minmax_normalize(raw_scores)
        for (word_id, idf, csim, ncos, esim, spec, sq, _raw), w in zip(entries, weights):
            insert_rows.append((town_id, word_id, idf, csim, ncos, esim, spec, sq, w))

    con.executemany("""
        INSERT INTO TownWordExports
            (town_id, word_id, idf, centroid_sim, name_cos, effective_sim,
             specificity, source_quality, export_weight, export_town_weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 128)
    """, insert_rows)
    con.commit()

    print(f"  Inserted {len(insert_rows):,} town export rows ({time.time()-t0:.1f}s)")

    # ===================================================================
    # Step 6: Populate IslandWordExports
    # ===================================================================
    print("Step 6: Populating IslandWordExports...")
    t0 = time.time()

    con.execute("DELETE FROM IslandWordExports")

    # For island-level words, aggregate across all reefs within each island
    island_word_rows = con.execute("""
        SELECT t.island_id, rw.word_id, rw.cosine_sim,
               rw.source_quality, rw.specificity
        FROM ReefWords rw
        JOIN Reefs r USING (reef_id)
        JOIN Towns t USING (town_id)
        WHERE rw.cosine_sim IS NOT NULL
    """).fetchall()

    # Group by (island_id, word_id), take MAX of cosine_sim and source_quality
    island_word_agg = defaultdict(lambda: {"max_csim": -1, "max_sq": 0.0, "spec": None})
    for island_id, word_id, csim, sq, spec in island_word_rows:
        if word_level.get(word_id) != "island":
            continue
        key = (island_id, word_id)
        agg = island_word_agg[key]
        if csim is not None and csim > agg["max_csim"]:
            agg["max_csim"] = csim
        if sq is not None and sq > agg["max_sq"]:
            agg["max_sq"] = sq
        if spec is not None:
            agg["spec"] = spec

    # Compute scores, group by island for normalization
    # Island-level uses island_idf (better differentiation) + post-normalization exclusivity
    a_island = ISLAND_ONLY_ALPHA
    island_exports = defaultdict(list)  # island_id -> [(word_id, iidf, csim, ncos, esim, spec, sq, raw, n_isl)]
    for (island_id, word_id), agg in island_word_agg.items():
        if word_id not in word_embs:
            continue
        n_isl = word_stats.get(word_id, {}).get("n_islands", 1)
        iidf = compute_island_idf(total_islands, n_isl)
        centroid_sim = agg["max_csim"]
        name_cos = cosine_sim(word_embs[word_id], island_name_embs.get(island_id, np.zeros(EMBEDDING_DIM)))
        effective_sim = (1 - a_island) * centroid_sim + a_island * name_cos
        sq = max(agg["max_sq"], SOURCE_QUALITY_FLOOR)
        raw_score = iidf * sq * effective_sim

        island_exports[island_id].append(
            (word_id, iidf, centroid_sim, name_cos, effective_sim, agg["spec"], sq, raw_score, n_isl)
        )

    # Hybrid normalization: blend per-island min-max with global min-max.
    # Per-island preserves within-island differentiation (every island uses full
    # [0,255] range).  Global preserves cross-island comparability (a word with
    # higher raw_score in Biology keeps a higher weight than in Computer Science).
    # The blend balances both needs.
    ISLAND_GLOBAL_BLEND = 0.8  # 0.0 = pure per-island, 1.0 = pure global

    # Collect all entries with their raw scores
    all_entries = []  # [(island_id, word_id, idf, csim, ncos, esim, spec, sq, raw, n_isl)]
    for island_id, entries in island_exports.items():
        for e in entries:
            all_entries.append((island_id, *e))

    # Global range
    all_raw = [e[8] for e in all_entries]
    g_lo = min(all_raw) if all_raw else 0
    g_hi = max(all_raw) if all_raw else 0
    g_rng = g_hi - g_lo if g_hi > g_lo else 1.0

    # Per-island ranges
    island_ranges = {}
    for island_id, entries in island_exports.items():
        raws = [e[7] for e in entries]
        lo = min(raws) if raws else 0
        hi = max(raws) if raws else 0
        island_ranges[island_id] = (lo, hi)

    insert_rows = []
    blend = ISLAND_GLOBAL_BLEND
    for island_id, word_id, idf, csim, ncos, esim, spec, sq, raw, n_isl in all_entries:
        # Per-island normalized weight
        lo, hi = island_ranges[island_id]
        rng = hi - lo if hi > lo else 1.0
        w_local = 255 * (raw - lo) / rng

        # Global normalized weight
        w_global = 255 * (raw - g_lo) / g_rng

        # Blend
        w = round((1 - blend) * w_local + blend * w_global)
        w = round(w * exclusivity_factor(total_islands, n_isl))
        insert_rows.append((island_id, word_id, idf, csim, ncos, esim, spec, sq, w))

    con.executemany("""
        INSERT INTO IslandWordExports
            (island_id, word_id, idf, centroid_sim, name_cos, effective_sim,
             specificity, source_quality, export_weight, export_island_weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 128)
    """, insert_rows)
    con.commit()

    print(f"  Inserted {len(insert_rows):,} island export rows ({time.time()-t0:.1f}s)")

    # ===================================================================
    # Step 7: Summary report
    # ===================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Counts per level
    n_reef = con.execute("SELECT COUNT(*) FROM ReefWordExports").fetchone()[0]
    n_town = con.execute("SELECT COUNT(*) FROM TownWordExports").fetchone()[0]
    n_island = con.execute("SELECT COUNT(*) FROM IslandWordExports").fetchone()[0]
    print(f"\n--- Export row counts ---")
    print(f"  ReefWordExports:   {n_reef:>8,}")
    print(f"  TownWordExports:   {n_town:>8,}")
    print(f"  IslandWordExports: {n_island:>8,}")
    print(f"  Total:             {n_reef + n_town + n_island:>8,}")

    # Unique word counts per level
    n_reef_w = con.execute("SELECT COUNT(DISTINCT word_id) FROM ReefWordExports").fetchone()[0]
    n_town_w = con.execute("SELECT COUNT(DISTINCT word_id) FROM TownWordExports").fetchone()[0]
    n_island_w = con.execute("SELECT COUNT(DISTINCT word_id) FROM IslandWordExports").fetchone()[0]
    print(f"\n--- Unique words per level ---")
    print(f"  Reef:   {n_reef_w:>8,}")
    print(f"  Town:   {n_town_w:>8,}")
    print(f"  Island: {n_island_w:>8,}")
    print(f"  Total:  {n_reef_w + n_town_w + n_island_w:>8,}")

    # Export weight distribution
    print(f"\n--- export_weight distribution ---")
    for level, table in [("reef", "ReefWordExports"), ("town", "TownWordExports"),
                         ("island", "IslandWordExports")]:
        row = con.execute(f"""
            SELECT MIN(export_weight), MAX(export_weight),
                   ROUND(AVG(export_weight), 1),
                   COUNT(CASE WHEN export_weight = 0 THEN 1 END),
                   COUNT(CASE WHEN export_weight = 255 THEN 1 END)
            FROM {table}
        """).fetchone()
        print(f"  {level:7s}: min={row[0]}, max={row[1]}, avg={row[2]}, "
              f"zeros={row[3]:,}, maxed={row[4]:,}")

    # Cross-level overlap check (should be 0)
    print(f"\n--- Cross-level overlap (should be 0) ---")
    overlap_rt = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT word_id FROM ReefWordExports
            INTERSECT SELECT word_id FROM TownWordExports
        )
    """).fetchone()[0]
    overlap_ti = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT word_id FROM TownWordExports
            INTERSECT SELECT word_id FROM IslandWordExports
        )
    """).fetchone()[0]
    overlap_ri = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT word_id FROM ReefWordExports
            INTERSECT SELECT word_id FROM IslandWordExports
        )
    """).fetchone()[0]
    print(f"  Reef ∩ Town:   {overlap_rt}")
    print(f"  Town ∩ Island: {overlap_ti}")
    print(f"  Reef ∩ Island: {overlap_ri}")
    if overlap_rt + overlap_ti + overlap_ri > 0:
        print("  WARNING: words appear at multiple export levels!")
    else:
        print("  OK — all words at exactly one level")

    # Per-archipelago coverage
    print(f"\n--- Per-archipelago coverage ---")
    for row in con.execute("""
        SELECT archipelago, COUNT(DISTINCT word_id) AS words
        FROM ExportIndex
        GROUP BY archipelago
        ORDER BY words DESC
    """).fetchall():
        print(f"  {row[0]:30s} {row[1]:>8,} words")

    # Sample lookups
    print(f"\n--- Sample word lookups ---")
    sample_words = ["violin", "neuron", "photosynthesis", "democracy", "algorithm"]
    for word in sample_words:
        results = con.execute("""
            SELECT export_level, island, town, reef, export_weight
            FROM WordSearch WHERE word = ?
            ORDER BY export_weight DESC LIMIT 5
        """, (word,)).fetchall()
        if results:
            print(f"  {word}:")
            for r in results:
                loc = r[3] if r[3] else (r[2] if r[2] else r[1])
                print(f"    {r[0]:6s} {loc:30s} weight={r[4]}")
        else:
            print(f"  {word}: (not exported)")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
