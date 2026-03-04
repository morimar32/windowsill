"""
V3 Reef Clustering — Leiden community detection within towns.

For each town in a given island:
  1. Gather all words (seeds + XGBoost predictions)
  2. Split into core (seeds + high-confidence XGBoost) vs non-core
  3. Build hybrid similarity (embedding cosine + PMI from town co-membership)
  4. kNN graph → Leiden → reef communities
  5. Assign non-core words to nearest reef centroid
  6. Write Reefs + ReefWords tables

Usage:
    python v3/cluster_reefs.py --island "Sport"
    python v3/cluster_reefs.py --island "Sport" --dry-run
"""

import argparse
import os
import struct
import sys
import time
from collections import Counter, defaultdict

import numpy as np

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project)

from lib.reef import (
    compute_global_pmi,
    build_pmi_word_vectors,
    compute_hybrid_similarity,
    build_knn_graph,
    leiden_cluster,
    compute_reef_centroids,
    assign_non_core,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
V3_DB = os.path.join(_project, "v3/windowsill.db")

CORE_SCORE_THRESHOLD = 0.6     # XGBoost score threshold for "core"
ALPHA = 0.7                    # blend: 0.7 * emb_cos + 0.3 * pmi_cos
KNN_K = 15
LEIDEN_RESOLUTION = 1.0
MIN_COMMUNITY_SIZE = 3
MIN_TOWN_SIZE = 10             # skip towns with fewer total words
CHARACTERISTIC_WORDS_N = 10


def unpack_embedding(blob):
    return np.array(struct.unpack(f"{768}f", blob), dtype=np.float64)


def pack_embedding(arr):
    return struct.pack(f"{768}f", *arr.astype(np.float32))


def load_town_words(v3_con, town_id):
    """Load all words for a town: seeds + xgboost predictions.

    Returns:
        core_wids: list of word_ids (seeds + high-confidence xgboost)
        non_core_wids: list of word_ids (low-confidence xgboost)
        all_wids: list of all word_ids
    """
    # Seeds are always core (exclude stop words)
    seed_wids = set()
    rows = v3_con.execute("""
        SELECT s.word_id FROM SeedWords s
        JOIN Words w USING (word_id)
        WHERE s.town_id = ? AND s.word_id IS NOT NULL AND w.is_stop = 0
    """, (town_id,)).fetchall()
    seed_wids = {r[0] for r in rows}

    # XGBoost predictions: core if score >= threshold, else non-core (exclude stop words)
    xgb_core = set()
    xgb_non_core = set()
    rows = v3_con.execute("""
        SELECT a.word_id, a.score FROM AugmentedTowns a
        JOIN Words w USING (word_id)
        WHERE a.town_id = ? AND w.is_stop = 0
    """, (town_id,)).fetchall()
    for wid, score in rows:
        if wid in seed_wids:
            continue  # already in seeds
        if score >= CORE_SCORE_THRESHOLD:
            xgb_core.add(wid)
        else:
            xgb_non_core.add(wid)

    core_wids = sorted(seed_wids | xgb_core)
    non_core_wids = sorted(xgb_non_core)

    return core_wids, non_core_wids


def build_word_town_membership(v3_con, island_name):
    """Build {word_id: set of town_names} for PMI computation.

    Considers all towns in the island.
    """
    word_towns = defaultdict(set)

    # From seeds (exclude stop words)
    rows = v3_con.execute("""
        SELECT s.word_id, t.name
        FROM SeedWords s
        JOIN Towns t USING (town_id)
        JOIN Islands i USING (island_id)
        JOIN Words w ON w.word_id = s.word_id
        WHERE i.name = ? AND s.word_id IS NOT NULL AND w.is_stop = 0
    """, (island_name,)).fetchall()
    for wid, town in rows:
        word_towns[wid].add(town)

    # From xgboost predictions (exclude stop words)
    rows = v3_con.execute("""
        SELECT a.word_id, t.name
        FROM AugmentedTowns a
        JOIN Towns t USING (town_id)
        JOIN Islands i USING (island_id)
        JOIN Words w ON w.word_id = a.word_id
        WHERE i.name = ? AND w.is_stop = 0
    """, (island_name,)).fetchall()
    for wid, town in rows:
        word_towns[wid].add(town)

    return dict(word_towns)


def main():
    import sqlite3

    parser = argparse.ArgumentParser(description="V3 Reef Clustering")
    parser.add_argument("--island", required=True, help="Island name to cluster")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    v3_con = sqlite3.connect(V3_DB)
    v3_con.execute("PRAGMA foreign_keys = ON")

    # Load island towns
    towns = v3_con.execute("""
        SELECT t.town_id, t.name
        FROM Towns t JOIN Islands i USING (island_id)
        WHERE i.name = ?
        ORDER BY t.town_id
    """, (args.island,)).fetchall()

    if not towns:
        print(f"No towns found for island '{args.island}'")
        return

    print(f"Island: {args.island}")
    print(f"Towns: {len(towns)}")

    # Load all embeddings
    print("Loading embeddings...")
    t0 = time.time()
    emb_map = {}
    rows = v3_con.execute(
        "SELECT word_id, embedding FROM Words WHERE embedding IS NOT NULL"
    ).fetchall()
    for wid, blob in rows:
        emb_map[wid] = unpack_embedding(blob)
    print(f"  {len(emb_map):,} embeddings ({time.time()-t0:.1f}s)")

    # Load word texts for labeling
    word_texts = {}
    for wid, word in v3_con.execute("SELECT word_id, word FROM Words").fetchall():
        word_texts[wid] = word

    # Load curated word_ids (wordnet + claude_augmented seeds) for label priority
    curated_wids = set()
    for r in v3_con.execute("SELECT word_id FROM SeedWords WHERE word_id IS NOT NULL").fetchall():
        curated_wids.add(r[0])

    # Build PMI from town co-membership
    print("Computing PMI...")
    word_towns = build_word_town_membership(v3_con, args.island)
    town_names = [t[1] for t in towns]
    pmi_matrix, town_index = compute_global_pmi(word_towns, town_names)
    print(f"  PMI matrix: {pmi_matrix.shape}, {(pmi_matrix > 0).sum()} positive entries")

    if args.dry_run:
        for town_id, town_name in towns:
            core, non_core = load_town_words(v3_con, town_id)
            total = len(core) + len(non_core)
            print(f"  {town_name}: {len(core)} core + {len(non_core)} non-core = {total} total")
        return

    # Process each town
    total_reefs = 0
    total_words_assigned = 0

    for town_id, town_name in towns:
        core_wids, non_core_wids = load_town_words(v3_con, town_id)
        total = len(core_wids) + len(non_core_wids)

        if total < MIN_TOWN_SIZE:
            print(f"  {town_name}: SKIP — only {total} words")
            continue

        # Filter to words with embeddings
        core_wids = [w for w in core_wids if w in emb_map]
        non_core_wids = [w for w in non_core_wids if w in emb_map]

        if len(core_wids) < 3:
            print(f"  {town_name}: SKIP — only {len(core_wids)} core words with embeddings")
            continue

        t0 = time.time()

        # Build embedding matrices
        core_embeddings = np.vstack([emb_map[w] for w in core_wids])
        non_core_embeddings = np.vstack([emb_map[w] for w in non_core_wids]) if non_core_wids else np.zeros((0, 768))

        # Build PMI vectors for core words
        core_word_towns = {wid: word_towns.get(wid, set()) for wid in core_wids}
        pmi_vectors = build_pmi_word_vectors(
            core_wids, core_word_towns, town_name,
            pmi_matrix, town_index
        )

        # Hybrid similarity
        sim_matrix = compute_hybrid_similarity(core_embeddings, pmi_vectors, ALPHA)

        # kNN graph
        graph = build_knn_graph(sim_matrix, KNN_K)

        # Leiden clustering
        core_labels, modularity = leiden_cluster(
            graph, LEIDEN_RESOLUTION, MIN_COMMUNITY_SIZE
        )

        # Centroids
        centroids = compute_reef_centroids(core_embeddings, core_labels)

        # Core cosine similarities
        core_norms = np.linalg.norm(core_embeddings, axis=1, keepdims=True)
        core_norms[core_norms == 0] = 1.0
        core_normed = core_embeddings / core_norms
        core_sims = np.zeros(len(core_wids))
        for i, label in enumerate(core_labels):
            if label >= 0 and label in centroids:
                core_sims[i] = float(core_normed[i] @ centroids[label])

        # Assign non-core
        non_core_labels, non_core_sims = assign_non_core(centroids, non_core_embeddings)

        n_reefs = len(centroids)
        n_noise_core = int((core_labels == -1).sum())
        n_noise_nc = int((non_core_labels == -1).sum())

        # Build reef stats for display
        # Priority: curated words (wordnet + claude_augmented) first, then xgboost
        reef_top_words = {}
        for rid in sorted(centroids.keys()):
            curated_entries = []
            xgboost_entries = []
            for i, wid in enumerate(core_wids):
                if core_labels[i] == rid:
                    bucket = curated_entries if wid in curated_wids else xgboost_entries
                    bucket.append((wid, core_sims[i]))
            for i, wid in enumerate(non_core_wids):
                if non_core_labels[i] == rid:
                    xgboost_entries.append((wid, non_core_sims[i]))
            curated_entries.sort(key=lambda x: x[1], reverse=True)
            xgboost_entries.sort(key=lambda x: x[1], reverse=True)
            # Curated words first, fill remaining slots with xgboost if needed
            combined = curated_entries + xgboost_entries
            reef_top_words[rid] = [word_texts.get(wid, str(wid)) for wid, _ in combined[:CHARACTERISTIC_WORDS_N]]

        # Persist to database
        cur = v3_con.cursor()
        cur.execute("BEGIN TRANSACTION")

        # Delete existing reefs for this town
        cur.execute("""
            DELETE FROM ReefWords WHERE reef_id IN
            (SELECT reef_id FROM Reefs WHERE town_id = ?)
        """, (town_id,))
        cur.execute("DELETE FROM Reefs WHERE town_id = ?", (town_id,))

        # Insert reefs
        reef_id_map = {}  # local_id → db reef_id
        for local_rid in sorted(centroids.keys()):
            label = "_".join(reef_top_words[local_rid][:3])
            centroid_blob = pack_embedding(centroids[local_rid])

            core_mask = core_labels == local_rid
            nc_mask = non_core_labels == local_rid
            n_core_in_reef = int(core_mask.sum())
            n_total_in_reef = n_core_in_reef + int(nc_mask.sum())

            cur.execute("""
                INSERT INTO Reefs (town_id, name, centroid, word_count, core_word_count)
                VALUES (?, ?, ?, ?, ?)
            """, (town_id, label, centroid_blob, n_total_in_reef, n_core_in_reef))
            reef_id_map[local_rid] = cur.lastrowid

        # Insert ReefWords — core (distinguish curated vs xgboost-promoted)
        for i, wid in enumerate(core_wids):
            local_rid = int(core_labels[i])
            if local_rid < 0:
                continue  # noise
            db_rid = reef_id_map[local_rid]
            source = "curated" if wid in curated_wids else "xgboost"
            cur.execute("""
                INSERT INTO ReefWords (reef_id, word_id, cosine_sim, source, is_core)
                VALUES (?, ?, ?, ?, 1)
            """, (db_rid, wid, float(core_sims[i]), source))

        # Insert ReefWords — non-core (always xgboost)
        for i, wid in enumerate(non_core_wids):
            local_rid = int(non_core_labels[i])
            if local_rid < 0:
                continue
            db_rid = reef_id_map[local_rid]
            cur.execute("""
                INSERT INTO ReefWords (reef_id, word_id, cosine_sim, source, is_core)
                VALUES (?, ?, ?, 'xgboost', 0)
            """, (db_rid, wid, float(non_core_sims[i])))

        cur.execute("COMMIT")

        elapsed = time.time() - t0
        total_reefs += n_reefs
        n_assigned = len(core_wids) - n_noise_core + len(non_core_wids) - n_noise_nc
        total_words_assigned += n_assigned

        print(f"  {town_name}: {n_reefs} reefs, {n_assigned}/{total} words assigned, "
              f"mod={modularity:.3f}, noise={n_noise_core}+{n_noise_nc} ({elapsed:.1f}s)")

        for local_rid in sorted(centroids.keys()):
            top = reef_top_words[local_rid]
            core_count = int((core_labels == local_rid).sum())
            nc_count = int((non_core_labels == local_rid).sum())
            print(f"    reef {local_rid}: {core_count}+{nc_count} words — {', '.join(top[:5])}")

    print(f"\nDone. {total_reefs} reefs, {total_words_assigned:,} words assigned across {len(towns)} towns")

    # Update town stats
    v3_con.execute("""
        UPDATE Towns SET reef_count = (SELECT COUNT(*) FROM Reefs WHERE Reefs.town_id = Towns.town_id),
                         word_count = (SELECT COUNT(*) FROM ReefWords rw JOIN Reefs r USING(reef_id) WHERE r.town_id = Towns.town_id)
        WHERE town_id IN (SELECT town_id FROM Towns JOIN Islands USING(island_id) WHERE Islands.name = ?)
    """, (args.island,))
    v3_con.commit()

    v3_con.close()


if __name__ == "__main__":
    main()
