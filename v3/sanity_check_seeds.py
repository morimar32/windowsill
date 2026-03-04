"""
Seed Sanity Check — redistribute shared seeds to their most specific town.

For each word that appears as a seed in 2+ towns within the same island:
  - If town sizes differ by >2x: always remove from the larger town.
    (Catch-all towns have contaminated centroids that pull everything in.)
  - If town sizes are within 2x: use embedding cosine similarity to each
    town's centroid (from non-shared seeds) to pick the best home.

This fixes the systematic contamination where broad v2 catch-all domains
("sport", "religion", "biology", etc.) dumped all their words into a
single town, including words that belong in more specific sibling towns.

Usage:
    python v3/sanity_check_seeds.py              # preview (dry-run by default)
    python v3/sanity_check_seeds.py --apply      # actually modify the database
    python v3/sanity_check_seeds.py --island Sport  # single island
"""

import argparse
import os
import sqlite3
import struct
import sys
from collections import defaultdict

import numpy as np

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project)

V3_DB = os.path.join(_project, "v3/windowsill.db")


def unpack_embedding(blob):
    return np.array(struct.unpack(f"{768}f", blob), dtype=np.float64)


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def main():
    parser = argparse.ArgumentParser(description="Seed sanity check")
    parser.add_argument("--apply", action="store_true",
                        help="Actually modify the database (default: dry-run)")
    parser.add_argument("--island", type=str, default=None,
                        help="Process single island (default: all)")
    args = parser.parse_args()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    # Load all embeddings
    print("Loading embeddings...")
    emb_map = {}
    for word, blob in con.execute(
        "SELECT word, embedding FROM Words WHERE embedding IS NOT NULL"
    ).fetchall():
        emb_map[word] = unpack_embedding(blob)
    print(f"  {len(emb_map):,} word embeddings loaded")

    # Get islands to process
    if args.island:
        islands = con.execute(
            "SELECT island_id, name FROM Islands WHERE name = ?",
            (args.island,)
        ).fetchall()
    else:
        islands = con.execute(
            "SELECT island_id, name FROM Islands ORDER BY island_id"
        ).fetchall()

    total_removed = 0
    total_shared_words = 0
    island_summaries = []

    for island_id, island_name in islands:
        # Get all towns in this island
        towns = con.execute(
            "SELECT town_id, name FROM Towns WHERE island_id = ? ORDER BY town_id",
            (island_id,)
        ).fetchall()
        town_ids = {t[0] for t in towns}
        town_names = {t[0]: t[1] for t in towns}

        # Load seeds per town: {town_id: {word: rowid}}
        town_seeds = defaultdict(dict)
        for rowid, town_id, word in con.execute("""
            SELECT rowid, town_id, word FROM SeedWords
            WHERE town_id IN ({})
        """.format(",".join("?" * len(town_ids))), list(town_ids)).fetchall():
            town_seeds[town_id][word] = rowid

        # Find shared words: words in 2+ towns within this island
        word_towns = defaultdict(list)  # word -> [(town_id, rowid)]
        for tid in town_ids:
            for word, rowid in town_seeds[tid].items():
                word_towns[word].append((tid, rowid))

        shared_words = {w: towns for w, towns in word_towns.items() if len(towns) >= 2}
        if not shared_words:
            continue

        # Build exclusive seed sets (words NOT shared with any sibling)
        shared_word_set = set(shared_words.keys())
        exclusive_seeds = {}  # town_id -> set of exclusive words
        for tid in town_ids:
            exclusive_seeds[tid] = {w for w in town_seeds[tid] if w not in shared_word_set}

        # Compute town centroids from exclusive seeds only
        town_centroids = {}
        for tid in town_ids:
            exc_words = exclusive_seeds[tid]
            vecs = [emb_map[w] for w in exc_words if w in emb_map]
            if vecs:
                town_centroids[tid] = np.mean(vecs, axis=0)

        # Count total seeds per town (for size-based decisions)
        town_seed_counts = {tid: len(town_seeds[tid]) for tid in town_ids}

        # For each shared word, decide which town keeps it
        # Asymmetric pairs (>2x size): always remove from larger (catch-all protection)
        # Symmetric pairs (≤2x size): use cosine similarity to centroid
        SIZE_RATIO_THRESHOLD = 2.0
        removals = []  # (rowid, word, from_town_id, to_town_id, cos_winner, cos_loser)
        no_embedding = 0

        for word, claiming_towns in shared_words.items():
            if len(claiming_towns) != 2:
                # 3+ towns: shouldn't happen but handle gracefully
                no_embedding += 1
                continue

            (tid1, rowid1), (tid2, rowid2) = claiming_towns
            n1 = town_seed_counts.get(tid1, 0)
            n2 = town_seed_counts.get(tid2, 0)

            # Ensure tid1 is the larger town
            if n2 > n1:
                tid1, rowid1, n1, tid2, rowid2, n2 = tid2, rowid2, n2, tid1, rowid1, n1

            ratio = n1 / max(n2, 1)

            if ratio > SIZE_RATIO_THRESHOLD:
                # Asymmetric: always remove from larger town
                cos1 = cosine_sim(emb_map[word], town_centroids[tid1]) if word in emb_map and tid1 in town_centroids else -1.0
                cos2 = cosine_sim(emb_map[word], town_centroids[tid2]) if word in emb_map and tid2 in town_centroids else -1.0
                removals.append((rowid1, word, tid1, tid2, cos2, cos1))
            else:
                # Symmetric: use cosine similarity
                if word not in emb_map:
                    no_embedding += 1
                    continue

                word_vec = emb_map[word]
                cos1 = cosine_sim(word_vec, town_centroids[tid1]) if tid1 in town_centroids else -1.0
                cos2 = cosine_sim(word_vec, town_centroids[tid2]) if tid2 in town_centroids else -1.0

                if cos1 >= cos2:
                    # Keep in tid1 (larger), remove from tid2
                    removals.append((rowid2, word, tid2, tid1, cos1, cos2))
                else:
                    # Keep in tid2 (smaller), remove from tid1
                    removals.append((rowid1, word, tid1, tid2, cos2, cos1))

        # Summarize for this island
        n_shared = len(shared_words)
        n_removals = len(removals)
        total_shared_words += n_shared
        total_removed += n_removals

        if n_removals == 0:
            continue

        # Group removals by source town for reporting
        removal_by_town = defaultdict(list)
        for rowid, word, from_tid, to_tid, cos_w, cos_l in removals:
            removal_by_town[from_tid].append((word, town_names[to_tid], cos_w, cos_l))

        before_counts = {tid: len(town_seeds[tid]) for tid in town_ids}
        after_counts = {tid: before_counts[tid] - len(removal_by_town.get(tid, []))
                        for tid in town_ids}

        print(f"\n{'='*60}")
        print(f"Island: {island_name}")
        print(f"  Shared words: {n_shared}, removals: {n_removals}" +
              (f", no embedding: {no_embedding}" if no_embedding else ""))

        # Show towns that lose the most seeds
        losers = [(tid, len(removal_by_town.get(tid, [])))
                   for tid in town_ids if removal_by_town.get(tid)]
        losers.sort(key=lambda x: -x[1])

        for tid, n_lost in losers:
            name = town_names[tid]
            before = before_counts[tid]
            after = after_counts[tid]
            pct = 100 * n_lost / before if before > 0 else 0
            print(f"  {name}: {before} → {after} seeds ({n_lost} removed, {pct:.0f}%)")

            # Show sample removals (top 5 by cosine difference)
            samples = sorted(removal_by_town[tid],
                             key=lambda x: x[2] - x[3], reverse=True)[:5]
            for word, dest, cos_w, cos_l in samples:
                print(f"    '{word}' → {dest} (cos {cos_l:.3f} → {cos_w:.3f})")

        island_summaries.append({
            "island": island_name,
            "shared": n_shared,
            "removals": n_removals,
            "losers": [(town_names[tid], before_counts[tid], after_counts[tid])
                       for tid, _ in losers],
        })

    # Final summary
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_shared_words:,} shared words, {total_removed:,} seed removals across {len(island_summaries)} islands")

    if not args.apply:
        print("\nDry run — no changes made. Use --apply to modify the database.")
        return

    # Apply removals
    print("\nApplying changes...")
    cur = con.cursor()
    cur.execute("BEGIN TRANSACTION")

    # Re-compute removals (same logic, just collect rowids)
    all_rowids_to_delete = []
    for island_id, island_name in islands:
        towns = con.execute(
            "SELECT town_id, name FROM Towns WHERE island_id = ? ORDER BY town_id",
            (island_id,)
        ).fetchall()
        town_ids = {t[0] for t in towns}

        town_seeds = defaultdict(dict)
        for rowid, town_id, word in con.execute("""
            SELECT rowid, town_id, word FROM SeedWords
            WHERE town_id IN ({})
        """.format(",".join("?" * len(town_ids))), list(town_ids)).fetchall():
            town_seeds[town_id][word] = rowid

        word_towns = defaultdict(list)
        for tid in town_ids:
            for word, rowid in town_seeds[tid].items():
                word_towns[word].append((tid, rowid))

        shared_words = {w: t for w, t in word_towns.items() if len(t) >= 2}
        if not shared_words:
            continue

        shared_word_set = set(shared_words.keys())
        exclusive_seeds = {}
        for tid in town_ids:
            exclusive_seeds[tid] = {w for w in town_seeds[tid] if w not in shared_word_set}

        town_centroids = {}
        for tid in town_ids:
            vecs = [emb_map[w] for w in exclusive_seeds[tid] if w in emb_map]
            if vecs:
                town_centroids[tid] = np.mean(vecs, axis=0)

        town_seed_counts = {tid: len(town_seeds[tid]) for tid in town_ids}

        for word, claiming_towns in shared_words.items():
            if len(claiming_towns) != 2:
                continue

            (tid1, rowid1), (tid2, rowid2) = claiming_towns
            n1 = town_seed_counts.get(tid1, 0)
            n2 = town_seed_counts.get(tid2, 0)
            if n2 > n1:
                tid1, rowid1, n1, tid2, rowid2, n2 = tid2, rowid2, n2, tid1, rowid1, n1

            ratio = n1 / max(n2, 1)

            if ratio > 2.0:
                all_rowids_to_delete.append(rowid1)
            else:
                if word not in emb_map:
                    continue
                word_vec = emb_map[word]
                cos1 = cosine_sim(word_vec, town_centroids[tid1]) if tid1 in town_centroids else -1.0
                cos2 = cosine_sim(word_vec, town_centroids[tid2]) if tid2 in town_centroids else -1.0
                if cos1 >= cos2:
                    all_rowids_to_delete.append(rowid2)
                else:
                    all_rowids_to_delete.append(rowid1)

    # Delete in batches
    for i in range(0, len(all_rowids_to_delete), 500):
        batch = all_rowids_to_delete[i:i+500]
        cur.execute(
            "DELETE FROM SeedWords WHERE rowid IN ({})".format(",".join("?" * len(batch))),
            batch
        )

    cur.execute("COMMIT")
    print(f"Deleted {len(all_rowids_to_delete):,} seed entries.")

    # Verify
    remaining = con.execute("SELECT COUNT(*) FROM SeedWords").fetchone()[0]
    print(f"Remaining seeds: {remaining:,}")

    con.close()


if __name__ == "__main__":
    main()
