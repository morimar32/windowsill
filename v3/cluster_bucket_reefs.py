"""Cluster bucket island seeds into reefs.

Bucket islands (is_bucket=1) skip XGBoost training, so their only words
are curated seeds.  Instead of full Leiden clustering (overkill for
25-150 words), each town gets a single reef with all its seeds as core
members.

Populates: Reefs, ReefWords (source='curated', is_core=1)
Updates:   Town/Island stats

Pipeline position: runs AFTER seed_from_v2.py populates SeedWords,
INSTEAD OF cluster_reefs.py (which handles topical islands).

Usage:
    python v3/cluster_bucket_reefs.py --dry-run
    python v3/cluster_bucket_reefs.py
"""

import os
import struct
import sys
import time

import numpy as np

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_DB = os.path.join(_project, "v3/windowsill.db")

CHARACTERISTIC_WORDS_N = 5


def unpack_embedding(blob):
    return np.array(struct.unpack(f"{768}f", blob), dtype=np.float64)


def pack_embedding(arr):
    return struct.pack(f"{768}f", *arr.astype(np.float32))


def main():
    import argparse
    import sqlite3

    parser = argparse.ArgumentParser(description="Cluster bucket island seeds into reefs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing")
    parser.add_argument("--island", type=str, default=None,
                        help="Process a specific bucket island (default: all)")
    args = parser.parse_args()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    # Find bucket islands
    if args.island:
        islands = con.execute("""
            SELECT island_id, name FROM Islands
            WHERE is_bucket = 1 AND name = ?
        """, (args.island,)).fetchall()
        if not islands:
            print(f"No bucket island named '{args.island}'")
            return
    else:
        islands = con.execute("""
            SELECT island_id, name FROM Islands
            WHERE is_bucket = 1
            ORDER BY island_id
        """).fetchall()

    print(f"Bucket islands: {len(islands)}")
    for island_id, island_name in islands:
        print(f"  {island_name} (id={island_id})")
    print()

    # Load embeddings
    print("Loading embeddings...")
    t0 = time.time()
    emb_map = {}
    for wid, blob in con.execute(
        "SELECT word_id, embedding FROM Words WHERE embedding IS NOT NULL"
    ).fetchall():
        emb_map[wid] = unpack_embedding(blob)
    print(f"  {len(emb_map):,} embeddings ({time.time()-t0:.1f}s)")

    # Load word texts
    word_texts = {wid: word for wid, word in
                  con.execute("SELECT word_id, word FROM Words").fetchall()}

    total_reefs = 0
    total_words = 0

    for island_id, island_name in islands:
        print(f"\n=== {island_name} ===")

        towns = con.execute("""
            SELECT town_id, name FROM Towns
            WHERE island_id = ?
            ORDER BY town_id
        """, (island_id,)).fetchall()

        for town_id, town_name in towns:
            # Get seed word_ids with embeddings (exclude stop words)
            seed_rows = con.execute("""
                SELECT DISTINCT s.word_id
                FROM SeedWords s
                JOIN Words w USING (word_id)
                WHERE s.town_id = ? AND s.word_id IS NOT NULL AND w.is_stop = 0
            """, (town_id,)).fetchall()
            seed_wids = [r[0] for r in seed_rows if r[0] in emb_map]

            if not seed_wids:
                print(f"  {town_name}: SKIP — no seeds with embeddings")
                continue

            # Build embedding matrix
            embeddings = np.vstack([emb_map[w] for w in seed_wids])

            # Compute centroid (L2-normalized mean)
            centroid = embeddings.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            # Cosine similarities to centroid
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = embeddings / norms
            cosines = normed @ centroid

            # Reef name from top characteristic words
            top_indices = np.argsort(cosines)[::-1][:CHARACTERISTIC_WORDS_N]
            top_words = [word_texts.get(seed_wids[i], str(seed_wids[i]))
                         for i in top_indices]
            reef_name = "_".join(top_words[:3])

            avg_cos = float(cosines.mean())
            min_cos = float(cosines.min())

            if args.dry_run:
                print(f"  {town_name}: {len(seed_wids)} seeds, "
                      f"avg_cos={avg_cos:.3f}, min_cos={min_cos:.3f}, "
                      f"top: {', '.join(top_words)}")
                continue

            # Write to database
            cur = con.cursor()

            # Delete existing reefs for this town (idempotent)
            cur.execute("""
                DELETE FROM ReefWords WHERE reef_id IN
                (SELECT reef_id FROM Reefs WHERE town_id = ?)
            """, (town_id,))
            cur.execute("DELETE FROM Reefs WHERE town_id = ?", (town_id,))

            # Insert single reef
            cur.execute("""
                INSERT INTO Reefs (town_id, name, centroid, word_count, core_word_count)
                VALUES (?, ?, ?, ?, ?)
            """, (town_id, reef_name, pack_embedding(centroid),
                  len(seed_wids), len(seed_wids)))
            reef_id = cur.lastrowid

            # Insert all seeds as core ReefWords
            for i, wid in enumerate(seed_wids):
                cur.execute("""
                    INSERT INTO ReefWords (reef_id, word_id, cosine_sim, source, is_core)
                    VALUES (?, ?, ?, 'curated', 1)
                """, (reef_id, wid, float(cosines[i])))

            con.commit()

            total_reefs += 1
            total_words += len(seed_wids)

            print(f"  {town_name}: 1 reef, {len(seed_wids)} words, "
                  f"avg_cos={avg_cos:.3f}, min_cos={min_cos:.3f}, "
                  f"top: {', '.join(top_words)}")

    if args.dry_run:
        print("\nDry run — no changes made.")
        con.close()
        return

    # Update town stats
    print("\nUpdating stats...")
    for island_id, island_name in islands:
        con.execute("""
            UPDATE Towns SET
                reef_count = (SELECT COUNT(*) FROM Reefs WHERE Reefs.town_id = Towns.town_id),
                word_count = (SELECT COUNT(*) FROM ReefWords rw
                           JOIN Reefs r USING(reef_id)
                           WHERE r.town_id = Towns.town_id)
            WHERE island_id = ?
        """, (island_id,))

        con.execute("""
            UPDATE Islands SET
                town_count = (SELECT COUNT(*) FROM Towns WHERE Towns.island_id = Islands.island_id),
                reef_count = (SELECT COUNT(*) FROM Reefs r
                           JOIN Towns t USING(town_id)
                           WHERE t.island_id = Islands.island_id),
                word_count = (SELECT COUNT(DISTINCT rw.word_id) FROM ReefWords rw
                           JOIN Reefs r USING(reef_id)
                           JOIN Towns t USING(town_id)
                           WHERE t.island_id = Islands.island_id)
            WHERE island_id = ?
        """, (island_id,))
    con.commit()

    # Summary
    print(f"\nDone. {total_reefs} reefs, {total_words:,} words across {len(islands)} bucket islands")

    # Verify
    for island_id, island_name in islands:
        row = con.execute("""
            SELECT town_count, reef_count, word_count FROM Islands WHERE island_id = ?
        """, (island_id,)).fetchone()
        print(f"  {island_name}: {row[0]} towns, {row[1]} reefs, {row[2]} words")

    con.close()


if __name__ == "__main__":
    main()
