"""
Re-embed all words in the v3 database using the clustering task prefix.

The original embeddings were generated with nomic-embed-text-v1.5 using
"classification: {word}" as input text.  This script replaces them with
"clustering: {word}" — nomic's intended prefix for clustering tasks,
which better matches our use case (topical word clustering).

All words get the same prefix to keep embeddings in a uniform space.
Hierarchy context enters through the classification pipeline (centroids,
XGBoost, Leiden), not through the embedding text.

Requires GPU for reasonable speed (~1 min for 158K words on RTX 4070 Ti).

Usage:
    python v3/reembed_words.py              # re-embed all words
    python v3/reembed_words.py --dry-run    # show stats, don't modify
    python v3/reembed_words.py --batch-size 512
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project)

V3_DB = os.path.join(_project, "v3/windowsill.db")
EMBEDDING_PREFIX = "clustering: "
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
BATCH_SIZE = 256
CHECKPOINT_INTERVAL = 50   # save progress every N batches


def pack_embedding(arr):
    return struct.pack(f"{EMBEDDING_DIM}f", *arr.astype(np.float32))


def unpack_embedding(blob):
    return np.array(struct.unpack(f"{EMBEDDING_DIM}f", blob), dtype=np.float32)


def main():
    import sqlite3

    parser = argparse.ArgumentParser(description="Re-embed words with clustering prefix")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats only, don't modify database")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Embedding batch size (default: {BATCH_SIZE})")
    parser.add_argument("--verify", action="store_true",
                        help="Verify embeddings changed after re-embedding")
    parser.add_argument("--missing-only", action="store_true",
                        help="Only embed words with no embedding (fast backfill)")
    args = parser.parse_args()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    # Load words
    if args.missing_only:
        print("Loading words WITHOUT embeddings...")
        rows = con.execute(
            "SELECT word_id, word FROM Words WHERE embedding IS NULL ORDER BY word_id"
        ).fetchall()
    else:
        print("Loading all words from database...")
        rows = con.execute(
            "SELECT word_id, word FROM Words ORDER BY word_id"
        ).fetchall()

    word_ids = [r[0] for r in rows]
    words = [r[1] for r in rows]
    n_words = len(words)
    print(f"  {n_words:,} words to embed")

    if n_words == 0:
        print("  Nothing to embed.")
        con.close()
        return

    # Check current embedding state
    n_total = con.execute("SELECT COUNT(*) FROM Words").fetchone()[0]
    n_with_emb = con.execute(
        "SELECT COUNT(*) FROM Words WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    n_without_emb = n_total - n_with_emb
    print(f"  {n_with_emb:,} currently have embeddings, {n_without_emb:,} without")

    if args.dry_run:
        # Sample: show what the old vs new embeddings look like
        sample_words = ["rook", "pitch", "batting", "outside", "photon"]
        print(f"\nSample comparison (old 'classification:' vs new 'clustering:'):")

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

        for word in sample_words:
            row = con.execute(
                "SELECT word_id, embedding FROM Words WHERE word = ?", (word,)
            ).fetchone()
            if not row or not row[1]:
                continue

            old_emb = unpack_embedding(row[1])
            new_emb = model.encode([f"{EMBEDDING_PREFIX}{word}"],
                                   show_progress_bar=False)[0]

            cos = float(np.dot(old_emb, new_emb) /
                       (np.linalg.norm(old_emb) * np.linalg.norm(new_emb)))
            print(f"  {word:15s} cos(old, new) = {cos:.4f}")

        print("\nDry run — no changes made.")
        con.close()
        return

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print("  Model loaded")

    # Check for checkpoint
    checkpoint_dir = os.path.join(_project, "v3/intermediates/reembed")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "progress.npy")
    meta_file = os.path.join(checkpoint_dir, "meta.txt")

    start_idx = 0
    if os.path.exists(checkpoint_file) and os.path.exists(meta_file):
        with open(meta_file) as f:
            start_idx = int(f.read().strip())
        print(f"  Resuming from word index {start_idx:,}")

    # Embed in batches
    total_batches = (n_words - start_idx + args.batch_size - 1) // args.batch_size
    print(f"\nEmbedding {n_words - start_idx:,} words in {total_batches} batches...")
    t0 = time.time()

    cur = con.cursor()
    n_updated = 0

    for batch_idx in range(total_batches):
        batch_start = start_idx + batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, n_words)
        batch_words = words[batch_start:batch_end]
        batch_wids = word_ids[batch_start:batch_end]

        # Prepare texts with clustering prefix
        texts = [f"{EMBEDDING_PREFIX}{w}" for w in batch_words]

        # Embed
        embeddings = model.encode(texts, show_progress_bar=False)

        # Update database
        cur.execute("BEGIN TRANSACTION")
        for i, (wid, emb) in enumerate(zip(batch_wids, embeddings)):
            blob = pack_embedding(emb)
            cur.execute(
                "UPDATE Words SET embedding = ? WHERE word_id = ?",
                (blob, wid)
            )
            n_updated += 1
        cur.execute("COMMIT")

        # Progress
        elapsed = time.time() - t0
        rate = n_updated / elapsed if elapsed > 0 else 0
        pct = 100 * (batch_start + len(batch_words)) / n_words

        if (batch_idx + 1) % 20 == 0 or batch_idx == total_batches - 1:
            eta = (n_words - start_idx - n_updated) / rate if rate > 0 else 0
            print(f"  {pct:5.1f}% | {n_updated:,}/{n_words - start_idx:,} | "
                  f"{rate:.0f} words/sec | ETA {eta:.0f}s")

        # Checkpoint
        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
            with open(meta_file, "w") as f:
                f.write(str(batch_end))

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if os.path.exists(meta_file):
        os.remove(meta_file)
    try:
        os.rmdir(checkpoint_dir)
    except OSError:
        pass

    elapsed = time.time() - t0
    print(f"\nDone. {n_updated:,} embeddings updated in {elapsed:.1f}s "
          f"({n_updated/elapsed:.0f} words/sec)")

    # Verify
    if args.verify:
        print("\nVerification: comparing old vs new for sample words...")
        sample = ["rook", "pitch", "batting", "outside", "photon"]
        for word in sample:
            row = con.execute(
                "SELECT embedding FROM Words WHERE word = ?", (word,)
            ).fetchone()
            if row and row[0]:
                emb = unpack_embedding(row[0])
                print(f"  {word:15s} norm={np.linalg.norm(emb):.4f} "
                      f"first_5={emb[:5]}")

    con.close()


if __name__ == "__main__":
    main()
