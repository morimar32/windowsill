"""Link SeedWords to the Words table — safety net after all seed generation.

Runs after all seed sources have populated the SeedWords table and ensures
every seed that CAN be linked to a Words record IS linked.  Two passes:

  Pass 1 — Link existing:  For SeedWords with word_id IS NULL, look up the
           word in the Words table and set word_id if found.

  Pass 2 — Insert missing: Single words (no spaces) that are NOT in the Words
           table get inserted (with word_hash and word_count, but no embedding).
           Then their SeedWords.word_id is updated.

After this script runs, the only SeedWords with word_id = NULL should be
multi-word phrases and hyphenated compounds that don't match a Words entry.
Those are expected — compound seeds are used by Lagoon's Aho-Corasick
tokenizer, not by XGBoost training.

NOTE: Newly inserted Words will NOT have embeddings.  Run reembed_words.py
afterward to generate embeddings for them (it processes all Words where
embedding IS NULL).

Pipeline position: runs AFTER all seed generation, BEFORE sanity_check_seeds.py.
Fully idempotent — safe to re-run at any time.

Usage:
    python v3/link_seed_words.py              # preview (dry-run by default)
    python v3/link_seed_words.py --apply      # actually modify the database
"""

import argparse
import os
import sqlite3
import sys

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project)

V3_DB = os.path.join(_project, "v3/windowsill.db")

# FNV-1a u64 constants (must match load_wordnet_vocab.py / config.py)
FNV1A_OFFSET = 14695981039346656037
FNV1A_PRIME = 1099511628211


def fnv1a_u64(s):
    """FNV-1a u64 hash of a string."""
    h = FNV1A_OFFSET
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * FNV1A_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def fnv1a_i64(s):
    """FNV-1a hash stored as signed i64 (SQLite convention)."""
    h = fnv1a_u64(s)
    if h >= (1 << 63):
        h -= (1 << 64)
    return h


def main():
    parser = argparse.ArgumentParser(
        description="Link SeedWords to Words table (safety net after seed generation)"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Actually modify the database (default is dry-run)")
    args = parser.parse_args()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    # ── Current state ────────────────────────────────────────────────
    total_seeds = con.execute("SELECT COUNT(*) FROM SeedWords").fetchone()[0]
    linked_seeds = con.execute(
        "SELECT COUNT(*) FROM SeedWords WHERE word_id IS NOT NULL"
    ).fetchone()[0]
    unlinked_seeds = total_seeds - linked_seeds

    print(f"SeedWords: {total_seeds:,} total, {linked_seeds:,} linked, "
          f"{unlinked_seeds:,} unlinked")

    if unlinked_seeds == 0:
        print("Nothing to do — all seeds already linked.")
        con.close()
        return

    # Build vocab lookup: word (lowercase) → word_id
    vocab = {}
    for word_id, word in con.execute("SELECT word_id, word FROM Words").fetchall():
        vocab[word.lower()] = word_id

    # ── Pass 1: Link seeds to existing Words ─────────────────────────
    print("\nPass 1: Linking seeds to existing Words entries...")

    unlinked_rows = con.execute(
        "SELECT rowid, word FROM SeedWords WHERE word_id IS NULL"
    ).fetchall()

    link_updates = []
    for rowid, seed_word in unlinked_rows:
        word_id = vocab.get(seed_word.lower())
        if word_id is not None:
            link_updates.append((word_id, rowid))

    print(f"  {len(link_updates):,} seeds match existing Words entries")

    # ── Pass 2: Insert missing single words into Words ───────────────
    print("\nPass 2: Finding single words not in Words table...")

    # Collect unique unlinked single words not already matched
    matched_rowids = {rowid for _, rowid in link_updates}
    missing_words = set()
    for rowid, seed_word in unlinked_rows:
        if rowid in matched_rowids:
            continue
        word = seed_word.lower().strip()
        # Only insert single words (no spaces, no hyphens)
        if " " in word or "-" in word:
            continue
        if not word:
            continue
        if word not in vocab:
            missing_words.add(word)

    print(f"  {len(missing_words):,} single words not in Words table")

    # Classify remaining unlinked that we WON'T insert
    still_unlinked = unlinked_seeds - len(link_updates) - len(missing_words)
    multi_word = sum(1 for rowid, w in unlinked_rows
                     if rowid not in matched_rowids
                     and (" " in w or "-" in w))
    print(f"  {multi_word:,} multi-word/hyphenated seeds (expected, no action needed)")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Pass 1 — Link to existing Words:  {len(link_updates):,}")
    print(f"  Pass 2 — Insert new Words:         {len(missing_words):,}")
    total_fixes = len(link_updates) + len(missing_words)
    print(f"  Total seeds that will gain word_id: {total_fixes:,}")
    print(f"  Remaining unlinked (multi-word):    {multi_word:,}")

    if not args.apply:
        # Show samples of what would be inserted
        if missing_words:
            sample = sorted(missing_words)[:20]
            print(f"\n  Sample new Words to insert: {', '.join(sample)}")

        # Show per-source breakdown of what gets fixed
        print(f"\n  Linkage by source (after fix):")
        for source, total, currently_linked in con.execute("""
            SELECT source, COUNT(*), SUM(CASE WHEN word_id IS NOT NULL THEN 1 ELSE 0 END)
            FROM SeedWords GROUP BY source
        """).fetchall():
            # Estimate how many would be newly linked
            newly_linked = sum(1 for rowid, word in unlinked_rows
                               if (word_id := vocab.get(word.lower())) is not None
                               or (word.lower().strip() in missing_words
                                   and " " not in word and "-" not in word))
            # This is an approximation; just show current state
            print(f"    {source:20s}: {currently_linked:>6,} / {total:>6,} linked")

        print(f"\nDry run — no changes made. Use --apply to modify the database.")
        con.close()
        return

    # ── Apply Pass 1: link existing ──────────────────────────────────
    print(f"\nApplying Pass 1: linking {len(link_updates):,} seeds...")
    con.executemany(
        "UPDATE SeedWords SET word_id = ? WHERE rowid = ?",
        link_updates
    )
    con.commit()
    print(f"  Done.")

    # ── Apply Pass 2: insert new Words + link ────────────────────────
    if missing_words:
        print(f"Applying Pass 2: inserting {len(missing_words):,} new Words...")

        insert_rows = []
        for word in sorted(missing_words):
            word_hash = fnv1a_i64(word)
            word_count = 1  # guaranteed single word (no spaces)
            insert_rows.append((word, word_hash, word_count))

        con.executemany(
            "INSERT OR IGNORE INTO Words (word, word_hash, word_count) VALUES (?, ?, ?)",
            insert_rows
        )
        con.commit()

        # Refresh vocab lookup with newly inserted words
        new_vocab = {}
        for word_id, word in con.execute("SELECT word_id, word FROM Words").fetchall():
            new_vocab[word.lower()] = word_id

        # Link the newly inserted words
        new_link_updates = []
        for rowid, seed_word in unlinked_rows:
            if rowid in matched_rowids:
                continue  # already handled in pass 1
            word = seed_word.lower().strip()
            if " " in word or "-" in word:
                continue
            word_id = new_vocab.get(word)
            if word_id is not None:
                new_link_updates.append((word_id, rowid))

        con.executemany(
            "UPDATE SeedWords SET word_id = ? WHERE rowid = ?",
            new_link_updates
        )
        con.commit()
        print(f"  Inserted {len(insert_rows):,} Words, linked {len(new_link_updates):,} seeds.")
        print(f"  NOTE: New Words have no embeddings. Run reembed_words.py to generate them.")

    # ── Final state ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"FINAL STATE")
    print(f"{'=' * 60}")

    for source, total, now_linked in con.execute("""
        SELECT source, COUNT(*), SUM(CASE WHEN word_id IS NOT NULL THEN 1 ELSE 0 END)
        FROM SeedWords GROUP BY source
    """).fetchall():
        pct = (now_linked / total * 100) if total > 0 else 0
        print(f"  {source:20s}: {now_linked:>6,} / {total:>6,} linked ({pct:.0f}%)")

    total_now = con.execute(
        "SELECT COUNT(*) FROM SeedWords WHERE word_id IS NOT NULL"
    ).fetchone()[0]
    total_null = con.execute(
        "SELECT COUNT(*) FROM SeedWords WHERE word_id IS NULL"
    ).fetchone()[0]
    print(f"\n  Total linked:   {total_now:,}")
    print(f"  Total unlinked: {total_null:,} (multi-word phrases / not-in-vocab)")

    n_words = con.execute("SELECT COUNT(*) FROM Words").fetchone()[0]
    n_no_emb = con.execute(
        "SELECT COUNT(*) FROM Words WHERE embedding IS NULL"
    ).fetchone()[0]
    print(f"\n  Words table: {n_words:,} total, {n_no_emb:,} without embeddings")
    if n_no_emb > 0:
        print(f"  → Run reembed_words.py to generate embeddings for new words")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
