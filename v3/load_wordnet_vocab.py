"""Load vocabulary from NLTK WordNet into the Words table.

Populates Words with all lemmas from WordNet 3.0 (via NLTK).
Computes POS, word_count, category, and FNV-1a word_hash.

No dependency on v2 database — pulls directly from WordNet.

Pipeline position: runs AFTER schema.sql, BEFORE reembed_words.py.

Usage:
    python v3/load_wordnet_vocab.py --dry-run
    python v3/load_wordnet_vocab.py
"""

import os
import sys
from collections import Counter

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_DB = os.path.join(_project, "v3/windowsill.db")

# FNV-1a u64 constants (must match config.py / word_list.py)
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


# WordNet POS tag → our POS label
WN_POS_MAP = {
    "n": "noun",
    "v": "verb",
    "a": "adj",
    "r": "adv",
    "s": "adj",  # satellite adjective → adj
}


def classify_word(lemma_name):
    """Classify a lemma into category: single, compound, phrasal_verb, etc."""
    if " " in lemma_name:
        # Multi-word expression
        tokens = lemma_name.split()
        # Phrasal verbs: 2-3 words where last word is a common particle
        particles = {"up", "down", "out", "in", "on", "off", "over", "away",
                     "back", "through", "around", "about", "along", "across",
                     "aside", "forth", "together", "apart"}
        if len(tokens) in (2, 3) and tokens[-1].lower() in particles:
            return "phrasal_verb"
        return "compound"
    elif "_" in lemma_name:
        return "compound"
    else:
        return "single"


def determine_dominant_pos(pos_counts):
    """Given POS frequency counts, return the dominant POS.

    Priority: noun > verb > adj > adv (in case of ties).
    """
    priority = ["noun", "verb", "adj", "adv"]
    best = None
    best_count = 0
    for pos in priority:
        c = pos_counts.get(pos, 0)
        if c > best_count:
            best = pos
            best_count = c
    return best


def main():
    import argparse
    import sqlite3
    import time

    from nltk.corpus import wordnet as wn

    parser = argparse.ArgumentParser(description="Load WordNet vocabulary into Words table")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats without writing to database")
    args = parser.parse_args()

    print("Loading WordNet lemmas from NLTK...")
    t0 = time.time()

    # Collect all unique lemma names with their POS frequencies
    # Key: normalized lemma text, Value: Counter of POS labels
    lemma_pos = {}
    total_synsets = 0

    for synset in wn.all_synsets():
        total_synsets += 1
        pos_label = WN_POS_MAP.get(synset.pos(), None)
        if pos_label is None:
            continue
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name not in lemma_pos:
                lemma_pos[name] = Counter()
            lemma_pos[name][pos_label] += 1

    print(f"  {total_synsets:,} synsets, {len(lemma_pos):,} unique lemmas ({time.time()-t0:.1f}s)")

    # Build word records
    words = []
    category_counts = Counter()
    pos_counts = Counter()

    for lemma_text, pos_freqs in sorted(lemma_pos.items()):
        word_hash = fnv1a_i64(lemma_text)
        pos = determine_dominant_pos(pos_freqs)
        word_count = len(lemma_text.split())
        category = classify_word(lemma_text)

        words.append((lemma_text, word_hash, pos, word_count, category))
        category_counts[category] += 1
        if pos:
            pos_counts[pos] += 1

    print(f"\n  POS distribution:")
    for pos, count in pos_counts.most_common():
        print(f"    {pos}: {count:,}")
    print(f"\n  Category distribution:")
    for cat, count in category_counts.most_common():
        print(f"    {cat}: {count:,}")

    if args.dry_run:
        print(f"\nDry run — {len(words):,} words would be inserted.")
        return

    # Write to database
    print(f"\nWriting {len(words):,} words to database...")
    t0 = time.time()

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    # Check if Words already has data
    existing = con.execute("SELECT COUNT(*) FROM Words").fetchone()[0]
    if existing > 0:
        print(f"  WARNING: Words table already has {existing:,} rows.")
        print(f"  Clearing existing data...")
        con.execute("DELETE FROM Words")
        con.commit()

    con.executemany("""
        INSERT INTO Words (word, word_hash, pos, word_count, category)
        VALUES (?, ?, ?, ?, ?)
    """, words)
    con.commit()

    final_count = con.execute("SELECT COUNT(*) FROM Words").fetchone()[0]
    print(f"  Inserted {final_count:,} words ({time.time()-t0:.1f}s)")

    # Verify
    print(f"\nVerification:")
    for row in con.execute("""
        SELECT pos, COUNT(*) as n FROM Words
        GROUP BY pos ORDER BY n DESC
    """).fetchall():
        print(f"  {row[0] or '(none)':>8}: {row[1]:>8,}")

    for row in con.execute("""
        SELECT category, COUNT(*) as n FROM Words
        GROUP BY category ORDER BY n DESC
    """).fetchall():
        print(f"  {row[0]:>15}: {row[1]:>8,}")

    # Sample
    print(f"\nSample words:")
    for row in con.execute(
        "SELECT word_id, word, pos, category FROM Words ORDER BY RANDOM() LIMIT 10"
    ).fetchall():
        print(f"  [{row[0]}] {row[1]} ({row[2]}, {row[3]})")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
