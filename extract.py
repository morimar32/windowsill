"""Entry point: full v2 database bootstrap.

Creates a fresh v2.db (sqlite3) with 5 tables:
  words, dim_stats, word_variants, wordnet_domains, augmented_domains

Usage:
    python extract.py
"""

import os
import time

from lib import db, vocab, domains

_here = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_here, "v2.db")
AUGMENTED_DIR = os.path.join(_here, "augmented_domains")

TOTAL_STEPS = 13


def main():
    t0 = time.time()

    # 1. Create fresh database
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"  Removed existing {DB_PATH}")
    con = db.get_connection(DB_PATH)
    db.create_schema(con)

    # 2. Load words + embeddings (WordNet, is_wordnet=1)
    print(f"\n[2/{TOTAL_STEPS}] Loading words + embeddings...")
    vocab.load_words(con)

    # 3. Load Claude-generated words (embed + insert, is_wordnet=0)
    print(f"\n[3/{TOTAL_STEPS}] Loading Claude-generated words...")
    new_words = vocab.collect_unmatched_claude_words(con, AUGMENTED_DIR)
    vocab.embed_and_insert_claude_words(con, new_words, AUGMENTED_DIR)

    # 4. Compute dim_stats (all words)
    print(f"\n[4/{TOTAL_STEPS}] Computing dim_stats...")
    vocab.compute_dim_stats(con)

    # 5. Compute total_dims per word
    print(f"\n[5/{TOTAL_STEPS}] Computing total_dims...")
    vocab.compute_total_dims(con)

    # 6. Backfill POS + category
    print(f"\n[6/{TOTAL_STEPS}] Backfilling POS + category...")
    vocab.backfill_pos_and_category(con)

    # 7. Flag stop words
    print(f"\n[7/{TOTAL_STEPS}] Flagging stop words...")
    vocab.flag_stop_words(con)

    # 8. Compute synset counts
    print(f"\n[8/{TOTAL_STEPS}] Computing synset counts...")
    vocab.compute_synset_counts(con)

    # 9. Compute word hashes
    print(f"\n[9/{TOTAL_STEPS}] Computing word hashes...")
    vocab.compute_word_hashes(con)

    # 10. Expand morphy variants
    print(f"\n[10/{TOTAL_STEPS}] Expanding morphy variants...")
    vocab.expand_morphy_variants(con)

    # 11. Populate wordnet_domains
    print(f"\n[11/{TOTAL_STEPS}] Populating wordnet_domains...")
    vocab.populate_wordnet_domains(con)

    # 12. Load augmented_domains
    print(f"\n[12/{TOTAL_STEPS}] Loading augmented_domains...")
    domains.create_augmented_domains_table(con)
    domains.load_wordnet_domains(con)
    domains.load_claude_domains(con, AUGMENTED_DIR)
    domains.print_coverage_report(con)

    # 13. Backfill domain counts (must come after augmented_domains)
    print(f"\n[13/{TOTAL_STEPS}] Backfilling domain counts...")
    vocab.backfill_domain_counts(con)

    con.close()

    elapsed = time.time() - t0
    size_mb = os.path.getsize(DB_PATH) / 1_048_576
    print(f"\nDone. {DB_PATH} ({size_mb:.1f} MB) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
