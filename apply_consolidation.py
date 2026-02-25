"""Apply domain consolidation mapping to the database.

One-off script that merges redundant domains by remapping old domain names
to canonical names. Run once after extract.py, before transform.py.

Usage:
    python apply_consolidation.py          # apply to v2.db
    python apply_consolidation.py --dry    # preview changes without writing
"""

import argparse
import json
import os
import sqlite3

from lib import db

_here = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_here, "v2.db")
MAP_PATH = os.path.join(_here, "domain_consolidation_map.json")

# Source priority: higher = keep when duplicates arise after merge.
SOURCE_PRIORITY = {
    "wordnet": 5,
    "claude_augmented": 4,
    "domain_name": 3,
    "morphy": 2,
    "xgboost": 1,
}


def load_mapping():
    with open(MAP_PATH) as f:
        return json.load(f)


def apply(con, mapping, dry=False):
    """Remap domains in augmented_domains and wordnet_domains."""

    # --- Snapshot before ---
    n_domains_before = con.execute(
        "SELECT COUNT(DISTINCT domain) FROM augmented_domains"
    ).fetchone()[0]

    # Only process old domains that actually exist in the DB
    existing = {r[0] for r in con.execute(
        "SELECT DISTINCT domain FROM augmented_domains"
    ).fetchall()}
    active = {old: new for old, new in mapping.items() if old in existing}
    skipped = len(mapping) - len(active)
    if skipped:
        print(f"  Skipping {skipped} domains not present in DB")

    total_remapped = 0
    total_dupes_removed = 0

    for old_domain, new_domain in sorted(active.items()):
        # --- augmented_domains ---
        # Find rows that would collide after rename (same word_id in new_domain)
        collisions = con.execute("""
            SELECT a_old.rowid, a_old.word_id, a_old.source,
                   a_new.rowid, a_new.source
            FROM augmented_domains a_old
            JOIN augmented_domains a_new
              ON a_old.word_id = a_new.word_id
             AND a_new.domain = ?
            WHERE a_old.domain = ?
              AND a_old.word_id IS NOT NULL
        """, (new_domain, old_domain)).fetchall()

        # Resolve collisions: keep higher-priority source
        dupes_to_delete = set()
        for old_rowid, wid, old_src, new_rowid, new_src in collisions:
            old_pri = SOURCE_PRIORITY.get(old_src, 0)
            new_pri = SOURCE_PRIORITY.get(new_src, 0)
            if old_pri > new_pri:
                # Old row wins: delete the existing new-domain row, rename old
                dupes_to_delete.add(new_rowid)
            else:
                # New row wins (or tie): delete old row
                dupes_to_delete.add(old_rowid)

        if not dry:
            # Delete losing duplicates
            if dupes_to_delete:
                placeholders = ",".join("?" * len(dupes_to_delete))
                con.execute(
                    f"DELETE FROM augmented_domains WHERE rowid IN ({placeholders})",
                    list(dupes_to_delete),
                )

            # Also handle (domain, word) PK collisions for rows without word_id
            # by using INSERT OR IGNORE + DELETE pattern
            # First, rename remaining old-domain rows
            n_renamed = con.execute(
                "UPDATE OR IGNORE augmented_domains SET domain = ? WHERE domain = ?",
                (new_domain, old_domain),
            ).rowcount

            # Delete any stragglers that couldn't be renamed due to PK conflict
            n_stragglers = con.execute(
                "DELETE FROM augmented_domains WHERE domain = ?",
                (old_domain,),
            ).rowcount

            total_remapped += n_renamed
            total_dupes_removed += len(dupes_to_delete) + n_stragglers

            # --- wordnet_domains ---
            con.execute(
                "UPDATE OR IGNORE wordnet_domains SET domain = ? WHERE domain = ?",
                (new_domain, old_domain),
            )
            con.execute(
                "DELETE FROM wordnet_domains WHERE domain = ?",
                (old_domain,),
            )

    if not dry:
        con.commit()

    n_domains_after = con.execute(
        "SELECT COUNT(DISTINCT domain) FROM augmented_domains"
    ).fetchone()[0]

    print(f"\n  Domains: {n_domains_before} → {n_domains_after} "
          f"({n_domains_before - n_domains_after} removed)")
    print(f"  Rows remapped: {total_remapped:,}")
    print(f"  Duplicate rows removed: {total_dupes_removed:,}")

    return n_domains_before, n_domains_after


def main():
    parser = argparse.ArgumentParser(description="Apply domain consolidation")
    parser.add_argument("--dry", action="store_true",
                        help="Preview changes without writing")
    args = parser.parse_args()

    mapping = load_mapping()
    print(f"Loaded {len(mapping)} domain mappings from {MAP_PATH}")

    con = db.get_connection(DB_PATH)

    if args.dry:
        print("\n[DRY RUN] No changes will be written.\n")

    apply(con, mapping, dry=args.dry)

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
