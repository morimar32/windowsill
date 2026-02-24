"""Export v2 data to lagoon format (v6.0).

Reads from v2.db (SQLite) and produces MessagePack-serialized .bin files
plus a manifest.json. Domains become reefs, archipelagos become archs,
reefs-within-domains become sub-reefs.

Usage:
    python load.py [--output DIR] [--verify] [--bg-samples N]
"""

import argparse
import math
import os
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

import config
from export import (
    IDF_SCALE,
    WEIGHT_SCALE,
    SUB_REEF_SENTINEL,
    EXPORT_FILES,
    clamp,
    write_msgpack,
    sha256_file,
    write_all_files,
    compute_background_model,
    adjust_background_model,
    build_word_lookup,
    expand_snowball_stems,
)
from lib import db
from word_list import fnv1a_u64

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2.db")
LAGOON_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "lagoon", "src", "lagoon", "data"
)
FORMAT_VERSION = "6.0"


def pack_hierarchy_addr(arch_id, reef_id):
    """Pack arch(16)|island(16) into u32. island_id = reef_id (1:1)."""
    return (arch_id << 16) | reef_id


# ---------------------------------------------------------------------------
# Phase 1: Load hierarchy + build ID remaps
# ---------------------------------------------------------------------------

def load_hierarchy(con):
    """Load archipelagos and domains, build contiguous ID remaps.

    Returns (reef_records, island_records, id_remap, domain_to_arch).
    """
    # Archipelagos
    archs = con.execute("""
        SELECT archipelago_id, label
        FROM domain_archipelago_stats
        ORDER BY archipelago_id
    """).fetchall()
    n_archs = len(archs)
    assert n_archs > 0, "No archipelagos found"

    arch_remap = {}
    for export_id, (db_arch_id, _label) in enumerate(
        sorted(archs, key=lambda r: r[0])
    ):
        arch_remap[db_arch_id] = export_id

    arch_names = {db_id: label for db_id, label in archs}

    # Domain-to-archipelago mapping
    domain_arch_rows = con.execute("""
        SELECT domain, archipelago_id
        FROM domain_archipelagos
        WHERE archipelago_id >= 0
    """).fetchall()
    domain_to_arch = {d: aid for d, aid in domain_arch_rows}

    # All domains with scores (these become reefs)
    scored_domains = con.execute("""
        SELECT DISTINCT domain FROM domain_word_scores
    """).fetchall()
    scored_domain_names = sorted(d[0] for d in scored_domains)

    # Per-domain stats from domain_word_scores
    domain_stats = {}
    for domain, n_words in con.execute("""
        SELECT domain, COUNT(*) FROM domain_word_scores GROUP BY domain
    """).fetchall():
        domain_stats[domain] = {"n_words": n_words}

    # Per-domain POS fractions from words table
    pos_rows = con.execute("""
        SELECT d.domain,
               AVG(w.is_noun), AVG(w.is_verb), AVG(w.is_adj), AVG(w.is_adv)
        FROM domain_word_scores d
        JOIN words w ON d.word_id = w.word_id
        GROUP BY d.domain
    """).fetchall()
    domain_pos = {d: (noun, verb, adj, adv) for d, noun, verb, adj, adv in pos_rows}

    # Build reef remap: sort by (arch_id, domain_name) for determinism
    reef_remap = {}
    reef_records = []
    island_records = []

    for domain in sorted(
        scored_domain_names,
        key=lambda d: (arch_remap.get(domain_to_arch.get(d, -1), 999), d)
    ):
        export_id = len(reef_remap)
        reef_remap[domain] = export_id

        db_arch_id = domain_to_arch.get(domain, -1)
        e_arch = arch_remap.get(db_arch_id, 0)

        stats = domain_stats.get(domain, {"n_words": 0})
        pos = domain_pos.get(domain, (0.0, 0.0, 0.0, 0.0))

        reef_records.append({
            "hierarchy_addr": pack_hierarchy_addr(e_arch, export_id),
            "n_words": stats["n_words"],
            "name": domain,
            "export_island_id": export_id,
            "export_arch_id": e_arch,
            "valence": 0.0,
            "avg_specificity": 0.0,
            "noun_frac": pos[0] or 0.0,
            "verb_frac": pos[1] or 0.0,
            "adj_frac": pos[2] or 0.0,
            "adv_frac": pos[3] or 0.0,
        })
        island_records.append({
            "arch_id": e_arch,
            "name": domain,
        })

    # Compute avg_specificity per domain from word n_word_domains
    spec_rows = con.execute("""
        SELECT domain, AVG(
            CASE
                WHEN n_word_domains <= 3 THEN 2
                WHEN n_word_domains <= 10 THEN 1
                WHEN n_word_domains <= 50 THEN 0
                WHEN n_word_domains <= 150 THEN -1
                ELSE -2
            END
        ) FROM domain_word_scores GROUP BY domain
    """).fetchall()
    domain_avg_spec = {d: avg for d, avg in spec_rows}
    for rec in reef_records:
        rec["avg_specificity"] = domain_avg_spec.get(rec["name"], 0.0)

    id_remap = {"reef": reef_remap, "arch": arch_remap}

    print(f"  Hierarchy: {n_archs} archs, {len(reef_remap)} domains-as-reefs")
    return reef_records, island_records, id_remap


# ---------------------------------------------------------------------------
# Phase 2: Load words
# ---------------------------------------------------------------------------

def load_words(con):
    """Load all words with IDF from domain_word_scores."""
    rows = con.execute("""
        SELECT word_id, word, word_hash, category, word_count
        FROM words ORDER BY word_id
    """).fetchall()
    assert len(rows) > 0, "No words found"

    # Get per-word IDF (same for all domains; just take any row)
    idf_rows = con.execute("""
        SELECT word_id, idf, n_word_domains
        FROM domain_word_scores
        GROUP BY word_id
    """).fetchall()
    word_idf = {wid: (idf, nwd) for wid, idf, nwd in idf_rows}

    words = {}
    for word_id, word, word_hash, category, word_count in rows:
        # SQLite stores all integers as signed i64; convert back to unsigned u64
        word_hash = word_hash & 0xFFFFFFFFFFFFFFFF
        idf, n_wd = word_idf.get(word_id, (0.0, 0))

        # Derive specificity from n_word_domains
        if n_wd <= 3:
            spec = 2
        elif n_wd <= 10:
            spec = 1
        elif n_wd <= 50:
            spec = 0
        elif n_wd <= 150:
            spec = -1
        else:
            spec = -2

        # Words not in any domain get spec 0
        if n_wd == 0:
            spec = 0

        idf_q = 0 if idf == 0 else clamp(round(idf * IDF_SCALE), 0, 255)

        # Derive actual word_count from spaces (v2.db has word_count=1 for all)
        actual_word_count = len(word.split()) if word else 1

        words[word_id] = {
            "word": word,
            "specificity": spec,
            "word_hash": word_hash,
            "reef_idf": idf,
            "idf_q": idf_q,
            "category": category,
            "word_count": actual_word_count,
        }

    print(f"  Loaded {len(words)} words")
    return words


# ---------------------------------------------------------------------------
# Phase 3: Load word variants
# ---------------------------------------------------------------------------

def load_word_variants(con):
    """Load all word variant mappings."""
    rows = con.execute("""
        SELECT variant_hash, variant, word_id, source
        FROM word_variants
    """).fetchall()
    assert len(rows) > 0, "No word variants found"
    # SQLite stores all integers as signed i64; convert hashes back to unsigned u64
    rows = [(vh & 0xFFFFFFFFFFFFFFFF, variant, wid, source)
            for vh, variant, wid, source in rows]
    print(f"  Loaded {len(rows)} word variants")
    return rows


# ---------------------------------------------------------------------------
# Phase 4: Build word_reefs from domain_word_scores
# ---------------------------------------------------------------------------

def build_word_reefs(con, id_remap):
    """Build word_reefs: word_id → list[(export_reef_id, weight_q, sub_reef_id)].

    Also builds sub_reef_remap and sub_reef_records.

    Returns (word_reefs, reef_stats, sub_reef_records, domainless_word_ids).
    """
    reef_remap = id_remap["reef"]

    # Load domain_word_scores
    score_rows = con.execute("""
        SELECT domain, word_id, reef_id, domain_score
        FROM domain_word_scores
    """).fetchall()

    # Build sub-reef remap: (domain, reef_id) → export sub_reef_id
    # Load all valid (domain, reef_id) pairs from domain_reef_stats
    sub_pairs = con.execute("""
        SELECT domain, reef_id, n_total, label
        FROM domain_reef_stats
        WHERE reef_id >= 0
        ORDER BY domain, reef_id
    """).fetchall()

    sub_reef_remap = {}  # (domain, reef_id) → export_sub_reef_id
    sub_reef_records = []
    for domain, reef_id, n_total, label in sorted(
        sub_pairs, key=lambda r: (reef_remap.get(r[0], 999999), r[1])
    ):
        if domain not in reef_remap:
            continue
        export_sub_id = len(sub_reef_remap)
        sub_reef_remap[(domain, reef_id)] = export_sub_id
        sub_reef_records.append({
            "parent_island_id": reef_remap[domain],
            "n_words": n_total,
            "name": label or "",
        })

    # Build word_reefs
    word_reefs_dict = defaultdict(list)
    for domain, word_id, db_reef_id, domain_score in score_rows:
        if domain not in reef_remap:
            continue
        export_reef_id = reef_remap[domain]
        weight_q = max(0, min(65535, round(domain_score * WEIGHT_SCALE)))
        if weight_q == 0:
            continue

        # Map sub-reef
        if db_reef_id >= 0 and (domain, db_reef_id) in sub_reef_remap:
            export_sub_reef_id = sub_reef_remap[(domain, db_reef_id)]
        else:
            export_sub_reef_id = SUB_REEF_SENTINEL

        word_reefs_dict[word_id].append(
            (export_reef_id, weight_q, export_sub_reef_id)
        )

    # Sort entries by reef_id within each word
    word_reefs = {}
    for word_id, entries in word_reefs_dict.items():
        word_reefs[word_id] = sorted(entries, key=lambda e: e[0])

    # Build reef_stats: export_reef_id → {"n_words", "n_dims"}
    reef_word_counts = defaultdict(int)
    for entries in word_reefs.values():
        for reef_id, _wq, _srid in entries:
            reef_word_counts[reef_id] += 1

    reef_stats = {}
    for rid, count in reef_word_counts.items():
        reef_stats[rid] = {"n_words": count, "n_dims": 0}

    # Domainless: words in vocabulary that have no domain scores
    all_vocab_wids = set(
        row[0] for row in con.execute(
            "SELECT word_id FROM words WHERE word_id IS NOT NULL"
        ).fetchall()
    )
    scoring_wids = set(word_reefs.keys())
    domainless_word_ids = sorted(all_vocab_wids - scoring_wids)

    print(f"  word_reefs: {len(word_reefs)} words with reef entries")
    print(f"  sub_reefs: {len(sub_reef_records)} sub-reef records")
    print(f"  domainless: {len(domainless_word_ids)} words (in vocab, no domain)")
    return word_reefs, reef_stats, sub_reef_records, domainless_word_ids


# ---------------------------------------------------------------------------
# Phase 5: Extract compounds
# ---------------------------------------------------------------------------

def extract_compounds(con):
    """Extract compound words for Aho-Corasick automaton.

    Uses space-in-word heuristic since word_count isn't populated in v2.db.
    """
    rows = con.execute("""
        SELECT word_id, word FROM words
        WHERE category IN ('compound', 'phrasal_verb', 'named_entity', 'taxonomic')
          AND word LIKE '% %'
        ORDER BY word_id
    """).fetchall()

    compounds = [[word, word_id] for word_id, word in rows]
    print(f"  Extracted {len(compounds)} compounds")
    return compounds


# ---------------------------------------------------------------------------
# Phase 6: Serialization helpers
# ---------------------------------------------------------------------------

def write_manifest(output_dir, words, word_lookup, word_reefs,
                   reef_records, island_records, compounds, reef_edges,
                   sub_reef_records):
    """Write manifest.json with checksums and stats."""
    import json

    checksums = {}
    for fname in EXPORT_FILES:
        path = os.path.join(output_dir, fname)
        checksums[fname] = sha256_file(path)

    n_archs = len(set(ir["arch_id"] for ir in island_records))
    manifest = {
        "version": FORMAT_VERSION,
        "format": "msgpack",
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": checksums,
        "stats": {
            "n_reefs": len(reef_records),
            "n_islands": len(island_records),
            "n_archs": n_archs,
            "n_words": len(words),
            "n_lookup_entries": len(word_lookup),
            "n_words_with_reefs": len(word_reefs),
            "n_compounds": len(compounds),
            "n_sub_reefs": len(sub_reef_records) if sub_reef_records else 0,
            "n_edges": len(reef_edges),
        },
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Wrote manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_export(output_dir, manifest, word_reefs, reef_records):
    """Post-export validation."""
    import json
    import msgpack as mp

    print("\n=== Verification ===")
    errors = 0

    # 1. Reload each .bin file
    print("  Checking deserialization...")
    for fname in EXPORT_FILES:
        path = os.path.join(output_dir, fname)
        try:
            with open(path, "rb") as f:
                mp.unpack(f, raw=False, strict_map_key=False)
        except Exception as e:
            print(f"    FAIL: {fname} — {e}")
            errors += 1
        else:
            print(f"    OK: {fname}")

    # 2. Verify checksums
    print("  Checking checksums...")
    for fname, expected_hash in manifest["files"].items():
        actual_hash = sha256_file(os.path.join(output_dir, fname))
        if actual_hash != expected_hash:
            print(f"    FAIL: {fname} checksum mismatch")
            errors += 1
        else:
            print(f"    OK: {fname}")

    # 3. Check reef_meta fields
    print("  Checking reef_meta...")
    with open(os.path.join(output_dir, "reef_meta.bin"), "rb") as f:
        rm_loaded = mp.unpack(f, raw=False, strict_map_key=False)
    if rm_loaded:
        required = {"hierarchy_addr", "n_words", "name", "valence",
                     "avg_specificity", "noun_frac", "verb_frac",
                     "adj_frac", "adv_frac"}
        missing = required - set(rm_loaded[0].keys())
        if missing:
            print(f"    FAIL: reef_meta[0] missing keys: {missing}")
            errors += 1
        else:
            print(f"    OK: reef_meta has all required fields ({len(rm_loaded)} reefs)")

    # 4. Check hierarchy_addr decoding
    print("  Checking hierarchy_addr v6 decoding...")
    for i, rm in enumerate(rm_loaded[:5]):
        addr = rm["hierarchy_addr"]
        island_id = addr & 0xFFFF
        arch_id = (addr >> 16) & 0xFFFF
        print(f"    reef[{i}] '{rm['name'][:30]}': arch={arch_id}, island={island_id}")

    # 5. Check constants for domainless_word_ids
    with open(os.path.join(output_dir, "constants.bin"), "rb") as f:
        constants = mp.unpack(f, raw=False, strict_map_key=False)
    n_domainless = len(constants.get("domainless_word_ids", []))
    print(f"  domainless_word_ids in constants: {n_domainless}")

    # 6. Sanity counts
    print("  Checking counts...")
    stats = manifest["stats"]
    for key, limit, desc in [
        ("n_reefs", 65535, "u16"),
        ("n_archs", 65535, "u16"),
    ]:
        val = stats[key]
        if val <= 0:
            print(f"    FAIL: {key}={val} (must be > 0)")
            errors += 1
        elif val > limit:
            print(f"    FAIL: {key}={val} exceeds {desc} limit ({limit})")
            errors += 1
        else:
            print(f"    OK: {key}={val}")

    if errors > 0:
        print(f"\n  {errors} verification errors!")
    else:
        print(f"\n  All checks passed.")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export(args):
    """Run the full v2 export pipeline."""
    start = time.time()
    print(f"Exporting from {DB_PATH} to {args.output}/\n")

    con = db.get_connection(DB_PATH)
    db.create_scoring_schema(con)

    # Phase 1: Hierarchy
    print("Phase 1: Loading hierarchy...")
    reef_records, island_records, id_remap = load_hierarchy(con)

    # Phase 2: Words
    print("Phase 2: Loading words...")
    words = load_words(con)

    # Phase 3: Word variants
    print("Phase 3: Loading word variants...")
    word_variants = load_word_variants(con)

    # Phase 4: Snowball stems
    print("Phase 4: Expanding snowball stems...")
    snowball_stems = expand_snowball_stems(words, word_variants)

    # Phase 5: Word-reefs + sub-reefs + domainless
    print("Phase 5: Building word-reefs from domain_word_scores...")
    word_reefs, reef_stats, sub_reef_records, domainless_word_ids = \
        build_word_reefs(con, id_remap)

    # Phase 6: Word lookup
    print("Phase 6: Building word_lookup...")
    word_lookup = build_word_lookup(words, word_variants, snowball_stems)

    # Phase 7: Compounds
    print("Phase 7: Extracting compounds...")
    compounds = extract_compounds(con)

    con.close()

    # Phase 8: Background model
    print("Phase 8: Computing background model...")
    bg_mean, bg_std = compute_background_model(
        word_reefs, words, n_reefs=len(reef_records),
        n_samples=args.bg_samples,
        words_per_sample=args.bg_words,
        seed=args.seed,
    )

    # Phase 9: Adjust background model
    print("Phase 9: Adjusting background model...")
    vocab_size = sum(1 for wid, info in words.items()
                     if info["word_count"] == 1 and wid in word_reefs)
    bg_std = adjust_background_model(
        bg_mean, bg_std, reef_records, reef_stats,
        vocab_size=vocab_size,
        n_samples=args.bg_samples,
        words_per_sample=args.bg_words,
    )

    # Phase 10: Reef edges (none for domain-level scoring)
    reef_edges = []

    # Phase 11: word_reef_detail (empty — each word has a single sub-reef per domain)
    word_reef_detail = None

    # Phase 12: Serialize
    print("Phase 12: Writing data files...")

    # Inject domainless_word_ids into constants before write
    # We override write_all_files' constants to include domainless
    avg_reef_words = sum(s["n_words"] for s in reef_stats.values()) / len(reef_stats) if reef_stats else 0.0
    print(f"  avg_reef_words = {avg_reef_words:.1f}")

    write_all_files(
        args.output, word_lookup, word_reefs, words,
        reef_records, island_records, bg_mean, bg_std,
        compounds, reef_stats, avg_reef_words, reef_edges,
        sub_reef_records=sub_reef_records,
        word_reef_detail=word_reef_detail,
    )

    # Patch constants.bin to add domainless_word_ids
    _patch_constants_with_domainless(args.output, domainless_word_ids)

    # Phase 13: Manifest
    print("Phase 13: Writing manifest...")
    manifest = write_manifest(
        args.output, words, word_lookup, word_reefs,
        reef_records, island_records, compounds, reef_edges,
        sub_reef_records,
    )

    elapsed = time.time() - start
    import json
    print(f"\nExport complete in {elapsed:.1f}s")
    print(f"  Stats: {json.dumps(manifest['stats'], indent=4)}")

    # Verification
    if args.verify:
        verify_errors = verify_export(
            args.output, manifest, word_reefs, reef_records,
        )
        if verify_errors > 0:
            sys.exit(1)

    # Copy to lagoon
    lagoon_dir = os.path.normpath(LAGOON_DATA_DIR)
    if os.path.isdir(lagoon_dir):
        print(f"\nCopying export files to {lagoon_dir}/...")
        for fname in EXPORT_FILES + ["manifest.json"]:
            src = os.path.join(args.output, fname)
            shutil.copy2(src, lagoon_dir)
        print(f"  Copied {len(EXPORT_FILES) + 1} files")
    else:
        print(f"\nWARNING: lagoon data dir not found at {lagoon_dir}, skipping copy")


def _patch_constants_with_domainless(output_dir, domainless_word_ids):
    """Re-read constants.bin, add domainless_word_ids, re-write."""
    import msgpack as mp

    path = os.path.join(output_dir, "constants.bin")
    with open(path, "rb") as f:
        constants = mp.unpackb(f.read(), raw=False)
    constants["domainless_word_ids"] = domainless_word_ids
    write_msgpack(constants, path)
    print(f"  Patched constants.bin with {len(domainless_word_ids)} domainless word_ids")


def main():
    parser = argparse.ArgumentParser(
        description="Export v2 data to lagoon format (v6.0)"
    )
    parser.add_argument(
        "--output", default="./v2_export",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--bg-samples", type=int, default=1000,
        help="Background model sample count (default: %(default)s)",
    )
    parser.add_argument(
        "--bg-words", type=int, default=15,
        help="Words per background sample (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run post-export validation",
    )
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
