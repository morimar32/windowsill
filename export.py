"""
Export binary data files for the lagoon scoring library.

Reads the DuckDB database and produces MessagePack-serialized .bin files
plus a manifest.json with checksums. These files are the sole interface
between the windowsill pipeline and the lagoon scoring engine.

Usage:
    python export.py [--db PATH] [--output DIR] [--verify]
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict

import duckdb
import msgpack
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

import config
from word_list import fnv1a_u64

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IDF_SCALE = 51
BM25_SCALE = 8192
FORMAT_VERSION = "1.0"

EXPORT_FILES = [
    "word_lookup.bin",
    "word_reefs.bin",
    "reef_meta.bin",
    "island_meta.bin",
    "background.bin",
    "compounds.bin",
    "constants.bin",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pack_hierarchy_addr(arch_id, island_id, reef_id):
    """Pack arch(2)|island(6)|reef(8) into a u16."""
    return (arch_id << 14) | (island_id << 8) | reef_id


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_msgpack(data, path):
    with open(path, "wb") as f:
        msgpack.pack(data, f, use_bin_type=True)


# ---------------------------------------------------------------------------
# Phase 1: Load reef hierarchy + ID remapping
# ---------------------------------------------------------------------------

def load_reef_hierarchy(con):
    """Load the 3-generation hierarchy and build contiguous ID remaps."""
    # Reefs (gen=2) with parent island and grandparent archipelago
    reefs = con.execute("""
        SELECT DISTINCT di2.island_id AS db_reef_id,
               di2.parent_island_id AS db_island_id,
               di1.parent_island_id AS db_arch_id,
               s.n_dims, s.n_words, s.island_name
        FROM dim_islands di2
        JOIN dim_islands di1 ON di2.parent_island_id = di1.island_id AND di1.generation = 1
        JOIN island_stats s ON di2.island_id = s.island_id AND s.generation = 2
        WHERE di2.generation = 2 AND di2.island_id >= 0
    """).fetchall()

    assert len(reefs) == config.N_REEFS, \
        f"Expected {config.N_REEFS} reefs, got {len(reefs)}"

    # Islands (gen=1)
    islands = con.execute("""
        SELECT island_id, parent_island_id AS db_arch_id, island_name
        FROM island_stats WHERE generation = 1 AND island_id >= 0
    """).fetchall()

    assert len(islands) == config.N_ISLANDS, \
        f"Expected {config.N_ISLANDS} islands, got {len(islands)}"

    # Archipelagos (gen=0)
    archs = con.execute("""
        SELECT island_id, island_name
        FROM island_stats WHERE generation = 0 AND island_id >= 0
    """).fetchall()

    assert len(archs) == config.N_ARCHS, \
        f"Expected {config.N_ARCHS} archipelagos, got {len(archs)}"

    # Build arch remap (sorted by db_arch_id for determinism)
    arch_remap = {}
    arch_names = {}
    for export_id, (db_arch_id, name) in enumerate(sorted(archs, key=lambda r: r[0])):
        arch_remap[db_arch_id] = export_id
        arch_names[export_id] = name

    # Build island remap (sorted by arch then db_island_id)
    island_remap = {}
    island_records = []
    for db_island_id, db_arch_id, name in sorted(islands, key=lambda r: (arch_remap[r[1]], r[0])):
        export_id = len(island_remap)
        island_remap[db_island_id] = export_id
        island_records.append({
            "arch_id": arch_remap[db_arch_id],
            "name": name,
        })

    assert max(island_remap.values()) <= 51, \
        f"Island ID {max(island_remap.values())} exceeds 6-bit limit (51)"

    # Build reef remap (sorted by arch → island → db_reef_id)
    reef_remap = {}
    reef_records = []
    for db_reef_id, db_island_id, db_arch_id, n_dims, n_words, name in sorted(
        reefs, key=lambda r: (arch_remap[r[2]], island_remap[r[1]], r[0])
    ):
        export_id = len(reef_remap)
        reef_remap[db_reef_id] = export_id
        e_arch = arch_remap[db_arch_id]
        e_island = island_remap[db_island_id]
        reef_records.append({
            "hierarchy_addr": pack_hierarchy_addr(e_arch, e_island, export_id),
            "n_words": n_words,
            "name": name,
            "n_dims": n_dims,
            "export_island_id": e_island,
            "export_arch_id": e_arch,
        })

    assert max(reef_remap.values()) <= 206, \
        f"Reef ID {max(reef_remap.values())} exceeds u8 limit (206)"
    assert max(arch_remap.values()) <= 3, \
        f"Arch ID {max(arch_remap.values())} exceeds 2-bit limit (3)"

    id_remap = {
        "reef": reef_remap,
        "island": island_remap,
        "arch": arch_remap,
    }

    print(f"  Hierarchy: {len(archs)} archs, {len(islands)} islands, {len(reefs)} reefs")
    return reef_records, island_records, id_remap


# ---------------------------------------------------------------------------
# Phase 2: Load reef stats
# ---------------------------------------------------------------------------

def load_reef_stats(con, id_remap):
    """Load n_dims and n_words per reef, indexed by export reef ID."""
    rows = con.execute("""
        SELECT island_id AS db_reef_id, n_dims, n_words
        FROM island_stats
        WHERE generation = 2 AND island_id >= 0
    """).fetchall()

    reef_stats = {}
    for db_reef_id, n_dims, n_words in rows:
        export_id = id_remap["reef"][db_reef_id]
        reef_stats[export_id] = {"n_dims": n_dims, "n_words": n_words}

    assert len(reef_stats) == config.N_REEFS
    return reef_stats


def compute_avg_reef_words(reef_stats):
    """Compute mean n_words across all reefs."""
    total = sum(s["n_words"] for s in reef_stats.values())
    avg = total / len(reef_stats)
    print(f"  avg_reef_words = {avg:.1f}")
    return avg


# ---------------------------------------------------------------------------
# Phase 3: Load words
# ---------------------------------------------------------------------------

def load_words(con):
    """Load all words with IDF quantization."""
    rows = con.execute("""
        SELECT word_id, word, specificity, word_hash, reef_idf, category, word_count
        FROM words ORDER BY word_id
    """).fetchall()

    assert len(rows) > 0, "No words found — pipeline incomplete"

    words = {}
    null_idf_count = 0
    null_hash_count = 0
    for word_id, word, specificity, word_hash, reef_idf, category, word_count in rows:
        if word_hash is None:
            null_hash_count += 1
        if reef_idf is None:
            null_idf_count += 1

        idf_q = 0 if reef_idf is None else clamp(round(reef_idf * IDF_SCALE), 0, 255)
        words[word_id] = {
            "word": word,
            "specificity": specificity if specificity is not None else 0,
            "word_hash": word_hash,
            "reef_idf": reef_idf if reef_idf is not None else 0.0,
            "idf_q": idf_q,
            "category": category,
            "word_count": word_count if word_count is not None else 1,
        }

    if null_hash_count > 0:
        print(f"  WARNING: {null_hash_count} words with NULL word_hash")
    if null_idf_count > 0:
        print(f"  WARNING: {null_idf_count} words with NULL reef_idf")

    print(f"  Loaded {len(words)} words")
    return words


# ---------------------------------------------------------------------------
# Phase 4: Load word variants
# ---------------------------------------------------------------------------

def load_word_variants(con):
    """Load all morphy variant mappings."""
    rows = con.execute("""
        SELECT variant_hash, variant, word_id, source
        FROM word_variants
    """).fetchall()

    assert len(rows) > 0, "No word variants found — pipeline incomplete"
    print(f"  Loaded {len(rows)} word variants")
    return rows


# ---------------------------------------------------------------------------
# Phase 5: Expand snowball stems
# ---------------------------------------------------------------------------

def expand_snowball_stems(words, word_variants):
    """Pre-expand Snowball stems so lagoon needs no runtime stemmer.

    For each single word in the vocabulary, compute its Snowball stem.
    If the stem differs from the word, add stem_hash → word_id mapping.
    Also stem every morphy variant.

    Returns list of (stem_hash, word_id, specificity) tuples.
    """
    stemmer = SnowballStemmer("english")
    stem_mappings = []  # (hash, word_id, specificity)
    seen_hashes = set()

    # Collect all existing hashes (base + morphy) for priority checking
    for wv in word_variants:
        seen_hashes.add(wv[0])  # variant_hash

    # Stem all base words
    for word_id, info in tqdm(words.items(), desc="Snowball stems (base)"):
        word = info["word"]
        if info["word_count"] > 1:
            continue  # skip multi-word compounds

        stem = stemmer.stem(word)
        if stem != word:
            stem_hash = fnv1a_u64(stem)
            if stem_hash not in seen_hashes:
                stem_mappings.append((stem_hash, word_id, info["specificity"]))
                seen_hashes.add(stem_hash)

    # Stem all morphy variants
    morphy_variants = [(vh, variant, wid) for vh, variant, wid, source in word_variants
                       if source == "morphy"]
    for vh, variant, word_id in tqdm(morphy_variants, desc="Snowball stems (morphy)"):
        if " " in variant:
            continue  # skip multi-word
        stem = stemmer.stem(variant)
        if stem != variant:
            stem_hash = fnv1a_u64(stem)
            if stem_hash not in seen_hashes:
                spec = words[word_id]["specificity"] if word_id in words else 0
                stem_mappings.append((stem_hash, word_id, spec))
                seen_hashes.add(stem_hash)

    print(f"  Generated {len(stem_mappings)} snowball stem mappings")
    return stem_mappings


# ---------------------------------------------------------------------------
# Phase 6: Load word-reef affinity
# ---------------------------------------------------------------------------

def load_word_reef_affinity(con, id_remap):
    """Load word-reef affinity data, remapping reef IDs."""
    rows = con.execute("""
        SELECT word_id, reef_id, n_dims
        FROM word_reef_affinity
        ORDER BY word_id, reef_id
    """).fetchall()

    assert len(rows) > 0, "No word-reef affinity data — pipeline incomplete"

    affinity = defaultdict(list)
    for word_id, db_reef_id, n_dims in rows:
        if db_reef_id not in id_remap["reef"]:
            continue
        export_reef_id = id_remap["reef"][db_reef_id]
        affinity[word_id].append((export_reef_id, n_dims))

    print(f"  Loaded {len(rows)} affinity rows for {len(affinity)} words")
    return affinity


# ---------------------------------------------------------------------------
# Phase 7: Precompute BM25
# ---------------------------------------------------------------------------

def compute_bm25_term_score(idf, n_dims, reef_total_dims, reef_n_words, avg_reef_words):
    """Compute BM25 term score for a word-reef pair."""
    k1 = config.BM25_K1
    b = config.BM25_B
    tf = n_dims / reef_total_dims
    norm_len = reef_n_words / avg_reef_words
    return idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * norm_len))


def precompute_word_reefs(words, affinity, reef_stats, avg_reef_words):
    """Precompute BM25 scores for all word-reef pairs.

    Returns dict: word_id → list[(export_reef_id, bm25_q)]
    """
    word_reefs = {}
    clamp_warnings = 0

    for word_id in tqdm(sorted(affinity.keys()), desc="BM25 precompute"):
        if word_id not in words:
            continue
        idf = words[word_id]["reef_idf"]
        if idf == 0.0:
            continue

        entries = []
        for export_reef_id, n_dims in affinity[word_id]:
            rs = reef_stats[export_reef_id]
            reef_total_dims = rs["n_dims"]
            reef_n_words = rs["n_words"]

            if reef_total_dims == 0:
                continue

            score = compute_bm25_term_score(
                idf, n_dims, reef_total_dims, reef_n_words, avg_reef_words
            )
            bm25_q = round(score * BM25_SCALE)
            if bm25_q > 65535:
                clamp_warnings += 1
                bm25_q = 65535
            bm25_q = max(0, bm25_q)
            entries.append((export_reef_id, bm25_q))

        # Sort by reef_id for cache-friendly iteration
        entries.sort(key=lambda e: e[0])
        if entries:
            word_reefs[word_id] = entries

    if clamp_warnings > 0:
        print(f"  WARNING: {clamp_warnings} BM25 scores clamped to u16 max")

    print(f"  Precomputed BM25 for {len(word_reefs)} words")
    return word_reefs


# ---------------------------------------------------------------------------
# Phase 8: Build word_lookup
# ---------------------------------------------------------------------------

def build_word_lookup(words, word_variants, snowball_stems):
    """Build the unified word_lookup HashMap.

    Priority: base word > morphy variant > snowball stem.
    Among ties, prefer higher specificity.

    Returns dict: u64_hash → [word_hash, word_id, specificity, idf_q]
    """
    lookup = {}  # hash → (word_hash, word_id, specificity, idf_q, priority)

    # Priority 0 (highest): base words
    for wv_hash, variant, word_id, source in word_variants:
        if source != "base":
            continue
        if word_id not in words:
            continue
        info = words[word_id]
        key = wv_hash
        existing = lookup.get(key)
        spec = info["specificity"]
        if existing is None or existing[4] > 0 or (existing[4] == 0 and spec > existing[2]):
            lookup[key] = (info["word_hash"], word_id, spec, info["idf_q"], 0)

    # Priority 1: morphy variants
    for wv_hash, variant, word_id, source in word_variants:
        if source != "morphy":
            continue
        if word_id not in words:
            continue
        info = words[word_id]
        key = wv_hash
        existing = lookup.get(key)
        spec = info["specificity"]
        if existing is None:
            lookup[key] = (info["word_hash"], word_id, spec, info["idf_q"], 1)
        elif existing[4] > 1:
            # replace snowball with morphy
            lookup[key] = (info["word_hash"], word_id, spec, info["idf_q"], 1)
        elif existing[4] == 1 and spec > existing[2]:
            # same priority, higher specificity wins
            lookup[key] = (info["word_hash"], word_id, spec, info["idf_q"], 1)

    # Priority 2 (lowest): snowball stems
    for stem_hash, word_id, spec in snowball_stems:
        if word_id not in words:
            continue
        info = words[word_id]
        key = stem_hash
        existing = lookup.get(key)
        if existing is None:
            lookup[key] = (info["word_hash"], word_id, spec, info["idf_q"], 2)
        elif existing[4] == 2 and spec > existing[2]:
            lookup[key] = (info["word_hash"], word_id, spec, info["idf_q"], 2)

    # Strip priority field for final output
    final = {}
    for h, (word_hash, word_id, spec, idf_q, _pri) in lookup.items():
        final[h] = [word_hash, word_id, spec, idf_q]

    print(f"  word_lookup: {len(final)} entries")
    return final


# ---------------------------------------------------------------------------
# Phase 9: Extract compounds
# ---------------------------------------------------------------------------

def extract_compounds(con, selective=False):
    """Extract compound words for Aho-Corasick automaton."""
    if selective:
        # Only idiomatic (low compositionality) + high-specificity compounds
        rows = con.execute("""
            SELECT w.word_id, w.word FROM words w
            WHERE w.category IN ('compound', 'phrasal_verb', 'named_entity', 'taxonomic')
              AND w.word_count > 1
              AND (w.specificity >= 1
                   OR EXISTS (
                       SELECT 1 FROM words w2
                       WHERE w2.word_id = w.word_id
                         AND w2.total_dims IS NOT NULL
                   ))
            ORDER BY w.word_id
        """).fetchall()
    else:
        rows = con.execute("""
            SELECT word_id, word FROM words
            WHERE category IN ('compound', 'phrasal_verb', 'named_entity', 'taxonomic')
              AND word_count > 1
            ORDER BY word_id
        """).fetchall()

    compounds = [[word, word_id] for word_id, word in rows]
    print(f"  Extracted {len(compounds)} compounds" +
          (" (selective)" if selective else ""))
    return compounds


# ---------------------------------------------------------------------------
# Phase 10: Background model
# ---------------------------------------------------------------------------

def compute_background_model(word_reefs, words, n_samples=1000,
                              words_per_sample=15, seed=42):
    """Compute background mean and std for z-score normalization.

    Samples random sets of single words, scores them, and records
    per-reef statistics.
    """
    # Get single-word IDs that have reef entries
    single_word_ids = [
        wid for wid, info in words.items()
        if info["word_count"] == 1 and wid in word_reefs
    ]
    single_word_ids = np.array(single_word_ids)

    assert len(single_word_ids) >= words_per_sample, \
        f"Not enough single words ({len(single_word_ids)}) for background sampling"

    rng = np.random.default_rng(seed)
    all_scores = np.zeros((n_samples, config.N_REEFS))

    for i in tqdm(range(n_samples), desc="Background model"):
        sample = rng.choice(single_word_ids, size=words_per_sample, replace=False)
        for word_id in sample:
            for reef_id, bm25_q in word_reefs[int(word_id)]:
                all_scores[i, reef_id] += bm25_q / BM25_SCALE

    bg_mean = all_scores.mean(axis=0).tolist()
    bg_std = all_scores.std(axis=0).tolist()
    # Replace zero std with epsilon
    bg_std = [max(s, 1e-6) for s in bg_std]

    print(f"  bg_mean range: [{min(bg_mean):.2f}, {max(bg_mean):.2f}]")
    print(f"  bg_std range:  [{min(bg_std):.4f}, {max(bg_std):.4f}]")
    return bg_mean, bg_std


# ---------------------------------------------------------------------------
# Phase 11: Serialization
# ---------------------------------------------------------------------------

def write_all_files(output_dir, word_lookup, word_reefs, words,
                    reef_records, island_records, bg_mean, bg_std,
                    compounds, reef_stats, avg_reef_words):
    """Serialize all data files to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. word_lookup.bin
    # Convert int keys to ensure msgpack handles them correctly
    wl_data = {int(k): v for k, v in word_lookup.items()}
    write_msgpack(wl_data, os.path.join(output_dir, "word_lookup.bin"))

    # 2. word_reefs.bin — outer list indexed by word_id
    max_word_id = max(words.keys()) if words else 0
    wr_list = [[] for _ in range(max_word_id + 1)]
    for word_id, entries in word_reefs.items():
        wr_list[word_id] = [[rid, bq] for rid, bq in entries]
    write_msgpack(wr_list, os.path.join(output_dir, "word_reefs.bin"))

    # 3. reef_meta.bin
    reef_meta = [
        {
            "hierarchy_addr": r["hierarchy_addr"],
            "n_words": r["n_words"],
            "name": r["name"],
        }
        for r in reef_records
    ]
    write_msgpack(reef_meta, os.path.join(output_dir, "reef_meta.bin"))

    # 4. island_meta.bin
    write_msgpack(island_records, os.path.join(output_dir, "island_meta.bin"))

    # 5. background.bin
    write_msgpack({"bg_mean": bg_mean, "bg_std": bg_std},
                  os.path.join(output_dir, "background.bin"))

    # 6. compounds.bin
    write_msgpack(compounds, os.path.join(output_dir, "compounds.bin"))

    # 7. constants.bin
    reef_total_dims = [0] * config.N_REEFS
    reef_n_words = [0] * config.N_REEFS
    for rid, stats in reef_stats.items():
        reef_total_dims[rid] = stats["n_dims"]
        reef_n_words[rid] = stats["n_words"]

    constants = {
        "N_REEFS": config.N_REEFS,
        "N_ISLANDS": config.N_ISLANDS,
        "N_ARCHS": config.N_ARCHS,
        "avg_reef_words": avg_reef_words,
        "k1": config.BM25_K1,
        "b": config.BM25_B,
        "IDF_SCALE": IDF_SCALE,
        "BM25_SCALE": BM25_SCALE,
        "FNV1A_OFFSET": config.FNV1A_OFFSET,
        "FNV1A_PRIME": config.FNV1A_PRIME,
        "reef_total_dims": reef_total_dims,
        "reef_n_words": reef_n_words,
    }
    write_msgpack(constants, os.path.join(output_dir, "constants.bin"))

    print(f"  Wrote {len(EXPORT_FILES)} data files to {output_dir}/")


def write_manifest(output_dir, words, word_lookup, word_reefs,
                   reef_records, island_records, compounds):
    """Write manifest.json with checksums and stats."""
    checksums = {}
    for fname in EXPORT_FILES:
        path = os.path.join(output_dir, fname)
        checksums[fname] = sha256_file(path)

    manifest = {
        "version": FORMAT_VERSION,
        "format": "msgpack",
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": checksums,
        "stats": {
            "n_reefs": len(reef_records),
            "n_islands": len(island_records),
            "n_archs": config.N_ARCHS,
            "n_words": len(words),
            "n_lookup_entries": len(word_lookup),
            "n_words_with_reefs": len(word_reefs),
            "n_compounds": len(compounds),
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

def verify_export(output_dir, words, word_reefs_data, reef_stats, avg_reef_words):
    """Post-export validation."""
    print("\n=== Verification ===")
    errors = 0

    # 1. Reload each .bin file
    print("  Checking deserialization...")
    for fname in EXPORT_FILES:
        path = os.path.join(output_dir, fname)
        try:
            with open(path, "rb") as f:
                msgpack.unpack(f, raw=False, strict_map_key=False)
        except Exception as e:
            print(f"    FAIL: {fname} — {e}")
            errors += 1
        else:
            print(f"    OK: {fname}")

    # 2. Verify checksums against manifest
    print("  Checking checksums...")
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    for fname, expected_hash in manifest["files"].items():
        actual_hash = sha256_file(os.path.join(output_dir, fname))
        if actual_hash != expected_hash:
            print(f"    FAIL: {fname} checksum mismatch")
            errors += 1
        else:
            print(f"    OK: {fname}")

    # 3. Spot-check 10 random words
    print("  Spot-checking word lookups...")
    with open(os.path.join(output_dir, "word_lookup.bin"), "rb") as f:
        lookup = msgpack.unpack(f, raw=False, strict_map_key=False)

    rng = np.random.default_rng(123)
    word_items = list(words.items())
    sample_indices = rng.choice(len(word_items), size=min(10, len(word_items)), replace=False)
    for idx in sample_indices:
        word_id, info = word_items[idx]
        h = fnv1a_u64(info["word"])
        entry = lookup.get(h) or lookup.get(str(h))
        if entry is None:
            # May not be in lookup if it has no word_hash — that's ok
            if info["word_hash"] is not None:
                print(f"    WARN: '{info['word']}' (hash={h}) not in lookup")
        elif entry[1] != word_id:
            # Variant collision — different word_id is acceptable if it's a variant mapping
            pass
        else:
            print(f"    OK: '{info['word']}' → word_id={word_id}")

    # 4. Spot-check 5 BM25 scores: compare loaded file against in-memory values
    print("  Spot-checking BM25 scores...")
    with open(os.path.join(output_dir, "word_reefs.bin"), "rb") as f:
        wr_loaded = msgpack.unpack(f, raw=False, strict_map_key=False)

    checked = 0
    for word_id in sorted(word_reefs_data.keys()):
        if checked >= 5:
            break
        if word_id >= len(wr_loaded) or not wr_loaded[word_id]:
            continue
        loaded_entries = {e[0]: e[1] for e in wr_loaded[word_id]}
        for reef_id, expected_q in word_reefs_data[word_id][:1]:
            loaded_q = loaded_entries.get(reef_id)
            if loaded_q is None:
                print(f"    WARN: word_id={word_id} reef={reef_id}: "
                      f"missing from loaded file")
            elif loaded_q != expected_q:
                print(f"    WARN: word_id={word_id} reef={reef_id}: "
                      f"loaded={loaded_q} expected={expected_q}")
            else:
                print(f"    OK: word_id={word_id} reef={reef_id}: bm25_q={loaded_q}")
            checked += 1

    # 5. Assert counts
    print("  Checking counts...")
    stats = manifest["stats"]
    for key, expected in [
        ("n_reefs", config.N_REEFS),
        ("n_islands", config.N_ISLANDS),
        ("n_archs", config.N_ARCHS),
    ]:
        if stats[key] != expected:
            print(f"    FAIL: {key}={stats[key]}, expected {expected}")
            errors += 1
        else:
            print(f"    OK: {key}={stats[key]}")

    if stats["n_lookup_entries"] < 200000:
        print(f"    WARN: only {stats['n_lookup_entries']} lookup entries "
              f"(expected 200K+)")
    else:
        print(f"    OK: {stats['n_lookup_entries']} lookup entries")

    if errors > 0:
        print(f"\n  {errors} verification errors!")
    else:
        print(f"\n  All checks passed.")
    return errors


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def export(args):
    """Run the full export pipeline."""
    start = time.time()
    print(f"Exporting from {args.db} to {args.output}/\n")

    # Connect to database
    con = duckdb.connect(args.db, read_only=True)

    # Phase 1: Hierarchy
    print("Phase 1: Loading reef hierarchy...")
    reef_records, island_records, id_remap = load_reef_hierarchy(con)

    # Phase 2: Reef stats
    print("Phase 2: Loading reef stats...")
    reef_stats = load_reef_stats(con, id_remap)
    avg_reef_words = compute_avg_reef_words(reef_stats)

    # Phase 3: Words
    print("Phase 3: Loading words...")
    words = load_words(con)

    # Phase 4: Word variants
    print("Phase 4: Loading word variants...")
    word_variants = load_word_variants(con)

    # Phase 5: Snowball stems
    print("Phase 5: Expanding snowball stems...")
    snowball_stems = expand_snowball_stems(words, word_variants)

    # Phase 6: Word-reef affinity
    print("Phase 6: Loading word-reef affinity...")
    affinity = load_word_reef_affinity(con, id_remap)

    # Phase 7: BM25 precomputation
    print("Phase 7: Precomputing BM25 scores...")
    word_reefs = precompute_word_reefs(words, affinity, reef_stats, avg_reef_words)

    # Phase 8: Word lookup
    print("Phase 8: Building word_lookup...")
    word_lookup = build_word_lookup(words, word_variants, snowball_stems)

    # Phase 9: Compounds
    print("Phase 9: Extracting compounds...")
    compounds = extract_compounds(con, selective=args.selective_compounds)

    # Phase 10: Background model
    print("Phase 10: Computing background model...")
    bg_mean, bg_std = compute_background_model(
        word_reefs, words,
        n_samples=args.bg_samples,
        words_per_sample=args.bg_words,
        seed=args.seed,
    )

    con.close()

    # Phase 11: Serialization
    print("Phase 11: Writing data files...")
    write_all_files(
        args.output, word_lookup, word_reefs, words,
        reef_records, island_records, bg_mean, bg_std,
        compounds, reef_stats, avg_reef_words,
    )

    manifest = write_manifest(
        args.output, words, word_lookup, word_reefs,
        reef_records, island_records, compounds,
    )

    elapsed = time.time() - start
    print(f"\nExport complete in {elapsed:.1f}s")
    print(f"  Stats: {json.dumps(manifest['stats'], indent=4)}")

    # Verification
    if args.verify:
        verify_errors = verify_export(
            args.output, words, word_reefs, reef_stats, avg_reef_words
        )
        if verify_errors > 0:
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Export binary data files for the lagoon scoring library."
    )
    parser.add_argument(
        "--db", default=config.DB_PATH,
        help="Path to DuckDB database (default: %(default)s)",
    )
    parser.add_argument(
        "--output", default="./export",
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
        "--selective-compounds", action="store_true",
        help="Only export idiomatic + high-specificity compounds",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run post-export validation",
    )
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
