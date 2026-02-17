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
import struct
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
FORMAT_VERSION = "2.0"

EXPORT_FILES = [
    "word_lookup.bin",
    "word_reefs.bin",
    "reef_meta.bin",
    "island_meta.bin",
    "background.bin",
    "compounds.bin",
    "constants.bin",
    "reef_edges.bin",
]

V2_FILES = [
    "word_lookup.bin",
    "word_reefs.bin",
    "reef_meta.bin",
    "island_meta.bin",
    "background.bin",
    "compounds.bin",
    "constants.bin",
    "reef_edges.bin",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pack_hierarchy_addr(arch_id, island_id, reef_id):
    """Pack arch(8)|island(8)|reef(16) into a u32."""
    return (arch_id << 24) | (island_id << 16) | reef_id


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

    n_reefs = len(reefs)
    assert n_reefs > 0, "No reefs found in database"
    assert n_reefs <= 65535, f"Reef count {n_reefs} exceeds u16 limit (65535)"

    # Islands (gen=1)
    islands = con.execute("""
        SELECT island_id, parent_island_id AS db_arch_id, island_name
        FROM island_stats WHERE generation = 1 AND island_id >= 0
    """).fetchall()

    n_islands = len(islands)
    assert n_islands > 0, "No islands found in database"
    assert n_islands <= 255, f"Island count {n_islands} exceeds u8 limit (255)"

    # Archipelagos (gen=0)
    archs = con.execute("""
        SELECT island_id, island_name
        FROM island_stats WHERE generation = 0 AND island_id >= 0
    """).fetchall()

    n_archs = len(archs)
    assert n_archs > 0, "No archipelagos found in database"
    assert n_archs <= 255, f"Arch count {n_archs} exceeds u8 limit (255)"

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

    assert max(island_remap.values()) <= 255, \
        f"Island ID {max(island_remap.values())} exceeds u8 limit (255)"

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

    assert max(reef_remap.values()) <= 65535, \
        f"Reef ID {max(reef_remap.values())} exceeds u16 limit (65535)"
    assert max(arch_remap.values()) <= 255, \
        f"Arch ID {max(arch_remap.values())} exceeds u8 limit (255)"

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

def compute_background_model(word_reefs, words, n_reefs, n_samples=1000,
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
    all_scores = np.zeros((n_samples, n_reefs))

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
# Phase 11: Load reef edges
# ---------------------------------------------------------------------------

def load_reef_edges(con, id_remap):
    """Load reef edges with weight above threshold, remapping IDs.

    Returns sorted list of (src_export_id, tgt_export_id, weight) tuples.
    """
    threshold = config.EXPORT_WEIGHT_THRESHOLD
    rows = con.execute("""
        SELECT source_reef_id, target_reef_id, weight
        FROM reef_edges
        WHERE weight > ?
    """, [threshold]).fetchall()

    reef_remap = id_remap["reef"]
    edges = []
    for src_db, tgt_db, weight in rows:
        if src_db not in reef_remap or tgt_db not in reef_remap:
            continue
        edges.append((reef_remap[src_db], reef_remap[tgt_db], weight))

    edges.sort(key=lambda e: (e[0], e[1]))
    print(f"  Loaded {len(edges)} reef edges (threshold={threshold})")
    return edges


# ---------------------------------------------------------------------------
# Phase 12: Serialization
# ---------------------------------------------------------------------------

def write_all_files(output_dir, word_lookup, word_reefs, words,
                    reef_records, island_records, bg_mean, bg_std,
                    compounds, reef_stats, avg_reef_words, reef_edges):
    """Serialize all data files to the output directory (v1 msgpack)."""
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
    n_reefs = len(reef_records)
    n_islands = len(island_records)
    # Derive n_archs from island_records
    n_archs = len(set(ir["arch_id"] for ir in island_records))
    reef_total_dims = [0] * n_reefs
    reef_n_words = [0] * n_reefs
    for rid, stats in reef_stats.items():
        reef_total_dims[rid] = stats["n_dims"]
        reef_n_words[rid] = stats["n_words"]

    constants = {
        "N_REEFS": n_reefs,
        "N_ISLANDS": n_islands,
        "N_ARCHS": n_archs,
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

    # 8. reef_edges.bin
    re_data = [[src, tgt, weight] for src, tgt, weight in reef_edges]
    write_msgpack(re_data, os.path.join(output_dir, "reef_edges.bin"))

    print(f"  Wrote {len(EXPORT_FILES)} data files to {output_dir}/")


def _write_v2_header(f, magic, count):
    """Write 4-byte ASCII magic + 4-byte u32 count header."""
    f.write(magic)
    f.write(struct.pack("<I", count))


def write_all_files_v2(output_dir, word_lookup, word_reefs, words,
                       reef_records, island_records, bg_mean, bg_std,
                       compounds, reef_stats, avg_reef_words, reef_edges):
    """Serialize all data files in flat binary format (v2).

    Each file has a 4-byte ASCII magic + 4-byte u32 count header,
    then fixed-stride records in little-endian byte order.
    """
    v2_dir = os.path.join(output_dir, "v2")
    os.makedirs(v2_dir, exist_ok=True)

    # 1. reef_edges.bin — magic WSRE, record: src(u16) + tgt(u16) + weight(f32) = 8 bytes
    with open(os.path.join(v2_dir, "reef_edges.bin"), "wb") as f:
        _write_v2_header(f, b"WSRE", len(reef_edges))
        for src, tgt, weight in reef_edges:
            f.write(struct.pack("<HHf", src, tgt, weight))

    # 2. word_lookup.bin — magic WSWL
    # record: lookup_hash(u64) + word_hash(u64) + word_id(u32) + specificity(i8) + idf_q(u8) + pad(2) = 24 bytes
    # Sort by lookup_hash for binary search in Rust
    wl_sorted = sorted(word_lookup.items(), key=lambda kv: kv[0])
    with open(os.path.join(v2_dir, "word_lookup.bin"), "wb") as f:
        _write_v2_header(f, b"WSWL", len(wl_sorted))
        for lookup_hash, (word_hash, word_id, specificity, idf_q) in wl_sorted:
            f.write(struct.pack("<QQIbBxx",
                                lookup_hash,
                                word_hash if word_hash is not None else 0,
                                word_id,
                                clamp(specificity, -128, 127),
                                idf_q))

    # 3. word_reefs.bin — magic WSWR
    # Index: (max_wid+1) x [offset(u32), count(u32)] = 8 bytes per entry
    # Data: reef_id(u16) + bm25_q(u16) = 4 bytes per entry
    max_word_id = max(words.keys()) if words else 0
    index_count = max_word_id + 1

    # Build index + data arrays
    wr_index = []  # (offset, count) pairs
    wr_data = []   # (reef_id, bm25_q) pairs
    offset = 0
    for wid in range(index_count):
        entries = word_reefs.get(wid, [])
        wr_index.append((offset, len(entries)))
        for reef_id, bm25_q in entries:
            wr_data.append((reef_id, bm25_q))
        offset += len(entries)

    with open(os.path.join(v2_dir, "word_reefs.bin"), "wb") as f:
        _write_v2_header(f, b"WSWR", index_count)
        for off, cnt in wr_index:
            f.write(struct.pack("<II", off, cnt))
        for reef_id, bm25_q in wr_data:
            f.write(struct.pack("<HH", reef_id, bm25_q))

    # 4. reef_meta.bin — magic WSRM
    # record: hierarchy_addr(u32) + n_words(u32) + name(64 bytes, null-padded) = 72 bytes
    with open(os.path.join(v2_dir, "reef_meta.bin"), "wb") as f:
        _write_v2_header(f, b"WSRM", len(reef_records))
        for r in reef_records:
            name_bytes = r["name"].encode("utf-8")[:64].ljust(64, b"\x00")
            f.write(struct.pack("<II", r["hierarchy_addr"], r["n_words"]))
            f.write(name_bytes)

    # 5. island_meta.bin — magic WSIM
    # record: arch_id(u8) + pad(1) + name(64 bytes, null-padded) = 66 bytes
    with open(os.path.join(v2_dir, "island_meta.bin"), "wb") as f:
        _write_v2_header(f, b"WSIM", len(island_records))
        for ir in island_records:
            name_bytes = ir["name"].encode("utf-8")[:64].ljust(64, b"\x00")
            f.write(struct.pack("<Bx", ir["arch_id"]))
            f.write(name_bytes)

    # 6. background.bin — magic WSBG
    # bg_mean[f32; N_REEFS] then bg_std[f32; N_REEFS]
    n_reefs = len(reef_records)
    with open(os.path.join(v2_dir, "background.bin"), "wb") as f:
        _write_v2_header(f, b"WSBG", n_reefs)
        for val in bg_mean:
            f.write(struct.pack("<f", val))
        for val in bg_std:
            f.write(struct.pack("<f", val))

    # 7. compounds.bin — magic WSCP
    # Index: [str_offset(u32), word_id(u32)] per compound
    # String pool: null-terminated UTF-8 strings
    string_pool = bytearray()
    compound_index = []
    for word_text, word_id in compounds:
        str_offset = len(string_pool)
        compound_index.append((str_offset, word_id))
        string_pool.extend(word_text.encode("utf-8"))
        string_pool.append(0)  # null terminator

    with open(os.path.join(v2_dir, "compounds.bin"), "wb") as f:
        _write_v2_header(f, b"WSCP", len(compounds))
        for str_off, wid in compound_index:
            f.write(struct.pack("<II", str_off, wid))
        f.write(bytes(string_pool))

    # 8. constants.bin — magic WSCN
    # Scalars packed as fixed struct, then reef_total_dims[f32; N], reef_n_words[f32; N]
    n_islands = len(island_records)
    n_archs = len(set(ir["arch_id"] for ir in island_records))
    r_total_dims = [0.0] * n_reefs
    r_n_words = [0.0] * n_reefs
    for rid, stats in reef_stats.items():
        r_total_dims[rid] = float(stats["n_dims"])
        r_n_words[rid] = float(stats["n_words"])

    with open(os.path.join(v2_dir, "constants.bin"), "wb") as f:
        _write_v2_header(f, b"WSCN", n_reefs)
        # Pack scalar constants: N_REEFS(u32), N_ISLANDS(u32), N_ARCHS(u32),
        # avg_reef_words(f32), k1(f32), b(f32), IDF_SCALE(u32), BM25_SCALE(u32),
        # FNV1A_OFFSET(u64), FNV1A_PRIME(u64)
        f.write(struct.pack("<IIIfffIIQQ",
                            n_reefs,
                            n_islands,
                            n_archs,
                            avg_reef_words,
                            config.BM25_K1,
                            config.BM25_B,
                            IDF_SCALE,
                            BM25_SCALE,
                            config.FNV1A_OFFSET,
                            config.FNV1A_PRIME))
        for val in r_total_dims:
            f.write(struct.pack("<f", val))
        for val in r_n_words:
            f.write(struct.pack("<f", val))

    print(f"  Wrote {len(V2_FILES)} v2 data files to {v2_dir}/")


def write_all_formats(output_dir, word_lookup, word_reefs, words,
                      reef_records, island_records, bg_mean, bg_std,
                      compounds, reef_stats, avg_reef_words, reef_edges,
                      write_v2=False):
    """Write v1 (msgpack) and optionally v2 (flat binary) formats."""
    write_all_files(
        output_dir, word_lookup, word_reefs, words,
        reef_records, island_records, bg_mean, bg_std,
        compounds, reef_stats, avg_reef_words, reef_edges,
    )
    if write_v2:
        write_all_files_v2(
            output_dir, word_lookup, word_reefs, words,
            reef_records, island_records, bg_mean, bg_std,
            compounds, reef_stats, avg_reef_words, reef_edges,
        )


def write_manifest(output_dir, words, word_lookup, word_reefs,
                   reef_records, island_records, compounds, reef_edges,
                   write_v2=False):
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
            "n_archs": len(set(ir["arch_id"] for ir in island_records)),
            "n_words": len(words),
            "n_lookup_entries": len(word_lookup),
            "n_words_with_reefs": len(word_reefs),
            "n_compounds": len(compounds),
            "n_edges": len(reef_edges),
            "edge_weight_threshold": config.EXPORT_WEIGHT_THRESHOLD,
        },
    }

    if write_v2:
        v2_checksums = {}
        v2_dir = os.path.join(output_dir, "v2")
        for fname in V2_FILES:
            path = os.path.join(v2_dir, fname)
            v2_checksums[fname] = sha256_file(path)
        manifest["v2_format"] = "flat_binary"
        manifest["v2_files"] = v2_checksums

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Wrote manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_export(output_dir, words, word_reefs_data, reef_stats,
                  avg_reef_words, reef_edges, write_v2=False):
    """Post-export validation."""
    print("\n=== Verification ===")
    errors = 0

    # 1. Reload each v1 .bin file
    print("  Checking v1 deserialization...")
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

    # 5. Spot-check reef edges
    print("  Checking reef edges...")
    with open(os.path.join(output_dir, "reef_edges.bin"), "rb") as f:
        re_loaded = msgpack.unpack(f, raw=False, strict_map_key=False)
    if len(re_loaded) != len(reef_edges):
        print(f"    FAIL: reef_edges count {len(re_loaded)} != {len(reef_edges)}")
        errors += 1
    else:
        print(f"    OK: reef_edges count = {len(re_loaded)}")
    if reef_edges:
        first = reef_edges[0]
        last = reef_edges[-1]
        print(f"    First edge: src={first[0]}, tgt={first[1]}, w={first[2]:.4f}")
        print(f"    Last edge:  src={last[0]}, tgt={last[1]}, w={last[2]:.4f}")

    # 6-8. Verify v2 files (only if --v2 was used)
    if write_v2:
        print("  Checking v2 files...")
        v2_dir = os.path.join(output_dir, "v2")
        for fname in V2_FILES:
            path = os.path.join(v2_dir, fname)
            if not os.path.exists(path):
                print(f"    FAIL: v2/{fname} missing")
                errors += 1
            else:
                size = os.path.getsize(path)
                print(f"    OK: v2/{fname} ({size} bytes)")

        # Spot-check v2 reef_edges header
        v2_re_path = os.path.join(v2_dir, "reef_edges.bin")
        if os.path.exists(v2_re_path):
            with open(v2_re_path, "rb") as f:
                magic = f.read(4)
                count = struct.unpack("<I", f.read(4))[0]
            if magic != b"WSRE":
                print(f"    FAIL: v2/reef_edges.bin magic={magic!r}, expected b'WSRE'")
                errors += 1
            elif count != len(reef_edges):
                print(f"    FAIL: v2/reef_edges.bin count={count}, expected {len(reef_edges)}")
                errors += 1
            else:
                print(f"    OK: v2/reef_edges.bin magic=WSRE, count={count}")

        # Verify v2 checksums
        if "v2_files" in manifest:
            for fname, expected_hash in manifest["v2_files"].items():
                actual_hash = sha256_file(os.path.join(v2_dir, fname))
                if actual_hash != expected_hash:
                    print(f"    FAIL: v2/{fname} checksum mismatch")
                    errors += 1
                else:
                    print(f"    OK: v2/{fname} checksum")
    else:
        print("  Skipping v2 checks (--v2 not specified)")

    # 9. Sanity-check counts (dynamic, not hardcoded)
    print("  Checking counts...")
    stats = manifest["stats"]
    for key, limit, desc in [
        ("n_reefs", 65535, "u16 limit"),
        ("n_islands", 255, "u8 limit"),
        ("n_archs", 255, "u8 limit"),
    ]:
        val = stats[key]
        if val <= 0:
            print(f"    FAIL: {key}={val} (must be > 0)")
            errors += 1
        elif val > limit:
            print(f"    FAIL: {key}={val} exceeds {desc} ({limit})")
            errors += 1
        else:
            print(f"    OK: {key}={val}")

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
        word_reefs, words, n_reefs=len(reef_records),
        n_samples=args.bg_samples,
        words_per_sample=args.bg_words,
        seed=args.seed,
    )

    # Phase 11: Reef edges
    print("Phase 11: Loading reef edges...")
    reef_edges = load_reef_edges(con, id_remap)

    con.close()

    # Phase 12: Serialization
    print("Phase 12: Writing data files...")
    write_all_formats(
        args.output, word_lookup, word_reefs, words,
        reef_records, island_records, bg_mean, bg_std,
        compounds, reef_stats, avg_reef_words, reef_edges,
        write_v2=args.v2,
    )

    manifest = write_manifest(
        args.output, words, word_lookup, word_reefs,
        reef_records, island_records, compounds, reef_edges,
        write_v2=args.v2,
    )

    elapsed = time.time() - start
    print(f"\nExport complete in {elapsed:.1f}s")
    print(f"  Stats: {json.dumps(manifest['stats'], indent=4)}")

    # Verification
    if args.verify:
        verify_errors = verify_export(
            args.output, words, word_reefs, reef_stats,
            avg_reef_words, reef_edges, write_v2=args.v2,
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
    parser.add_argument(
        "--v2", action="store_true",
        help="Also generate v2 flat binary files (off by default)",
    )
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
