"""Export v3 data to lagoon format (v3.1).

Reads from v3/windowsill.db and produces MessagePack-serialized .bin files
plus a manifest.json.  The 4-tier hierarchy (archipelago > island > town > reef)
replaces v2's flat domain-as-reef structure.

Exports both island-level (40 topical) and town-level (298 topical) word
association files.  word_towns.bin is the primary scoring unit for lagoon;
word_islands.bin is kept for rollup.  Bucket islands (is_bucket=1) are
carved out into a separate file.

Self-contained — no imports from v2 code.

Usage:
    python v3/export.py                   # export only
    python v3/export.py --verify          # export + verify checksums
    python v3/export.py --output DIR      # custom output directory
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict

import msgpack
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
V3_DB = os.path.join(_HERE, "windowsill.db")
DEFAULT_OUTPUT = os.path.join(_HERE, "exports")
FORMAT_VERSION = "3.1"

IDF_SCALE = 21          # max idf ~11.94 → round(11.94*21) = 251, fits u8
WEIGHT_SCALE = 255.0

FNV1A_OFFSET = 14695981039346656037
FNV1A_PRIME = 1099511628211

BG_STD_FLOOR = 0.1

EXPORT_FILES = [
    "word_lookup.bin",
    "word_islands.bin",
    "word_towns.bin",
    "island_meta.bin",
    "background.bin",
    "constants.bin",
    "compounds.bin",
    "word_detail.bin",
    "town_meta.bin",
    "reef_meta.bin",
    "bucket_words.bin",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def fnv1a_u64(s):
    """FNV-1a u64 hash of a string."""
    h = FNV1A_OFFSET
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * FNV1A_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


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
# Phase 1: Load hierarchy + build ID remaps
# ---------------------------------------------------------------------------

def load_hierarchy(con):
    """Load 4-tier hierarchy and build contiguous 0-based ID remaps.

    Returns (arch_remap, island_remap, town_remap, reef_remap,
             bucket_island_remap, arch_rows, island_rows, town_rows, reef_rows,
             bucket_island_rows).
    """
    # Archipelagos
    arch_rows = con.execute(
        "SELECT archipelago_id, name FROM Archipelagos ORDER BY archipelago_id"
    ).fetchall()
    arch_remap = {row[0]: i for i, row in enumerate(arch_rows)}

    # Topical islands (is_bucket=0)
    island_rows = con.execute(
        "SELECT island_id, archipelago_id, name FROM Islands "
        "WHERE is_bucket = 0 ORDER BY island_id"
    ).fetchall()
    island_remap = {row[0]: i for i, row in enumerate(island_rows)}

    # Bucket islands (is_bucket=1)
    bucket_island_rows = con.execute(
        "SELECT island_id, archipelago_id, name FROM Islands "
        "WHERE is_bucket = 1 ORDER BY island_id"
    ).fetchall()
    bucket_island_remap = {row[0]: i for i, row in enumerate(bucket_island_rows)}

    # Towns in topical islands only
    town_rows = con.execute("""
        SELECT t.town_id, t.island_id, t.name
        FROM Towns t JOIN Islands i USING(island_id)
        WHERE i.is_bucket = 0
        ORDER BY t.town_id
    """).fetchall()
    town_remap = {row[0]: i for i, row in enumerate(town_rows)}

    # Reefs in topical islands only
    reef_rows = con.execute("""
        SELECT r.reef_id, r.town_id, r.name
        FROM Reefs r
        JOIN Towns t USING(town_id)
        JOIN Islands i USING(island_id)
        WHERE i.is_bucket = 0
        ORDER BY r.reef_id
    """).fetchall()
    reef_remap = {row[0]: i for i, row in enumerate(reef_rows)}

    print(f"  {len(arch_rows)} archipelagos, {len(island_rows)} topical islands, "
          f"{len(bucket_island_rows)} bucket islands")
    print(f"  {len(town_rows)} topical towns, {len(reef_rows)} topical reefs")

    return (arch_remap, island_remap, town_remap, reef_remap,
            bucket_island_remap, arch_rows, island_rows, town_rows, reef_rows,
            bucket_island_rows)


# ---------------------------------------------------------------------------
# Phase 2: Build word_lookup
# ---------------------------------------------------------------------------

def build_word_lookup(con):
    """Build word_lookup: {u64_hash: [word_hash, word_id, specificity, idf_q]}.

    All words included (topical, bucket-only, and domainless).
    If WordVariants is populated, adds morphy (priority 1) and snowball (priority 2).
    Otherwise, base words only (priority 0).
    """
    rows = con.execute(
        "SELECT word_id, word, word_hash, specificity, idf FROM Words"
    ).fetchall()

    # Check if WordVariants is populated
    n_variants = con.execute("SELECT COUNT(*) FROM WordVariants").fetchone()[0]

    # Build base lookup (priority 0)
    lookup = {}  # u64_hash → (word_hash_u64, word_id, specificity, idf_q, priority)
    words_info = {}  # word_id → {word, word_hash, specificity, idf_q, word_count}

    for word_id, word, word_hash_i64, specificity, idf in rows:
        # Convert signed i64 → unsigned u64
        word_hash = word_hash_i64 & 0xFFFFFFFFFFFFFFFF
        spec = specificity if specificity is not None else 0
        idf_q = clamp(round((idf or 0) * IDF_SCALE), 0, 255)

        words_info[word_id] = {
            "word": word,
            "word_hash": word_hash,
            "specificity": spec,
            "idf_q": idf_q,
        }

        existing = lookup.get(word_hash)
        if existing is None or (existing[4] > 0) or (existing[4] == 0 and spec > existing[2]):
            lookup[word_hash] = (word_hash, word_id, spec, idf_q, 0)

    if n_variants > 0:
        # Add morphy variants (priority 1)
        morphy_rows = con.execute(
            "SELECT variant_hash, word_id FROM WordVariants WHERE source = 'morphy'"
        ).fetchall()
        for vh_i64, word_id in morphy_rows:
            if word_id not in words_info:
                continue
            vh = vh_i64 & 0xFFFFFFFFFFFFFFFF
            info = words_info[word_id]
            existing = lookup.get(vh)
            if existing is None:
                lookup[vh] = (info["word_hash"], word_id, info["specificity"], info["idf_q"], 1)
            elif existing[4] > 1:
                lookup[vh] = (info["word_hash"], word_id, info["specificity"], info["idf_q"], 1)
            elif existing[4] == 1 and info["specificity"] > existing[2]:
                lookup[vh] = (info["word_hash"], word_id, info["specificity"], info["idf_q"], 1)

        # Add snowball variants (priority 2)
        snowball_rows = con.execute(
            "SELECT variant_hash, word_id FROM WordVariants WHERE source = 'snowball'"
        ).fetchall()
        for vh_i64, word_id in snowball_rows:
            if word_id not in words_info:
                continue
            vh = vh_i64 & 0xFFFFFFFFFFFFFFFF
            info = words_info[word_id]
            existing = lookup.get(vh)
            if existing is None:
                lookup[vh] = (info["word_hash"], word_id, info["specificity"], info["idf_q"], 2)
            elif existing[4] == 2 and info["specificity"] > existing[2]:
                lookup[vh] = (info["word_hash"], word_id, info["specificity"], info["idf_q"], 2)

        print(f"  word_lookup: {len(lookup)} entries "
              f"(base + {len(morphy_rows)} morphy + {len(snowball_rows)} snowball)")
    else:
        print(f"  word_lookup: {len(lookup)} entries (base words only, WordVariants empty)")

    # Strip priority field
    final = {h: [wh, wid, spec, idfq]
             for h, (wh, wid, spec, idfq, _) in lookup.items()}
    return final, words_info


# ---------------------------------------------------------------------------
# Phase 3: Build word_islands (topical) + partition words
# ---------------------------------------------------------------------------

def build_word_islands(con, island_remap, town_remap, reef_remap):
    """Build word_islands: sparse list word_id → [[export_island_id, weight_q]].

    Queries all three export tables for topical islands, groups by
    (word_id, island_id), takes MAX(export_weight).

    Returns (word_islands, topical_word_ids).
    """
    # Reef-level exports → island
    reef_rows = con.execute("""
        SELECT e.word_id, t.island_id, e.export_weight
        FROM ReefWordExports e
        JOIN Reefs r ON e.reef_id = r.reef_id
        JOIN Towns t ON r.town_id = t.town_id
        JOIN Islands i ON t.island_id = i.island_id
        WHERE i.is_bucket = 0
    """).fetchall()

    # Town-level exports → island
    town_rows = con.execute("""
        SELECT e.word_id, t.island_id, e.export_weight
        FROM TownWordExports e
        JOIN Towns t ON e.town_id = t.town_id
        JOIN Islands i ON t.island_id = i.island_id
        WHERE i.is_bucket = 0
    """).fetchall()

    # Island-level exports (already at island)
    island_rows = con.execute("""
        SELECT e.word_id, e.island_id, e.export_weight
        FROM IslandWordExports e
        JOIN Islands i ON e.island_id = i.island_id
        WHERE i.is_bucket = 0
    """).fetchall()

    # Group by (word_id, island_id), take MAX
    word_island_max = defaultdict(lambda: defaultdict(int))
    for word_id, db_island_id, weight in reef_rows + town_rows + island_rows:
        if db_island_id not in island_remap:
            continue
        eid = island_remap[db_island_id]
        if weight > word_island_max[word_id][eid]:
            word_island_max[word_id][eid] = weight

    # Build sparse list
    word_islands = {}
    for word_id, island_weights in word_island_max.items():
        entries = sorted([[iid, w] for iid, w in island_weights.items()], key=lambda x: x[0])
        word_islands[word_id] = entries

    topical_word_ids = set(word_islands.keys())
    print(f"  word_islands: {len(word_islands)} words with topical island entries")
    return word_islands, topical_word_ids


# ---------------------------------------------------------------------------
# Phase 3b: Build word_towns (topical)
# ---------------------------------------------------------------------------

def build_word_towns(con, island_remap, town_remap, reef_remap):
    """Build word_towns: sparse list word_id → [[export_town_id, weight_q]].

    Aggregates all three export tables to town level:
    - ReefWordExports: reef→town via Reefs table
    - TownWordExports: already at town level
    - IslandWordExports: distributed to all child towns of the island

    IslandWordExports words (spec≤0, generic) get their weight spread to every
    town in the parent island. This preserves signal for common domain words
    (e.g. "photosynthesis" → all Biology towns) while reef/town-level exports
    provide finer discrimination via MAX.

    Returns (word_towns, topical_town_word_ids).
    """
    # Reef-level exports → town
    reef_rows = con.execute("""
        SELECT e.word_id, r.town_id, e.export_weight
        FROM ReefWordExports e
        JOIN Reefs r ON e.reef_id = r.reef_id
        JOIN Towns t ON r.town_id = t.town_id
        JOIN Islands i ON t.island_id = i.island_id
        WHERE i.is_bucket = 0
    """).fetchall()

    # Town-level exports (already at town)
    town_rows = con.execute("""
        SELECT e.word_id, e.town_id, e.export_weight
        FROM TownWordExports e
        JOIN Towns t ON e.town_id = t.town_id
        JOIN Islands i ON t.island_id = i.island_id
        WHERE i.is_bucket = 0
    """).fetchall()

    # Island-level exports → distribute to all child towns
    island_rows = con.execute("""
        SELECT e.word_id, e.island_id, e.export_weight
        FROM IslandWordExports e
        JOIN Islands i ON e.island_id = i.island_id
        WHERE i.is_bucket = 0
    """).fetchall()

    # Build island→towns mapping (db_island_id → list of db_town_ids)
    island_to_towns = defaultdict(list)
    for db_town_id, db_island_id, _ in con.execute("""
        SELECT t.town_id, t.island_id, t.name
        FROM Towns t JOIN Islands i USING(island_id)
        WHERE i.is_bucket = 0
    """).fetchall():
        island_to_towns[db_island_id].append(db_town_id)

    # Group by (word_id, town_id), take MAX
    word_town_max = defaultdict(lambda: defaultdict(int))
    for word_id, db_town_id, weight in reef_rows + town_rows:
        if db_town_id not in town_remap:
            continue
        tid = town_remap[db_town_id]
        if weight > word_town_max[word_id][tid]:
            word_town_max[word_id][tid] = weight

    # Distribute island-level exports to child towns
    n_island_distributed = 0
    for word_id, db_island_id, weight in island_rows:
        if db_island_id not in island_remap:
            continue
        child_towns = island_to_towns.get(db_island_id, [])
        for db_town_id in child_towns:
            if db_town_id not in town_remap:
                continue
            tid = town_remap[db_town_id]
            if weight > word_town_max[word_id][tid]:
                word_town_max[word_id][tid] = weight
        if child_towns:
            n_island_distributed += 1

    # Build sparse list
    word_towns = {}
    for word_id, town_weights in word_town_max.items():
        entries = sorted([[tid, w] for tid, w in town_weights.items()], key=lambda x: x[0])
        word_towns[word_id] = entries

    topical_town_word_ids = set(word_towns.keys())
    print(f"  word_towns: {len(word_towns)} words with topical town entries "
          f"({n_island_distributed} via island distribution)")
    return word_towns, topical_town_word_ids


# ---------------------------------------------------------------------------
# Phase 4: Build bucket_words
# ---------------------------------------------------------------------------

def build_bucket_words(con, bucket_island_remap, topical_word_ids):
    """Build bucket_words: sparse list word_id → [[bucket_idx, weight_q]].

    Returns (bucket_words, bucket_only_word_ids).
    """
    # All three export tables for bucket islands
    reef_rows = con.execute("""
        SELECT e.word_id, t.island_id, e.export_weight
        FROM ReefWordExports e
        JOIN Reefs r ON e.reef_id = r.reef_id
        JOIN Towns t ON r.town_id = t.town_id
        JOIN Islands i ON t.island_id = i.island_id
        WHERE i.is_bucket = 1
    """).fetchall()

    town_rows = con.execute("""
        SELECT e.word_id, t.island_id, e.export_weight
        FROM TownWordExports e
        JOIN Towns t ON e.town_id = t.town_id
        JOIN Islands i ON t.island_id = i.island_id
        WHERE i.is_bucket = 1
    """).fetchall()

    island_rows = con.execute("""
        SELECT e.word_id, e.island_id, e.export_weight
        FROM IslandWordExports e
        JOIN Islands i ON e.island_id = i.island_id
        WHERE i.is_bucket = 1
    """).fetchall()

    # Group by (word_id, island_id), take MAX
    word_bucket_max = defaultdict(lambda: defaultdict(int))
    for word_id, db_island_id, weight in reef_rows + town_rows + island_rows:
        if db_island_id not in bucket_island_remap:
            continue
        bidx = bucket_island_remap[db_island_id]
        if weight > word_bucket_max[word_id][bidx]:
            word_bucket_max[word_id][bidx] = weight

    bucket_words = {}
    for word_id, bw in word_bucket_max.items():
        entries = sorted([[bidx, w] for bidx, w in bw.items()], key=lambda x: x[0])
        bucket_words[word_id] = entries

    bucket_only_word_ids = sorted(set(bucket_words.keys()) - topical_word_ids)
    print(f"  bucket_words: {len(bucket_words)} words, "
          f"{len(bucket_only_word_ids)} bucket-only")
    return bucket_words, bucket_only_word_ids


# ---------------------------------------------------------------------------
# Phase 5: Build word_detail (optional)
# ---------------------------------------------------------------------------

def build_word_detail(con, island_remap, town_remap, reef_remap):
    """Build word_detail: sparse list word_id → [[island_id, town_id, reef_id, weight, level]].

    level_code: 0=reef, 1=town, 2=island.
    Sentinel -1 used for missing town_id/reef_id at higher export levels.
    """
    entries = defaultdict(list)

    # Reef-level (level=0): all three IDs present
    for row in con.execute("""
        SELECT e.word_id, t.island_id, r.town_id, e.reef_id, e.export_weight
        FROM ReefWordExports e
        JOIN Reefs r ON e.reef_id = r.reef_id
        JOIN Towns t ON r.town_id = t.town_id
        JOIN Islands i ON t.island_id = i.island_id
        WHERE i.is_bucket = 0
    """).fetchall():
        word_id, db_iid, db_tid, db_rid, w = row
        if db_iid in island_remap and db_tid in town_remap and db_rid in reef_remap:
            entries[word_id].append([island_remap[db_iid], town_remap[db_tid],
                                    reef_remap[db_rid], w, 0])

    # Town-level (level=1): reef_id = -1
    for row in con.execute("""
        SELECT e.word_id, t.island_id, e.town_id, e.export_weight
        FROM TownWordExports e
        JOIN Towns t ON e.town_id = t.town_id
        JOIN Islands i ON t.island_id = i.island_id
        WHERE i.is_bucket = 0
    """).fetchall():
        word_id, db_iid, db_tid, w = row
        if db_iid in island_remap and db_tid in town_remap:
            entries[word_id].append([island_remap[db_iid], town_remap[db_tid], -1, w, 1])

    # Island-level (level=2): town_id = -1, reef_id = -1
    for row in con.execute("""
        SELECT e.word_id, e.island_id, e.export_weight
        FROM IslandWordExports e
        JOIN Islands i ON e.island_id = i.island_id
        WHERE i.is_bucket = 0
    """).fetchall():
        word_id, db_iid, w = row
        if db_iid in island_remap:
            entries[word_id].append([island_remap[db_iid], -1, -1, w, 2])

    word_detail = dict(entries)
    total_entries = sum(len(v) for v in word_detail.values())
    print(f"  word_detail: {len(word_detail)} words, {total_entries} entries")
    return word_detail


# ---------------------------------------------------------------------------
# Phase 6: Build metadata
# ---------------------------------------------------------------------------

def build_island_meta(con, island_rows, arch_remap):
    """Build island_meta: [{arch_id, name, n_words, iqf, avg_specificity, noun_frac, ...}]."""
    meta = []
    for db_island_id, db_arch_id, name in island_rows:
        row = con.execute("""
            SELECT word_count, avg_specificity, noun_frac, verb_frac, adj_frac, adv_frac
            FROM Islands WHERE island_id = ?
        """, (db_island_id,)).fetchone()

        n_words = row[0] or 0
        avg_spec = row[1] or 0.0
        noun_frac = row[2] or 0.0
        verb_frac = row[3] or 0.0
        adj_frac = row[4] or 0.0
        adv_frac = row[5] or 0.0

        meta.append({
            "arch_id": arch_remap[db_arch_id],
            "name": name,
            "n_words": n_words,
            "iqf": 128,  # placeholder — no IQF column yet
            "avg_specificity": round(avg_spec, 4),
            "noun_frac": round(noun_frac, 4),
            "verb_frac": round(verb_frac, 4),
            "adj_frac": round(adj_frac, 4),
            "adv_frac": round(adv_frac, 4),
        })

    print(f"  island_meta: {len(meta)} entries")
    return meta


def build_town_meta(con, town_rows, island_remap):
    """Build town_meta: [{island_id, name, n_words, tqf, avg_specificity}]."""
    meta = []
    for db_town_id, db_island_id, name in town_rows:
        row = con.execute("""
            SELECT word_count, avg_specificity
            FROM Towns WHERE town_id = ?
        """, (db_town_id,)).fetchone()

        meta.append({
            "island_id": island_remap[db_island_id],
            "name": name,
            "n_words": row[0] or 0,
            "tqf": 128,  # placeholder — no TQF column yet
            "avg_specificity": round(row[1] or 0.0, 4),
        })

    print(f"  town_meta: {len(meta)} entries")
    return meta


def build_reef_meta(con, reef_rows, town_remap):
    """Build reef_meta: [{town_id, name, n_words, avg_specificity}]."""
    meta = []
    for db_reef_id, db_town_id, name in reef_rows:
        row = con.execute("""
            SELECT word_count, avg_specificity
            FROM Reefs WHERE reef_id = ?
        """, (db_reef_id,)).fetchone()

        meta.append({
            "town_id": town_remap[db_town_id],
            "name": name or "",
            "n_words": row[0] or 0,
            "avg_specificity": round(row[1] or 0.0, 4),
        })

    print(f"  reef_meta: {len(meta)} entries")
    return meta


# ---------------------------------------------------------------------------
# Phase 7: Extract compounds
# ---------------------------------------------------------------------------

def extract_compounds(con):
    """Extract compound words for Aho-Corasick automaton."""
    rows = con.execute("""
        SELECT word, word_id FROM Words
        WHERE word_count > 1 AND is_stop = 0
        ORDER BY word_id
    """).fetchall()

    compounds = [[word, word_id] for word, word_id in rows]
    print(f"  compounds: {len(compounds)} entries")
    return compounds


# ---------------------------------------------------------------------------
# Phase 8: Background model
# ---------------------------------------------------------------------------

def compute_background_model(word_islands, words_info, n_islands,
                             n_samples=1000, words_per_sample=15, seed=42):
    """Compute background mean and std per island.

    Samples random sets of single words weighted by island-association count
    (proxy for word frequency), accumulates per-island, computes mean/std.
    """
    # Get single-word IDs with topical island entries
    single_word_ids = []
    sample_weights = []

    # Build word_count lookup
    word_counts = {}
    for row in _con_ref.execute("SELECT word_id, word_count FROM Words").fetchall():
        word_counts[row[0]] = row[1] or 1

    for wid, entries in word_islands.items():
        wc = word_counts.get(wid, 1)
        if wc == 1:
            single_word_ids.append(wid)
            sample_weights.append(len(entries))

    single_word_ids = np.array(single_word_ids)
    sample_weights = np.array(sample_weights, dtype=np.float64)
    sample_weights /= sample_weights.sum()

    assert len(single_word_ids) >= words_per_sample, \
        f"Not enough single words ({len(single_word_ids)}) for background sampling"

    print(f"  Frequency-weighted sampling: {len(single_word_ids)} single words")

    rng = np.random.default_rng(seed)
    all_scores = np.zeros((n_samples, n_islands))

    for i in tqdm(range(n_samples), desc="Background model"):
        sample = rng.choice(single_word_ids, size=words_per_sample,
                            replace=False, p=sample_weights)
        for word_id in sample:
            for island_id, weight_q in word_islands[int(word_id)]:
                all_scores[i, island_id] += weight_q / WEIGHT_SCALE

    bg_mean = all_scores.mean(axis=0).tolist()
    bg_std = all_scores.std(axis=0).tolist()
    bg_std = [max(s, 1e-6) for s in bg_std]

    print(f"  bg_mean range: [{min(bg_mean):.2f}, {max(bg_mean):.2f}]")
    print(f"  bg_std range:  [{min(bg_std):.4f}, {max(bg_std):.4f}]")
    return bg_mean, bg_std


def adjust_background_model(bg_mean, bg_std, island_meta, word_islands,
                            n_samples=1000, words_per_sample=15):
    """Apply regression adjustment, specificity modulation, and std floor."""
    n_islands = len(bg_mean)

    # Build vocab_size (single words with island entries)
    word_counts = {}
    for row in _con_ref.execute("SELECT word_id, word_count FROM Words").fetchall():
        word_counts[row[0]] = row[1] or 1
    vocab_size = sum(1 for wid in word_islands if word_counts.get(wid, 1) == 1)

    # Step 1: Identify unreliable islands by expected hit rate
    reliable = [False] * n_islands
    for iid in range(n_islands):
        n_words = island_meta[iid].get("n_words", 0)
        if vocab_size > 0 and n_words > 0:
            p_miss_per_word = 1.0 - words_per_sample / vocab_size
            p_miss_all = p_miss_per_word ** n_words
            p_hit = 1.0 - p_miss_all
            expected_hits = p_hit * n_samples
            reliable[iid] = expected_hits >= 30

    n_reliable = sum(reliable)
    print(f"  Background adjustment: {n_reliable} reliable, "
          f"{n_islands - n_reliable} unreliable islands")

    # Step 2: Log-log regression on reliable islands
    log_means = []
    log_stds = []
    for iid in range(n_islands):
        if reliable[iid] and bg_std[iid] > 0.01 and bg_mean[iid] > 0.01:
            log_means.append(math.log(bg_mean[iid]))
            log_stds.append(math.log(bg_std[iid]))

    if len(log_means) >= 5:
        n = len(log_means)
        sum_x = sum(log_means)
        sum_y = sum(log_stds)
        sum_xy = sum(x * y for x, y in zip(log_means, log_stds))
        sum_x2 = sum(x * x for x in log_means)
        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) > 1e-12:
            a = (n * sum_xy - sum_x * sum_y) / denom
            b = (sum_y - a * sum_x) / n
        else:
            a, b = 0.513, 1.482
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in log_stds)
        ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(log_means, log_stds))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        print(f"  Regression: log(std) = {a:.3f} * log(mean) + {b:.3f}  "
              f"(R²={r_sq:.3f}, n={n})")
    else:
        a, b = 0.513, 1.482
        print(f"  Regression: using fallback coefficients")

    # Step 3: Replace unreliable stds
    reliable_stds = sorted(
        bg_std[iid] for iid in range(n_islands)
        if reliable[iid] and bg_std[iid] > 0.01
    )
    median_reliable_std = reliable_stds[len(reliable_stds) // 2] if reliable_stds else 1.0

    adjusted_std = list(bg_std)
    replaced = 0
    for iid in range(n_islands):
        if not reliable[iid] and bg_std[iid] > 1e-6:
            mean_val = max(bg_mean[iid], 0.01)
            predicted = math.exp(a * math.log(mean_val) + b)
            adjusted_std[iid] = max(predicted, median_reliable_std)
            replaced += 1
    if replaced:
        print(f"  Replaced {replaced} unreliable bg_std values")

    # Step 4: Specificity modulation
    for iid in range(n_islands):
        if adjusted_std[iid] > 1e-6:
            avg_spec = island_meta[iid].get("avg_specificity", 0.0)
            spec_factor = 1.0 + 0.3 * avg_spec
            adjusted_std[iid] = adjusted_std[iid] / spec_factor

    # Step 5: Floor
    floored = 0
    for iid in range(n_islands):
        if adjusted_std[iid] < BG_STD_FLOOR:
            adjusted_std[iid] = BG_STD_FLOOR
            floored += 1
    if floored:
        print(f"  Floored {floored} island bg_std values to {BG_STD_FLOOR}")

    print(f"  Adjusted bg_std range: [{min(adjusted_std):.4f}, {max(adjusted_std):.4f}]")
    return adjusted_std


# ---------------------------------------------------------------------------
# Phase 9: Build constants
# ---------------------------------------------------------------------------

def build_constants(n_islands, n_towns, n_reefs, n_archs,
                    island_meta, town_meta, domainless_word_ids,
                    town_domainless_word_ids,
                    bucket_only_word_ids, bucket_island_names):
    """Build the constants dict."""
    island_n_words = [m["n_words"] for m in island_meta]
    town_n_words = [m["n_words"] for m in town_meta]

    constants = {
        "N_ISLANDS": n_islands,
        "N_TOWNS": n_towns,
        "N_REEFS": n_reefs,
        "N_ARCHS": n_archs,
        "IDF_SCALE": IDF_SCALE,
        "WEIGHT_SCALE": WEIGHT_SCALE,
        "FNV1A_OFFSET": FNV1A_OFFSET,
        "FNV1A_PRIME": FNV1A_PRIME,
        "island_n_words": island_n_words,
        "town_n_words": town_n_words,
        "domainless_word_ids": domainless_word_ids,
        "town_domainless_word_ids": town_domainless_word_ids,
        "bucket_only_word_ids": bucket_only_word_ids,
        "bucket_island_names": bucket_island_names,
    }
    print(f"  constants: {n_islands} islands, {n_towns} towns, {n_reefs} reefs, "
          f"{n_archs} archs")
    print(f"  {len(domainless_word_ids)} island-domainless, "
          f"{len(town_domainless_word_ids)} town-domainless, "
          f"{len(bucket_only_word_ids)} bucket-only")
    return constants


# ---------------------------------------------------------------------------
# Phase 10: Write files + manifest
# ---------------------------------------------------------------------------

def write_all_files(output_dir, word_lookup, word_islands, word_towns,
                    words_info, island_meta,
                    bg_mean, bg_std, town_bg_mean, town_bg_std,
                    constants, compounds,
                    word_detail, town_meta, reef_meta, bucket_words):
    """Serialize all 11 data files."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. word_lookup.bin — {u64_hash: [word_hash, word_id, specificity, idf_q]}
    write_msgpack({int(k): v for k, v in word_lookup.items()},
                  os.path.join(output_dir, "word_lookup.bin"))

    max_word_id = max(words_info.keys()) if words_info else 0

    # 2. word_islands.bin — sparse list indexed by word_id
    wi_list = [[] for _ in range(max_word_id + 1)]
    for word_id, entries in word_islands.items():
        wi_list[word_id] = entries
    write_msgpack(wi_list, os.path.join(output_dir, "word_islands.bin"))

    # 3. word_towns.bin — sparse list indexed by word_id
    wt_list = [[] for _ in range(max_word_id + 1)]
    for word_id, entries in word_towns.items():
        wt_list[word_id] = entries
    write_msgpack(wt_list, os.path.join(output_dir, "word_towns.bin"))

    # 4. island_meta.bin
    write_msgpack(island_meta, os.path.join(output_dir, "island_meta.bin"))

    # 5. background.bin — island + town level
    write_msgpack({
        "bg_mean": bg_mean, "bg_std": bg_std,
        "town_bg_mean": town_bg_mean, "town_bg_std": town_bg_std,
    }, os.path.join(output_dir, "background.bin"))

    # 6. constants.bin
    write_msgpack(constants, os.path.join(output_dir, "constants.bin"))

    # 7. compounds.bin
    write_msgpack(compounds, os.path.join(output_dir, "compounds.bin"))

    # 8. word_detail.bin — sparse list indexed by word_id
    wd_list = [[] for _ in range(max_word_id + 1)]
    for word_id, entries in word_detail.items():
        wd_list[word_id] = entries
    write_msgpack(wd_list, os.path.join(output_dir, "word_detail.bin"))

    # 9. town_meta.bin
    write_msgpack(town_meta, os.path.join(output_dir, "town_meta.bin"))

    # 10. reef_meta.bin
    write_msgpack(reef_meta, os.path.join(output_dir, "reef_meta.bin"))

    # 11. bucket_words.bin — sparse list indexed by word_id
    bw_list = [[] for _ in range(max_word_id + 1)]
    for word_id, entries in bucket_words.items():
        bw_list[word_id] = entries
    write_msgpack(bw_list, os.path.join(output_dir, "bucket_words.bin"))

    print(f"  Wrote {len(EXPORT_FILES)} data files to {output_dir}/")


def write_manifest(output_dir, stats):
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
        "stats": stats,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Wrote manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# Phase 11: Verification
# ---------------------------------------------------------------------------

def verify_export(output_dir, manifest):
    """Post-export validation: checksums + deserialization + spot checks."""
    print("\n=== Verification ===")
    errors = 0

    # 1. Deserialize each file
    print("  Checking deserialization...")
    for fname in EXPORT_FILES:
        path = os.path.join(output_dir, fname)
        try:
            with open(path, "rb") as f:
                msgpack.unpack(f, raw=False, strict_map_key=False)
            print(f"    OK: {fname}")
        except Exception as e:
            print(f"    FAIL: {fname} — {e}")
            errors += 1

    # 2. Checksums
    print("  Checking checksums...")
    for fname, expected_hash in manifest["files"].items():
        actual_hash = sha256_file(os.path.join(output_dir, fname))
        if actual_hash != expected_hash:
            print(f"    FAIL: {fname} checksum mismatch")
            errors += 1
        else:
            print(f"    OK: {fname}")

    # 3. Load and spot-check key structures
    print("  Spot-checking structures...")

    with open(os.path.join(output_dir, "constants.bin"), "rb") as f:
        constants = msgpack.unpack(f, raw=False, strict_map_key=False)
    print(f"    N_ISLANDS={constants['N_ISLANDS']}, N_TOWNS={constants['N_TOWNS']}, "
          f"N_REEFS={constants['N_REEFS']}, N_ARCHS={constants['N_ARCHS']}")
    print(f"    domainless_word_ids: {len(constants['domainless_word_ids'])}")
    print(f"    bucket_only_word_ids: {len(constants['bucket_only_word_ids'])}")
    print(f"    bucket_island_names: {constants['bucket_island_names']}")

    with open(os.path.join(output_dir, "island_meta.bin"), "rb") as f:
        im = msgpack.unpack(f, raw=False, strict_map_key=False)
    required_keys = {"arch_id", "name", "n_words", "iqf", "avg_specificity",
                     "noun_frac", "verb_frac", "adj_frac", "adv_frac"}
    if im:
        missing = required_keys - set(im[0].keys())
        if missing:
            print(f"    FAIL: island_meta[0] missing keys: {missing}")
            errors += 1
        else:
            print(f"    OK: island_meta has all required fields ({len(im)} islands)")
        # Print first 3 islands
        for rec in im[:3]:
            print(f"      arch={rec['arch_id']} {rec['name']:25s} "
                  f"n_words={rec['n_words']}")

    with open(os.path.join(output_dir, "word_lookup.bin"), "rb") as f:
        wl = msgpack.unpack(f, raw=False, strict_map_key=False)
    print(f"    word_lookup: {len(wl)} entries")

    with open(os.path.join(output_dir, "background.bin"), "rb") as f:
        bg = msgpack.unpack(f, raw=False, strict_map_key=False)
    print(f"    background (island): {len(bg['bg_mean'])} mean, {len(bg['bg_std'])} std values")
    print(f"    background (town):   {len(bg['town_bg_mean'])} mean, "
          f"{len(bg['town_bg_std'])} std values")

    if errors > 0:
        print(f"\n  {errors} verification errors!")
    else:
        print(f"\n  All checks passed.")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Module-level ref so background model functions can access connection
_con_ref = None


def export(args):
    """Run the full v3 export pipeline."""
    global _con_ref

    import sqlite3

    start = time.time()
    output_dir = args.output
    print(f"Exporting from {V3_DB} to {output_dir}/\n")

    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")
    _con_ref = con

    # Phase 1: Hierarchy
    print("Phase 1: Loading hierarchy...")
    (arch_remap, island_remap, town_remap, reef_remap,
     bucket_island_remap, arch_rows, island_rows, town_rows, reef_rows,
     bucket_island_rows) = load_hierarchy(con)

    # Phase 2: Word lookup
    print("\nPhase 2: Building word_lookup...")
    word_lookup, words_info = build_word_lookup(con)

    # Phase 3: Word-islands (topical)
    print("\nPhase 3: Building word_islands...")
    word_islands, topical_word_ids = build_word_islands(
        con, island_remap, town_remap, reef_remap)

    # Phase 3b: Word-towns (topical)
    print("\nPhase 3b: Building word_towns...")
    word_towns, topical_town_word_ids = build_word_towns(
        con, island_remap, town_remap, reef_remap)

    # Phase 4: Bucket words
    print("\nPhase 4: Building bucket_words...")
    bucket_words, bucket_only_word_ids = build_bucket_words(
        con, bucket_island_remap, topical_word_ids)

    # Phase 5: Word detail
    print("\nPhase 5: Building word_detail...")
    word_detail = build_word_detail(con, island_remap, town_remap, reef_remap)

    # Phase 6: Metadata
    print("\nPhase 6: Building metadata...")
    island_meta = build_island_meta(con, island_rows, arch_remap)
    town_meta = build_town_meta(con, town_rows, island_remap)
    reef_meta = build_reef_meta(con, reef_rows, town_remap)

    # Phase 7: Compounds
    print("\nPhase 7: Extracting compounds...")
    compounds = extract_compounds(con)

    # Phase 8: Background model (island-level)
    print("\nPhase 8: Computing island background model...")
    bg_mean, bg_std = compute_background_model(
        word_islands, words_info, n_islands=len(island_rows),
        n_samples=args.bg_samples, words_per_sample=args.bg_words, seed=args.seed)

    # Phase 9: Adjust island background model
    print("\nPhase 9: Adjusting island background model...")
    bg_std = adjust_background_model(
        bg_mean, bg_std, island_meta, word_islands,
        n_samples=args.bg_samples, words_per_sample=args.bg_words)

    # Phase 9b: Town-level background model
    print("\nPhase 9b: Computing town background model...")
    town_bg_mean, town_bg_std = compute_background_model(
        word_towns, words_info, n_islands=len(town_rows),
        n_samples=args.bg_samples, words_per_sample=args.bg_words, seed=args.seed)

    print("\nPhase 9c: Adjusting town background model...")
    town_bg_std = adjust_background_model(
        town_bg_mean, town_bg_std, town_meta, word_towns,
        n_samples=args.bg_samples, words_per_sample=args.bg_words)

    # Partition: domainless = in vocab, not in topical, not in bucket
    all_word_ids = set(words_info.keys())
    exported_word_ids = topical_word_ids | set(bucket_words.keys())
    domainless_word_ids = sorted(all_word_ids - exported_word_ids)

    # Town-level domainless: words not in topical_town_word_ids and not in bucket
    town_domainless_word_ids = sorted(
        all_word_ids - topical_town_word_ids - set(bucket_words.keys()))

    # Phase 10: Constants
    print("\nPhase 10: Building constants...")
    bucket_island_names = [row[2] for row in bucket_island_rows]
    constants = build_constants(
        n_islands=len(island_rows),
        n_towns=len(town_rows),
        n_reefs=len(reef_rows),
        n_archs=len(arch_rows),
        island_meta=island_meta,
        town_meta=town_meta,
        domainless_word_ids=domainless_word_ids,
        town_domainless_word_ids=town_domainless_word_ids,
        bucket_only_word_ids=bucket_only_word_ids,
        bucket_island_names=bucket_island_names,
    )

    # Phase 11: Write files
    print("\nPhase 11: Writing data files...")
    write_all_files(output_dir, word_lookup, word_islands, word_towns,
                    words_info, island_meta,
                    bg_mean, bg_std, town_bg_mean, town_bg_std,
                    constants, compounds,
                    word_detail, town_meta, reef_meta, bucket_words)

    # Phase 12: Manifest
    print("\nPhase 12: Writing manifest...")
    stats = {
        "n_islands": len(island_rows),
        "n_towns": len(town_rows),
        "n_reefs": len(reef_rows),
        "n_archs": len(arch_rows),
        "n_bucket_islands": len(bucket_island_rows),
        "n_words": len(words_info),
        "n_lookup_entries": len(word_lookup),
        "n_topical_words": len(topical_word_ids),
        "n_topical_town_words": len(topical_town_word_ids),
        "n_bucket_only_words": len(bucket_only_word_ids),
        "n_domainless_words": len(domainless_word_ids),
        "n_town_domainless_words": len(town_domainless_word_ids),
        "n_compounds": len(compounds),
        "n_detail_words": len(word_detail),
    }
    manifest = write_manifest(output_dir, stats)

    con.close()
    _con_ref = None

    elapsed = time.time() - start
    print(f"\nExport complete in {elapsed:.1f}s")
    print(f"  Stats: {json.dumps(stats, indent=4)}")

    # Verification
    if args.verify:
        verify_errors = verify_export(output_dir, manifest)
        if verify_errors > 0:
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Export v3 data to lagoon format (v3.1)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
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
        help="Run post-export verification",
    )
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
