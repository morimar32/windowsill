"""
Shared export utilities for the lagoon scoring library.

Constants, helpers, and core functions used by v2/load.py.
"""

import hashlib
import math
import os

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
WEIGHT_SCALE = 100.0      # final_calc_weight * 100 → u16
SUB_REEF_SENTINEL = 0xFFFF

EXPORT_FILES = [
    "word_lookup.bin",
    "word_reefs.bin",
    "reef_meta.bin",
    "island_meta.bin",
    "background.bin",
    "compounds.bin",
    "constants.bin",
    "reef_edges.bin",
    "word_reef_detail.bin",
    "sub_reef_meta.bin",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# Expand snowball stems
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
# Build word_lookup
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
# Background model
# ---------------------------------------------------------------------------

def compute_background_model(word_reefs, words, n_reefs, n_samples=1000,
                              words_per_sample=15, seed=42):
    """Compute background mean and std for z-score normalization.

    Samples random sets of single words weighted by reef-association count
    (proxy for word frequency — common words appear in more reefs), scores
    them, and records per-reef statistics.
    """
    # Get single-word IDs that have reef entries
    single_word_ids = []
    sample_weights = []
    for wid, info in words.items():
        if info["word_count"] == 1 and wid in word_reefs:
            single_word_ids.append(wid)
            sample_weights.append(len(word_reefs[wid]))

    single_word_ids = np.array(single_word_ids)
    sample_weights = np.array(sample_weights, dtype=np.float64)
    sample_weights /= sample_weights.sum()  # normalize to probabilities

    assert len(single_word_ids) >= words_per_sample, \
        f"Not enough single words ({len(single_word_ids)}) for background sampling"

    print(f"  Frequency-weighted sampling: {len(single_word_ids)} single words, "
          f"weight range [{sample_weights.min():.6f}, {sample_weights.max():.6f}]")

    rng = np.random.default_rng(seed)
    all_scores = np.zeros((n_samples, n_reefs))

    for i in tqdm(range(n_samples), desc="Background model"):
        sample = rng.choice(single_word_ids, size=words_per_sample,
                            replace=False, p=sample_weights)
        for word_id in sample:
            for entry in word_reefs[int(word_id)]:
                reef_id, weight_q = entry[0], entry[1]
                all_scores[i, reef_id] += weight_q / WEIGHT_SCALE

    bg_mean = all_scores.mean(axis=0).tolist()
    bg_std = all_scores.std(axis=0).tolist()
    # Replace zero std with epsilon
    bg_std = [max(s, 1e-6) for s in bg_std]

    print(f"  bg_mean range: [{min(bg_mean):.2f}, {max(bg_mean):.2f}]")
    print(f"  bg_std range:  [{min(bg_std):.4f}, {max(bg_std):.4f}]")
    return bg_mean, bg_std


def adjust_background_model(bg_mean, bg_std, reef_records, reef_stats,
                            vocab_size, n_samples=1000, words_per_sample=15):
    """Pre-bake tiny-reef correction and specificity modulation into bg_std.

    1. Fit a power-law regression on reliable reefs: log(std) ~ log(mean)
    2. For unreliable reefs (expected sample hits < 30), replace observed
       std with the regression prediction
    3. Divide all stds by the reef's specificity factor, so z-scores
       already include the specificity boost

    After this, the scorer can do z = (raw - mean) / std with no
    conditionals and no separate quality modulation phase.
    """
    n_reefs = len(bg_mean)

    # --- Step 1: Identify unreliable reefs by expected hit rate ---
    reliable = [False] * n_reefs
    for rid in range(n_reefs):
        n_words = reef_stats.get(rid, {}).get("n_words", 0)
        # p_hit = 1 - (1 - words_per_sample / vocab_size) ^ n_words
        if vocab_size > 0 and n_words > 0:
            p_miss_per_word = 1.0 - words_per_sample / vocab_size
            p_miss_all = p_miss_per_word ** n_words
            p_hit = 1.0 - p_miss_all
            expected_hits = p_hit * n_samples
            reliable[rid] = expected_hits >= 30
        else:
            reliable[rid] = False

    n_reliable = sum(reliable)
    n_unreliable = n_reefs - n_reliable
    print(f"  Background adjustment: {n_reliable} reliable, {n_unreliable} unreliable reefs")

    # --- Step 2: Fit log-log regression on reliable reefs ---
    # Collect (log_mean, log_std) for reliable reefs with credible std
    log_means = []
    log_stds = []
    for rid in range(n_reefs):
        if reliable[rid] and bg_std[rid] > 0.01 and bg_mean[rid] > 0.01:
            log_means.append(math.log(bg_mean[rid]))
            log_stds.append(math.log(bg_std[rid]))

    if len(log_means) >= 5:
        # Simple linear regression: log(std) = a * log(mean) + b
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
            a, b = 0.513, 1.482  # fallback from analysis
        # R² for logging
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in log_stds)
        ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(log_means, log_stds))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        print(f"  Regression: log(std) = {a:.3f} * log(mean) + {b:.3f}  (R²={r_sq:.3f}, n={n})")
    else:
        a, b = 0.513, 1.482  # fallback from analysis
        print(f"  Regression: using fallback coefficients (too few reliable reefs)")

    # --- Step 3: Replace unreliable stds with regression prediction ---
    # Compute median std of reliable reefs as floor for unreliable predictions.
    # Unreliable reefs have near-zero observed means (sampling rarely hits them),
    # so regression predictions from those means can be too low.  The median of
    # reliable stds provides a principled floor.
    reliable_stds = sorted(
        bg_std[rid] for rid in range(n_reefs) if reliable[rid] and bg_std[rid] > 0.01
    )
    median_reliable_std = reliable_stds[len(reliable_stds) // 2] if reliable_stds else 1.0
    print(f"  Median reliable bg_std: {median_reliable_std:.4f}")

    adjusted_std = list(bg_std)
    replaced = 0
    for rid in range(n_reefs):
        if not reliable[rid] and bg_std[rid] > 1e-6:
            # Predict std from mean using regression, floored at median
            mean_val = max(bg_mean[rid], 0.01)
            predicted = math.exp(a * math.log(mean_val) + b)
            adjusted_std[rid] = max(predicted, median_reliable_std)
            replaced += 1
    print(f"  Replaced {replaced} unreliable bg_std values with regression predictions")

    # --- Step 4: Apply specificity modulation to ALL reefs ---
    for rid in range(n_reefs):
        if adjusted_std[rid] > 1e-6:
            avg_spec = reef_records[rid].get("avg_specificity", 0.0) if isinstance(reef_records[rid], dict) else reef_records[rid].avg_specificity
            spec_factor = 1.0 + 0.3 * avg_spec
            adjusted_std[rid] = adjusted_std[rid] / spec_factor

    # --- Step 5: Apply bg_std floor to cap z-score sensitivity ---
    # Without a floor, niche reefs with tiny std get disproportionate z-score
    # amplification (1/std), causing them to dominate over legitimate reefs.
    floored = 0
    for rid in range(n_reefs):
        if adjusted_std[rid] < config.BG_STD_FLOOR:
            adjusted_std[rid] = config.BG_STD_FLOOR
            floored += 1
    if floored:
        print(f"  Floored {floored} reef bg_std values to {config.BG_STD_FLOOR}")

    print(f"  Adjusted bg_std range: [{min(adjusted_std):.4f}, {max(adjusted_std):.4f}]")
    return adjusted_std


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def write_all_files(output_dir, word_lookup, word_reefs, words,
                    reef_records, island_records, bg_mean, bg_std,
                    compounds, reef_stats, avg_reef_words, reef_edges,
                    sub_reef_records=None, word_reef_detail=None):
    """Serialize all data files to the output directory (msgpack)."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. word_lookup.bin
    # Convert int keys to ensure msgpack handles them correctly
    wl_data = {int(k): v for k, v in word_lookup.items()}
    write_msgpack(wl_data, os.path.join(output_dir, "word_lookup.bin"))

    # 2. word_reefs.bin — outer list indexed by word_id, 3-element tuples
    max_word_id = max(words.keys()) if words else 0
    wr_list = [[] for _ in range(max_word_id + 1)]
    for word_id, entries in word_reefs.items():
        wr_list[word_id] = [[rid, wq, srid] for rid, wq, srid in entries]
    write_msgpack(wr_list, os.path.join(output_dir, "word_reefs.bin"))

    # 3. reef_meta.bin
    reef_meta = [
        {
            "hierarchy_addr": r["hierarchy_addr"],
            "n_words": r["n_words"],
            "name": r["name"],
            "valence": r["valence"],
            "avg_specificity": r["avg_specificity"],
            "noun_frac": r["noun_frac"],
            "verb_frac": r["verb_frac"],
            "adj_frac": r["adj_frac"],
            "adv_frac": r["adv_frac"],
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

    n_sub_reefs = len(sub_reef_records) if sub_reef_records else 0
    constants = {
        "N_REEFS": n_reefs,
        "N_ISLANDS": n_islands,
        "N_ARCHS": n_archs,
        "N_SUB_REEFS": n_sub_reefs,
        "avg_reef_words": avg_reef_words,
        "IDF_SCALE": IDF_SCALE,
        "WEIGHT_SCALE": WEIGHT_SCALE,
        "FNV1A_OFFSET": config.FNV1A_OFFSET,
        "FNV1A_PRIME": config.FNV1A_PRIME,
        "reef_total_dims": reef_total_dims,
        "reef_n_words": reef_n_words,
    }
    write_msgpack(constants, os.path.join(output_dir, "constants.bin"))

    # 8. reef_edges.bin
    re_data = [[src, tgt, weight] for src, tgt, weight in reef_edges]
    write_msgpack(re_data, os.path.join(output_dir, "reef_edges.bin"))

    # 9. word_reef_detail.bin — per-word detail for multi-reef words
    if word_reef_detail is not None:
        wrd_list = [[] for _ in range(max_word_id + 1)]
        for word_id, entries in word_reef_detail.items():
            wrd_list[word_id] = [[iid, srid, wq] for iid, srid, wq in entries]
        write_msgpack(wrd_list, os.path.join(output_dir, "word_reef_detail.bin"))
    else:
        write_msgpack([[] for _ in range(max_word_id + 1)],
                      os.path.join(output_dir, "word_reef_detail.bin"))

    # 10. sub_reef_meta.bin — gen-2 reef metadata
    write_msgpack(sub_reef_records or [],
                  os.path.join(output_dir, "sub_reef_meta.bin"))

    print(f"  Wrote {len(EXPORT_FILES)} data files to {output_dir}/")
