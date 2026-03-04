# Windowsill Technical Reference

Deep technical documentation for the windowsill project. This file is optimized for getting an LLM up to speed on the codebase -- it covers the database schema, scoring formulas, feature engineering, clustering algorithms, and export format in full detail.

For a human-readable overview, see [README.md](README.md).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Database Schema](#database-schema)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1: Hierarchy Population](#stage-1-hierarchy-population)
  - [Stage 2: Vocabulary & Embeddings](#stage-2-vocabulary--embeddings)
  - [Stage 3: Seed Population](#stage-3-seed-population)
  - [Stage 4: Classification & Clustering](#stage-4-classification--clustering)
  - [Stage 5: Stats & Export Promotion](#stage-5-stats--export-promotion)
  - [Stage 6: Binary Export](#stage-6-binary-export)
- [Scoring Formulas](#scoring-formulas)
- [Feature Engineering (776 Features)](#feature-engineering-776-features)
- [Name Cosine Blending](#name-cosine-blending)
- [Clustering Algorithms](#clustering-algorithms)
- [Export Promotion Rules](#export-promotion-rules)
- [Export Format v3.1](#export-format-v31)
- [Configuration Reference](#configuration-reference)
- [Lagoon Integration](#lagoon-integration)
- [Important Implementation Details](#important-implementation-details)

---

## Architecture Overview

Windowsill is a multi-stage pipeline that builds a four-tier semantic hierarchy from word embeddings, then distills it into compact binary files for the Lagoon scoring library.

```
WordNet vocab (147K lemmas)
  + WDH seed words (10K+ pairs)
  + Claude seed words (100-150 per town)
        |
        v
  [load.sh steps 1-11]  ──> windowsill.db
        |                    (words, embeddings, hierarchy, seeds)
        |
  [steps 12-16]          ──> + XGBoost predictions
        |                    + Leiden-clustered reefs
        |                    + ReefWords (450K associations)
        |
  [steps 17-19]          ──> ReefWordExports (30K rows)
        |                    TownWordExports (241K rows)
        |                    IslandWordExports (145K rows)
        |
  [export.py]            ──> v3/exports/*.bin (11 msgpack files + manifest.json)
        |
        v
  [Lagoon]               ──> real-time domain scoring of free text
```

The four-tier hierarchy:

```
Archipelagos (6)   — broad knowledge families
  └── Islands (44)   — major disciplines (41 topical + 3 bucket)
        └── Towns (332)  — specific subfields
              └── Reefs (3,919)  — Leiden-discovered subclusters
```

Three curated levels (archipelago, island, town) provide the named, stable structure. One discovered level (reef) provides fine-grained subclustering within towns. Each word exports at exactly one level (reef, town, or island) based on its specificity and spread across the hierarchy.

---

## Database Schema

All data lives in `v3/windowsill.db` (SQLite, ~780 MB). Schema defined in `v3/schema.sql`.

### Hierarchy Tables

#### Archipelagos (6 rows)
```sql
CREATE TABLE Archipelagos (
    archipelago_id  INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    island_count    INTEGER DEFAULT 0,
    town_count      INTEGER DEFAULT 0,
    reef_count      INTEGER DEFAULT 0,
    word_count      INTEGER DEFAULT 0
);
```

#### Islands (44 rows)
```sql
CREATE TABLE Islands (
    island_id       INTEGER PRIMARY KEY,
    archipelago_id  INTEGER NOT NULL REFERENCES Archipelagos,
    name            TEXT NOT NULL,
    is_bucket       INTEGER NOT NULL DEFAULT 0,  -- 1 = non-topical (e.g. Linguistic Register)
    town_count      INTEGER DEFAULT 0,
    reef_count      INTEGER DEFAULT 0,
    word_count      INTEGER DEFAULT 0,
    noun_frac       REAL,
    verb_frac       REAL,
    adj_frac        REAL,
    adv_frac        REAL,
    avg_specificity REAL,
    UNIQUE (archipelago_id, name)
);
```

Bucket islands (is_bucket=1) contain non-topical vocabulary: Languages, Regional, Miscellaneous. These words are identified and exported separately, but excluded from topical scoring.

#### Towns (332 rows)
```sql
CREATE TABLE Towns (
    town_id         INTEGER PRIMARY KEY,
    island_id       INTEGER NOT NULL REFERENCES Islands,
    name            TEXT NOT NULL,
    is_capital      INTEGER DEFAULT 0,  -- 1 = catch-all town for island
    model_f1        REAL,               -- XGBoost validation F1
    reef_count      INTEGER DEFAULT 0,
    word_count      INTEGER DEFAULT 0,
    noun_frac       REAL,
    verb_frac       REAL,
    adj_frac        REAL,
    adv_frac        REAL,
    avg_specificity REAL,
    UNIQUE (island_id, name)
);
```

Capital towns (is_capital=1) are catch-all towns within an island for words that don't fit any specific town. Their seeds go through a triage process in `detect_island_words.py`.

#### Reefs (3,919 rows)
```sql
CREATE TABLE Reefs (
    reef_id         INTEGER PRIMARY KEY,
    town_id         INTEGER NOT NULL REFERENCES Towns,
    name            TEXT,           -- set after Leiden clustering (top 3 words)
    centroid        BLOB,           -- L2-normalized float32 embedding
    word_count      INTEGER DEFAULT 0,
    core_word_count INTEGER DEFAULT 0,
    avg_specificity REAL,
    noun_frac       REAL,
    verb_frac       REAL,
    adj_frac        REAL,
    adv_frac        REAL
);
```

### Dictionary Tables

#### Words (149,691 rows)
```sql
CREATE TABLE Words (
    word_id         INTEGER PRIMARY KEY,
    word            TEXT NOT NULL,
    word_hash       INTEGER NOT NULL,   -- FNV-1a u64 (stored as signed i64)
    pos             TEXT,               -- dominant POS: noun, verb, adj, adv
    specificity     INTEGER,            -- based on total reef_count
    cosine_sim      REAL,               -- best cosine to any assigned reef centroid
    idf             REAL,               -- log2(N_total_reefs / reef_count)
    embedding       BLOB,               -- float32 x 768 = 3072 bytes
    word_count      INTEGER DEFAULT 1,  -- number of space-separated tokens
    category        TEXT,               -- single, compound, phrasal_verb
    is_stop         INTEGER DEFAULT 0,
    reef_count      INTEGER DEFAULT 0,
    town_count      INTEGER DEFAULT 0,
    island_count    INTEGER DEFAULT 0
);
```

Embeddings are stored as raw float32 bytes (3072 bytes per word). All words are embedded with `"clustering: "` prefix using nomic-embed-text-v1.5.

#### ReefWords (449,925 rows)
```sql
CREATE TABLE ReefWords (
    reef_id         INTEGER NOT NULL REFERENCES Reefs,
    word_id         INTEGER NOT NULL REFERENCES Words,
    pos             TEXT,               -- contextual POS (overrides Words.pos)
    specificity     INTEGER,            -- within this reef
    cosine_sim      REAL,               -- cosine to this reef's centroid
    idf             REAL,               -- within this reef's context
    island_idf      REAL,               -- log2(island_total_reefs / word_reefs_in_island)
    source          TEXT NOT NULL,       -- curated, xgboost
    source_quality  REAL NOT NULL DEFAULT 1.0,
    is_core         INTEGER DEFAULT 0,  -- 1 = Leiden core member
    PRIMARY KEY (reef_id, word_id)
);
```

The central word-domain association table. Sources:
- `curated`: from WordNet domains (via WDH) + Claude-generated seeds
- `xgboost`: XGBoost classifier predictions above threshold

### Export Tables

Three export tables form a promotion chain. Each word appears at **exactly one** level:

#### ReefWordExports (~30K rows)
```sql
CREATE TABLE ReefWordExports (
    reef_id         INTEGER NOT NULL REFERENCES Reefs,
    word_id         INTEGER NOT NULL REFERENCES Words,
    idf             REAL,
    centroid_sim    REAL,           -- cosine to reef centroid
    name_cos        REAL,           -- cosine to hierarchy name embedding
    effective_sim   REAL,           -- blended similarity
    specificity     INTEGER,
    source_quality  REAL,
    export_weight   INTEGER NOT NULL CHECK (export_weight BETWEEN 0 AND 255),
    PRIMARY KEY (reef_id, word_id)
);
```

#### TownWordExports (~241K rows)
```sql
CREATE TABLE TownWordExports (
    town_id         INTEGER NOT NULL REFERENCES Towns,
    word_id         INTEGER NOT NULL REFERENCES Words,
    idf             REAL,
    centroid_sim    REAL,
    name_cos        REAL,
    effective_sim   REAL,
    specificity     INTEGER,
    source_quality  REAL,
    export_weight   INTEGER NOT NULL CHECK (export_weight BETWEEN 0 AND 255),
    export_town_weight INTEGER NOT NULL DEFAULT 128,
    PRIMARY KEY (town_id, word_id)
);
```

#### IslandWordExports (~145K rows)
```sql
CREATE TABLE IslandWordExports (
    island_id       INTEGER NOT NULL REFERENCES Islands,
    word_id         INTEGER NOT NULL REFERENCES Words,
    idf             REAL,
    centroid_sim    REAL,
    name_cos        REAL,
    effective_sim   REAL,
    specificity     INTEGER,
    source_quality  REAL,
    export_weight   INTEGER NOT NULL CHECK (export_weight BETWEEN 0 AND 255),
    export_island_weight INTEGER NOT NULL DEFAULT 128,
    PRIMARY KEY (island_id, word_id)
);
```

### Pipeline Support Tables

#### SeedWords
Town-level seed vocabulary before XGBoost expansion. Populated from WDH, Claude API, and Wikipedia.

#### AugmentedTowns
XGBoost predictions above threshold, per town. Populated by `train_town_xgboost.py`.

#### IslandWords
Words that are non-discriminative across towns within an island. Detected by cosine-std analysis and XGBoost post-filtering. These feed into IslandWordExports.

#### DimStats (768 rows)
Per-embedding-dimension statistics for z-score feature computation.

#### WordIslandStats
Per (word, island) concentration metrics, populated by `compute_word_stats.py`.

#### WordVariants
Morphological and stemming variants for tokenization. Feeds into EquivalencesExport.

### Views

#### ExportIndex
Unified view across all three export tables, resolving every exported word to its full hierarchy path. This is the primary query target for search validation:

```sql
-- Single word lookup
SELECT * FROM ExportIndex
WHERE word_id = (SELECT word_id FROM Words WHERE word = 'violin')
ORDER BY export_weight DESC;

-- Multi-word search (accumulate per island)
SELECT island, SUM(export_weight) AS score
FROM WordSearch
WHERE word IN ('hockey', 'stick', 'ice', 'rink')
GROUP BY island_id
ORDER BY score DESC LIMIT 10;
```

#### WordSearch
Joins ExportIndex with word text for human-readable results.

#### HierarchyPath
Full hierarchy path for any reef (archipelago > island > town > reef with word counts).

#### Compounds
Multi-word expressions for Aho-Corasick tokenization in Lagoon.

---

## Pipeline Stages

### Stage 1: Hierarchy Population

**Scripts:** `schema.sql`, `populate_archipelagos_islands.sql`, `populate_towns.sql`, `populate_bucket_islands.sql`

Creates the database and populates the three curated hierarchy levels. The hierarchy is grounded in the FBK WordNet Domain Hierarchy (WDH) for archipelagos and islands, extended for modern topics. Towns are curated from Wikipedia category analysis and domain expertise.

Six archipelagos: Applied Science, Pure Science, Social Science, Humanities, Free Time, Doctrines.

44 islands: 41 topical + 3 bucket (Languages, Regional, Miscellaneous).

332 towns: ~308 topical + 24 bucket.

All hierarchy data is defined in SQL INSERT scripts for version control and manual review. Scripts use subselects with readable labels instead of hardcoded IDs.

### Stage 2: Vocabulary & Embeddings

**Scripts:** `load_wordnet_vocab.py`, `reembed_words.py`, `compute_dimstats.py`

1. **Load vocabulary:** Extracts ~147K unique lemmas from NLTK WordNet 3.0. Computes POS, word_count, category, and FNV-1a word hashes.

2. **Embed words:** Uses nomic-embed-text-v1.5 with `"clustering: "` prefix (better suited for topical clustering than `"classification: "`). GPU-accelerated, ~1 min on RTX 4070 Ti. Supports checkpointing and `--missing-only` for incremental embedding.

3. **Dimension statistics:** Computes per-dimension mean, std, threshold (mean + 2.0*std), member count, and selectivity. Used for z-score feature engineering in XGBoost.

### Stage 3: Seed Population

**Scripts:** `import_wdh.py`, `populate_claude_seeds.sql`, `populate_compound_seeds.sql`, `link_seed_words.py`, `generate_seeds.py`, `generate_compound_seeds.py`, `sanity_check_seeds.py`, `detect_island_words.py`, `flag_stop_words.py --baseline`

Seeds are the starting vocabulary for each town, before XGBoost expansion.

1. **WDH import:** Maps 168 WDH domain labels to (island, town) pairs via the SKI cross-version bridge (WN 2.0 offsets → WN 3.0 synsets → lemmas). Resolution rate: ~98.3%.

2. **Claude seeds:** 80-150 discriminative single words per town, generated via Claude API with hierarchy context. Words are classified as core (unmistakably belongs to town) or peripheral (could appear in related towns). Claude is given sibling town context to ensure discrimination.

3. **Compound seeds:** Multi-word expressions generated separately.

4. **Linking:** `link_seed_words.py` matches seed words against the vocabulary table and sets word_ids. New words inserted during seeding are re-embedded.

5. **Sanity check:** Deduplicates words across sibling towns within the same island.

6. **Island word detection** (`detect_island_words.py`): Two-phase cleanup:
   - **Phase 1 (Capital triage):** For each capital town, computes cosine-std of seeds against non-capital town centroids. Generic seeds (std < NC p25) are promoted to IslandWords. Misplaced seeds (closer to a specific non-capital centroid) are moved.
   - **Phase 2 (Cross-town):** For all remaining seeds, words with low cosine-std across all town centroids are non-discriminative and promoted to IslandWords.

7. **Stop words:** Flags lagoon's function words (the, and, ...) in Words.is_stop.

### Stage 4: Classification & Clustering

**Scripts:** `train_town_xgboost.py`, `post_process_xgb.py`, `flag_stop_words.py --ubiquity`, `cluster_reefs.py`, `cluster_bucket_reefs.py`

#### XGBoost Training (per island)

Trains one binary classifier per town within each island.

**Feature engineering (776 features):**
- 768 z-score columns from embeddings
- 4 POS flags (noun, verb, adj, adv)
- 2 domain-specific: centroid cosine + KNN-5 mean cosine

**Negative sampling strategy:**
- 50% hard negatives from sibling towns in same island
- 50% easy negatives from global vocabulary
- 5:1 negative-to-positive ratio

**Post-training filters:**
1. **Island-level word detection:** Words predicted by ≥80% of towns in an island are non-discriminative → promoted to IslandWords, removed from AugmentedTowns.
2. **Capital starving:** For capital towns, predictions where any sibling also has the word are removed. Capitals get only words that no specific town claimed.

**Hyperparameters:** Same as v2 production (max_depth=8, n_estimators=2000, learning_rate=0.08, CUDA-accelerated).

**Model validation:** StratifiedGroupKFold (7-fold) cross-validation with morphological group awareness. F1 stored in Towns.model_f1.

#### XGBoost Post-Processing

For predictions with raw_score < 0.7:
```
idf = log2(N_towns / n_word_towns) / log2(N_towns)
adjusted = raw_score * idf
```
High-confidence predictions (≥ 0.7) pass through unchanged. Rows below 0.4 are pruned.

#### Ubiquity Pruning

Words appearing in 20+ towns:
- Score < 0.80: DELETE
- Score 0.80-0.95: multiply by 0.5
- Score ≥ 0.95: untouched

Additionally sets Words.is_stop for ubiquitous words.

#### Reef Clustering (per island)

For each town with ≥10 total words:

1. **Core/non-core split:** Seeds + high-confidence XGBoost (score ≥ 0.6) are core. Lower-confidence XGBoost predictions are non-core.
2. **PMI computation:** Pointwise Mutual Information from town co-membership within the island.
3. **Hybrid similarity:** `0.7 * embedding_cosine + 0.3 * pmi_cosine`
4. **kNN graph** (k=15), symmetrized
5. **Leiden clustering** (resolution=1.0, min_community_size=3)
6. **Centroid computation** from core members
7. **Non-core assignment** to nearest reef centroid
8. **Persist** to Reefs + ReefWords tables

### Stage 5: Stats & Export Promotion

**Scripts:** `compute_word_stats.py`, `compute_hierarchy_stats.py`, `populate_exports.py`

#### Word Stats

Computes per-word: reef_count, town_count, island_count, IDF, specificity, best cosine_sim. Also populates WordIslandStats with per (word, island) concentration metrics.

**Specificity categories** (from reef_count):
| reef_count | specificity | Meaning |
|------------|-------------|---------|
| 1 | 3 | Highly specific |
| 2-3 | 2 | Very specific |
| 4-5 | 1 | Specific |
| 6-10 | 0 | Moderate |
| 11-20 | -1 | Generic |
| 21+ | -2 | Very generic |

#### Export Promotion

`populate_exports.py` classifies each word into exactly one export level, then computes and normalizes weights. See [Export Promotion Rules](#export-promotion-rules) and [Scoring Formulas](#scoring-formulas).

### Stage 6: Binary Export

**Script:** `export.py`

Reads from the three export tables and serializes to 11 MessagePack binary files for Lagoon consumption. See [Export Format v3.1](#export-format-v31).

---

## Scoring Formulas

### Raw Score

```python
raw_score = idf * source_quality * effective_sim
```

### IDF

Two variants depending on export level:

```python
# Reef/town exports: reef-level IDF
global_idf = log2(total_reefs / reef_count)   # range ~8-11

# Island exports: island-level IDF (better differentiation)
island_idf = log2(N_islands / n_islands)       # range ~2.6-5.5
```

### Source Quality

```python
source_quality = max(raw_source_quality, SOURCE_QUALITY_FLOOR)  # floor = 0.9

# Raw source_quality values:
#   1.0  — curated (WordNet + Claude seeds)
#   0.9  — XGBoost core (score >= 0.6)
#   0.7  — XGBoost non-core
```

The floor of 0.9 softens the gap between XGBoost and curated words, preventing cross-island scoring artifacts.

### Effective Similarity

**Reef/town exports (three-signal blend):**
```python
a1 = 0.2  # GROUP_NAME_COS_ALPHA (reef/town name weight)
a2 = 0.3  # ISLAND_NAME_COS_ALPHA (island name weight)
effective_sim = (1 - a1 - a2) * centroid_sim + a1 * group_name_cos + a2 * island_name_cos
#             = 0.5 * centroid_sim + 0.2 * group_name_cos + 0.3 * island_name_cos
```

**Island exports (two-signal blend):**
```python
a = 0.5  # ISLAND_ONLY_ALPHA
effective_sim = (1 - a) * centroid_sim + a * island_name_cos
#             = 0.5 * centroid_sim + 0.5 * island_name_cos
```

**Constraint:** centroid_sim must be ≥ 50% of effective_sim at all levels. This is currently at exactly 50%.

### Normalization

**Reef/town exports:** Per-group min-max to [0, 255]. Each reef normalizes independently (for reef exports), each town for town exports.

**Island exports:** Hybrid normalization to solve cross-island comparability:
```python
ISLAND_GLOBAL_BLEND = 0.8  # 80% global + 20% per-island
w = round((1 - blend) * w_local + blend * w_global)
w = round(w * exclusivity_factor(n_islands))
```

Where `exclusivity_factor = 1 / n_islands^0.33` (cube root dampening).

**Why hybrid normalization:** Pure per-island min-max makes cross-island weight comparison unreliable. Islands with wider raw score ranges (e.g., Biology with 7,822 island-level words) compress mid-range words more than narrow islands. The global component preserves absolute score ordering across islands.

---

## Feature Engineering (776 Features)

XGBoost classifiers use 776 features per word per town.

### Global Features (772 columns, same for all towns)

**Embedding Z-scores (768 columns):**
```python
z_score[d] = (embedding[d] - dim_mean[d]) / dim_std[d]
```

**POS Flags (4 columns):**
- `is_noun`, `is_verb`, `is_adj`, `is_adv`: binary indicators derived from Words.pos

### Town-Specific Features (2 columns, recomputed per town)

**Centroid Cosine Similarity (1 column):**
```python
core_centroid = normalize(mean(embeddings[positive_word_ids]))
centroid_cos = normalize(embedding) @ core_centroid
```

**KNN Top-5 Mean Cosine (1 column):**
```python
cos_to_core = normalize(embedding) @ normalize(core_embeddings).T
top5_mean = mean(top_5_values(cos_to_core, axis=1))
```

### Cross-Validation

StratifiedGroupKFold (7-fold) preserves label distribution. Groups are word_ids (not morphological equivalence classes as in v2, since WordVariants is populated later in the v3 pipeline).

---

## Name Cosine Blending

A multi-level scoring signal that captures "if someone says this word, which domain comes to mind?" -- nearly orthogonal to centroid_sim (Pearson r ≈ 0.073).

### Three Signals

1. **centroid_sim**: "Does this word statistically belong in this reef/town cluster?" Strongest, most reliable signal.
2. **group_name_cos**: `cos(word_embedding, reef_or_town_name_embedding)`. Local topic steering.
3. **island_name_cos**: `cos(word_embedding, island_name_embedding)`. Broad domain steering -- captures that "violin" should go to Music even if it clusters well with Italian cultural words.

### Implementation

All hierarchy names (reef, town, island) are embedded at export time using the same nomic-embed-text-v1.5 model with `"clustering: "` prefix. Cosine similarities are computed per-word against each relevant name embedding.

### Signal Independence

centroid_sim and name_cos are nearly independent signals (Pearson r ≈ 0.073), making them ideal for blending. The name signal overwhelmingly dampens rather than boosts -- most words score lower against the domain name than against the cluster centroid. Domains with concrete, intuitive names benefit most (music, anatomy, astronomy).

---

## Clustering Algorithms

### PMI (Pointwise Mutual Information)

Computed from word-town co-membership within an island:
```python
P(t) = count_words_in(t) / total_vocab
P(t1, t2) = count_words_in_both(t1, t2) / total_vocab
PMI(t1, t2) = log2(P(t1,t2) / (P(t1) * P(t2)))
```
Only positive PMI retained. Produces a symmetric (n_towns × n_towns) matrix per island.

### Hybrid Similarity

```python
similarity = 0.7 * cosine_sim(embeddings) + 0.3 * cosine_sim(pmi_vectors)
```

### kNN Graph + Leiden

1. Compute pairwise hybrid similarity matrix
2. For each node, keep top-k=15 neighbors
3. Symmetrize (union of directed edges)
4. Run Leiden clustering (resolution=1.0, min_community_size=3)
5. Compute L2-normalized centroids from core members
6. Assign non-core words to nearest centroid

---

## Export Promotion Rules

Each word is classified into exactly one export level. No word appears in multiple export tables.

| Condition | Level | Rationale |
|-----------|-------|-----------|
| spec ≥ 0, 1 town in island | reef | Specific, single-town |
| spec ≥ 1, 2+ towns | town | Specific, multi-town |
| spec = 0, 2+ towns | island | Moderate, multi-town |
| spec = -1, 1 reef in island | reef | Singleton rescue |
| spec = -1, 2+ reefs in island | island | Generic + spread |
| spec ≤ -2 | island | Very generic |

Current distribution: ~30K reef rows, ~241K town rows, ~145K island rows.

---

## Export Format v3.1

**Format:** MessagePack binary files, deserialized by Lagoon.

### File Descriptions (11 files)

#### word_lookup.bin
```python
{u64_hash: [word_hash, word_id, specificity, idf_q], ...}
```
Hash table for O(1) word lookup. Priority: base words > morphy variants > snowball stems. `idf_q = clamp(round(idf * 21), 0, 255)`.

#### word_islands.bin
```python
[[], [], [[island_id, weight_q], ...], ...]
```
Sparse list indexed by word_id. Each entry lists island memberships with u8 weights. MAX(export_weight) per (word, island) across all three export tables.

#### word_towns.bin
```python
[[], [], [[town_id, weight_q], ...], ...]
```
Sparse list indexed by word_id. Town-level associations. Island-level exports are distributed to all child towns. This is the primary scoring unit for Lagoon.

#### island_meta.bin
```python
[{
    "arch_id": int,
    "name": str,
    "n_words": int,
    "iqf": int,             # placeholder
    "avg_specificity": float,
    "noun_frac": float,
    "verb_frac": float,
    "adj_frac": float,
    "adv_frac": float,
}, ...]
```

#### town_meta.bin
```python
[{
    "island_id": int,
    "name": str,
    "n_words": int,
    "tqf": int,             # placeholder
    "avg_specificity": float,
}, ...]
```

#### reef_meta.bin
```python
[{
    "town_id": int,
    "name": str,
    "n_words": int,
    "avg_specificity": float,
}, ...]
```

#### background.bin
```python
{
    "bg_mean": [float, ...],        # per-island background mean
    "bg_std": [float, ...],         # per-island background std (adjusted)
    "town_bg_mean": [float, ...],   # per-town background mean
    "town_bg_std": [float, ...],    # per-town background std (adjusted)
}
```

Background model for z-score normalization in Lagoon. Computed by sampling 1000 random 15-word queries, then adjusted via:
1. Power-law regression on reliable islands (expected hits ≥ 30)
2. Replace unreliable stds with regression predictions
3. Specificity modulation: `std / (1 + 0.3 * avg_specificity)`
4. Floor at BG_STD_FLOOR (0.1)

Both island-level and town-level background models are computed.

#### constants.bin
```python
{
    "N_ISLANDS": 41,                    # topical islands
    "N_TOWNS": 308,                     # topical towns
    "N_REEFS": int,
    "N_ARCHS": 6,
    "IDF_SCALE": 21,
    "WEIGHT_SCALE": 255.0,
    "FNV1A_OFFSET": 14695981039346656037,
    "FNV1A_PRIME": 1099511628211,
    "island_n_words": [n, ...],
    "town_n_words": [n, ...],
    "domainless_word_ids": [...],       # words not in any topical island
    "town_domainless_word_ids": [...],  # words not in any topical town
    "bucket_only_word_ids": [...],      # words only in bucket islands
    "bucket_island_names": [...],       # names of bucket islands
}
```

#### compounds.bin
```python
[[word_text, word_id], ...]
```
Multi-word expressions for Aho-Corasick tokenization in Lagoon.

#### word_detail.bin
```python
[[], [], [[island_id, town_id, reef_id, weight, level_code], ...], ...]
```
Sparse list indexed by word_id. Full hierarchy detail for each word. `level_code`: 0=reef, 1=town, 2=island. Sentinel -1 for missing town_id/reef_id at higher export levels.

#### bucket_words.bin
```python
[[], [], [[bucket_idx, weight_q], ...], ...]
```
Sparse list indexed by word_id. Non-topical bucket island associations, exported separately for optional loading.

#### manifest.json
```json
{
    "version": "3.1",
    "format": "msgpack",
    "build_timestamp": "ISO-8601",
    "files": {"filename.bin": "sha256_hex", ...},
    "stats": {
        "n_islands": 41,
        "n_towns": 308,
        "n_reefs": int,
        "n_archs": 6,
        "n_bucket_islands": 3,
        "n_words": 149691,
        "n_lookup_entries": int,
        "n_topical_words": int,
        "n_compounds": int,
        ...
    }
}
```

---

## Configuration Reference

### Scoring Constants (in `populate_exports.py`)

```python
GROUP_NAME_COS_ALPHA = 0.2    # weight of reef/town name cosine
ISLAND_NAME_COS_ALPHA = 0.3   # weight of island name cosine (reef/town exports)
ISLAND_ONLY_ALPHA = 0.5       # weight of island name cosine (island exports)
SOURCE_QUALITY_FLOOR = 0.9    # min source_quality for scoring
ISLAND_GLOBAL_BLEND = 0.8     # 80% global + 20% per-island normalization
```

**Constraint:** `GROUP_NAME_COS_ALPHA + ISLAND_NAME_COS_ALPHA ≤ 0.5` (centroid_sim must be ≥ 50%).

### Embedding Model

```python
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_PREFIX = "clustering: "   # changed from "classification: " in v2
EMBEDDING_DIM = 768
```

### XGBoost

```python
NEG_RATIO = 5              # 50% hard (sibling), 50% easy (global)
SCORE_THRESHOLD = 0.4      # min probability for predictions
CORE_SCORE_THRESHOLD = 0.6 # core vs non-core in ReefWords

# Hyperparameters (same as v2)
max_depth = 8, n_estimators = 2000, learning_rate = 0.08
subsample = 0.82, colsample_bytree = 0.84
reg_alpha = 0.12, reg_lambda = 0.06
early_stopping_rounds = 200
```

### Clustering

```python
ALPHA = 0.7                 # hybrid: 70% embedding, 30% PMI
KNN_K = 15
LEIDEN_RESOLUTION = 1.0
MIN_COMMUNITY_SIZE = 3
MIN_TOWN_SIZE = 10
```

### Ubiquity Pruning

```python
UBIQUITY_TOWN_THRESHOLD = 20
UBIQUITY_SCORE_FLOOR = 0.80
UBIQUITY_SCORE_CEILING = 0.95
UBIQUITY_PENALTY = 0.5
```

### Export

```python
FORMAT_VERSION = "3.1"
IDF_SCALE = 21              # max idf ~11.94 → 251, fits u8
WEIGHT_SCALE = 255.0
BG_STD_FLOOR = 0.1
```

---

## Lagoon Integration

Lagoon is the downstream consumer -- a Python search scoring library. It loads the 11 `.bin` files and provides:

- **Tokenization:** Aho-Corasick multi-word matching using `compounds.bin`, then FNV-1a lookup using `word_lookup.bin`
- **Domain scoring:** For each query word, look up island/town memberships from `word_islands.bin` / `word_towns.bin`, accumulate weighted scores
- **Z-score normalization:** Compare accumulated scores against `background.bin` (bg_mean, bg_std) to produce z-scores
- **Hierarchy-aware coherence:** Use island/town structure to detect coherent multi-level activations

The export format provides two scoring granularities:
- **Island-level** (`word_islands.bin`): 41 scoring units, coarser but more stable
- **Town-level** (`word_towns.bin`): 308 scoring units, finer discrimination

All weights are pre-computed and quantized to u8. The `export_weight` values already incorporate IDF, source quality, centroid similarity, name cosine blending, normalization, and exclusivity -- so Lagoon uses them as-is with no re-derivation.

---

## Important Implementation Details

### FNV-1a Hashing
```python
def fnv1a_u64(s):
    h = FNV1A_OFFSET  # 14695981039346656037
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * FNV1A_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h
```
SQLite stores as signed i64. Convert on read: `hash & 0xFFFFFFFFFFFFFFFF`.

### Embedding Prefix Change

V3 uses `"clustering: "` instead of v2's `"classification: "`. This is nomic-embed-text-v1.5's intended prefix for clustering tasks and produces better topical grouping. All word embeddings and hierarchy name embeddings use the same prefix.

### Capital Town Triage

Capital towns are catch-all towns for generic island-level vocabulary. The `detect_island_words.py` script uses an adaptive threshold (NC p25 of cosine-std across non-capital town centroids) to identify generic capital seeds and promote them to IslandWords. Misplaced seeds are relocated to their best-matching non-capital town.

### Post-Training Capital Starving

After XGBoost training, `train_town_xgboost.py` performs a second pass: for capital towns, any prediction where a sibling town also has the word is removed. This ensures capitals only contain words that no specific town claimed.

### Island-Level Word Detection (Two Sources)

Words become island-level from two independent detections:
1. **Pre-XGBoost** (`detect_island_words.py`): cosine-std analysis on seed words
2. **Post-XGBoost** (`train_town_xgboost.py`): words predicted by ≥80% of towns in the island

Both feed into the IslandWords table, which flows into IslandWordExports during the export phase.

### Bucket Islands

Three non-topical islands (Languages, Regional, Miscellaneous) are marked `is_bucket=1`. They:
- Skip XGBoost training entirely
- Are clustered separately by `cluster_bucket_reefs.py`
- Are exported in a separate `bucket_words.bin` file
- Are optionally loaded by Lagoon for annotation (not topical scoring)

### Test Battery

53 queries covering: shoal xfail regressions, core domain convergence, cross-domain disambiguation, low-health island stress tests, high-health island stress tests, and ambiguous word resolution. Current score: 47/53 pass (89%). The 6 failures are all upstream data gaps, not formula-fixable.

### Weight Tuning Loop

The fast iteration loop for weight quality:
```
populate_exports.py   (~15s, idempotent, clears + rebuilds all export tables)
test_battery.py       (~1s, 53 pass/fail queries with margin analysis)
```

See `v3/tuning.md` for the full tuning guide including diagnostic queries, parameter sensitivity, and approaches tried and rejected.

### Source Quality Distribution in ReefWords

| Source | quality | Fraction |
|--------|---------|----------|
| WordNet/curated | 1.0 | 60% |
| Claude augmented | 0.9 | 25% |
| XGBoost predicted | 0.7 | 15% |

### WDH Bridge (WN 2.0 → 3.0)

The FBK WordNet Domains use WN 2.0 synset offsets. NLTK ships WN 3.0 (NOT 3.1). The SKI (Sense Key Index) bridge maps 2.0→3.0 offsets with 98.3% resolution. Bridge data is in `v3/data/ski-pwn-sets.txt`.
