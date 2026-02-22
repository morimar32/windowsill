# Windowsill Technical Reference

Deep technical documentation for the windowsill project. This file is optimized for getting an LLM up to speed on the codebase -- it covers the database schema, scoring formulas, feature engineering, clustering algorithms, and export format in full detail.

For a human-readable overview, see [README.md](README.md).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Database Schema](#database-schema)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1: Extract](#stage-1-extract)
  - [Stage 2: Transform (XGBoost)](#stage-2-transform-xgboost)
  - [Stage 3: Reef Subdivision](#stage-3-reef-subdivision)
  - [Stage 4: Archipelago Clustering](#stage-4-archipelago-clustering)
  - [Stage 5: Scoring](#stage-5-scoring)
  - [Stage 6: Export (Load)](#stage-6-export-load)
- [Scoring Formulas](#scoring-formulas)
- [Feature Engineering (777 Features)](#feature-engineering-777-features)
- [Clustering Algorithms](#clustering-algorithms)
- [Export Format v6.0](#export-format-v60)
- [Configuration Reference](#configuration-reference)
- [Lagoon Integration](#lagoon-integration)
- [Important Implementation Details](#important-implementation-details)

---

## Architecture Overview

Windowsill is a six-stage pipeline that transforms raw word embeddings into a scored domain taxonomy, exported as compact binary files for the Lagoon search scoring library.

```
WordNet vocab + Claude domain words
        |
        v
  [extract.py]  ──> v2.db (words, embeddings, variants, domains)
        |
        v
  [transform.py] ──> XGBoost classifiers (models/*.json)
        |               augmented_domains table updated with predictions
        v
  [reef.py]     ──> domain_reefs, domain_reef_stats (sub-domain clusters)
        |
        v
  [archipelago.py] ──> domain_archipelagos, domain_archipelago_stats
        |
        v
  [score.py]    ──> domain_word_scores (materialized scores)
        |
        v
  [load.py]     ──> v2_export/*.bin (10 msgpack files + manifest.json)
        |
        v
  [Lagoon]      ──> real-time domain scoring of free text
```

The hierarchy is: **words** belong to **domains** (444), domains contain **sub-reefs** (5,150), and domains are grouped into **archipelagos** (11).

In the export format, domains are called "reefs" and archipelagos are called "archs" (legacy naming from the v1 pipeline where the hierarchy was dimensions → reefs → islands → archipelagos).

---

## Database Schema

All data lives in `v2.db` (SQLite, ~1 GB). Schema defined in `lib/db.py`.

### Core Tables

#### words (158,060 rows)
```sql
CREATE TABLE words (
    word_id    INTEGER PRIMARY KEY,
    word       TEXT NOT NULL,
    pos        TEXT,                -- dominant POS: n, v, a, r, or NULL
    category   TEXT,                -- single, compound, phrasal_verb, named_entity, taxonomic
    word_count INTEGER DEFAULT 1,   -- number of space-separated tokens
    word_hash  INTEGER,             -- FNV-1a u64 hash of word text
    total_dims INTEGER DEFAULT 0,   -- count of z-score dimensions where word is a member
    is_noun    INTEGER DEFAULT 0,
    is_verb    INTEGER DEFAULT 0,
    is_adj     INTEGER DEFAULT 0,
    is_adv     INTEGER DEFAULT 0,
    is_stop    INTEGER DEFAULT 0,   -- flagged as stop word
    is_wordnet INTEGER DEFAULT 1,   -- 1 = from WordNet, 0 = Claude-generated
    n_synsets  INTEGER DEFAULT 0,   -- WordNet synset count
    n_domains  INTEGER DEFAULT 0,   -- distinct domain count from augmented_domains
    embedding  BLOB                 -- float32 x 768 = 3072 bytes
)
```

Embeddings are stored as raw float32 bytes. Use `lib.db.pack_embedding(ndarray)` to serialize and `lib.db.unpack_embedding(blob)` to deserialize.

#### dim_stats (768 rows)
```sql
CREATE TABLE dim_stats (
    dim_id      INTEGER PRIMARY KEY,  -- 0..767
    mean        REAL,                 -- mean activation across all words
    std         REAL,                 -- standard deviation
    threshold   REAL,                 -- mean + 2.0 * std (z-score membership cutoff)
    n_members   INTEGER,              -- words exceeding threshold
    selectivity REAL                  -- 1.0 - (n_members / total_words)
)
```

#### word_variants (509,579 rows)
```sql
CREATE TABLE word_variants (
    variant_hash INTEGER NOT NULL,    -- FNV-1a u64 of variant text (stored as signed i64)
    variant      TEXT NOT NULL,
    word_id      INTEGER NOT NULL,
    source       TEXT NOT NULL,        -- "base", "morphy", or "snowball"
    PRIMARY KEY (variant_hash, word_id)
)
```

Note: SQLite stores the u64 hash as signed i64. The export pipeline converts back: `vh & 0xFFFFFFFFFFFFFFFF`.

### Domain Tables

#### wordnet_domains (10,463 rows)
```sql
CREATE TABLE wordnet_domains (
    domain      TEXT NOT NULL,
    word_id     INTEGER NOT NULL,
    word        TEXT NOT NULL,
    synset_name TEXT NOT NULL,
    gloss       TEXT NOT NULL,
    PRIMARY KEY (domain, word_id, synset_name)
)
```

Ground-truth domain associations from WordNet's topic and usage domain pointers.

#### augmented_domains (820,804 rows)
```sql
CREATE TABLE augmented_domains (
    domain        TEXT NOT NULL,
    word          TEXT NOT NULL,
    word_id       INTEGER,            -- NULL if word not in vocabulary
    matched_word  TEXT,               -- actual matched word (may differ from input)
    source        TEXT NOT NULL,       -- "wordnet", "claude_augmented", "xgboost", "pipeline", "morphy"
    confidence    TEXT,               -- "core" or "peripheral" for Claude words
    has_embedding INTEGER DEFAULT 0,
    score         REAL,               -- XGBoost prediction probability (for source=xgboost)
    PRIMARY KEY (domain, word)
)
```

The central domain membership table. Sources:
- `wordnet`: from wordnet_domains table
- `claude_augmented`: Claude-generated domain vocabularies
- `xgboost`: XGBoost classifier predictions above threshold
- `pipeline`/`morphy`: morphological variant expansion

### Clustering Tables

#### domain_reefs (807,988 rows)
```sql
CREATE TABLE domain_reefs (
    domain       TEXT NOT NULL,
    reef_id      INTEGER NOT NULL,    -- sub-reef ID within domain (0-based)
    word_id      INTEGER NOT NULL,
    word         TEXT NOT NULL,
    is_core      INTEGER NOT NULL,    -- 1 = Leiden core member, 0 = assigned to nearest
    centroid_sim REAL,                -- cosine similarity to sub-reef centroid
    PRIMARY KEY (domain, word_id)
)
```

#### domain_reef_stats (5,150 rows)
```sql
CREATE TABLE domain_reef_stats (
    domain     TEXT NOT NULL,
    reef_id    INTEGER NOT NULL,
    n_core     INTEGER NOT NULL,
    n_assigned INTEGER NOT NULL,
    n_total    INTEGER NOT NULL,
    label      TEXT,                  -- top 3 words joined with underscores
    top_words  TEXT,                  -- JSON array of top-10 characteristic words
    centroid   BLOB,                  -- L2-normalized float32 embedding
    PRIMARY KEY (domain, reef_id)
)
```

#### domain_archipelagos (444 rows)
```sql
CREATE TABLE domain_archipelagos (
    domain          TEXT NOT NULL PRIMARY KEY,
    archipelago_id  INTEGER NOT NULL,
    centroid_sim    REAL              -- cosine similarity to archipelago centroid
)
```

#### domain_archipelago_stats (11 rows)
```sql
CREATE TABLE domain_archipelago_stats (
    archipelago_id  INTEGER NOT NULL PRIMARY KEY,
    n_domains       INTEGER NOT NULL,
    label           TEXT,             -- top 3 domains joined with underscores
    top_domains     TEXT,             -- JSON array of top-10 domains
    centroid        BLOB              -- L2-normalized float32 embedding
)
```

### Scoring Table

#### domain_word_scores (807,988 rows)
```sql
CREATE TABLE domain_word_scores (
    domain          TEXT NOT NULL,
    word_id         INTEGER NOT NULL,
    reef_id         INTEGER NOT NULL,   -- sub-reef within domain (-1 if no reef)
    source          TEXT NOT NULL,
    source_quality  REAL NOT NULL,      -- 1.0 for curated, xgb_score for ML
    centroid_sim    REAL,               -- cosine sim to sub-reef centroid (default 0.8)
    n_word_domains  INTEGER NOT NULL,   -- how many domains this word appears in
    idf             REAL NOT NULL,      -- log2(N/n), floored at 0.1
    domain_score    REAL NOT NULL,      -- idf * source_quality * centroid_sim
    reef_score      REAL NOT NULL,      -- source_quality * centroid_sim
    PRIMARY KEY (domain, word_id)
)
```

This is the final pre-computed table that `load.py` reads for export.

---

## Pipeline Stages

### Stage 1: Extract

**Entry point:** `extract.py` (13 internal steps)

Builds `v2.db` from scratch:

1. Create fresh database with schema
2. Load WordNet vocabulary + compute embeddings via nomic-embed-text-v1.5 (`lib/vocab.py`)
3. Load Claude-generated domain words, embed any unmatched ones
4. Compute `dim_stats`: per-dimension mean, std, threshold (mean + 2.0*std), member count
5. Compute `total_dims` per word (how many dimensions each word is a member of)
6. Backfill POS and category from WordNet
7. Flag stop words
8. Compute synset counts
9. Compute FNV-1a word hashes
10. Expand morphy variants (WordNet morphological analysis)
11. Populate `wordnet_domains` from WordNet topic/usage domain pointers
12. Load `augmented_domains` (WordNet domains + Claude domains)
13. Backfill `n_domains` per word

**Key dependency:** `lib/vocab.py` imports `embedder` for embedding computation. The `embedder` module uses sentence-transformers with nomic-embed-text-v1.5 (Matryoshka 768-dim). This is only needed when building the database from scratch.

### Stage 2: Transform (XGBoost)

**Entry point:** `transform.py`

Trains one binary XGBoost classifier per domain and runs inference on all words.

#### Training pipeline:
1. Load all word embeddings, compute z-scores using `dim_stats`
2. Build global feature matrix (775 columns -- see [Feature Engineering](#feature-engineering-777-features))
3. For each domain with >= `AUGMENT_MIN_DOMAIN_WORDS` (20) positive examples:
   - Positive set: domain members with embeddings
   - Negative set: random non-members at `AUGMENT_NEG_RATIO` (5:1) ratio
   - Add 2 domain-specific features (centroid cosine + KNN-5 mean cosine)
   - Train XGBoost with StratifiedGroupKFold CV (morphological groups prevent data leakage)
   - Save model to `models/{domain}.json`

#### Inference pipeline:
1. Load each trained model
2. Score all words (build domain-specific features per domain)
3. Insert predictions above `XGBOOST_SCORE_THRESHOLD` (0.4) into `augmented_domains`

#### IDF Post-Processing (`post_process_xgb.py`):
For XGBoost predictions with raw_score < 0.7:
```
idf = log2(N_total_domains / n_word_domains) / log2(N_total_domains)
adjusted_score = raw_score * idf
```
High-confidence predictions (>= 0.7) pass through unchanged. Words appearing in many domains get penalized. After adjustment, rows below 0.4 are pruned.

Domainless words (simple single words with embeddings that aren't in any domain) are tagged with pseudo-domain "domainless".

### Stage 3: Reef Subdivision

**Entry point:** `reef.py`

Subdivides each domain into semantic sub-reefs using Leiden community detection.

For each domain with >= `REEF_MIN_DOMAIN_SIZE` (10) core words (score >= `REEF_SCORE_THRESHOLD` 0.6):

1. **Compute hybrid similarity matrix:**
   ```
   sim = 0.7 * embedding_cosine + 0.3 * pmi_cosine
   ```
   PMI vectors encode domain co-membership patterns (see [Clustering Algorithms](#clustering-algorithms))

2. **Build kNN graph** (k=15), symmetrized

3. **Run Leiden clustering** (resolution=1.0, min_community_size=3)

4. **Compute sub-reef centroids** from core members

5. **Assign non-core words** to nearest centroid (if cosine >= 0.05)

6. **Store results** in `domain_reefs` and `domain_reef_stats`

### Stage 4: Archipelago Clustering

**Entry point:** `archipelago.py`

Groups the 444 domains into ~11 higher-level archipelagos.

1. **Compute domain embeddings** as weighted average of sub-reef centroids (weighted by sub-reef size)
2. **Compute PMI matrix** from word-domain co-memberships
3. **Build hybrid similarity** (same formula: 70% embedding + 30% PMI)
4. **kNN graph** (k=10) + **Leiden clustering** (resolution=1.0, min_community_size=2)
5. **Store results** in `domain_archipelagos` and `domain_archipelago_stats`

### Stage 5: Scoring

**Entry point:** `score.py`

Materializes `domain_word_scores` from `augmented_domains` + `domain_reefs`.

For each (domain, word_id) pair:
1. Resolve `source_quality` (1.0 for curated, xgb_score for ML predictions)
2. Look up `centroid_sim` from `domain_reefs` (default 0.8 if no reef assignment)
3. Compute `idf = max(log2(N_total / n_word_domains), 0.1)`
4. Compute `domain_score = idf * source_quality * centroid_sim`
5. Compute `reef_score = source_quality * centroid_sim`

Pairs are deduplicated by (domain, word_id), keeping the highest source_quality.

Includes 17 built-in test queries for verification (e.g., "DNA genetic mutation heredity" should surface biology/genetics domains in top 10).

### Stage 6: Export (Load)

**Entry point:** `load.py`

Exports the database into 10 MessagePack binary files for Lagoon consumption. See [Export Format v6.0](#export-format-v60) for full details.

---

## Scoring Formulas

### Source Quality
```python
if source in ("wordnet", "claude_augmented"):
    source_quality = 1.0
else:  # xgboost predictions
    source_quality = max(xgb_score or 0.5, 0.1)
```

### IDF (Inverse Domain Frequency)
```python
idf = max(log2(n_total_domains / n_word_domains), 0.1)
```
Words in fewer domains score higher. Floored at 0.1.

### Domain Score
```python
domain_score = idf * source_quality * centroid_sim
```
This is the primary weight exported to Lagoon. Quantized as:
```python
weight_q = clamp(round(domain_score * WEIGHT_SCALE), 0, 65535)  # u16
```
Where `WEIGHT_SCALE = 100.0`.

### Reef Score
```python
reef_score = source_quality * centroid_sim
```
Used internally for clustering quality assessment.

### IDF Quantization (for word lookup)
```python
idf_q = clamp(round(idf * IDF_SCALE), 0, 255)  # u8
```
Where `IDF_SCALE = 51`.

### Specificity Categories
Based on n_word_domains (number of domains a word appears in):
```
<= 3 domains:   specificity = 2  (highly specific)
<= 10 domains:  specificity = 1
<= 50 domains:  specificity = 0
<= 150 domains: specificity = -1
> 150 domains:  specificity = -2 (very generic)
```

---

## Feature Engineering (777 Features)

XGBoost classifiers use 777 features per word per domain.

### Global Features (775 columns, same for all domains)

**Embedding Z-scores (768 columns):**
```python
z_score[i] = (embedding[i] - dim_mean[i]) / dim_std[i]
```
Using `dim_stats` mean and std per dimension.

**Metadata Features (7 columns):**
- `total_dims`: number of dimension memberships
- `is_noun`, `is_verb`, `is_adj`, `is_adv`: binary POS indicators
- `n_synsets`: WordNet synset count
- `n_domains`: domain count

### Domain-Specific Features (2 columns, recomputed per domain)

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
- **StratifiedGroupKFold** preserves label distribution and morphological groups
- Groups: words linked via shared `variant_hash` values (morphy equivalence classes)
- Union-find builds equivalence classes to prevent train/test leakage between word forms

### XGBoost Hyperparameters
```python
objective = "binary:logistic"
device = "cuda"
max_depth = 8
n_estimators = 2000
learning_rate = 0.08
subsample = 0.82
colsample_bytree = 0.84
scale_pos_weight = n_neg / n_pos
reg_alpha = 0.12
reg_lambda = 0.06
early_stopping_rounds = 200
eval_metric = "logloss"
```

---

## Clustering Algorithms

### PMI (Pointwise Mutual Information)

Global co-occurrence metric between domains, computed from word-domain memberships:

```python
P(d) = count_words_in(d) / total_vocab
P(d1, d2) = count_words_in_both(d1, d2) / total_vocab
PMI(d1, d2) = log2(P(d1,d2) / (P(d1) * P(d2)))
```

Only positive PMI retained (negative clamped to 0). Produces a symmetric (n_domains x n_domains) matrix.

**PMI word vectors:** For reef clustering within a domain, each word gets a PMI vector of length n_domains, where entry[i] = PMI(parent_domain, domain_i) if the word is also in domain_i, else 0.

### Hybrid Similarity

Used for both reef subdivision and archipelago clustering:

```python
similarity = alpha * cosine_sim(embeddings) + (1 - alpha) * cosine_sim(pmi_vectors)
```

- Reef level: `alpha = 0.7` (REEF_ALPHA)
- Archipelago level: `alpha = 0.7` (ARCH_ALPHA)
- Fallback: embedding-only cosine if a word/domain lacks PMI neighbors

### kNN Graph Construction

1. Compute pairwise hybrid similarity matrix
2. For each node, keep top-k neighbors (excluding self)
3. Symmetrize: union of directed k-nearest edges
4. Deduplicate: keep maximum weight for duplicate edges
5. Result: undirected weighted iGraph graph

Parameters: k=15 for reefs, k=10 for archipelagos.

### Leiden Community Detection

Uses `leidenalg` with `RBConfigurationVertexPartition`:

```python
partition = la.find_partition(graph, RBConfigurationVertexPartition,
                              weights="weight", resolution_parameter=resolution)
```

Post-processing:
1. Communities smaller than `min_community_size` are merged into noise (label = -1)
2. Remaining communities relabeled contiguously (0, 1, 2, ...)
3. Returns: label array and modularity score

Parameters:
- Reefs: resolution=1.0, min_community_size=3
- Archipelagos: resolution=1.0, min_community_size=2

### Centroid Computation and Assignment

For each cluster (sub-reef or archipelago):
1. Centroid = L2-normalized mean of member embeddings
2. Non-core words assigned to nearest centroid (if cosine >= 0.05)
3. Unassigned words labeled as noise (reef_id = -1)

---

## Export Format v6.0

**Format:** MessagePack binary files, deserialized by Lagoon.

### Terminology Mapping (DB → Export)

| Database concept | Export concept | Reason |
|-----------------|---------------|--------|
| Domain (444) | Reef | Lagoon's hierarchy: reef → island → arch |
| Domain (same) | Island | 1:1 with reef in v2 (no separate island layer) |
| Sub-reef (5,150) | Sub-reef | Subdivision within a domain |
| Archipelago (11) | Arch | Top-level grouping |

### File Descriptions

#### word_lookup.bin (~4.8 MB)
```python
{word_hash: [word_hash, word_id, specificity, idf_q], ...}
```
Hash table for O(1) word lookup. Priority resolution: base words > morphy variants > snowball stems. Within same priority, higher specificity wins.

#### word_reefs.bin (~7.4 MB)
```python
[
    [],                                    # word_id 0 (unused)
    [[reef_id, weight_q, sub_reef_id], ...],  # word_id 1
    ...
]
```
Indexed by word_id. Each entry lists domain memberships with quantized weights (u16).

#### reef_meta.bin (~72 KB)
```python
[{
    "hierarchy_addr": u32,    # pack: (arch_id << 16) | reef_id
    "n_words": int,
    "name": str,              # domain name
    "valence": 0.0,           # placeholder (no valence in v2)
    "avg_specificity": float,
    "noun_frac": float,
    "verb_frac": float,
    "adj_frac": float,
    "adv_frac": float,
}, ...]
```

#### island_meta.bin (~11 KB)
```python
[{"arch_id": int, "name": str}, ...]
```
1:1 with reef_meta in v2 (each domain is both a "reef" and an "island").

#### sub_reef_meta.bin (~347 KB)
```python
[{"parent_island_id": int, "n_words": int, "name": str}, ...]
```
5,150 sub-reef records with Leiden cluster labels as names.

#### background.bin (~8 KB)
```python
{"bg_mean": [float, ...], "bg_std": [float, ...]}
```
Per-reef background model for z-score normalization in Lagoon. Computed by sampling 1000 random 15-word queries and measuring per-reef score distributions.

**Background adjustment algorithm:**
1. Identify "unreliable" reefs: expected sample hit rate < 30 occurrences
2. Fit power-law regression on reliable reefs: `log(std) = a * log(mean) + b` (defaults: a=0.513, b=1.482)
3. Replace unreliable stds with regression prediction (floored at median reliable std)
4. Specificity modulation: `adjusted_std[r] = std[r] / (1.0 + 0.3 * avg_specificity[r])`

#### constants.bin (~131 KB)
```python
{
    "N_REEFS": 444,
    "N_ISLANDS": 444,
    "N_ARCHS": 11,
    "N_SUB_REEFS": 5150,
    "avg_reef_words": float,
    "IDF_SCALE": 51,
    "WEIGHT_SCALE": 100.0,
    "FNV1A_OFFSET": 14695981039346656037,
    "FNV1A_PRIME": 1099511628211,
    "reef_total_dims": [0, ...],      # always 0 in v2
    "reef_n_words": [n, ...],         # words per reef
    "domainless_word_ids": [id, ...], # words not in any domain
}
```

#### compounds.bin (~1.4 MB)
```python
[[word_text, word_id], ...]
```
69,793 multi-word expressions for Aho-Corasick tokenization in Lagoon.

#### reef_edges.bin (~1 byte)
Empty in v2 (no inter-reef edges at the domain level).

#### word_reef_detail.bin (~158 KB)
```python
[[], [], [[island_id, sub_reef_id, weight_q], ...], ...]
```
Indexed by word_id. Sub-reef level detail for words that have sub-reef assignments.

#### manifest.json
```json
{
    "version": "6.0",
    "format": "msgpack",
    "build_timestamp": "ISO-8601",
    "files": {"filename.bin": "sha256_hex", ...},
    "stats": {
        "n_reefs": 444,
        "n_islands": 444,
        "n_archs": 11,
        "n_words": 158060,
        "n_lookup_entries": 186184,
        "n_words_with_reefs": 126163,
        "n_compounds": 69793,
        "n_sub_reefs": 5150,
        "n_edges": 0
    }
}
```

### Hierarchy Address Packing
```python
def pack_hierarchy_addr(arch_id, reef_id):
    return (arch_id << 16) | reef_id  # u32: arch(16) | reef(16)
```

---

## Configuration Reference

All constants in `config.py`:

```python
# FNV-1a u64 hashing
FNV1A_OFFSET = 14695981039346656037
FNV1A_PRIME  = 1099511628211

# Domain augmentation (Claude API)
AUGMENT_CLAUDE_MODEL     = "claude-sonnet-4-5-20250929"
AUGMENT_BATCH_SIZE       = 5       # domains per API call
AUGMENT_API_DELAY        = 0.5     # seconds between calls
AUGMENT_MIN_DOMAIN_WORDS = 20      # min matched words for XGBoost training
AUGMENT_NEG_RATIO        = 5       # negative:positive sampling ratio

# XGBoost threshold
XGBOOST_SCORE_THRESHOLD  = 0.4    # min score for domain membership

# Reef subdivision (Leiden clustering within domains)
REEF_SCORE_THRESHOLD       = 0.6  # min xgb score for core words
REEF_ALPHA                 = 0.7  # hybrid: 70% embedding, 30% PMI
REEF_KNN_K                 = 15   # kNN neighbors
REEF_LEIDEN_RESOLUTION     = 1.0
REEF_MIN_COMMUNITY_SIZE    = 3    # merge smaller communities into noise
REEF_MIN_DOMAIN_SIZE       = 10   # skip domains with fewer core words
REEF_CHARACTERISTIC_WORDS_N = 10  # top words per sub-reef label

# Archipelago clustering (domain-level)
ARCH_ALPHA                 = 0.7  # hybrid: 70% embedding, 30% PMI
ARCH_KNN_K                 = 10   # kNN neighbors
ARCH_LEIDEN_RESOLUTION     = 1.0
ARCH_MIN_COMMUNITY_SIZE    = 2
ARCH_CHARACTERISTIC_DOMAINS_N = 10

# Export
EXPORT_WEIGHT_THRESHOLD    = 0.01  # min weight to include in export
```

---

## Lagoon Integration

Lagoon is the downstream consumer -- a Python search scoring library at `../lagoon/src/lagoon/`. It loads the 10 `.bin` files and provides:

- **Tokenization:** Aho-Corasick multi-word matching using `compounds.bin`, then FNV-1a lookup using `word_lookup.bin`
- **Domain scoring:** For each query word, look up domain memberships from `word_reefs.bin`, accumulate weighted scores per domain
- **Z-score normalization:** Compare accumulated scores against `background.bin` (bg_mean, bg_std) to produce z-scores
- **Ranking:** Return domains ranked by z-score

The export format is designed for fast deserialization and O(1) lookups. All numeric values are pre-quantized to minimize runtime computation.

---

## Important Implementation Details

### FNV-1a Hashing (`word_list.py`)
```python
def fnv1a_u64(s):
    h = FNV1A_OFFSET  # 14695981039346656037
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * FNV1A_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h
```
Used for all word lookups. Deterministic, fast, good distribution. The same hash function must be used by Lagoon.

### Embedding Model
- **Model:** nomic-embed-text-v1.5
- **Dimensions:** 768 (Matryoshka)
- **Prefix:** `"search_document: "` for indexing, `"search_query: "` for queries
- **Storage:** float32 blobs in SQLite (3072 bytes per word)

### Z-Score Dimension Membership
A word is a "member" of dimension d if:
```
embedding[d] > dim_stats.mean[d] + 2.0 * dim_stats.std[d]
```
This produces ~1-2% membership rate per dimension, with `total_dims` averaging ~17.5 per word.

### Word Priority in Lookup Table
When multiple variants map to the same hash:
1. Base words (priority 0) > morphy variants (priority 1) > snowball stems (priority 2)
2. Within same priority: higher specificity wins
3. Ensures the most informative word form is returned

### Signed/Unsigned Hash Conversion
SQLite stores integers as signed i64. FNV-1a produces u64 values. The export pipeline converts:
```python
unsigned_hash = signed_hash & 0xFFFFFFFFFFFFFFFF
```

### Morphological Groups for CV
XGBoost cross-validation uses morphological equivalence classes (union-find on shared variant_hash values) to prevent data leakage. Words like "run", "running", "ran" are always in the same CV fold.

### Background Model Sampling
The background model represents "what does a random query look like?" for each domain:
- 1000 random samples of 15 single words each
- Per-domain mean and std of raw accumulated scores
- Unreliable domains (low expected hit rate) get regression-predicted stds
- Specificity modulation: more specific domains get tighter (lower) stds, making it easier to achieve a significant z-score

### The "domainless" Pseudo-Domain
Words with embeddings that don't belong to any real domain are tagged as "domainless" in `augmented_domains`. Their word_ids are stored in `constants.bin["domainless_word_ids"]` for Lagoon to handle specially (they contribute to tokenization but not domain scoring).
