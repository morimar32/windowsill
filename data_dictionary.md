# Windowsill Data Dictionary

## 1. Overview

This document is the authoritative column-level reference for the Windowsill database (`vector_distillery.duckdb`). It covers every table, view, and index in the schema.

| Metric | Value |
|--------|-------|
| Database engine | DuckDB |
| Database file | `vector_distillery.duckdb` (~2.6 GB) |
| Tables | 14 |
| Views | 6 |
| Indexes | 21 |
| Words | ~146,000 (WordNet lemmas) |
| Embedding dimensions | 768 (nomic-embed-text-v1.5, Matryoshka) |
| Total memberships | ~2.56M |
| Island hierarchy | 4 archipelagos → 52 islands → 208 reefs |
| Reef affinity | Continuous word-reef affinity scores in `word_reef_affinity` table |
| Reef refinement | Iterative dim loyalty analysis (phase 10): dims with higher Jaccard affinity to a sibling reef than their own are reassigned |

---

## 2. Entity Relationships

### ER Diagram

```
                          ┌──────────────────────┐
                          │        words          │
                          │  word_id (PK)         │
                          │  word, embedding, ... │
                          └──────────┬────────────┘
                 ┌───────────┬───────┼────────┬──────────────┬──────────────┐
                 │           │       │        │              │              │
                 ▼           ▼       ▼        ▼              ▼              ▼
          ┌────────────┐ ┌───────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐
          │  word_pos   │ │word_  │ │dim_member-   │ │word_pair_    │ │compositionality │
          │(word_id,pos)│ │compo- │ │ships         │ │overlap       │ │(word_id PK)     │
          │  PK         │ │nents  │ │(dim_id,      │ │(word_id_a,   │ └─────────────────┘
          └────────────┘ │(comp- │ │ word_id) PK  │ │ word_id_b)PK │
                         │ound_  │ └──────┬───────┘ └──────────────┘
                         │word_id│        │
                         │,pos)  │        │  ┌───────────────┐
                         │PK     │        │  │   dim_stats    │
                         └───────┘        │  │  dim_id (PK)   │
                                          │  └───────┬────────┘
                  ┌───────────────┐       │          │
                  │  word_senses  │       │          │
                  │ sense_id (PK) │◄──────┘          │
                  │ word_id (FK)  │                   │
                  └───────┬───────┘                   │
                          │                           │
                          ▼                           ▼
                 ┌──────────────────┐        ┌──────────────┐
                 │sense_dim_member- │        │  dim_jaccard  │
                 │ships             │        │(dim_id_a,     │
                 │(dim_id,sense_id) │        │ dim_id_b) PK  │
                 │PK                │        └──────────────┘
                 └──────────────────┘                │
                                                     ▼
                                             ┌──────────────┐
                                             │ dim_islands   │
                                             │(dim_id,       │
                                             │ generation)PK │
                                             └───────┬───────┘
                                                     │
                                                     ▼
                                             ┌──────────────────────┐
                                             │    island_stats       │
                                             │(island_id,generation) │
                                             │PK                     │
                                             └───────┬───────────────┘
                                                     │
                                              ┌──────┴──────┐
                                              ▼             ▼
                                  ┌──────────────────────────┐  ┌──────────────────────┐
                                  │island_characteristic_words│  │ word_reef_affinity    │
                                  │(island_id, generation,    │  │(word_id, reef_id) PK  │
                                  │ word_id) PK               │  │ word_id FK → words    │
                                  └──────────────────────────┘  └──────────────────────┘
```

### Foreign Key Relationships

All FK relationships are enforced by application-level integrity checks (`database.run_integrity_checks`), not by DuckDB constraints.

| Child Table | Child Column | Parent Table | Parent Column | Notes |
|-------------|-------------|--------------|---------------|-------|
| dim_memberships | word_id | words | word_id | |
| dim_memberships | dim_id | dim_stats | dim_id | Implicit; both use 0-767 |
| word_pos | word_id | words | word_id | |
| word_components | compound_word_id | words | word_id | |
| word_components | component_word_id | words | word_id | NULL if component not in vocabulary |
| word_senses | word_id | words | word_id | |
| sense_dim_memberships | sense_id | word_senses | sense_id | |
| sense_dim_memberships | dim_id | dim_stats | dim_id | Implicit |
| compositionality | word_id | words | word_id | |
| word_pair_overlap | word_id_a | words | word_id | |
| word_pair_overlap | word_id_b | words | word_id | |
| dim_jaccard | dim_id_a | dim_stats | dim_id | |
| dim_jaccard | dim_id_b | dim_stats | dim_id | |
| dim_islands | dim_id | dim_stats | dim_id | |
| island_characteristic_words | word_id | words | word_id | |
| word_reef_affinity | word_id | words | word_id | |
| word_reef_affinity | reef_id | island_stats | island_id (generation=2) | Reef-level islands only |
| island_stats → dim_islands | island_id + generation | dim_islands | island_id + generation | Logical, not enforced |

### Cardinality

```
words (1) ──< dim_memberships (M)    ~17.5 memberships per word on average
dim_stats (1) ──< dim_memberships (M)    ~3,339 members per dimension on average
words (1) ──< word_pos (M)              1-4 POS tags per word
words (1) ──< word_components (M)       Multi-word compounds only; 2+ components
words (1) ──< word_senses (M)           Ambiguous words only (~8% of vocab)
word_senses (1) ──< sense_dim_memberships (M)
dim_stats (1) ──< dim_islands (M)       1 row per generation (up to 3 generations)
island_stats (1) ──< island_characteristic_words (M)    Up to 100 words per island
words (M) ──< word_pair_overlap (M)     Symmetric pairs where a < b
words (1) ──< word_reef_affinity (M)   One row per word-reef pair (every reef the word touches)
island_stats (1) ──< word_reef_affinity (M)   reef_id references island_stats(island_id, generation=2)
```

### Common Join Patterns

**Word → its dimension memberships → dimension statistics:**
```sql
SELECT w.word, dm.dim_id, dm.z_score, ds.selectivity, ds.threshold
FROM words w
JOIN dim_memberships dm ON w.word_id = dm.word_id
JOIN dim_stats ds ON dm.dim_id = ds.dim_id
WHERE w.word = 'guitar';
```

**Word → its island/reef assignments (via dimension memberships):**
```sql
SELECT w.word, di.island_id, di.generation, ist.island_name, COUNT(*) as n_dims
FROM words w
JOIN dim_memberships dm ON w.word_id = dm.word_id
JOIN dim_islands di ON dm.dim_id = di.dim_id
JOIN island_stats ist ON di.island_id = ist.island_id AND di.generation = ist.generation
WHERE w.word = 'guitar' AND di.island_id >= 0
GROUP BY w.word, di.island_id, di.generation, ist.island_name
ORDER BY di.generation, n_dims DESC;
```

**Word → its reefs (direct, using denormalized columns — no dim_islands join needed):**
```sql
SELECT w.word, dm.reef_id, ist.island_name, COUNT(*) as n_dims
FROM words w
JOIN dim_memberships dm ON w.word_id = dm.word_id
JOIN island_stats ist ON dm.reef_id = ist.island_id AND ist.generation = 2
WHERE w.word = 'guitar' AND dm.reef_id IS NOT NULL
GROUP BY w.word, dm.reef_id, ist.island_name
ORDER BY n_dims DESC;
```

**Compound word → its components → component dimension overlap:**
```sql
SELECT wc.component_text, w_comp.total_dims, w_comp.pos
FROM word_components wc
LEFT JOIN words w_comp ON wc.component_word_id = w_comp.word_id
WHERE wc.compound_word_id = (SELECT word_id FROM words WHERE word = 'heart attack')
ORDER BY wc.position;
```

**Two words' relationship via denormalized columns:**
```sql
SELECT
    CASE
        WHEN COUNT(DISTINCT CASE WHEN a.reef_id = b.reef_id THEN a.reef_id END) > 0
            THEN 'same reef'
        WHEN COUNT(DISTINCT CASE WHEN a.island_id = b.island_id THEN a.island_id END) > 0
            THEN 'reef neighbors (same island)'
        WHEN COUNT(DISTINCT CASE WHEN a.archipelago_id = b.archipelago_id THEN a.archipelago_id END) > 0
            THEN 'island neighbors (same archipelago)'
        ELSE 'different archipelagos'
    END as relationship
FROM dim_memberships a
JOIN dim_memberships b ON a.dim_id = b.dim_id
JOIN words wa ON wa.word_id = a.word_id
JOIN words wb ON wb.word_id = b.word_id
WHERE wa.word = 'cat' AND wb.word = 'dog';
```

---

## 3. Core Tables

### 3.1 `words`

**Purpose:** One row per vocabulary entry. The central fact table linking to all other tables.

**Row count:** ~146,000

**Primary key:** `word_id`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `word_id` | INTEGER | No (PK) | Sequential ID, 1-indexed, assigned in alphabetical order. | `1`, `72345`, `146000` |
| `word` | TEXT | No | The word or multi-word expression. Lowercase; spaces separate tokens in compounds (WordNet underscores converted). | `'cat'`, `'heart attack'`, `'hot dog'` |
| `total_dims` | INTEGER | No | Number of dimensions this word is a member of. Computed in phase 6 from `COUNT(*) FROM dim_memberships`. Range: ~1–44. | `5`, `17`, `38` |
| `embedding` | FLOAT[768] | No | Raw 768-dimensional embedding vector from nomic-embed-text-v1.5, encoded with `"classification: "` prefix. | `[0.0123, -0.0456, ...]` |
| `pos` | TEXT | Yes | Dominant part-of-speech. Set to the unambiguous POS if the word has only one POS in WordNet; **NULL if the word has multiple POS** (e.g., "bank" is both noun and verb). | `'noun'`, `'verb'`, `'adj'`, `'adv'`, `NULL` |
| `category` | TEXT | Yes | Word classification. Set in phase 4b by `word_list.classify_word()`. | `'single'`, `'compound'`, `'taxonomic'`, `'phrasal_verb'`, `'named_entity'` |
| `word_count` | INTEGER | No | Number of space-separated tokens. `DEFAULT 1`. | `1` (single), `2` (heart attack), `3` (out of bounds) |
| `specificity` | INTEGER | No | Sigma-band classification based on distance from mean `total_dims`. Positive = specific (fewer dims than normal), negative = universal (more dims). Computed as: `+2` if `total_dims <= mean - 2*std`, `+1` if `<= mean - std`, `-1` if `>= mean + 2*std`, `-2` if `>= mean + std`, else `0`. `DEFAULT 0`. | `2`, `1`, `0`, `-1`, `-2` |
| `sense_spread` | INTEGER | Yes | `MAX(total_dims) - MIN(total_dims)` across all senses in `word_senses` for this word. **NULL if the word has fewer than 2 senses.** Only meaningful for ambiguous words. | `NULL`, `3`, `22` |
| `polysemy_inflated` | BOOLEAN | No | `TRUE` if `specificity < 0` AND `sense_spread >= 15` (config `SENSE_SPREAD_INFLATED_THRESHOLD`). Flags universal words whose high dim count may be driven by having very different senses. `DEFAULT FALSE`. | `TRUE`, `FALSE` |
| `arch_concentration` | DOUBLE | Yes | Fraction of this word's dimension memberships concentrated in a single gen-0 archipelago: `MAX(count_per_archipelago) / SUM(count_per_archipelago)`. **Only computed for universal words** (`specificity < 0`) that have ≥ 2 dimension memberships in islands. NULL for specific/typical words and words with insufficient island data. Range: ~0.3–1.0. | `NULL`, `0.65`, `0.92` |


### 3.2 `dim_stats`

**Purpose:** One row per embedding dimension. Stores distribution statistics, membership thresholds, POS enrichment ratios, and abstractness metrics.

**Row count:** 768 (exactly one per dimension)

**Primary key:** `dim_id`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `dim_id` | INTEGER | No (PK) | Dimension index, 0-based. Range: 0–767. | `0`, `384`, `767` |
| `mean` | DOUBLE | Yes | Mean activation value across all ~146K words for this dimension. | `-0.0012` |
| `std` | DOUBLE | Yes | Standard deviation of activation values. | `0.0234` |
| `min_val` | DOUBLE | Yes | Minimum activation value observed. | `-0.0891` |
| `max_val` | DOUBLE | Yes | Maximum activation value observed. | `0.1456` |
| `median` | DOUBLE | Yes | Median activation value. | `-0.0008` |
| `skewness` | DOUBLE | Yes | Skewness of activation distribution (scipy). | `0.34`, `1.87` |
| `kurtosis` | DOUBLE | Yes | Excess kurtosis of activation distribution (scipy). | `2.15`, `8.73` |
| `threshold` | DOUBLE | Yes | Activation threshold for membership. Computed as `mean + ZSCORE_THRESHOLD * std` where `ZSCORE_THRESHOLD = 2.0`. A word is a member of this dimension if its raw value ≥ this threshold. | `0.0456` |
| `threshold_method` | TEXT | Yes | Method used to compute the threshold. Always `'zscore'`. | `'zscore'` |
| `n_members` | INTEGER | Yes | Number of words that exceed the threshold (members of this dimension). Range: typically 1,000–8,000. | `2450`, `5120` |
| `selectivity` | DOUBLE | Yes | Fraction of vocabulary that are members: `n_members / total_words`. Lower = more selective/specific dimension. Range: ~0.007–0.055. | `0.0168`, `0.0350` |
| `verb_enrichment` | DOUBLE | Yes | Ratio of verb rate among this dimension's members vs. corpus-wide verb base rate. `> 1.0` means verbs are overrepresented. Added in phase 6b. | `0.45`, `3.21` |
| `adj_enrichment` | DOUBLE | Yes | Same ratio for adjectives. | `0.67`, `2.84` |
| `adv_enrichment` | DOUBLE | Yes | Same ratio for adverbs. | `0.31`, `4.12` |
| `noun_pct` | DOUBLE | Yes | Fraction of unambiguous members (where `pos IS NOT NULL`) that are nouns. Range: 0.0–1.0. | `0.72`, `0.45` |
| `universal_pct` | DOUBLE | Yes | Fraction of this dimension's members that are universal words (`specificity < 0`). Higher = more "abstract" dimension. Added in phase 6b/9d. | `0.08` (concrete), `0.42` (abstract) |
| `dim_weight` | DOUBLE | Yes | Information-theoretic weight: `-log2(MAX(universal_pct, 0.01))`. Higher = more informative/concrete dimension. Range: ~1.0–6.6. Added in phase 6b/9d. | `1.25` (abstract), `5.50` (concrete) |
| `avg_specificity` | DOUBLE | Yes | Mean `specificity` value across all member words in this dimension (from `dim_memberships dm JOIN words w`). Positive values indicate concrete/taxonomic dimensions whose members are specific words appearing in few dims; negative values indicate abstract/diffuse dimensions whose members are universal words appearing in many dims. Range: ~-0.611 to 0.319. Added in phase 6b. | `-0.350` (abstract), `0.261` (concrete) |


### 3.3 `dim_memberships`

**Purpose:** Junction table recording which words are members of which dimensions, based on z-score thresholding. This is the largest table and the backbone of most queries.

**Row count:** ~2,560,000

**Primary key:** `(dim_id, word_id)`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `dim_id` | INTEGER | No (PK) | FK → `dim_stats.dim_id`. The dimension this membership is in. Range: 0–767. | `42`, `500` |
| `word_id` | INTEGER | No (PK) | FK → `words.word_id`. The word that is a member. | `1234`, `89012` |
| `value` | DOUBLE | Yes | Raw activation value of this word in this dimension (from the embedding vector). Always ≥ the dimension's threshold. | `0.0567`, `0.1234` |
| `z_score` | DOUBLE | Yes | Number of standard deviations above the dimension mean: `(value - mean) / std`. Always ≥ 2.0 (the `ZSCORE_THRESHOLD`). | `2.05`, `4.87` |
| `compound_support` | INTEGER | No | Number of compound words containing this word that also activate this dimension AND whose residual embedding (`compound - word`) still exceeds the threshold. Nonzero values suggest the membership may be compound-contaminated rather than intrinsic. `DEFAULT 0`. Added in phase 6b. | `0`, `3`, `12` |
| `archipelago_id` | INTEGER | Yes | Gen-0 island_id for this dimension, denormalized from `dim_islands` where `generation = 0`. **NULL for noise dimensions** (those assigned `island_id = -1` in Leiden clustering). Every membership in the same `dim_id` shares the same value. Backfilled in phase 9/9b. | `NULL`, `0`, `1`, `3` |
| `island_id` | INTEGER | Yes | Gen-1 island_id for this dimension, denormalized from `dim_islands` where `generation = 1`. **NULL for noise dimensions.** | `NULL`, `0`, `15`, `51` |
| `reef_id` | INTEGER | Yes | Gen-2 island_id for this dimension, denormalized from `dim_islands` where `generation = 2`. **NULL for noise dimensions.** Enables direct reef-level queries without joining through `dim_islands`. | `NULL`, `0`, `42`, `207` |


### 3.4 `word_pair_overlap`

**Purpose:** Precomputed pairwise word similarity based on shared dimension memberships. Only stores pairs sharing ≥ `PAIR_OVERLAP_THRESHOLD` (3) dimensions. Expensive to compute; optional.

**Row count:** Varies (~millions of pairs)

**Primary key:** `(word_id_a, word_id_b)`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `word_id_a` | INTEGER | No (PK) | FK → `words.word_id`. Enforced: `word_id_a < word_id_b` (no duplicates, no self-pairs). | `100` |
| `word_id_b` | INTEGER | No (PK) | FK → `words.word_id`. | `5432` |
| `shared_dims` | INTEGER | Yes | Number of dimensions both words are members of. Always ≥ 3 (the threshold). | `3`, `12`, `28` |

---

## 4. Enrichment Tables

### 4.1 `word_pos`

**Purpose:** All parts-of-speech for each word from WordNet, including ambiguous words. Unlike `words.pos` (which is NULL for ambiguous words), this table stores every POS a word has.

**Row count:** ~175,000 (more rows than words because ambiguous words have multiple POS entries)

**Primary key:** `(word_id, pos)`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `word_id` | INTEGER | No (PK) | FK → `words.word_id`. | `45678` |
| `pos` | TEXT | No (PK) | Part-of-speech tag from WordNet. | `'noun'`, `'verb'`, `'adj'`, `'adv'` |
| `synset_count` | INTEGER | Yes | Number of WordNet synsets for this word+POS combination. Higher count = more senses for that POS. | `1`, `5`, `14` |


### 4.2 `word_components`

**Purpose:** Decomposition of multi-word expressions into their constituent tokens. Only populated for words where `word_count > 1`.

**Row count:** ~100,000+

**Primary key:** `(compound_word_id, position)`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `compound_word_id` | INTEGER | No (PK) | FK → `words.word_id`. The multi-word compound being decomposed. | `50000` (for "heart attack") |
| `component_word_id` | INTEGER | Yes | FK → `words.word_id`. The single-word component if it exists in the vocabulary. **NULL if the component word is not in the `words` table** (rare but possible for obscure tokens). | `30000` (for "heart"), `NULL` |
| `component_text` | TEXT | No | The text of the component token. Always populated even if `component_word_id` is NULL. | `'heart'`, `'attack'` |
| `position` | INTEGER | No (PK) | 0-indexed position of this component within the compound. | `0` (first word), `1` (second word) |


### 4.3 `word_senses`

**Purpose:** Sense-specific embeddings for ambiguous words (those with `words.pos IS NULL`). Each WordNet synset for an ambiguous word gets its own row with a contextualized embedding.

**Row count:** ~61,000

**Primary key:** `sense_id`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `sense_id` | INTEGER | No (PK) | Sequential sense ID, globally unique across all senses. | `1`, `30000`, `61000` |
| `word_id` | INTEGER | No | FK → `words.word_id`. The parent word this sense belongs to. | `45678` |
| `pos` | TEXT | No | Part-of-speech for this specific sense. | `'noun'`, `'verb'` |
| `synset_name` | TEXT | No | WordNet synset identifier. | `'bank.n.01'`, `'bank.n.02'`, `'bank.v.01'` |
| `gloss` | TEXT | No | WordNet definition text used to contextualize the sense embedding. | `'a financial institution that accepts deposits'` |
| `sense_embedding` | FLOAT[768] | Yes | Embedding of `"classification: {word}: {gloss}"`. NULL should not occur in practice but the column is nullable. | `[0.0234, -0.0123, ...]` |
| `total_dims` | INTEGER | No | Number of dimensions this sense belongs to, computed using the same thresholds as words. `DEFAULT 0`. | `8`, `15`, `22` |


### 4.4 `sense_dim_memberships`

**Purpose:** Dimension memberships for word senses, analogous to `dim_memberships` but for sense-specific embeddings. Uses the same dimension thresholds from `dim_stats`.

**Row count:** ~1,000,000+

**Primary key:** `(dim_id, sense_id)`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `dim_id` | INTEGER | No (PK) | FK → `dim_stats.dim_id`. Range: 0–767. | `42`, `500` |
| `sense_id` | INTEGER | No (PK) | FK → `word_senses.sense_id`. | `1234` |
| `value` | DOUBLE | Yes | Raw activation value for this sense in this dimension. | `0.0678` |
| `z_score` | DOUBLE | Yes | Standard deviations above the dimension mean. Always ≥ 2.0. | `2.34`, `5.12` |


### 4.5 `compositionality`

**Purpose:** Compositionality analysis for compound words and phrasal verbs. Measures whether a compound's meaning is derivable from its parts (compositional) or emergent (idiomatic).

**Row count:** ~50,000 (compounds/phrasal verbs with resolvable components)

**Primary key:** `word_id`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `word_id` | INTEGER | No (PK) | FK → `words.word_id`. Only words with `category IN ('compound', 'phrasal_verb')`. | `50000` |
| `word` | TEXT | Yes | The compound word text (denormalized for convenience). | `'heart attack'`, `'hot dog'` |
| `jaccard` | DOUBLE | Yes | Jaccard similarity between the compound's dimension set and the union of its components' dimension sets: `\|shared\| / \|union\|`. Range: 0.0–1.0. | `0.05` (idiomatic), `0.45` (compositional) |
| `is_compositional` | BOOLEAN | Yes | `TRUE` if `jaccard >= COMPOSITIONALITY_THRESHOLD` (0.20). | `TRUE`, `FALSE` |
| `compound_dims` | INTEGER | Yes | Total dimensions the compound word belongs to. | `12`, `25` |
| `component_union_dims` | INTEGER | Yes | Total dimensions in the union of all component words' dimension sets. | `30`, `55` |
| `shared_dims` | INTEGER | Yes | Dimensions present in both the compound's set and the component union. | `3`, `18` |
| `emergent_dims` | INTEGER | Yes | Dimensions in the compound but not in any component: `compound_dims - shared_dims`. High values suggest emergent meaning. | `2`, `9` |

---

## 5. Island Hierarchy Tables

### 5.1 `dim_jaccard`

**Purpose:** Pairwise Jaccard similarity between all 768 embedding dimensions based on their member word sets. Only pairs with `intersection_size > 0` are stored. Computed using single-token words only (excludes multi-word expressions).

**Row count:** ~250,000–295,000 (depends on membership density)

**Primary key:** `(dim_id_a, dim_id_b)`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `dim_id_a` | INTEGER | No (PK) | FK → `dim_stats.dim_id`. Enforced: `dim_id_a < dim_id_b`. | `0`, `42` |
| `dim_id_b` | INTEGER | No (PK) | FK → `dim_stats.dim_id`. | `1`, `500` |
| `intersection_size` | INTEGER | No | Number of single-token words that are members of both dimensions. Always > 0. | `50`, `1200` |
| `union_size` | INTEGER | No | Size of the union of both dimensions' single-token member sets. | `3000`, `6500` |
| `jaccard` | DOUBLE | No | `intersection_size / union_size`. Range: >0.0 to ~0.5. | `0.0167`, `0.1846` |
| `expected_intersection` | DOUBLE | Yes | Hypergeometric expected intersection under random assignment: `n_i * n_j / N` where N is total single-token words. | `120.5`, `450.2` |
| `z_score` | DOUBLE | Yes | Hypergeometric z-score: `(intersection_size - expected_intersection) / sqrt(variance)`. Edges with `z_score >= 3.0` (`ISLAND_JACCARD_ZSCORE`) are used for Leiden clustering. | `-1.2`, `8.5`, `15.3` |


### 5.2 `dim_islands`

**Purpose:** Maps each dimension to its island assignment at each generation of the hierarchy. A dimension can belong to one island per generation (gen-0 archipelago, gen-1 island, gen-2 reef). Gen-2 (reef) assignments may be updated by Phase 10 reef refinement, which reassigns misplaced dims to better-fitting sibling reefs based on Jaccard affinity.

**Row count:** ~2,304 (768 dims × up to 3 generations)

**Primary key:** `(dim_id, generation)`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `dim_id` | INTEGER | No (PK) | FK → `dim_stats.dim_id`. Range: 0–767. | `42`, `500` |
| `island_id` | INTEGER | No | Island assignment. `>= 0` for a valid island; **`-1` for noise/singleton** dimensions that were not assigned to any community (community size < `ISLAND_MIN_COMMUNITY_SIZE`). | `0`, `3`, `51`, `-1` |
| `generation` | INTEGER | No (PK) | Hierarchy level. `0` = archipelago, `1` = island, `2` = reef. `DEFAULT 0`. | `0`, `1`, `2` |
| `parent_island_id` | INTEGER | Yes | The parent island from the previous generation. **NULL for generation 0.** For gen-1, references a gen-0 island_id; for gen-2, references a gen-1 island_id. | `NULL`, `0`, `15` |


### 5.3 `island_stats`

**Purpose:** Aggregate statistics for each island at each generation. Includes naming and word depth metrics.

**Row count:** ~264 (4 archipelagos + 52 islands + 208 reefs)

**Primary key:** `(island_id, generation)`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `island_id` | INTEGER | No (PK) | Island identifier within its generation. Sequentially assigned starting from 0 by Leiden clustering. | `0`, `15`, `207` |
| `generation` | INTEGER | No (PK) | Hierarchy level. `0` = archipelago, `1` = island, `2` = reef. `DEFAULT 0`. | `0`, `1`, `2` |
| `n_dims` | INTEGER | No | Number of dimensions assigned to this island. | `4` (small reef), `65` (large island), `250` (archipelago) |
| `n_words` | INTEGER | No | Number of unique single-token words across all dims in this island (union count). | `5000`, `45000` |
| `avg_internal_jaccard` | DOUBLE | Yes | Mean Jaccard similarity among all dimension pairs within this island. NULL if `n_dims < 2`. | `0.0234`, `0.0891` |
| `max_internal_jaccard` | DOUBLE | Yes | Maximum pairwise Jaccard within the island. NULL if `n_dims < 2`. | `0.15`, `0.35` |
| `min_internal_jaccard` | DOUBLE | Yes | Minimum pairwise Jaccard within the island. NULL if `n_dims < 2`. | `0.001`, `0.02` |
| `modularity_contribution` | DOUBLE | Yes | Placeholder for future use. Currently always NULL. | `NULL` |
| `parent_island_id` | INTEGER | Yes | Parent island from the previous generation. **NULL for generation 0.** For gen-1, references a gen-0 island_id; for gen-2, references a gen-1 island_id. | `NULL`, `0`, `25` |
| `island_name` | TEXT | Yes | Human-readable name assigned by the LLM naming pipeline (phase 9c). 2-4 words, lowercase. NULL if naming has not been run. | `'natural sciences and taxonomy'`, `'musical instruments'`, `'string instruments'` |
| `n_core_words` | INTEGER | Yes | Words appearing in ≥ `max(2, ceil(n_dims * 0.10))` of this island's dimensions. "Core" words deeply embedded in the island's semantic theme. | `150`, `2000` |
| `median_word_depth` | DOUBLE | Yes | Median number of island dimensions each word appears in. Higher = tighter cluster. | `1.0`, `2.5` |


### 5.4 `island_characteristic_words`

**Purpose:** Top PMI-ranked words per island per generation. PMI (pointwise mutual information) measures how much more likely a word is to appear in this island vs. the corpus overall. Up to `ISLAND_CHARACTERISTIC_WORDS_N` (100) words stored per island.

**Row count:** ~26,000+ (up to 100 per island × ~264 islands)

**Primary key:** `(island_id, generation, word_id)`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `island_id` | INTEGER | No (PK) | The island this characteristic word belongs to. | `0`, `15` |
| `generation` | INTEGER | No (PK) | Hierarchy level. `DEFAULT 0`. | `0`, `1`, `2` |
| `word_id` | INTEGER | No (PK) | FK → `words.word_id`. | `12345` |
| `word` | TEXT | No | Denormalized word text. | `'guitar'`, `'enzyme'` |
| `pmi` | DOUBLE | No | `log2(island_freq / corpus_freq)`. Higher = more characteristic of this island vs. corpus background. Can be negative. | `2.5`, `5.8`, `-0.3` |
| `island_freq` | DOUBLE | No | `n_dims_in_island / n_island_dims`. Fraction of this island's dimensions that contain this word. | `0.15`, `0.65` |
| `corpus_freq` | DOUBLE | No | `total_dims / 768`. Fraction of all 768 dimensions that contain this word. | `0.005`, `0.025` |
| `n_dims_in_island` | INTEGER | No | How many of this island's dimensions contain this word. Always ≥ 2 (filtered in computation). | `2`, `8`, `15` |


### 5.5 `word_reef_affinity`

**Purpose:** Continuous affinity scores for every word-reef pair. Computed by joining `dim_memberships` with `dim_islands` (gen=2) and `dim_stats`, aggregating per (word, reef). Used by the explorer's `affinity` command and `evaluate` battery for relationship classification. Recomputed in phase 9b or after phase 10 reef refinement.

**Row count:** ~4-6M (every word × every reef it touches, even at depth 1)

**Primary key:** `(word_id, reef_id)`

| Column | Type | Nullable | Description | Example Values |
|--------|------|----------|-------------|----------------|
| `word_id` | INTEGER | No (PK) | FK → `words.word_id`. | `1234`, `89012` |
| `reef_id` | INTEGER | No (PK) | FK → `island_stats.island_id` where `generation = 2`. The gen-2 reef. | `0`, `42`, `207` |
| `n_dims` | INTEGER | No | Number of the reef's dimensions that this word activates in (depth). Range: 1 to reef `n_dims`. | `1`, `3`, `8` |
| `max_z` | DOUBLE | Yes | Maximum `z_score` across all of this word's memberships in this reef's dimensions. | `2.15`, `6.87` |
| `sum_z` | DOUBLE | Yes | Sum of `z_score` across all memberships. | `2.15`, `18.5` |
| `max_weighted_z` | DOUBLE | Yes | Maximum of `z_score * dim_weight` across memberships. A strong discriminator for meaningful depth-1 memberships. | `3.2`, `12.8` |
| `sum_weighted_z` | DOUBLE | Yes | Sum of `z_score * dim_weight` across memberships. | `3.2`, `35.6` |

---

## 6. Views

### 6.1 `v_unique_words`

**Purpose:** Words with positive specificity — those belonging to fewer dimensions than average (1σ+ below mean), indicating niche/specific vocabulary.

**Definition:**
```sql
SELECT w.word_id, w.word, w.total_dims, w.specificity
FROM words w
WHERE w.specificity > 0
ORDER BY w.total_dims ASC
```

| Column | Type | Description |
|--------|------|-------------|
| `word_id` | INTEGER | FK → words |
| `word` | TEXT | The word |
| `total_dims` | INTEGER | Number of dimensions (low end of distribution) |
| `specificity` | INTEGER | `1` or `2` |


### 6.2 `v_universal_words`

**Purpose:** Words with negative specificity — those belonging to more dimensions than average (1σ+ above mean), indicating broadly-used/general vocabulary.

**Definition:**
```sql
SELECT w.word_id, w.word, w.total_dims, w.specificity
FROM words w
WHERE w.specificity < 0
ORDER BY w.total_dims DESC
```

| Column | Type | Description |
|--------|------|-------------|
| `word_id` | INTEGER | FK → words |
| `word` | TEXT | The word |
| `total_dims` | INTEGER | Number of dimensions (high end of distribution) |
| `specificity` | INTEGER | `-1` or `-2` |


### 6.3 `v_selective_dims`

**Purpose:** Dimensions with selectivity < 5% — the sharpest, most concept-specific dimensions.

**Definition:**
```sql
SELECT ds.*
FROM dim_stats ds
WHERE ds.selectivity < 0.05
ORDER BY ds.selectivity ASC
```

| Column | Type | Description |
|--------|------|-------------|
| (all columns from `dim_stats`) | | See `dim_stats` table definition |


### 6.4 `v_abstract_dims`

**Purpose:** Dimensions where ≥ 30% of members are universal words (`universal_pct >= 0.30`). These dimensions encode abstract/social concepts.

**Definition:**
```sql
SELECT ds.* FROM dim_stats ds
WHERE ds.universal_pct >= 0.30
ORDER BY ds.universal_pct DESC
```

| Column | Type | Description |
|--------|------|-------------|
| (all columns from `dim_stats`) | | See `dim_stats` table definition |


### 6.5 `v_concrete_dims`

**Purpose:** Dimensions where ≤ 15% of members are universal words (`universal_pct <= 0.15`). These dimensions encode concrete/taxonomic domains.

**Definition:**
```sql
SELECT ds.* FROM dim_stats ds
WHERE ds.universal_pct IS NOT NULL AND ds.universal_pct <= 0.15
ORDER BY ds.universal_pct ASC
```

| Column | Type | Description |
|--------|------|-------------|
| (all columns from `dim_stats`) | | See `dim_stats` table definition |


### 6.6 `v_domain_generals`

**Purpose:** Universal words (`specificity < 0`) with high archipelago concentration (≥ 0.75). These are "domain generals" — broadly-used words that are topically focused on one archipelago.

**Definition:**
```sql
SELECT w.word_id, w.word, w.total_dims, w.specificity,
       w.arch_concentration, w.sense_spread, w.polysemy_inflated
FROM words w
WHERE w.specificity < 0 AND w.arch_concentration >= 0.75
ORDER BY w.arch_concentration DESC
```

| Column | Type | Description |
|--------|------|-------------|
| `word_id` | INTEGER | FK → words |
| `word` | TEXT | The word |
| `total_dims` | INTEGER | Dimension membership count |
| `specificity` | INTEGER | Always `-1` or `-2` |
| `arch_concentration` | DOUBLE | Max fraction in one archipelago (≥ 0.75) |
| `sense_spread` | INTEGER | Spread across senses (NULL if < 2 senses) |
| `polysemy_inflated` | BOOLEAN | Whether flagged as polysemy-inflated |

---

## 7. Index Reference

All 21 indexes, listed with their table, indexed columns, and purpose.

| # | Index Name | Table | Column(s) | Purpose |
|---|-----------|-------|-----------|---------|
| 1 | `idx_dm_word` | `dim_memberships` | `word_id` | Fast lookup of all dimensions a word belongs to |
| 2 | `idx_dm_dim` | `dim_memberships` | `dim_id` | Fast lookup of all words in a dimension |
| 3 | `idx_words_total` | `words` | `total_dims DESC` | Rank words by dimension count (most/least universal) |
| 4 | `idx_ds_selectivity` | `dim_stats` | `selectivity` | Filter/sort dimensions by selectivity |
| 5 | `idx_wpo_a` | `word_pair_overlap` | `word_id_a` | Fast lookup of pairs for a given word (left side) |
| 6 | `idx_wpo_b` | `word_pair_overlap` | `word_id_b` | Fast lookup of pairs for a given word (right side) |
| 7 | `idx_wpo_shared` | `word_pair_overlap` | `shared_dims DESC` | Rank word pairs by overlap strength |
| 8 | `idx_wp_word` | `word_pos` | `word_id` | Fast lookup of POS tags for a word |
| 9 | `idx_wc_compound` | `word_components` | `compound_word_id` | Find components of a compound |
| 10 | `idx_wc_component` | `word_components` | `component_word_id` | Find which compounds contain a component |
| 11 | `idx_ws_word` | `word_senses` | `word_id` | Find all senses for a word |
| 12 | `idx_sdm_sense` | `sense_dim_memberships` | `sense_id` | Find all dimensions for a sense |
| 13 | `idx_sdm_dim` | `sense_dim_memberships` | `dim_id` | Find all senses in a dimension |
| 14 | `idx_dj_a` | `dim_jaccard` | `dim_id_a` | Fast lookup of Jaccard pairs for a dimension (left side) |
| 15 | `idx_dj_b` | `dim_jaccard` | `dim_id_b` | Fast lookup of Jaccard pairs for a dimension (right side) |
| 16 | `idx_di_island` | `dim_islands` | `island_id, generation` | Find all dimensions in an island at a generation |
| 17 | `idx_is_gen` | `island_stats` | `generation` | Filter islands by hierarchy level |
| 18 | `idx_icw_island` | `island_characteristic_words` | `island_id, generation` | Find characteristic words for an island |
| 19 | `idx_icw_word` | `island_characteristic_words` | `word_id` | Find which islands a word is characteristic of |
| 20 | `idx_wra_reef` | `word_reef_affinity` | `reef_id` | Find all words with affinity to a given reef |
| 21 | `idx_wra_wz` | `word_reef_affinity` | `max_weighted_z DESC` | Rank word-reef pairs by weighted z-score |
