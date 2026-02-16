# Windowsill Technical Reference

> Detailed schema, design decisions, and verification queries for the windowsill pipeline.
> For project overview and usage, see [README.md](README.md).
> For live database metrics, see [ANALYSIS.md](ANALYSIS.md).

## Key Design Decisions

- **DuckDB** for storage -- embeddings stored as `FLOAT[768]` arrays directly in the DB, enabling SQL queries alongside vector operations without a separate vector store.
- **Z-score thresholding (mean + 2.0σ)** -- Each dimension's membership threshold is `mean + 2.0 * std`. The threshold was lowered from 2.45σ (~6 dims/word, ~800K memberships) to 2.0σ (~17 dims/word, ~2.5M memberships) to provide richer data for island/reef clustering. At 2.45σ, dimension-pair Jaccard similarities were too sparse for coherent reef formation — semantically related dimensions (e.g., musical instrument dims) had jaccard < 0.01 and couldn't cluster together. At 2.0σ, the 3x membership increase raises pairwise Jaccard into the significant range (hyper_z 5-15 for related dims), producing tighter, more semantically coherent reefs. The noise introduced by the lower threshold is addressed by a secondary **reef depth filter** (`REEF_MIN_DEPTH = 2`): a word's reef bit is only encoded if the word appears in ≥ 2 of the reef's dimensions. At 2.0σ, genuine concept membership manifests as multi-dimension overlap (e.g., "guitar" activates 7/17 music-related dims), while noise connections are single-dim and get pruned.
- **Matryoshka at 768** -- nomic-embed-text-v1.5 supports Matryoshka dimensionality reduction. We use the full 768 for maximum resolution.
- **`"classification: "` prefix** -- The Nomic model uses task-specific prefixes. All words are embedded with this prefix to activate the classification head, which produces more discriminative dimensions.
- **Sense embedding format** -- Ambiguous words are re-embedded as `"classification: {word}: {gloss}"` where the gloss comes from WordNet. This contextualizes the word without changing the embedding space.
- **Compositionality via Jaccard** -- A compound is compositional if the Jaccard similarity between its dimension set and the union of its components' dimension sets is >= 0.20. Below that threshold, it's idiomatic (meaning not derivable from parts).
- **Contamination via residual activation** -- For a component word W in dimension D, compound_support counts how many compounds containing W are also in D AND whose residual (compound_embedding - W_embedding) still exceeds D's threshold. High support means D might be compound-derived, not intrinsic to W.
- **Specificity bands** -- Words are classified into sigma-based bands (`+2` to `-2`) based on their `total_dims` distance from the population mean. This replaces the original hardcoded thresholds in `v_unique_words` (was `<= 15`) and `v_universal_words` (was `>= 36`) with statistically derived boundaries that adapt to the actual distribution.
- **Noise dim recovery** -- After Leiden community detection, singleton dimensions classified as noise (island_id = -1) are recovered by computing their average Jaccard similarity to each sibling reef's dimensions. If the best match exceeds `NOISE_RECOVERY_MIN_JACCARD`, the dim is assigned to that reef. This captures coherent dimensions that Leiden's minimum community size constraint would otherwise discard.
- **Domain-anchored sense enrichment** -- Words with WordNet `topic_domains()` or `usage_domains()` get synthetic compound embeddings (e.g., `"classification: chess rook"`). Only these domain-anchored senses contribute to `word_reef_affinity`, preventing the generic UNION ALL approach from inflating polysemous words' reef counts 3-7x.
- **Dynamic hierarchy counts** -- Reef, island, and archipelago counts are computed at runtime from the `island_stats` table via `config.get_hierarchy_counts()`, rather than hardcoded. This makes the system robust to changes in subdivision thresholds, noise recovery, or input data.

## Database Schema

> For full column-level detail — types, nullability, value ranges, example values, FK relationships, and join patterns — see the [Data Dictionary](data_dictionary.md).

### Core Tables (Phases 1-4)

**`words`** -- One row per vocabulary entry
```
word_id       INTEGER PK    Sequential ID (1-indexed, alphabetical order)
word          TEXT          The word (spaces for multi-word, lowercase)
total_dims    INTEGER       Number of dimensions this word belongs to
embedding     FLOAT[768]    Raw embedding vector
pos           TEXT          Part-of-speech (NULL if ambiguous across POS)
category      TEXT          'single' | 'compound' | 'taxonomic' | 'phrasal_verb' | 'named_entity'
word_count    INTEGER       Number of space-separated tokens
specificity        INTEGER    Sigma band: +2/+1/0/-1/-2 (positive = specific, negative = universal)
sense_spread       INTEGER    MAX(total_dims) - MIN(total_dims) across senses (NULL if < 2 senses)
polysemy_inflated  BOOLEAN    TRUE if universal (specificity < 0) and sense_spread >= 15
arch_concentration DOUBLE     Max fraction of dims in any single archipelago (universal words only)
word_hash          UBIGINT    FNV-1a u64 hash (zero collisions confirmed across full vocabulary)
reef_idf           DOUBLE     BM25 IDF: ln((N_reefs - n + 0.5) / (n + 0.5) + 1) where n = reefs containing word
```

**`dim_stats`** -- One row per embedding dimension (768 total)
```
dim_id           INTEGER PK    Dimension index (0-767)
mean, std, min_val, max_val, median, skewness, kurtosis   -- distribution stats
threshold        DOUBLE        Activation threshold for membership (mean + 2.0σ)
threshold_method TEXT          'zscore'
n_members        INTEGER       How many words exceed the threshold
selectivity      DOUBLE        n_members / total_words
verb_enrichment  DOUBLE        Ratio of verb rate in dim vs corpus base rate
adj_enrichment   DOUBLE        Same for adjectives
adv_enrichment   DOUBLE        Same for adverbs
noun_pct         DOUBLE        Fraction of dim members that are nouns
universal_pct    DOUBLE        Fraction of dim members that are universal words (specificity < 0)
dim_weight       DOUBLE        -log2(max(universal_pct, 0.01)) — information-theoretic weight
avg_specificity  DOUBLE        Mean specificity of member words (positive = concrete, negative = abstract)
valence          DOUBLE        Projection onto negation vector: positive = negative-pole, negative = positive-pole
noun_frac        DOUBLE        Sense-aware fractional noun composition (0.0-1.0)
verb_frac        DOUBLE        Sense-aware fractional verb composition (0.0-1.0)
adj_frac         DOUBLE        Sense-aware fractional adjective composition (0.0-1.0)
adv_frac         DOUBLE        Sense-aware fractional adverb composition (0.0-1.0)
```

**`dim_memberships`** -- Which words belong to which dimensions
```
dim_id           INTEGER       (PK with word_id)
word_id          INTEGER
value            DOUBLE        Raw activation value
z_score          DOUBLE        Standard deviations above mean
compound_support INTEGER       How many compounds contribute to this membership
archipelago_id   INTEGER       Gen-0 island_id for this dim (NULL = noise)
island_id        INTEGER       Gen-1 island_id for this dim (NULL = noise)
reef_id          INTEGER       Gen-2 island_id for this dim (NULL = noise)
```

**`word_pair_overlap`** -- Precomputed word similarity (expensive, optional)
```
word_id_a    INTEGER    (PK with word_id_b, enforced a < b)
word_id_b    INTEGER
shared_dims  INTEGER    Number of shared dimension memberships
```

### Enrichment Tables (Phases 5-6)

**`word_pos`** -- All POS tags per word (even ambiguous ones)
```
word_id       INTEGER    (PK with pos)
pos           TEXT       'noun' | 'verb' | 'adj' | 'adv'
synset_count  INTEGER    Number of WordNet synsets for this word+pos
```

**`word_components`** -- Decomposition of multi-word expressions
```
compound_word_id   INTEGER    (PK with position)
component_word_id  INTEGER    FK to words (NULL if component not in vocabulary)
component_text     TEXT       The component word text
position           INTEGER    0-indexed position in the compound
```

**`word_senses`** -- Sense-specific embeddings for ambiguous words
```
sense_id            INTEGER PK    Sequential sense ID
word_id             INTEGER       FK to words
pos                 TEXT          POS for this specific sense
synset_name         TEXT          WordNet synset (e.g., 'bank.n.02')
gloss               TEXT          WordNet definition text
sense_embedding     FLOAT[768]    Embedding of "classification: word: gloss"
total_dims          INTEGER       Dimensions this sense belongs to
is_domain_anchored  BOOLEAN       TRUE if sense comes from WordNet topic/usage domain enrichment
```

**`sense_dim_memberships`** -- Dimension memberships per sense
```
dim_id    INTEGER    (PK with sense_id)
sense_id  INTEGER
value     DOUBLE
z_score   DOUBLE
```

**`compositionality`** -- Compositionality analysis results
```
word_id              INTEGER PK
word                 TEXT
jaccard              DOUBLE     Jaccard(compound_dims, union_of_component_dims)
is_compositional     BOOLEAN    jaccard >= 0.20
compound_dims        INTEGER    Total dims the compound belongs to
component_union_dims INTEGER    Total dims in union of all components
shared_dims          INTEGER    Dims in both compound and component union
emergent_dims        INTEGER    Dims in compound but not in any component
```

### Island Tables (Phases 7-10)

**`dim_jaccard`** -- Pairwise Jaccard similarity between all 768 dimensions
```
dim_id_a              INTEGER    (PK with dim_id_b, enforced a < b)
dim_id_b              INTEGER
intersection_size     INTEGER    Size of word set intersection
union_size            INTEGER    Size of word set union
jaccard               DOUBLE     intersection_size / union_size
expected_intersection DOUBLE     Hypergeometric expected intersection (n_i * n_j / N)
z_score               DOUBLE     (observed - expected) / sqrt(variance)
```

**`dim_islands`** -- Dimension-to-island assignments
```
dim_id             INTEGER    (PK with generation)
island_id          INTEGER    -1 = noise/singleton
generation         INTEGER    0 = archipelago, 1 = island, 2 = reef
parent_island_id   INTEGER    NULL for gen 0, FK to island_id for gen 1+
```

**`island_stats`** -- Per-island summary
```
island_id              INTEGER    (PK with generation)
generation             INTEGER    0 = archipelago, 1 = island, 2 = reef
n_dims                 INTEGER    Number of dimensions in island
n_words                INTEGER    Unique words across all dims in island (union count)
avg_internal_jaccard   DOUBLE     Mean Jaccard among island dims
max_internal_jaccard   DOUBLE
min_internal_jaccard   DOUBLE
modularity_contribution DOUBLE    Placeholder (NULL for now)
parent_island_id       INTEGER
island_name            TEXT       Human-readable label (set by phase 10 naming pipeline)
n_core_words           INTEGER    Words in >= max(2, ceil(n_dims*0.10)) island dims
median_word_depth      DOUBLE     Median island-dim count per word
valence                DOUBLE     Mean dimension valence across dims in this island/reef/archipelago
noun_frac              DOUBLE     Mean sense-aware noun fraction across dims (0.0-1.0)
verb_frac              DOUBLE     Mean sense-aware verb fraction across dims (0.0-1.0)
adj_frac               DOUBLE     Mean sense-aware adjective fraction across dims (0.0-1.0)
adv_frac               DOUBLE     Mean sense-aware adverb fraction across dims (0.0-1.0)
avg_specificity        DOUBLE     Mean dim-level avg_specificity (positive = concrete, negative = abstract)
```

**`island_characteristic_words`** -- PMI-ranked diagnostic words per island
```
island_id          INTEGER    (PK with generation, word_id)
generation         INTEGER
word_id            INTEGER
word               TEXT
pmi                DOUBLE     log2(P(word|island) / P(word))
island_freq        DOUBLE     P(word|island) = dims_in_island / n_island_dims
corpus_freq        DOUBLE     P(word) = total_dims / 768
n_dims_in_island   INTEGER    How many island dims contain this word
```

**`word_reef_affinity`** -- Continuous affinity scores for every word-reef pair
```
word_id            INTEGER    (PK with reef_id) FK to words
reef_id            INTEGER    FK to island_stats where generation=2
n_dims             INTEGER    Number of reef dims the word activates
max_z              DOUBLE     Max z_score across reef dims
sum_z              DOUBLE     Sum of z_scores across reef dims
max_weighted_z     DOUBLE     Max (z_score * dim_weight) across reef dims
sum_weighted_z     DOUBLE     Sum of (z_score * dim_weight) across reef dims
```

**`reef_edges`** -- Directed component scores for all reef pairs (N_reefs x (N_reefs - 1) rows)
```
source_reef_id  INTEGER    (PK with target_reef_id) FK to island_stats where generation=2
target_reef_id  INTEGER    FK to island_stats where generation=2
containment     DOUBLE     Fraction of source's words (depth >= 2) also in target
lift            DOUBLE     P(target | source) / P(target) — co-activation above baseline
pos_similarity  DOUBLE     Cosine similarity of [noun_frac, verb_frac, adj_frac, adv_frac] vectors
valence_gap     DOUBLE     Signed: target.valence - source.valence
specificity_gap DOUBLE     Signed: target.avg_specificity - source.avg_specificity
```

**`word_variants`** -- Morphy expansion mapping inflected forms to base word_ids
```
variant_hash   UBIGINT    (PK with word_id) FNV-1a hash of the variant string
variant        TEXT       The variant text (e.g., "running")
word_id        INTEGER    FK to words — the base word this variant maps to
source         TEXT       'base' for the word itself, 'morphy' for inflected forms
```

**`computed_vectors`** -- Stored analytical vectors (negation vector, etc.)
```
name           TEXT PK     Vector name (e.g., 'negation')
vector         FLOAT[768]  The vector itself
n_pairs        INTEGER     Number of word pairs used to compute it
description    TEXT        Human-readable description
```

### Views

- `v_unique_words` -- Words with positive specificity (1σ+ fewer dims than mean; specific words)
- `v_universal_words` -- Words with negative specificity (1σ+ more dims than mean; general words)
- `v_selective_dims` -- Dimensions with selectivity < 5% (sharp concepts)
- `v_abstract_dims` -- Dimensions where >= 30% of members are universal words (dominated by abstract/social concepts)
- `v_concrete_dims` -- Dimensions where <= 15% of members are universal words (concrete/taxonomic domains)
- `v_domain_generals` -- Universal words with arch_concentration >= 0.75 (concentrated in one archipelago)
- `v_positive_dims` -- Dimensions with valence <= -0.15 (positive-pole: negation decreases activation)
- `v_negative_dims` -- Dimensions with valence >= 0.15 (negative-pole: negation increases activation)

### Indexes

```
idx_dm_word        dim_memberships(word_id)
idx_dm_dim         dim_memberships(dim_id)
idx_words_total    words(total_dims DESC)
idx_ds_selectivity dim_stats(selectivity)
idx_wpo_a/b/shared word_pair_overlap indexes
idx_wp_word        word_pos(word_id)
idx_wc_compound    word_components(compound_word_id)
idx_wc_component   word_components(component_word_id)
idx_ws_word        word_senses(word_id)
idx_sdm_sense      sense_dim_memberships(sense_id)
idx_sdm_dim        sense_dim_memberships(dim_id)
idx_dj_a/b         dim_jaccard indexes
idx_di_island      dim_islands(island_id, generation)
idx_is_gen         island_stats(generation)
idx_icw_island     island_characteristic_words(island_id, generation)
idx_icw_word       island_characteristic_words(word_id)
idx_wra_reef       word_reef_affinity(reef_id)
idx_wra_wz         word_reef_affinity(max_weighted_z DESC)
idx_words_hash     words(word_hash)
idx_wv_hash        word_variants(variant_hash)
idx_wv_word        word_variants(word_id)
idx_re_source      reef_edges(source_reef_id)
idx_re_target      reef_edges(target_reef_id)
```

## Embedding Mechanics

### How words are embedded

Each word is embedded as `"classification: {word}"` using nomic-embed-text-v1.5 via sentence-transformers. The `"classification: "` prefix activates the model's classification head, producing more discriminative (less smooth) embeddings -- exactly what we want for dimension-level analysis.

### How senses are embedded

Ambiguous words (pos IS NULL, ~8% of vocabulary) get additional sense-specific embeddings. For each WordNet synset of an ambiguous word, we embed:
```
"classification: {word}: {gloss}"
```
For example:
```
"classification: bank: a financial institution that accepts deposits"
"classification: bank: sloping land beside a body of water"
```

These sense embeddings are evaluated against the **same dimension thresholds** computed from the full word set in phase 4. No thresholds are recomputed -- senses are measured against the existing statistical framework.

### How domain compounds are embedded

Words with WordNet topic or usage domains get additional domain-anchored sense embeddings. For a word like "rook" with the chess domain, the system generates:
```
"classification: chess rook"
```
These domain compounds use the domain label as a disambiguating prefix, producing sharper dimension activations in the relevant domain. Domain compound senses are marked with `is_domain_anchored = TRUE` in the `word_senses` table. Only domain-anchored senses contribute to `word_reef_affinity`, preventing generic sense overlap from inflating polysemous words' reef counts.

### Intermediate checkpointing

Embedding is the most expensive operation. The system saves `.npy` checkpoint files every 50 batches to `intermediates/` (or `intermediates/senses/` for sense embeddings, `intermediates/domain_senses/` for domain compound embeddings). If interrupted, it resumes from the last checkpoint. A `embeddings_final.npy` file is saved on completion and used for instant reload on re-runs.

## Configuration Reference

All constants are in `config.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `MODEL_NAME` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |
| `EMBEDDING_PREFIX` | `"classification: "` | Task prefix for Nomic model |
| `MATRYOSHKA_DIM` | `768` | Embedding dimensionality |
| `BATCH_SIZE` | `256` | Words per embedding batch |
| `DB_PATH` | `vector_distillery.duckdb` | Default database file |
| `ZSCORE_THRESHOLD` | `2.0` | Membership threshold: mean + N*std |
| `PAIR_OVERLAP_THRESHOLD` | `3` | Min shared dims for pair table |
| `COMMIT_INTERVAL` | `50` | Dimensions per DB batch commit |
| `INTERMEDIATE_DIR` | `intermediates` | Checkpoint directory for embeddings |
| `SENSE_EMBEDDING_PREFIX` | `"classification: "` | Same prefix, context from gloss |
| `SENSE_BATCH_SIZE` | `256` | Senses per embedding batch |
| `SENSE_INTERMEDIATE_DIR` | `intermediates/senses` | Checkpoint dir for sense embeddings |
| `DOMAIN_COMPOUND_INTERMEDIATE_DIR` | `intermediates/domain_senses` | Checkpoint dir for domain compound embeddings |
| `COMPOSITIONALITY_THRESHOLD` | `0.20` | Jaccard below this = idiomatic |
| `CONTAMINATION_ZSCORE_MIN` | `2.0` | Min residual activation to flag |
| `ISLAND_JACCARD_ZSCORE` | `3.0` | Min hypergeometric z-score to include edge in island graph |
| `ISLAND_LEIDEN_RESOLUTION` | `1.0` | Leiden resolution (higher = more/smaller islands) |
| `ISLAND_CHARACTERISTIC_WORDS_N` | `100` | Top N PMI-ranked words stored per island |
| `ISLAND_MIN_COMMUNITY_SIZE` | `2` | Communities smaller than this become noise (island_id = -1) |
| `ISLAND_SUB_LEIDEN_RESOLUTION` | `1.5` | Leiden resolution for sub-island detection (higher = more splitting) |
| `ISLAND_MIN_DIMS_FOR_SUBDIVISION` | `2` | Don't subdivide islands with fewer dims than this |
| `REEF_MIN_DEPTH` | `2` | Min dims a word must activate in a reef to count as meaningfully present |
| `REEF_REFINE_MIN_DIMS` | `4` | Min dims for a reef to be analyzed for misplaced dims |
| `REEF_REFINE_LOYALTY_THRESHOLD` | `1.0` | Dims with loyalty_ratio below this are considered misplaced |
| `REEF_REFINE_MAX_ITERATIONS` | `5` | Max refinement rounds before stopping |
| `SENSE_SPREAD_INFLATED_THRESHOLD` | `15` | Min sense_spread to flag a universal word as polysemy-inflated |
| `DOMAIN_GENERAL_THRESHOLD` | `0.75` | Min arch_concentration for `v_domain_generals` view |
| `ABSTRACT_DIM_THRESHOLD` | `0.30` | Min universal_pct for `v_abstract_dims` view |
| `CONCRETE_DIM_THRESHOLD` | `0.15` | Max universal_pct for `v_concrete_dims` view |
| `FNV1A_OFFSET` | `14695981039346656037` | FNV-1a 64-bit offset basis |
| `FNV1A_PRIME` | `1099511628211` | FNV-1a 64-bit prime |
| `NOISE_RECOVERY_MIN_JACCARD` | `0.01` | Min avg Jaccard for noise dim recovery to nearest sibling reef |
| `BM25_K1` | `1.2` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 length normalization |
| `NEGATION_PREFIXES` | `['un', 'non', 'in', ...]` | Morphological negation prefixes for directed pair extraction |
| `POSITIVE_DIM_VALENCE_THRESHOLD` | `-0.15` | Valence below this = positive-pole dimension |
| `NEGATIVE_DIM_VALENCE_THRESHOLD` | `0.15` | Valence above this = negative-pole dimension |

## Verification Queries

After running the full pipeline, these queries confirm everything worked:

```sql
-- POS coverage: expect ~135K (92%)
SELECT count(*) FROM words WHERE pos IS NOT NULL;

-- Category distribution
SELECT category, count(*) FROM words GROUP BY category ORDER BY count(*) DESC;

-- Component decomposition: expect ~100K+ rows
SELECT count(*) FROM word_components;

-- Sense count: expect ~61K (all synsets for ambiguous words)
SELECT count(*) FROM word_senses;

-- Sense memberships populated
SELECT count(*) FROM sense_dim_memberships;

-- Top verb-enriched dimensions (should see dims 687, 636, 727 near top)
SELECT dim_id, verb_enrichment FROM dim_stats
ORDER BY verb_enrichment DESC LIMIT 10;

-- Most idiomatic compounds
SELECT word, jaccard, emergent_dims FROM compositionality
WHERE NOT is_compositional ORDER BY jaccard ASC LIMIT 20;

-- Words with highest compound contamination
SELECT w.word, count(*) as contaminated_dims
FROM dim_memberships dm JOIN words w ON w.word_id = dm.word_id
WHERE dm.compound_support > 0
GROUP BY w.word ORDER BY contaminated_dims DESC LIMIT 20;

-- Island count
SELECT COUNT(DISTINCT island_id) FROM dim_islands WHERE generation = 0 AND island_id >= 0;

-- Largest islands
SELECT island_id, n_dims, n_words, avg_internal_jaccard
FROM island_stats WHERE generation = 0 ORDER BY n_dims DESC LIMIT 10;

-- Top words for island 0
SELECT word, pmi, island_freq, corpus_freq
FROM island_characteristic_words WHERE island_id = 0 AND generation = 0
ORDER BY pmi DESC LIMIT 20;

-- Sub-island count
SELECT COUNT(DISTINCT island_id) FROM dim_islands WHERE generation = 1 AND island_id >= 0;

-- Sub-islands per parent
SELECT parent_island_id, COUNT(DISTINCT island_id) as n_sub_islands
FROM dim_islands WHERE generation = 1 AND island_id >= 0
GROUP BY parent_island_id ORDER BY parent_island_id;

-- Gen-2 reef count
SELECT COUNT(DISTINCT island_id) FROM dim_islands WHERE generation = 2 AND island_id >= 0;

-- Specificity distribution
SELECT specificity, COUNT(*), MIN(total_dims), MAX(total_dims)
FROM words GROUP BY specificity ORDER BY specificity DESC;

-- Naming coverage: expect all non-noise islands named after phase 10
SELECT generation, COUNT(*) as total,
       COUNT(island_name) as named, COUNT(*) - COUNT(island_name) as unnamed
FROM island_stats WHERE island_id >= 0 GROUP BY generation ORDER BY generation;

-- Sample names per generation
SELECT generation, island_id, island_name
FROM island_stats WHERE island_id >= 0 AND island_name IS NOT NULL
ORDER BY generation, island_id LIMIT 20;

-- Universal word analytics: dimension abstractness (expect 768)
SELECT COUNT(*) FROM dim_stats WHERE universal_pct IS NOT NULL;

-- Most abstract dims (highest universal word fraction)
SELECT dim_id, universal_pct, dim_weight FROM dim_stats
ORDER BY universal_pct DESC LIMIT 10;

-- Most concrete dims (lowest universal word fraction)
SELECT dim_id, universal_pct, dim_weight FROM dim_stats
WHERE universal_pct IS NOT NULL ORDER BY universal_pct ASC LIMIT 10;

-- Sense spread: words with polysemy inflation
SELECT COUNT(*) FROM words WHERE sense_spread IS NOT NULL;
SELECT COUNT(*) FROM words WHERE polysemy_inflated = TRUE;

-- Arch concentration: domain generals
SELECT COUNT(*) FROM words WHERE arch_concentration IS NOT NULL;
SELECT * FROM v_domain_generals LIMIT 20;

-- View counts
SELECT COUNT(*) FROM v_abstract_dims;
SELECT COUNT(*) FROM v_concrete_dims;

-- Word hashes: expect 0 nulls, 0 collisions
SELECT COUNT(*) FROM words WHERE word_hash IS NULL;
SELECT COUNT(*) = COUNT(DISTINCT word_hash) FROM words WHERE word_hash IS NOT NULL;

-- Reef IDF: expect ~146K with IDF, range ~0.94 to ~5.05
SELECT COUNT(*) FROM words WHERE reef_idf IS NOT NULL;
SELECT MIN(reef_idf), MAX(reef_idf) FROM words WHERE reef_idf IS NOT NULL;

-- Word variants: expect ~490K total, ~146K base + morphy
SELECT source, COUNT(*) FROM word_variants GROUP BY source;
SELECT COUNT(DISTINCT variant_hash) FROM word_variants;

-- Negation vector: expect 1 row
SELECT name, n_pairs, description FROM computed_vectors;

-- Dimension valence: expect 768 with valence, range ~[-0.70, 0.98]
SELECT COUNT(*) FROM dim_stats WHERE valence IS NOT NULL;
SELECT MIN(valence), MAX(valence) FROM dim_stats;

-- Positive/negative pole dims
SELECT COUNT(*) FROM v_positive_dims;
SELECT COUNT(*) FROM v_negative_dims;

-- Spot-check: dim 47 (futility) should have high positive valence
SELECT dim_id, valence FROM dim_stats WHERE dim_id IN (47, 205) ORDER BY dim_id;

-- Reef valence: expect N_archs + N_islands + N_reefs
SELECT COUNT(*) FROM island_stats WHERE valence IS NOT NULL;
SELECT generation, MIN(valence), MAX(valence) FROM island_stats
WHERE valence IS NOT NULL GROUP BY generation ORDER BY generation;

-- Most positive-pole and negative-pole reefs
SELECT island_id, island_name, valence FROM island_stats
WHERE generation = 2 AND valence IS NOT NULL ORDER BY valence ASC LIMIT 5;
SELECT island_id, island_name, valence FROM island_stats
WHERE generation = 2 AND valence IS NOT NULL ORDER BY valence DESC LIMIT 5;

-- POS composition: expect 768 dims with fractions summing to ~1.0
SELECT COUNT(*), AVG(noun_frac + verb_frac + adj_frac + adv_frac)
FROM dim_stats WHERE noun_frac IS NOT NULL;

-- Compare sense-aware vs unambiguous-only (noun_frac should be slightly lower than noun_pct)
SELECT ROUND(AVG(noun_pct), 4) AS old_noun, ROUND(AVG(noun_frac), 4) AS new_noun,
       ROUND(AVG(noun_pct) - AVG(noun_frac), 4) AS delta
FROM dim_stats WHERE noun_frac IS NOT NULL;

-- Reef POS composition: most verb-heavy reefs
SELECT island_name, noun_frac, verb_frac, adj_frac, adv_frac
FROM island_stats WHERE generation = 2
ORDER BY verb_frac DESC LIMIT 10;

-- Hierarchy POS coverage: expect N_archs + N_islands + N_reefs
SELECT COUNT(*) FROM island_stats WHERE noun_frac IS NOT NULL;

-- Hierarchy specificity: expect N_archs + N_islands + N_reefs entities
SELECT COUNT(*) FROM island_stats WHERE avg_specificity IS NOT NULL;

-- Reef edges: expect N_reefs * (N_reefs - 1)
SELECT COUNT(*) FROM reef_edges;

-- Pairs with word overlap
SELECT COUNT(*) FROM reef_edges WHERE containment > 0;

-- Containment range: [0, ~0.197]
SELECT MIN(containment), MAX(containment), AVG(containment) FROM reef_edges;

-- Lift range: [0, ~21.4]
SELECT MIN(lift), MAX(lift), AVG(lift) FROM reef_edges WHERE lift > 0;

-- POS similarity: all non-null, range ~[0.7, 1.0]
SELECT COUNT(*) FILTER (WHERE pos_similarity IS NULL), MIN(pos_similarity), MAX(pos_similarity)
FROM reef_edges;

-- Asymmetry check: containment(A→B) != containment(B→A) for most pairs
SELECT COUNT(*) FROM reef_edges a
JOIN reef_edges b ON a.source_reef_id = b.target_reef_id
  AND a.target_reef_id = b.source_reef_id
WHERE ABS(a.containment - b.containment) > 0.01;

-- Specificity gap: signed, should span both directions
SELECT MIN(specificity_gap), MAX(specificity_gap) FROM reef_edges;

```

## File Artifacts

After a full run, the project directory will contain:

```
vector_distillery.duckdb       The database (~500MB+ with embeddings)
intermediates/
  embeddings_final.npy         Cached word embeddings (~146K x 768 float32)
intermediates/senses/
  embeddings_final.npy         Cached sense embeddings (~61K x 768 float32)
intermediates/domain_senses/
  embeddings_final.npy         Cached domain compound embeddings
```

The `.npy` files are caches. If deleted, phase 2 or 5 will regenerate them (expensive). The database is the source of truth once populated.
