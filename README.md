# 🪟 Windowsill

_what you can see shifts when you switch from a 🔭 to a 🪟_

A vector space distillation engine that decomposes word embeddings into interpretable dimensions. Takes the 768 dimensions of a transformer embedding model and answers the question: *what does each dimension actually encode?*

## What This Project Does

Windowsill embeds the entire WordNet vocabulary (~146K lemmas) using [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), then statistically analyzes each of the 768 embedding dimensions to determine which words "activate" in that dimension. The result is a database where every word has a discrete set of dimension memberships, and every dimension has a characterized set of member words -- turning opaque floating-point vectors into an inspectable symbolic structure.

The core insight: embedding dimensions are not random. Each dimension has a small set of high-activation words (typically ~1-2% of the vocabulary) that often correspond to a coherent semantic concept. Windowsill identifies these clusters via z-score thresholding (mean + 2.0σ), determines per-dimension membership boundaries, and records the results.

### Beyond basic embeddings

The enrichment pipeline adds six layers of analysis on top of the base embedding decomposition:

1. **Sense disambiguation** -- The ~8% of words with multiple parts-of-speech in WordNet (e.g., "bank" as noun vs verb) get separate sense-specific embeddings using WordNet glosses as context. Each sense gets its own dimension profile, revealing that "bank (financial institution)" and "bank (river edge)" activate completely different dimensions.

2. **Compound decomposition** -- 43% of WordNet's vocabulary is multi-word expressions ("heart attack", "hot dog", "red herring"). These get decomposed into components and analyzed for compositionality. Vector arithmetic (`compound - component`) reveals whether a compound is compositional (meaning derived from parts) or idiomatic (meaning emergent).

3. **Contamination detection** -- When "attack" is a member of a cardiac-related dimension, is that because of the word "attack" itself, or because "heart attack" drags it there? Compound contamination scoring answers this by checking whether the residual (`"heart attack" - "attack"`) still activates that dimension.

4. **Island detection** -- Many of the 768 dimensions share dense word overlap, forming natural "islands" (communities of co-activated dimensions). Pairwise Jaccard similarity between dimension member sets is computed, then a hypergeometric z-score filters edges: for each pair, the expected intersection under random assignment is subtracted and normalized by the standard deviation, so only pairs with statistically significant overlap (z >= 3) form edges. Jaccard values weight edges for Leiden community detection, and PMI scoring identifies characteristic words per island. The hierarchy is three generations deep: gen-0 **archipelagos** subdivide into gen-1 **islands**, which further subdivide into gen-2 **reefs**. (Exact counts depend on the z-score threshold; see Distillation Results.)

5. **Denormalized hierarchy access** -- Reef/island/archipelago IDs are denormalized onto `dim_memberships` for direct query access without joining through `dim_islands`. The `word_reef_affinity` table stores continuous affinity scores (n_dims, max_z, sum_z, max_weighted_z, sum_weighted_z) for every word-reef pair, enabling relationship classification and reef discovery without expensive multi-table joins.

6. **Island & reef naming** -- Every entity in the three-generation hierarchy gets a human-readable name via a bottom-up LLM-assisted pipeline. Reefs are named from their most exclusive words (words present in that reef but absent from all sibling reefs), islands are named by synthesizing their child reef names, and archipelagos are named from their child island names. This bottom-up approach ensures names are grounded in the most specific, distinctive vocabulary rather than diluted by shared terms.

7. **Universal word analytics** -- Universal words (specificity < 0, ~24,651 words appearing in 23-44 dims) carry meaningful signal: dimensions they avoid are biological taxonomy (5.8% universal), dimensions they dominate are abstract/social concepts (48.7% universal). Six features leverage this: per-dimension **abstractness** (`universal_pct` + information-theoretic `dim_weight`), **sense spread** detecting polysemy-inflated universals, **arch concentration** identifying "domain generals" (universal words concentrated in one archipelago), **exclusion fingerprints** (shared reef avoidance between universal word pairs), and **bridge profiles** (cross-archipelago reef distributions).

8. **Evaluative polarity (valence)** -- Analysis of WordNet antonym pairs reveals a consistent "negation vector" in 768-space: the average direction from positive to negated forms (e.g., friendly → unfriendly) across ~1,600 morphological negation pairs. This vector has 98.2% directional consistency for morphological pairs and 72% for semantic antonyms. Projecting each dimension onto this vector yields a **valence score**: positive valence means negation increases activation (negative-pole dimension, e.g., "futility and incompleteness"), negative valence means negation decreases activation (positive-pole dimension, e.g., "eloquence and celebration"). Valence propagates to reefs/islands/archipelagos as mean dimension valence, providing an evaluative polarity axis that cross-cuts specificity and abstractness. The negation vector also enables **antonym prediction via embedding arithmetic** (`word ± negation_vector` → nearest neighbor search).

### The Big Picture - Archipelagos, Islands, and Reefs

![Full island hierarchy: 52 named islands decomposed into 207 reefs across 4 archipelagos](great_chart.png)

The complete 3-generation decomposition of nomic-embed-text-v1.5's 768 dimensions into interpretable semantic structure. Each row is a named gen-1 **island**, segmented into its gen-2 **reefs** (numbers show reef size in dimensions). Rows are grouped by their gen-0 **archipelago**: *natural sciences and taxonomy* (blue), *physical world and materiality* (orange), *abstract processes and systems* (green), and *social order and assessment* (red). Gray segments are noise dimensions that didn't form a reef community. The chart reads as a table of contents for the embedding space -- every dimension in the model has an address in this hierarchy, and every structure has a human-readable name derived from its most characteristic words. 100% of words in the vocabulary are associated with at least one reef.

## Quick Start

initialize a venv and install the requirements
```bash
python3 -m venv ws
source ws/bin/activate
pip3 install -r requirements.txt
```

start the extraction
```bash
python3 main.py --from 2
```

The database that is created is approximately `13 gb`

## Even quicker start

If you do not want to build the database for yourself, you can download the latest snapshot from ~[Hugging Face](https://huggingface.co/datasets/morimar/windowsill/tree/main)

## Data Characteristics

Key numbers discovered through investigation of the WordNet source data:

| Metric | Value | Notes |
|--------|-------|-------|
| Total words | ~146K | WordNet lemmas after cleaning |
| Multi-word expressions | ~64K (43%) | Underscores converted to spaces |
| Unambiguous POS | ~135K (92%) | Single part-of-speech in WordNet |
| Ambiguous POS | ~11.7K (8%) | Multiple POS, need sense-specific embedding |
| Embedding dimensions | 768 | Matryoshka dimensionality |

### Distillation Results

Output from a full pipeline run (z-score threshold = 2.0, REEF_MIN_DEPTH = 2):

| Metric | Value | Notes |
|--------|-------|-------|
| Total memberships | ~2.56M | 3x increase from z=2.45 (~800K) |
| Avg members/dim | ~3,339 | Per dimension |
| Avg dims/word | ~17.5 | Up from ~6 at z=2.45 |
| Word-reef affinity rows | ~1.97M | Every word-reef pair where word activates in >= 1 reef dim |
| Reef coverage (any depth) | 100.0% | 146,695 / 146,697 words touch at least one reef |
| Reef coverage (depth >= 2) | 54.1% | 79,395 words with strong multi-dim reef membership |
| Gen-0 archipelagos | 4 | Leiden community detection |
| Gen-1 islands | 52 | Sub-island subdivision |
| Gen-2 reefs | 207 | Sub-island subdivision |
| All structures named | Yes | Bottom-up LLM naming (phase 9c) |
| Word variant mappings | ~490K | base (~146K) + morphy (~344K) in word_variants table |
| Universal words | 24,651 | specificity < 0 (23-44 dims) |
| Abstract dims | 128 | universal_pct >= 30% |
| Concrete dims | 46 | universal_pct <= 15% |
| Domain generals | 111 | Universal words with arch_concentration >= 0.75 |
| Polysemy-inflated | 293 | Universal + sense_spread >= 15 |
| Negation vector pairs | 1,639 | Directed morphological negation pairs (norm 6.13) |
| Positive-pole dims | 184 | Valence <= -0.15 (negation decreases activation) |
| Negative-pole dims | 182 | Valence >= 0.15 (negation increases activation) |
| Reef valence range | [-0.53, 0.69] | Mean dim valence per reef |

### Word categories

Each word is classified into one of five categories:

- **single** (~82K) -- Single words, no spaces
- **compound** (~47K) -- Multi-word expressions (default for multi-word)
- **taxonomic** (~5.5K) -- Starts with genus/family/order/class/etc.
- **phrasal_verb** (~2.8K) -- Multi-word with verb synsets
- **named_entity** (thousands) -- Multi-word with WordNet `instance_hypernyms()` (specific people/places/things)

## Architecture

### Pipeline Phases

The system runs as a sequential pipeline. Each phase can be run independently via `--phase`.

```
Phase 2:  Word list curation          (extract + clean WordNet lemmas)
Phase 3:  Embedding generation         (nomic-embed-text-v1.5, CPU, ~30 min)
Phase 4:  Database schema + bulk load  (create tables, insert words + embeddings)
Phase 4b: Schema migration + backfill  (POS/category/components on existing DB)
Phase 4c: Word hash computation        (FNV-1a u64 hashes for all words)
Phase 5:  Statistical analysis         (per-dimension z-score threshold)
Phase 5b: Sense embedding generation   (gloss-contextualized, ~5 min CPU)
Phase 5c: Sense analysis               (apply existing thresholds to senses)
Phase 6:  Post-processing              (dim counts, specificity bands, views, pair overlap)
Phase 6b: POS enrichment + compounds   (contamination scoring, compositionality, dimension abstractness/specificity, sense spread, negation vector, dimension valence)
Phase 9:  Island detection             (Jaccard matrix, Leiden clustering, 3-gen hierarchy, backfill + affinity)
Phase 9b: Backfill + affinity          (re-backfill denormalized columns + recompute word_reef_affinity)
Phase 9d: Universal word analytics     (arch concentration, reef valence, domain generals -- needs island data)
Phase 9e: Reef IDF computation         (BM25 IDF from word_reef_affinity)
Phase 10: Reef refinement              (dim loyalty analysis, iterative regrouping, re-backfill + affinity)
Phase 9f: POS composition              (sense-aware fractional POS fractions for dims, reefs, islands, archipelagos)
Phase 9g: Reef edges                   (directed component scores for all reef pairs: containment, lift, POS similarity, valence gap, specificity gap)
Phase 9c: Island & reef naming         (LLM-assisted naming via Claude API, bottom-up)
Phase 11: Morphy variant expansion     (WordNet morphy() inflection mapping)
Phase 7:  Database maintenance         (integrity checks, reindex, ANALYZE, CHECKPOINT)
explore:  Interactive explorer         (REPL for querying the database)
```

Phase order: `2 -> 3 -> 4 -> 4b -> 4c -> 5 -> 5b -> 5c -> 6 -> 6b -> 9 -> 9b -> 9d -> 9e -> 10 -> 9f -> 9g -> 9c -> 11 -> 7 -> explore`

Phases 4b/5b/5c/6b/9d are enrichment phases designed to run on an already-populated database. They use ALTER TABLE to add columns, so they're safe to run without re-running the expensive embedding pipeline.

**Dependency notes:**
- Phases 4b (POS backfill) is independent of 5b/5c (senses) and 6b (compounds)
- Phase 5b requires 4b to have run (needs `pos IS NULL` to identify ambiguous words)
- Phase 5c requires 5b (needs sense embeddings in the DB)
- Phase 6b requires 4b (needs category/components populated); sense spread gracefully skips if 5b hasn't run; negation vector + dimension valence need only word embeddings + WordNet
- Phase 9 requires phase 5 (needs dim_memberships populated)
- Phase 9b requires phase 9 (needs island hierarchy; standalone re-backfill + affinity recompute)
- Phase 9d requires phase 9 (needs island hierarchy for arch_concentration) + phase 6b (needs dim_stats.valence for reef valence); also re-creates views
- Phase 10 requires phase 9 (needs reef assignments + Jaccard data); re-backfills + recomputes affinity + reef valence after convergence
- Phase 9c requires phase 10 (or 9 if skipping refinement); needs island hierarchy + characteristic words post-refinement; requires `ANTHROPIC_API_KEY`. Runs after phase 10 because refinement recomputes reef stats (wiping names).
- Phase 9f requires phase 10 (finalized reef assignments) + phase 5c (sense analysis); computes sense-aware fractional POS composition at dim/reef/island/archipelago levels
- Phase 9g requires phase 9f (POS fractions) + phase 10 (finalized reefs) + phase 6b (valence, specificity); computes directed reef-pair component scores
- Phase 4c is independent of other phases (only needs words table populated)
- Phase 9e requires phase 9b (needs word_reef_affinity populated)
- Phase 11 requires phase 4c (needs word_hash column populated)

### File Layout

```
config.py        Constants: model name, thresholds, batch sizes, paths, FNV-1a hashing, BM25 params
database.py      DuckDB schema, migrations, insert/load functions
word_list.py     WordNet extraction, cleaning, POS/category classification, FNV-1a hashing, morphy expansion
embedder.py      Sentence-transformers encoding, checkpointing, sense embedding
analyzer.py      Per-dimension z-score thresholding, sense analysis
post_process.py  Dim counts, specificity bands, views, pair overlap, POS enrichment,
                 contamination, compositionality, dimension abstractness/specificity,
                 sense spread, arch concentration, reef IDF, negation vector,
                 dimension valence, reef valence, hierarchy specificity, reef edges
islands.py       Island detection: Jaccard matrix, Leiden clustering, PMI scoring,
                 denormalization, word-reef affinity, LLM-assisted bottom-up naming
reef_refine.py   Reef refinement: iterative dim loyalty analysis and reassignment
main.py          Pipeline orchestration, CLI argument parsing, explorer REPL
explore.py       Interactive query functions (what_is, words_like, archipelago,
                 relationship, exclusion, bridge_profile, senses, synonyms,
                 antonyms, etc.)
```

### Key Design Decisions

- **DuckDB** for storage -- embeddings stored as `FLOAT[768]` arrays directly in the DB, enabling SQL queries alongside vector operations without a separate vector store.
- **Z-score thresholding (mean + 2.0σ)** -- Each dimension's membership threshold is `mean + 2.0 * std`. The threshold was lowered from 2.45σ (~6 dims/word, ~800K memberships) to 2.0σ (~17 dims/word, ~2.5M memberships) to provide richer data for island/reef clustering. At 2.45σ, dimension-pair Jaccard similarities were too sparse for coherent reef formation — semantically related dimensions (e.g., musical instrument dims) had jaccard < 0.01 and couldn't cluster together. At 2.0σ, the 3x membership increase raises pairwise Jaccard into the significant range (hyper_z 5-15 for related dims), producing tighter, more semantically coherent reefs. The noise introduced by the lower threshold is addressed by a secondary **reef depth filter** (`REEF_MIN_DEPTH = 2`): a word's reef bit is only encoded if the word appears in ≥ 2 of the reef's dimensions. At 2.0σ, genuine concept membership manifests as multi-dimension overlap (e.g., "guitar" activates 7/17 music-related dims), while noise connections are single-dim and get pruned.
- **Matryoshka at 768** -- nomic-embed-text-v1.5 supports Matryoshka dimensionality reduction. We use the full 768 for maximum resolution.
- **`"classification: "` prefix** -- The Nomic model uses task-specific prefixes. All words are embedded with this prefix to activate the classification head, which produces more discriminative dimensions.
- **Sense embedding format** -- Ambiguous words are re-embedded as `"classification: {word}: {gloss}"` where the gloss comes from WordNet. This contextualizes the word without changing the embedding space.
- **Compositionality via Jaccard** -- A compound is compositional if the Jaccard similarity between its dimension set and the union of its components' dimension sets is >= 0.20. Below that threshold, it's idiomatic (meaning not derivable from parts).
- **Contamination via residual activation** -- For a component word W in dimension D, compound_support counts how many compounds containing W are also in D AND whose residual (compound_embedding - W_embedding) still exceeds D's threshold. High support means D might be compound-derived, not intrinsic to W.
- **Specificity bands** -- Words are classified into sigma-based bands (`+2` to `-2`) based on their `total_dims` distance from the population mean. This replaces the original hardcoded thresholds in `v_unique_words` (was `<= 15`) and `v_universal_words` (was `>= 36`) with statistically derived boundaries that adapt to the actual distribution.

## Database Schema

> For full column-level detail — types, nullability, value ranges, example values, FK relationships, and join patterns — see the [Data Dictionary](data_dictionary.md).

### Core Tables (Phases 2-6)

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
reef_idf           DOUBLE     BM25 IDF: ln((207 - n + 0.5) / (n + 0.5) + 1) where n = reefs containing word
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

### Enrichment Tables (Phases 4b-6b)

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
sense_id         INTEGER PK    Sequential sense ID
word_id          INTEGER       FK to words
pos              TEXT          POS for this specific sense
synset_name      TEXT          WordNet synset (e.g., 'bank.n.02')
gloss            TEXT          WordNet definition text
sense_embedding  FLOAT[768]    Embedding of "classification: word: gloss"
total_dims       INTEGER       Dimensions this sense belongs to
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

### Island Tables (Phase 9)

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
island_name            TEXT       Human-readable label (set by phase 9c naming pipeline)
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

**`reef_edges`** -- Directed component scores for all reef pairs (207 x 206 = 42,642 rows)
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

## Usage

### Full pipeline (fresh start)

```bash
python main.py
```

This runs phases 2 through explore. Phase 3 (embedding) takes ~30 min on CPU. Phase 6 pair overlap takes 10-30 min. Use `--skip-pair-overlap` to skip the expensive pair materialization.

### Run specific phases

```bash
python main.py --phase 4b          # Just schema migration + POS backfill
python main.py --from 5b           # Run phases 5b, 5c, 6, 6b, ..., explore
python main.py --phase explore     # Jump straight to explorer
python main.py --phase 10          # Run reef refinement only
python main.py --phase 4c           # Compute FNV-1a word hashes
```

### Enrichment pipeline on existing DB

If you already have the base pipeline done (phases 2-6), run the enrichment:

```bash
python main.py --phase 4b          # ~1 min: POS, categories, components
python main.py --phase 5b          # ~5 min: sense embeddings (needs model)
python main.py --phase 5c          # ~30 sec: sense analysis
python main.py --phase 6b          # ~2-5 min: POS enrichment, contamination, compositionality, dim abstractness, sense spread, negation vector, dimension valence
```

### Island detection

```bash
python main.py --phase 9             # Full: Jaccard matrix, Leiden (gen 0-2), backfill + affinity
python main.py --phase 9b            # Re-backfill denormalized columns + recompute affinity
```

Phase 9 runs the full three-generation hierarchy, backfills denormalized island/reef columns onto `dim_memberships`, and computes `word_reef_affinity` scores. Phase 9b is a lightweight standalone step that re-runs only the backfill and affinity computation -- useful if you've modified island assignments and need to refresh without recomputing the Jaccard matrix or re-running Leiden.

### Reef refinement

```bash
python main.py --phase 10            # Iterative dim loyalty refinement
```

Phase 10 examines every reef with 4+ dims and checks whether each dim has higher Jaccard affinity to its own reef or to a sibling reef (same parent island). Dims with a loyalty ratio below 1.0 are reassigned to their best-fit sibling. The process iterates until no more moves are needed (typically 2-3 rounds). After convergence, reef stats, characteristic words, denormalized columns, affinity scores, and reef valence are all refreshed automatically.

### Universal word analytics (post-island)

```bash
python main.py --phase 9d            # Arch concentration + domain general views
```

Phase 9d computes `arch_concentration` for all universal words (requires island data from phase 9). This measures how concentrated each universal word's dimensions are within a single archipelago -- a value of 0.75+ means the word is a "domain general" (universal but topically focused). Also computes **reef valence** (mean dimension valence for each island/reef/archipelago, requires `dim_stats.valence` from phase 6b) and re-creates views to include `v_domain_generals`, `v_positive_dims`, and `v_negative_dims`.

### POS composition (sense-aware)

```bash
python main.py --phase 9f            # Sense-aware fractional POS fractions
```

Phase 9f computes sense-aware POS composition fractions at every level of the hierarchy: dimension, reef, island, archipelago. Unambiguous words contribute 1.0 to their known POS; ambiguous words contribute fractional weights based on which of their senses activate in each dimension. Requires phase 10 (finalized reef assignments) and phase 5c (sense analysis). The fractions are also refreshed automatically by phase 10 (reef refinement) after convergence.

### Reef edges (directed pair scores)

```bash
python main.py --phase 9g            # Compute directed reef-pair component scores
```

Phase 9g computes five directed component scores for every reef pair (207 x 206 = 42,642 rows): **containment** (fraction of source words also in target), **lift** (co-activation above baseline), **POS similarity** (cosine of POS fraction vectors), **valence gap** (signed difference), and **specificity gap** (signed difference). These are stable data about embedding geometry — the composite weight used at runtime is NOT stored here but computed at export time. Requires phase 9f (POS fractions), phase 10 (finalized reefs), and phase 6b (valence, specificity). Also refreshed automatically by phase 10 (reef refinement) after convergence.

### Scoring engine prep

```bash
python main.py --phase 4c            # ~30 sec: compute FNV-1a word hashes
python main.py --phase 9e            # ~5 sec: compute reef IDF
python main.py --phase 11            # ~5 min: morphy variant expansion
```

Phase 4c computes FNV-1a u64 hashes for all words (zero collisions confirmed). Phase 9e computes BM25 IDF values from `word_reef_affinity`. Phase 11 expands WordNet `morphy()` variants, creating a mapping from inflected forms back to base word_ids in the `word_variants` table (~490K entries).

### Island & reef naming

```bash
python main.py --phase 9c            # Generate names for all islands and reefs via Claude API
```

Phase 9c generates human-readable names for all gen-1 islands and gen-2 reefs using a bottom-up approach:

1. **Reefs first** -- For each island, computes words exclusive to each child reef (present in that reef's dimensions but absent from all sibling reefs). These exclusive words plus PMI-ranked characteristic words are sent to Claude to generate a 2-4 word descriptive name per reef.
2. **Islands next** -- Each island is named by synthesizing the names of its constituent reefs. Claude sees all reef names and generates a broader label.
3. **Archipelagos last** -- Each archipelago is named by synthesizing its constituent island names.

Islands without child reefs fall back to their PMI-ranked characteristic words for naming. Requires `ANTHROPIC_API_KEY` environment variable. Names are stored in `island_stats.island_name` and are idempotent (re-running overwrites previous names).

### Database maintenance

Phase 7 optimizes the database before the interactive explorer session. It runs automatically as part of the pipeline, or standalone:

```bash
python main.py --phase 7           # Run maintenance on existing DB
```

This phase performs:
- **Integrity checks** -- FK validation across all tables, sanity checks on row counts
- **Index rebuild** -- Drops and recreates all 24 indexes for consistent state
- **ANALYZE** -- Recomputes query planner statistics after bulk inserts/updates
- **FORCE CHECKPOINT** -- Flushes WAL to disk and reclaims space from deleted row groups

### CLI options

```
--phase PHASE          Run only this phase (2, 3, 4, 4b, 4c, 5, 5b, 5c, 6, 6b, 9, 9b, 9c, 9d, 9e, 9f, 9g, 10, 11, 7, explore)
--from PHASE           Run from this phase through the end
--db PATH              Database path (default: vector_distillery.duckdb)
--skip-pair-overlap    Skip the expensive pair overlap materialization
--no-resume            Don't resume from intermediate .npy checkpoint files
```

### Interactive Explorer

The explore phase launches a REPL with a read-only database connection. Commands:

```
what_is <word>                Show all dimensions a word belongs to + island/reef summary
words_like <word> [n]         Find similar words by shared dimensions
dim <id> [n]                  List top members of a dimension + cluster name
compare <word1> <word2>       Jaccard similarity, shared/unique dims
disambiguate <word>           Cluster dimensions into potential senses
bridges <word1> <word2> [n]   Words that share dims with both inputs
dim_info <id>                 Full statistical profile + island hierarchy with names
search <pattern>              SQL LIKE search over vocabulary
senses <word>                 WordNet senses with per-sense dimension profiles
compositionality <compound>   Compositional vs idiomatic analysis
contamination <word>          Which dims have compound contamination support
pos_dims <pos>                Dimensions most enriched for verb/adj/adv
archipelago <word>            Nested island hierarchy with per-dimension stats
relationship <word1> <word2>  Classify relationship + named shared structures
exclusion <word1> <word2>     Shared reef exclusions between universal words (Jaccard of avoided reefs)
bridge_profile <word>         Reef distribution by archipelago + cross-archipelago bridge pairs
affinity <word>               Reef affinity profile — all reefs ranked by weighted z-score
synonyms <word> [n]          Synonym candidates via dimension Jaccard overlap (same POS)
antonyms <word> [n]          Antonym prediction via negation vector embedding arithmetic
```

Multi-word inputs work naturally: `what_is heart attack`, `senses bank`.

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

These sense embeddings are evaluated against the **same dimension thresholds** computed from the full word set in phase 5. No thresholds are recomputed -- senses are measured against the existing statistical framework.

### Intermediate checkpointing

Embedding is the most expensive operation. The system saves `.npy` checkpoint files every 50 batches to `intermediates/` (or `intermediates/senses/` for sense embeddings). If interrupted, it resumes from the last checkpoint. A `embeddings_final.npy` file is saved on completion and used for instant reload on re-runs.

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

-- Naming coverage: expect all non-noise islands named after phase 9c
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

-- Reef IDF: expect ~146K with IDF, range ~1.80 to ~4.93
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

-- Reef valence: expect 263 (4 archs + 52 islands + 207 reefs)
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

-- Hierarchy POS coverage: expect 4 + 52 + 207 = 263
SELECT COUNT(*) FROM island_stats WHERE noun_frac IS NOT NULL;

-- Hierarchy specificity: expect 263 (4+52+207) entities
SELECT COUNT(*) FROM island_stats WHERE avg_specificity IS NOT NULL;

-- Reef edges: expect exactly 42,642 (207 * 206)
SELECT COUNT(*) FROM reef_edges;

-- Pairs with word overlap: expect ~24,460
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

## Dependencies

- **duckdb** -- Embedded analytical database
- **numpy**, **pandas** -- Array operations and dataframe-based bulk inserts
- **sentence-transformers** -- Embedding model loading and encoding
- **nltk** -- WordNet access (auto-downloads `wordnet` and `omw-1.4` corpora)
- **scipy** -- Skewness/kurtosis statistics
- **scikit-learn** -- Agglomerative Clustering (sense disambiguation)
- **tqdm** -- Progress bars
- **leidenalg** -- Leiden community detection algorithm (may require a C compiler on some systems)
- **python-igraph** -- Graph library used by leidenalg
- **anthropic** -- Claude API client for LLM-assisted island/reef naming (phase 9c only)

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
| `COMPOSITIONALITY_THRESHOLD` | `0.20` | Jaccard below this = idiomatic |
| `CONTAMINATION_ZSCORE_MIN` | `2.0` | Min residual activation to flag |
| `ISLAND_JACCARD_ZSCORE` | `3.0` | Min hypergeometric z-score to include edge in island graph |
| `ISLAND_LEIDEN_RESOLUTION` | `1.0` | Leiden resolution (higher = more/smaller islands) |
| `ISLAND_CHARACTERISTIC_WORDS_N` | `100` | Top N PMI-ranked words stored per island |
| `ISLAND_MIN_COMMUNITY_SIZE` | `2` | Communities smaller than this become noise (island_id = -1) |
| `ISLAND_SUB_LEIDEN_RESOLUTION` | `1.5` | Leiden resolution for sub-island detection (higher = more splitting) |
| `ISLAND_MIN_DIMS_FOR_SUBDIVISION` | `10` | Don't subdivide islands with fewer dims than this |
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
| `N_REEFS` | `207` | Total reef count |
| `N_ISLANDS` | `52` | Total island count |
| `N_ARCHS` | `4` | Total archipelago count |
| `BM25_K1` | `1.2` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 length normalization |
| `NEGATION_PREFIXES` | `['un', 'non', 'in', ...]` | Morphological negation prefixes for directed pair extraction |
| `POSITIVE_DIM_VALENCE_THRESHOLD` | `-0.15` | Valence below this = positive-pole dimension |
| `NEGATIVE_DIM_VALENCE_THRESHOLD` | `0.15` | Valence above this = negative-pole dimension |

## File Artifacts

After a full run, the project directory will contain:

```
vector_distillery.duckdb       The database (~500MB+ with embeddings)
intermediates/
  embeddings_final.npy         Cached word embeddings (~146K x 768 float32)
intermediates/senses/
  embeddings_final.npy         Cached sense embeddings (~61K x 768 float32)
```

The `.npy` files are caches. If deleted, phase 3 or 5b will regenerate them (expensive). The database is the source of truth once populated.
