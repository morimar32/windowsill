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

5. **Archipelago encoding** -- The three-generation island hierarchy is pivoted from normalized tables into 8 BIGINT bitmask columns on the `words` table (512 bits available). The `archipelago` column packs gen-0 archipelago bits in the low positions and the first half of gen-1 island bits above them; `archipelago_ext` holds the second half of gen-1 islands. Each `reef_N` column is dedicated to one gen-0 archipelago's gen-2 reefs. One AND operation replaces multi-table JOINs: checking whether two words share a reef is `(a.reef_0 & b.reef_0) | ... != 0`. The encoding also enables instant relationship classification between any two words (same reef / reef neighbors / island neighbors / different archipelagos).

6. **Island & reef naming** -- Every entity in the three-generation hierarchy gets a human-readable name via a bottom-up LLM-assisted pipeline. Reefs are named from their most exclusive words (words present in that reef but absent from all sibling reefs), islands are named by synthesizing their child reef names, and archipelagos are named from their child island names. This bottom-up approach ensures names are grounded in the most specific, distinctive vocabulary rather than diluted by shared terms.

7. **Universal word analytics** -- Universal words (specificity < 0, ~24,651 words appearing in 23-44 dims) carry meaningful signal: dimensions they avoid are biological taxonomy (5.8% universal), dimensions they dominate are abstract/social concepts (48.7% universal). Six features leverage this: per-dimension **abstractness** (`universal_pct` + information-theoretic `dim_weight`), **sense spread** detecting polysemy-inflated universals, **arch concentration** identifying "domain generals" (universal words concentrated in one archipelago), **exclusion fingerprints** (shared reef avoidance between universal word pairs), and **bridge profiles** (cross-archipelago reef distributions).

### The Big Picture - Archipelagos, Islands, and Reefs

![Full island hierarchy: 52 named islands decomposed into 208 reefs across 4 archipelagos](great_chart.png)

The complete 3-generation decomposition of nomic-embed-text-v1.5's 768 dimensions into interpretable semantic structure. Each row is a named gen-1 **island**, segmented into its gen-2 **reefs** (numbers show reef size in dimensions). Rows are grouped by their gen-0 **archipelago**: *natural sciences and taxonomy* (blue), *physical world and materiality* (orange), *abstract processes and systems* (green), and *social order and assessment* (red). Gray segments are noise dimensions that didn't form a reef community. The chart reads as a table of contents for the embedding space -- every dimension in the model has an address in this hierarchy, and every structure has a human-readable name derived from its most characteristic words.

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

The database that is created is approximately `2.6 gb`

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
| Gen-0 archipelagos | 4 | Leiden community detection |
| Gen-1 islands | 52 | Sub-island subdivision |
| Gen-2 reefs | 208 | Sub-island subdivision |
| All structures named | Yes | Bottom-up LLM naming (phase 9c) |
| Archipelago encoding bits | 264 used | Across 8 BIGINT columns (512 available) |
| Universal words | 24,651 | specificity < 0 (23-44 dims) |
| Abstract dims | 128 | universal_pct >= 30% |
| Concrete dims | 46 | universal_pct <= 15% |
| Domain generals | 111 | Universal words with arch_concentration >= 0.75 |
| Polysemy-inflated | 293 | Universal + sense_spread >= 15 |

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
Phase 5:  Statistical analysis         (per-dimension z-score threshold)
Phase 5b: Sense embedding generation   (gloss-contextualized, ~5 min CPU)
Phase 5c: Sense analysis               (apply existing thresholds to senses)
Phase 6:  Post-processing              (dim counts, specificity bands, views, pair overlap)
Phase 6b: POS enrichment + compounds   (contamination scoring, compositionality, dimension abstractness, sense spread)
Phase 9:  Island detection             (Jaccard matrix, Leiden clustering, 3-gen hierarchy, encoding)
Phase 9b: Archipelago encoding         (standalone bitmask encoding for existing island data)
Phase 9c: Island & reef naming         (LLM-assisted naming via Claude API, bottom-up)
Phase 9d: Universal word analytics     (arch concentration, domain generals -- needs island data)
Phase 7:  Database maintenance         (integrity checks, reindex, ANALYZE, CHECKPOINT)
Phase 8:  Interactive explorer         (REPL for querying the database)
```

Phase order: `2 -> 3 -> 4 -> 4b -> 5 -> 5b -> 5c -> 6 -> 6b -> 9 -> 9b -> 9c -> 9d -> 7 -> 8`

Phases 4b/5b/5c/6b/9d are enrichment phases designed to run on an already-populated database. They use ALTER TABLE to add columns, so they're safe to run without re-running the expensive embedding pipeline.

**Dependency notes:**
- Phases 4b (POS backfill) is independent of 5b/5c (senses) and 6b (compounds)
- Phase 5b requires 4b to have run (needs `pos IS NULL` to identify ambiguous words)
- Phase 5c requires 5b (needs sense embeddings in the DB)
- Phase 6b requires 4b (needs category/components populated); sense spread gracefully skips if 5b hasn't run
- Phase 9 requires phase 5 (needs dim_memberships populated)
- Phase 9b requires phase 9 (needs island hierarchy; standalone re-encoding if schema changes)
- Phase 9c requires phase 9 (needs island hierarchy + characteristic words; requires `ANTHROPIC_API_KEY`)
- Phase 9d requires phase 9 (needs island hierarchy for arch_concentration); also re-creates views

### File Layout

```
config.py        Constants: model name, thresholds, batch sizes, paths
database.py      DuckDB schema, migrations, insert/load functions
word_list.py     WordNet extraction, cleaning, POS/category classification
embedder.py      Sentence-transformers encoding, checkpointing, sense embedding
analyzer.py      Per-dimension z-score thresholding, sense analysis
post_process.py  Dim counts, specificity bands, views, pair overlap, POS enrichment,
                 contamination, compositionality, dimension abstractness, sense spread,
                 arch concentration
islands.py       Island detection: Jaccard matrix, Leiden clustering, PMI scoring,
                 archipelago encoding, LLM-assisted bottom-up naming
main.py          Pipeline orchestration, CLI argument parsing, explorer REPL
explore.py       Interactive query functions (what_is, words_like, archipelago,
                 relationship, exclusion, bridge_profile, senses, etc.)
```

### Key Design Decisions

- **DuckDB** for storage -- embeddings stored as `FLOAT[768]` arrays directly in the DB, enabling SQL queries alongside vector operations without a separate vector store.
- **Z-score thresholding (mean + 2.0σ)** -- Each dimension's membership threshold is `mean + 2.0 * std`. The threshold was lowered from 2.45σ (~6 dims/word, ~800K memberships) to 2.0σ (~17 dims/word, ~2.5M memberships) to provide richer data for island/reef clustering. At 2.45σ, dimension-pair Jaccard similarities were too sparse for coherent reef formation — semantically related dimensions (e.g., musical instrument dims) had jaccard < 0.01 and couldn't cluster together. At 2.0σ, the 3x membership increase raises pairwise Jaccard into the significant range (hyper_z 5-15 for related dims), producing tighter, more semantically coherent reefs. The noise introduced by the lower threshold is addressed by a secondary **reef depth filter** (`REEF_MIN_DEPTH = 2`): a word's reef bit is only encoded if the word appears in ≥ 2 of the reef's dimensions. At 2.0σ, genuine concept membership manifests as multi-dimension overlap (e.g., "guitar" activates 7/17 music-related dims), while noise connections are single-dim and get pruned.
- **Matryoshka at 768** -- nomic-embed-text-v1.5 supports Matryoshka dimensionality reduction. We use the full 768 for maximum resolution.
- **`"classification: "` prefix** -- The Nomic model uses task-specific prefixes. All words are embedded with this prefix to activate the classification head, which produces more discriminative dimensions.
- **Sense embedding format** -- Ambiguous words are re-embedded as `"classification: {word}: {gloss}"` where the gloss comes from WordNet. This contextualizes the word without changing the embedding space.
- **Compositionality via Jaccard** -- A compound is compositional if the Jaccard similarity between its dimension set and the union of its components' dimension sets is >= 0.20. Below that threshold, it's idiomatic (meaning not derivable from parts).
- **Contamination via residual activation** -- For a component word W in dimension D, compound_support counts how many compounds containing W are also in D AND whose residual (compound_embedding - W_embedding) still exceeds D's threshold. High support means D might be compound-derived, not intrinsic to W.
- **Archipelago bitmask encoding** -- The 3-generation island hierarchy is encoded as 8 BIGINT columns (512 bits available). The `archipelago` column packs gen-0 bits in the low positions and the first half of gen-1 islands above them; `archipelago_ext` holds the second half of gen-1 islands. Gen-1 islands are split evenly across the two columns for headroom. Masking gen-0: `& ((1 << gen0_count) - 1)`; masking gen-1: `>> gen0_count` on `archipelago` plus all of `archipelago_ext`. Each `reef_N` column (0-5) is dedicated to one gen-0 archipelago's reefs, keeping per-archipelago masking as clean bit ranges. One AND replaces a multi-table JOIN for relationship checks.
- **Specificity bands** -- Words are classified into sigma-based bands (`+2` to `-2`) based on their `total_dims` distance from the population mean. This replaces the original hardcoded thresholds in `v_unique_words` (was `<= 15`) and `v_universal_words` (was `>= 36`) with statistically derived boundaries that adapt to the actual distribution.

## Database Schema

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
archipelago       BIGINT    Bitmask: gen-0 (low bits) + gen-1 first half (above gen-0)
archipelago_ext   BIGINT    Bitmask: gen-1 second half (bits 0+)
reef_0            BIGINT    Bitmask: gen-2 reefs for archipelago 0
reef_1            BIGINT    Bitmask: gen-2 reefs for archipelago 1
reef_2            BIGINT    Bitmask: gen-2 reefs for archipelago 2
reef_3            BIGINT    Bitmask: gen-2 reefs for archipelago 3
reef_4            BIGINT    Bitmask: gen-2 reefs for archipelago 4
reef_5            BIGINT    Bitmask: gen-2 reefs for archipelago 5
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
```

**`dim_memberships`** -- Which words belong to which dimensions
```
dim_id           INTEGER       (PK with word_id)
word_id          INTEGER
value            DOUBLE        Raw activation value
z_score          DOUBLE        Standard deviations above mean
compound_support INTEGER       How many compounds contribute to this membership
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
arch_column            TEXT       Encoding column: 'archipelago' | 'archipelago_ext' | 'reef_0'..'reef_5'
arch_bit               INTEGER    Bit position within that column
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

### Views

- `v_unique_words` -- Words with positive specificity (1σ+ fewer dims than mean; specific words)
- `v_universal_words` -- Words with negative specificity (1σ+ more dims than mean; general words)
- `v_selective_dims` -- Dimensions with selectivity < 5% (sharp concepts)
- `v_archipelago_profile` -- Words with non-zero encoding: archipelago/island/reef counts via `bit_count()` (spans all 8 bitmask columns)
- `v_abstract_dims` -- Dimensions where >= 30% of members are universal words (dominated by abstract/social concepts)
- `v_concrete_dims` -- Dimensions where <= 15% of members are universal words (concrete/taxonomic domains)
- `v_domain_generals` -- Universal words with arch_concentration >= 0.75 (concentrated in one archipelago)

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
```

## Usage

### Full pipeline (fresh start)

```bash
python main.py
```

This runs phases 2 through 8. Phase 3 (embedding) takes ~30 min on CPU. Phase 6 pair overlap takes 10-30 min. Use `--skip-pair-overlap` to skip the expensive pair materialization.

### Run specific phases

```bash
python main.py --phase 4b          # Just schema migration + POS backfill
python main.py --from 5b           # Run phases 5b, 5c, 6, 6b, 7, 8
python main.py --phase 8           # Jump straight to explorer
```

### Enrichment pipeline on existing DB

If you already have the base pipeline done (phases 2-6), run the enrichment:

```bash
python main.py --phase 4b          # ~1 min: POS, categories, components
python main.py --phase 5b          # ~5 min: sense embeddings (needs model)
python main.py --phase 5c          # ~30 sec: sense analysis
python main.py --phase 6b          # ~2-5 min: POS enrichment, contamination, compositionality, dim abstractness, sense spread
```

### Island detection + encoding

```bash
python main.py --phase 9             # Full: Jaccard matrix, Leiden (gen 0-2), encoding
python main.py --phase 9b            # Just re-encode bitmasks (if islands already exist)
```

Phase 9 now runs the full three-generation hierarchy and encodes the result into bitmask columns. Phase 9b is a lightweight standalone step that re-runs only the encoding -- useful if you've modified island assignments and need to refresh the bitmasks without recomputing the Jaccard matrix or re-running Leiden.

### Universal word analytics (post-island)

```bash
python main.py --phase 9d            # Arch concentration + domain general views
```

Phase 9d computes `arch_concentration` for all universal words (requires island data from phase 9). This measures how concentrated each universal word's dimensions are within a single archipelago -- a value of 0.75+ means the word is a "domain general" (universal but topically focused). Also re-creates views to include `v_domain_generals`.

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
- **Index rebuild** -- Drops and recreates all 14 indexes for consistent state
- **ANALYZE** -- Recomputes query planner statistics after bulk inserts/updates
- **FORCE CHECKPOINT** -- Flushes WAL to disk and reclaims space from deleted row groups

### CLI options

```
--phase PHASE          Run only this phase (2, 3, 4, 4b, 5, 5b, 5c, 6, 6b, 9, 9b, 9c, 9d, 7, 8)
--from PHASE           Run from this phase through the end
--db PATH              Database path (default: vector_distillery.duckdb)
--skip-pair-overlap    Skip the expensive pair overlap materialization
--no-resume            Don't resume from intermediate .npy checkpoint files
```

### Interactive Explorer

Phase 8 launches a REPL with a read-only database connection. Commands:

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

-- Archipelago encoding: bit position uniqueness (expect 0 rows)
SELECT arch_column, arch_bit, COUNT(*) FROM island_stats
WHERE arch_column IS NOT NULL GROUP BY 1, 2 HAVING COUNT(*) > 1;

-- Encoding coverage
SELECT COUNT(*) FILTER (WHERE archipelago != 0 OR archipelago_ext != 0) as encoded,
       COUNT(*) as total FROM words;

-- Specificity distribution
SELECT specificity, COUNT(*), MIN(total_dims), MAX(total_dims)
FROM words GROUP BY specificity ORDER BY specificity DESC;

-- Archipelago profile: words in the most structures
SELECT word, n_archipelagos, n_islands, n_reefs
FROM v_archipelago_profile ORDER BY n_reefs DESC LIMIT 20;

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

-- Relationship check via bitwise AND
-- gen0_mask and shift depend on actual gen0_count from island_stats
-- example below assumes 6 gen-0 archipelagos (gen0_count=6, mask=63)
SELECT
    CASE
        WHEN (a.reef_0 & b.reef_0) | (a.reef_1 & b.reef_1) |
             (a.reef_2 & b.reef_2) | (a.reef_3 & b.reef_3) |
             (a.reef_4 & b.reef_4) | (a.reef_5 & b.reef_5) != 0
            THEN 'same reef'
        WHEN ((a.archipelago & b.archipelago) >> 6) |
             (a.archipelago_ext & b.archipelago_ext) != 0
            THEN 'reef neighbors'
        WHEN (a.archipelago & b.archipelago & 63) != 0
            THEN 'island neighbors'
        ELSE 'different archipelagos'
    END as relationship
FROM words a, words b
WHERE a.word = 'cat' AND b.word = 'dog';
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
| `REEF_MIN_DEPTH` | `2` | Min dims a word must activate in a reef/island/archipelago to be encoded |
| `SENSE_SPREAD_INFLATED_THRESHOLD` | `15` | Min sense_spread to flag a universal word as polysemy-inflated |
| `DOMAIN_GENERAL_THRESHOLD` | `0.75` | Min arch_concentration for `v_domain_generals` view |
| `ABSTRACT_DIM_THRESHOLD` | `0.30` | Min universal_pct for `v_abstract_dims` view |
| `CONCRETE_DIM_THRESHOLD` | `0.15` | Max universal_pct for `v_concrete_dims` view |

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
