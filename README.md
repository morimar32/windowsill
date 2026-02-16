# 🪟 Windowsill

_what you can see shifts when you switch from a 🔭 to a 🪟_

A vector space distillation engine that decomposes word embeddings into interpretable dimensions. Takes the 768 dimensions of a transformer embedding model and answers the question: *what does each dimension actually encode?*

## What This Project Does

Windowsill embeds the entire WordNet vocabulary (~146K lemmas) using [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), then statistically analyzes each of the 768 embedding dimensions to determine which words "activate" in that dimension. The result is a database where every word has a discrete set of dimension memberships, and every dimension has a characterized set of member words -- turning opaque floating-point vectors into an inspectable symbolic structure.

The core insight: embedding dimensions are not random. Each dimension has a small set of high-activation words (typically ~1-2% of the vocabulary) that often correspond to a coherent semantic concept. Windowsill identifies these clusters via z-score thresholding (mean + 2.0σ), determines per-dimension membership boundaries, and records the results.

### Beyond basic embeddings

The enrichment pipeline adds seven layers of analysis on top of the base embedding decomposition. For full design rationale, see [DETAILS.md](DETAILS.md#key-design-decisions).

1. **Sense disambiguation** -- Words with multiple parts-of-speech (~8%) get separate sense-specific embeddings using WordNet glosses, revealing that e.g. "bank (financial)" and "bank (river edge)" activate completely different dimensions.

2. **Domain-anchored sense enrichment** -- Words with WordNet topic/usage domains get synthetic compound embeddings (e.g., `"classification: chess rook"`) that produce sharper activations in the relevant domain.

3. **Compound decomposition** -- Multi-word expressions (43% of vocabulary) are analyzed for compositionality via vector arithmetic, distinguishing compositional compounds from idiomatic ones.

4. **Contamination detection** -- Identifies when a word's dimension membership is driven by compound contamination rather than intrinsic meaning.

5. **Island detection** -- Dimensions with significant word overlap form communities via Leiden clustering, producing a 3-generation hierarchy: archipelagos, islands, and reefs. Noise dimensions are recovered to the nearest sibling reef.

6. **Denormalized hierarchy access** -- Reef/island/archipelago IDs are denormalized onto memberships for direct query access. The `word_reef_affinity` table stores continuous affinity scores for every word-reef pair.

7. **Island & reef naming** -- Every entity in the hierarchy gets a human-readable name via bottom-up LLM-assisted naming, grounded in each reef's most exclusive words.

8. **Universal word analytics** -- Universal words (~24K, appearing in 23-44 dims) are analyzed for abstractness, sense spread, archipelago concentration, exclusion fingerprints, and bridge profiles.

9. **Evaluative polarity (valence)** -- A negation vector derived from ~1,600 antonym pairs provides an evaluative polarity axis for dimensions, reefs, islands, and archipelagos.

### The Big Picture - Archipelagos, Islands, and Reefs

![Full island hierarchy: named islands decomposed into reefs across archipelagos](great_chart.png)

The complete 3-generation decomposition of nomic-embed-text-v1.5's 768 dimensions into interpretable semantic structure. Each row is a named gen-1 **island**, segmented into its gen-2 **reefs** (numbers show reef size in dimensions). Rows are grouped by their gen-0 **archipelago**: *natural sciences and taxonomy* (blue), *physical world and materiality* (orange), *abstract processes and systems* (green), and *social order and assessment* (red). Gray segments are noise dimensions that didn't form a reef community. The chart reads as a table of contents for the embedding space -- every dimension in the model has an address in this hierarchy, and every structure has a human-readable name derived from its most characteristic words. 100% of words in the vocabulary are associated with at least one reef. (Exact hierarchy counts are dynamic and depend on subdivision threshold and noise recovery -- see Distillation Results.)

## Quick Start

initialize a venv and install the requirements
```bash
python3 -m venv ws
source ws/bin/activate
pip3 install -r requirements.txt
```

start the extraction
```bash
python3 main.py --from 1
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

Output from a full pipeline run (z-score threshold = 2.0, REEF_MIN_DEPTH = 2). For full breakdown by archipelago/island/reef, see [ANALYSIS.md](ANALYSIS.md).

| Metric | Value | Notes |
|--------|-------|-------|
| Total memberships | ~2.56M | 3x increase from z=2.45 (~800K) |
| Avg members/dim | ~3,339 | Per dimension |
| Avg dims/word | ~17.5 | Up from ~6 at z=2.45 |
| Word-reef affinity rows | ~2.44M | Every word-reef pair where word activates in >= 1 reef dim |
| Reef coverage (any depth) | 100.0% | 146,697 / 146,698 words touch at least one reef |
| Reef coverage (depth >= 2) | 65.2% | 95,637 words with strong multi-dim reef membership |
| Gen-0 archipelagos | dynamic | Leiden community detection (counted at runtime) |
| Gen-1 islands | dynamic | Sub-island subdivision (counted at runtime) |
| Gen-2 reefs | dynamic | Sub-island subdivision + noise recovery (counted at runtime) |
| All structures named | Yes | Bottom-up LLM naming (phase 10) |
| Word variant mappings | ~490K | base (~146K) + morphy (~344K) in word_variants table |
| Universal words | 24,651 | specificity < 0 (23-44 dims) |
| Abstract dims | 128 | universal_pct >= 30% |
| Concrete dims | 46 | universal_pct <= 15% |
| Domain generals | 111 | Universal words with arch_concentration >= 0.75 |
| Polysemy-inflated | 363 | Universal + sense_spread >= 15 |
| Negation vector pairs | 1,639 | Directed morphological negation pairs (norm 6.13) |
| Positive-pole dims | 184 | Valence <= -0.15 (negation decreases activation) |
| Negative-pole dims | 182 | Valence >= 0.15 (negation increases activation) |
| Reef valence range | [-0.44, 0.64] | Mean dim valence per reef |

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
Phase 1:  Vocabulary                   (extract + clean WordNet lemmas)
Phase 2:  Embeddings                   (nomic-embed-text-v1.5, CPU, ~30 min)
Phase 3:  Database                     (schema, bulk load, POS backfill, word hashes)
Phase 4:  Analysis                     (z-score thresholds, dim counts, specificity, pair overlap)
Phase 5:  Senses                       (sense embeddings, sense analysis, domain compound enrichment)
Phase 6:  Enrichment                   (POS enrichment, contamination, compositionality, negation vector, valence)
Phase 7:  Islands                      (Jaccard matrix, Leiden clustering, 3-gen hierarchy + noise recovery)
Phase 8:  Refinement                   (iterative dim loyalty analysis + reassignment)
Phase 9:  Reef Analytics               (backfill, affinity, IDF, arch concentration, valence, POS composition,
                                        specificity, reef edges, composite weight, views)
Phase 10: Naming                       (LLM-assisted bottom-up naming via Claude API)
Phase 11: Finalization                 (morphy variant expansion, integrity checks, ANALYZE, CHECKPOINT)
explore:  Interactive explorer         (standalone REPL for querying the database)
```

Phase order: `1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11`

The `explore` command is standalone and not part of the pipeline sequence. Each phase builds on the previous ones in a clean linear dependency chain. Hierarchy counts (reefs, islands, archipelagos) are computed dynamically from the database rather than hardcoded.

### File Layout

```
config.py        Constants: model name, thresholds, batch sizes, paths, FNV-1a hashing, BM25 params
database.py      DuckDB schema, migrations, insert/load functions
word_list.py     WordNet extraction, cleaning, POS/category classification, FNV-1a hashing, morphy expansion
embedder.py      Sentence-transformers encoding, checkpointing, sense embedding,
                 domain compound embedding (WordNet topic/usage domains)
analyzer.py      Per-dimension z-score thresholding, sense analysis
post_process.py  Dim counts, specificity bands, views, pair overlap, POS enrichment,
                 contamination, compositionality, dimension abstractness/specificity,
                 sense spread, arch concentration, reef IDF, negation vector,
                 dimension valence, reef valence, hierarchy specificity, reef edges
islands.py       Island detection: Jaccard matrix, Leiden clustering, PMI scoring,
                 noise dim recovery, denormalization, word-reef affinity
reef_refine.py   Reef refinement: iterative dim loyalty analysis and reassignment
main.py          Pipeline orchestration, CLI argument parsing
explore.py       Interactive query functions (what_is, words_like, archipelago,
                 relationship, exclusion, bridge_profile, senses, synonyms,
                 antonyms, etc.)
```

## Database Schema

> For full column-level schema, see [DETAILS.md](DETAILS.md#database-schema). For the data dictionary with types, ranges, and examples, see [data_dictionary.md](data_dictionary.md).

| Table | Description |
|-------|-------------|
| `words` | One row per vocabulary entry (~146K), with embedding, POS, category, specificity, and hash |
| `dim_stats` | Per-dimension statistics (768 rows): thresholds, selectivity, POS enrichment, valence |
| `dim_memberships` | Which words belong to which dimensions (~2.56M rows), with z-scores and hierarchy IDs |
| `word_pair_overlap` | Precomputed pairwise word similarity by shared dimensions |
| `word_pos` | All POS tags per word (including ambiguous) |
| `word_components` | Decomposition of multi-word expressions into components |
| `word_senses` | Sense-specific embeddings for ambiguous words (~61K senses) |
| `sense_dim_memberships` | Dimension memberships per sense |
| `compositionality` | Compositionality analysis for compounds (Jaccard, emergent dims) |
| `dim_jaccard` | Pairwise Jaccard similarity between all 768 dimensions |
| `dim_islands` | Dimension-to-island assignments across 3 generations |
| `island_stats` | Per-island summary: size, Jaccard, valence, POS composition, name |
| `island_characteristic_words` | PMI-ranked diagnostic words per island |
| `word_reef_affinity` | Continuous affinity scores for every word-reef pair (~2.44M rows) |
| `reef_edges` | Directed relationship scores between all reef pairs |
| `word_variants` | Morphy expansion mapping inflected forms to base words (~490K) |
| `computed_vectors` | Stored analytical vectors (negation vector) |

See [DETAILS.md](DETAILS.md#key-design-decisions) for design rationale behind key choices (z-score threshold, noise recovery, domain-anchored enrichment, etc.).

## Usage

### Full pipeline (fresh start)

```bash
python main.py
```

This runs phases 1 through 11. Phase 2 (embedding) takes ~30 min on CPU. Phase 4 pair overlap takes 10-30 min. Use `--skip-pair-overlap` to skip the expensive pair materialization.

### Run specific phases

```bash
python main.py --phase 3           # Just database schema + bulk load
python main.py --from 5            # Run phases 5, 6, 7, ..., 11
python main.py --phase explore     # Jump straight to explorer
python main.py --phase 8           # Run reef refinement only
```

### Key phases

**Phase 5: Senses** -- Generates sense-specific embeddings for ambiguous words, runs sense analysis, and generates domain compound embeddings from WordNet topic/usage domains. Domain compounds (e.g., `"classification: chess rook"`) create sharper activations for specialized vocabulary.

**Phase 7: Islands** -- Computes the Jaccard similarity matrix, runs Leiden community detection for the 3-generation hierarchy, and recovers noise dimensions by assigning orphan dims to the nearest sibling reef if their average Jaccard similarity exceeds `NOISE_RECOVERY_MIN_JACCARD`.

**Phase 8: Refinement** -- Examines every reef with 4+ dims and checks whether each dim has higher Jaccard affinity to its own reef or to a sibling reef (same parent island). Dims with a loyalty ratio below 1.0 are reassigned to their best-fit sibling. Iterates until convergence (typically 2-3 rounds).

**Phase 9: Reef Analytics** -- Consolidates all post-island analytics into a single pass: backfill denormalized columns, compute word-reef affinity, reef IDF, arch concentration, reef valence, POS composition, hierarchy specificity, reef edges, composite weight, and recreate views. This runs once after refinement completes, avoiding redundant double-computation.

**Phase 10: Naming** -- Generates human-readable names for all gen-1 islands and gen-2 reefs using a bottom-up approach. Reefs are named from their most exclusive words, islands from their child reef names, archipelagos from their child island names. Requires `ANTHROPIC_API_KEY` environment variable.

**Phase 11: Finalization** -- Expands WordNet `morphy()` variants (~490K entries), runs integrity checks, rebuilds indexes, and performs ANALYZE + CHECKPOINT for database optimization.

### CLI options

```
--phase PHASE          Run only this phase (1-11, or explore)
--from PHASE           Run from this phase through the end
--db PATH              Database path (default: vector_distillery.duckdb)
--skip-pair-overlap    Skip the expensive pair overlap materialization
--no-resume            Don't resume from intermediate .npy checkpoint files
```

### Interactive Explorer

The `explore` command launches a standalone REPL with a read-only database connection (it is not part of the pipeline sequence). Commands:

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
- **anthropic** -- Claude API client for LLM-assisted island/reef naming (phase 10 only)

## Documentation

- **[ANALYSIS.md](ANALYSIS.md)** — Full database metrics: hierarchy details, valence, specificity, POS composition, reef quality, generated by `python analyze.py`
- **[DETAILS.md](DETAILS.md)** — Technical reference: database schema, design decisions, configuration, verification queries, embedding mechanics
- **[data_dictionary.md](data_dictionary.md)** — Column-level data dictionary with types, ranges, and examples
