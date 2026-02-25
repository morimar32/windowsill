# Windowsill

what you can see shifts when you switch from a 🔭 to a 🪟

A vector space distillation engine that maps 158K English words to 258 semantic domains using transformer embeddings, XGBoost classifiers, domain-name cosine blending, and graph clustering. Produces compact binary files consumed by [Lagoon](https://github.com/morimar32/lagoon) for real-time domain scoring of free text.

## What This Does

Windowsill takes the WordNet vocabulary plus Claude-generated domain words, embeds them with [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (768-dim), and builds a multi-layer classification system:

1. **WordNet domains** provide ground-truth word-domain associations (~10K pairs)
2. **Claude augmentation** generates 100-200 discriminative words per domain, classified as core/peripheral
3. **XGBoost classifiers** (one per domain, 777 features) extend coverage to the full vocabulary
4. **Domain-name cosine similarity** computes `cos(word_embedding, domain_name_embedding)` for every word-domain pair -- a "default implied association" signal blended into scoring
5. **Domain consolidation** merges 444 raw WordNet domains down to 258 via a curated mapping (e.g., "baseball", "softball", "cricket" → "ball game")
6. **Leiden clustering** subdivides each domain into semantic sub-reefs and groups domains into archipelagos
7. **Blended scoring** produces a final domain_score per word-domain pair using `effective_sim = 0.7 * centroid_sim + 0.3 * domain_name_cos`
8. **Per-reef min-max normalization** maps each reef's [score_min, score_max] → [0, 255] (u8), preserving within-reef dynamic range
9. **MessagePack export** serializes everything into 10 binary files (format v7.0) for Lagoon

The result: given any English word, Lagoon can instantly look up which domains it belongs to and with what strength.

## Pipeline

Three entry points:

```
extract.py              Build v2.db: load words, embeddings, variants, WordNet domains, Claude domains
apply_consolidation.py  Merge 444 raw domains → 258 via domain_consolidation_map.json
transform.py            Full 9-step pipeline: XGBoost, IDF, domain-name cos, ubiquity, domainless,
                        reef clustering, archipelago clustering, scoring, consolidation application
load.py                 Export to Lagoon v7.0 format (10 msgpack .bin files + manifest)
```

`transform.py` orchestrates everything after extraction. The 9 steps run sequentially:

| Step | Name | What it does |
|------|------|--------------|
| 1 | XGBoost | Train/load classifiers, inference, insert into augmented_domains |
| 2 | IDF | Adjust low-confidence xgb scores, prune below 0.4 |
| 3 | Domain Name Cos | Embed 258 domain names, compute cos(word, domain_name), store in augmented_domains |
| 4 | Ubiquity | Prune/penalize words appearing in 20+ domains |
| 5 | Domainless | Tag words with no domain |
| 6 | Reef | Leiden subdivision per domain -> centroid_sim |
| 7 | Archipelago | Group domains into higher-level topic families |
| 8 | Scoring | Blend centroid_sim + domain_name_cos -> effective_sim -> domain_score |
| 9 | Consolidation | Apply domain_consolidation_map.json (444 → 258 domains) |

### Quick Start

```bash
python3 -m venv ws && source ws/bin/activate
pip3 install -r requirements.txt

python extract.py         # ~40s, builds v2.db (~1 GB)
python apply_consolidation.py  # ~1s, merges 444 → 258 domains
python transform.py       # ~2 hrs (GPU), trains 258 XGBoost models + full pipeline
python load.py --output v2_export --verify   # ~7s, export + verify
```

The export produces `v2_export/` with 10 `.bin` files (~12 MB total) and a `manifest.json`.

### Verification

`transform.py` runs 17 built-in test queries at the end of step 8 (e.g., "python snake reptile" should rank fauna/zoology domains in top 10). `load.py --verify` checks deserialization and checksums.

## Current Stats

| Metric | Value |
|--------|-------|
| Vocabulary | 158,060 words |
| Embedding dimensions | 768 (nomic-embed-text-v1.5) |
| Domains | 258 (consolidated from 444 WordNet categories) |
| Archipelagos | 8 |
| Sub-reefs | 2,534 |
| Words with domain assignments | 89,715 |
| Word variants (morphy + base) | 509,579 |
| Lookup entries (incl. snowball stems) | 186,184 |
| Compound words | 69,793 |
| Database size | ~1 GB (SQLite) |
| Export size | ~12 MB (msgpack) |
| Export format | v7.0 |
| Weight encoding | u8 per-reef min-max normalized [0, 255] |

## File Layout

```
# Entry points (run in order)
extract.py              Database bootstrap (14 steps)
apply_consolidation.py  Merge 444 raw WordNet domains → 258 via domain_consolidation_map.json
transform.py            Full 9-step pipeline (XGBoost -> IDF -> domain_name_cos ->
                        ubiquity -> domainless -> reef -> archipelago -> scoring -> consolidation)
load.py                 Export to Lagoon v7.0 msgpack format

# Standalone scripts (also callable independently)
reef.py                 Domain subdivision via Leiden clustering
archipelago.py          Domain-level archipelago clustering
score.py                Pre-compute domain_word_scores (with --stats flag for verification)

# Supporting modules
config.py               All tunable constants
embedder.py             Embedding model loading + batch encoding (sentence-transformers)
export.py               Shared export utilities (used by load.py)
word_list.py            FNV-1a hashing, WordNet extraction
xgb.py                  Single-domain XGBoost training inspector
post_process_xgb.py     IDF score adjustment for XGBoost predictions

# Library
lib/db.py               SQLite schema, connection, migrations, embedding pack/unpack
lib/vocab.py            Word loading, embedding, dim_stats, variants
lib/domains.py          Domain table loading (WordNet + Claude)
lib/claude.py           Claude API: domain discovery + word generation
lib/xgboost.py          XGBoost feature engineering + training
lib/scoring.py          Score formulas (IDF, source quality, effective_sim, domain_score)
lib/score_pipeline.py   Scoring orchestration: load, dedup, compute, persist, verify
lib/reef.py             Hybrid similarity, kNN graph, Leiden clustering
lib/reef_pipeline.py    Reef subdivision orchestration
lib/archipelago.py      Domain embedding aggregation + clustering
lib/arch_pipeline.py    Archipelago clustering orchestration

# Data artifacts
v2.db                           SQLite database (~1 GB)
domain_consolidation_map.json   Curated 444→258 domain merge mapping
models/                         258 XGBoost model JSON files (one per domain)
augmented_domains/              Claude-generated domain vocabularies
intermediates/                  Embedding checkpoint files
v2_export/                      Exported binary files for Lagoon (v7.0)
```

## Key Technical Choices

- **nomic-embed-text-v1.5**: 768-dim Matryoshka embeddings, good balance of quality and size
- **XGBoost** over neural classifiers: fast training, interpretable feature importance, handles class imbalance well with scale_pos_weight
- **Domain-name cosine blending** (alpha=0.3): captures "if someone says this word, which domain comes to mind?" -- nearly orthogonal to centroid_sim (Pearson r=0.073), dramatically improves intuitive rankings (e.g., violin -> music instead of italian)
- **Leiden algorithm** over Louvain: better community quality, resolution parameter for tuning granularity
- **Hybrid similarity** (70% embedding cosine + 30% PMI): captures both semantic similarity and co-membership patterns
- **SQLite** over DuckDB: simpler deployment, adequate for the data volume
- **MessagePack** for export: compact, fast deserialization, language-agnostic
- **FNV-1a u64** for word hashing: deterministic, fast, good distribution for hash tables

## Dependencies

Core: `numpy`, `sqlite3` (stdlib), `msgpack`, `tqdm`, `nltk`
ML: `xgboost`, `scikit-learn`, `sentence-transformers`
Clustering: `leidenalg`, `python-igraph`
NLP: `nltk` (WordNet), `snowballstemmer`
API: `anthropic` (Claude, for domain augmentation only)

## Documentation

- **[DETAILS.md](DETAILS.md)** -- Technical reference: database schema, scoring formulas, feature engineering, clustering algorithms, export format. Optimized for getting an LLM up to speed on the codebase.
