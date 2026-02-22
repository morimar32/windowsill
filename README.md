# Windowsill

what you can see shifts when you switch from a 🔭 to a 🪟

A vector space distillation engine that decomposes word embeddings into a domain-aware vocabulary engine that maps 158K English words to 444 semantic domains using transformer embeddings, XGBoost classifiers, and graph clustering. Produces compact binary files consumed by [Lagoon](../lagoon/) for real-time domain scoring of free text.

## What This Does

Windowsill takes the WordNet vocabulary plus Claude-generated domain words, embeds them with [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (768-dim), and builds a multi-layer classification system:

1. **WordNet domains** provide ground-truth word-domain associations (~10K pairs)
2. **Claude augmentation** generates 100-200 discriminative words per domain, classified as core/peripheral
3. **XGBoost classifiers** (one per domain, 777 features) extend coverage to the full vocabulary
4. **Leiden clustering** subdivides each domain into semantic sub-reefs and groups domains into archipelagos
5. **IDF-weighted scoring** produces a final domain_score per word-domain pair
6. **MessagePack export** serializes everything into 10 binary files (format v6.0) for Lagoon

The result: given any English word, Lagoon can instantly look up which domains it belongs to and with what strength.

## Pipeline

Six entry points, run in order:

```
extract.py       Build v2.db: load words, embeddings, variants, WordNet domains, Claude domains
transform.py     Train XGBoost classifiers, run inference, post-process with IDF adjustment
reef.py          Subdivide domains into sub-reefs via hybrid similarity + Leiden clustering
archipelago.py   Group domains into ~11 archipelagos (higher-level topic families)
score.py         Pre-compute domain_score and reef_score for all word-domain pairs
load.py          Export to Lagoon v6.0 format (10 msgpack .bin files + manifest)
```

### Quick Start

```bash
python3 -m venv ws && source ws/bin/activate
pip3 install -r requirements.txt

python extract.py         # ~5 min, builds v2.db (~1 GB)
python transform.py       # ~2 hrs (GPU), trains 444 XGBoost models
python reef.py            # ~2 min, Leiden clustering per domain
python archipelago.py     # ~30s, domain-level clustering
python score.py           # ~1 min, materialize scores
python load.py --output v2_export --verify   # ~30s, export + verify
```

The export produces `v2_export/` with 10 `.bin` files (~14 MB total) and a `manifest.json`.

### Verification

`score.py --stats` and `load.py --verify` both run 17 built-in test queries (e.g., "python snake reptile" should rank fauna/zoology domains in top 10). Current status: 17/17 pass.

## Current Stats

| Metric | Value |
|--------|-------|
| Vocabulary | 158,060 words |
| Embedding dimensions | 768 (nomic-embed-text-v1.5) |
| Domains | 444 |
| Archipelagos | 11 |
| Sub-reefs | 5,150 |
| Scored word-domain pairs | 807,988 |
| Word variants (morphy + base) | 509,579 |
| Lookup entries (incl. snowball stems) | 186,184 |
| Words with domain assignments | 126,163 |
| Compound words | 69,793 |
| Database size | ~1 GB (SQLite) |
| Export size | ~14 MB (msgpack) |
| Export format | v6.0 |

## File Layout

```
# Entry points (run in order)
extract.py              Database bootstrap (13 steps)
transform.py            XGBoost training + inference + IDF post-processing
reef.py                 Domain subdivision via Leiden clustering
archipelago.py          Domain-level archipelago clustering
score.py                Pre-compute domain_word_scores
load.py                 Export to Lagoon v6.0 msgpack format

# Supporting modules
config.py               All tunable constants
export.py               Shared export utilities (used by load.py)
word_list.py            FNV-1a hashing, WordNet extraction
xgb.py                  Single-domain XGBoost training inspector
post_process_xgb.py     IDF score adjustment for XGBoost predictions

# Library
lib/db.py               SQLite schema, connection, embedding pack/unpack
lib/vocab.py            Word loading, embedding, dim_stats, variants
lib/domains.py          Domain table loading (WordNet + Claude)
lib/claude.py           Claude API: domain discovery + word generation
lib/xgboost.py          XGBoost feature engineering + training
lib/scoring.py          Score formulas (IDF, source quality, domain_score)
lib/reef.py             Hybrid similarity, kNN graph, Leiden clustering
lib/archipelago.py      Domain embedding aggregation + clustering

# Data artifacts
v2.db                   SQLite database (~1 GB)
models/                 777 XGBoost model JSON files (one per domain)
augmented_domains/      Claude-generated domain vocabularies
classifiers/            XGBoost training artifacts
v2_export/              Exported binary files for Lagoon
```

## Key Technical Choices

- **nomic-embed-text-v1.5**: 768-dim Matryoshka embeddings, good balance of quality and size
- **XGBoost** over neural classifiers: fast training, interpretable feature importance, handles class imbalance well with scale_pos_weight
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
