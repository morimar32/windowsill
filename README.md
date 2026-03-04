# Windowsill

what you can see shifts when you switch from a 🔭 to a 🪟

A vector space distillation engine that maps ~150K English words to a four-tier semantic hierarchy using transformer embeddings, XGBoost classifiers, and Leiden graph clustering. Produces compact binary files consumed by [Lagoon](https://github.com/morimar32/lagoon) for real-time domain scoring of free text.

## What This Does

Windowsill builds a curated four-tier topical hierarchy:

```
Archipelagos (6)  — broad knowledge families (Applied Science, Pure Science, ...)
  └── Islands (44)  — major disciplines (Medicine, Physics, Sport, ...)
        └── Towns (332)  — specific subfields (Ophthalmology, Quantum Mechanics, Ball Games, ...)
              └── Reefs (3,919)  — Leiden-discovered clusters within towns
```

Three levels are top-down, named, and curated. One level (reefs) is statistically discovered via Leiden community detection operating within small, coherent town groups.

The hierarchy is grounded in the [FBK WordNet Domain Hierarchy](https://wndomains.fbk.eu/) (WDH) for the top two levels, extended for modern topics (AI/ML, cryptocurrency, cybersecurity, etc.), and populated with town-level granularity from curated Wikipedia category pulls and Claude-generated seed vocabularies.

For each word, Windowsill determines which reefs it belongs to and with what strength, then classifies each word-reef association into one of three export levels (reef, town, or island) based on specificity and spread. The result is a set of MessagePack binary files that let Lagoon instantly score any English text against the hierarchy.

View the [Data Dictionary](DATADICTIONARY.md) for information about the schema.

## Pipeline

The full pipeline is orchestrated by `v3/load.sh` (20 steps):

```
v3/load.sh              Full rebuild from empty database to test battery
```

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `schema.sql` | Create fresh database with 4-tier schema |
| 2 | `populate_archipelagos_islands.sql` | Insert 6 archipelagos + 44 islands |
| 3 | `populate_towns.sql` | Insert 332 towns |
| 4 | `populate_bucket_islands.sql` | Insert 3 bucket islands + 24 non-topical towns |
| 5 | `load_wordnet_vocab.py` | Load ~147K WordNet lemmas into Words table |
| 6 | `reembed_words.py` | Embed all words with nomic-embed-text-v1.5 (GPU) |
| 7 | `compute_dimstats.py` | Per-dimension mean, std, threshold stats |
| 8 | `import_wdh.py` | Seed towns from WDH via SKI bridge (WN 2.0 -> 3.0) |
| 9 | `populate_claude_seeds.sql` | Load Claude-generated single-word seeds |
| 9b | `populate_compound_seeds.sql` | Load Claude-generated compound seeds |
| 9c | `link_seed_words.py` | Link seed words to vocabulary word_ids |
| 9d | `reembed_words.py --missing-only` | Embed newly inserted words |
| 10 | `sanity_check_seeds.py` | Deduplicate words across sibling towns |
| 11 | `detect_island_words.py` | Triage capital seeds + detect island-level words |
| 11b | `flag_stop_words.py --baseline` | Flag function words (the, and, ...) |
| 12 | `train_town_xgboost.py` | Train XGBoost classifiers per town (GPU, per island) |
| 13 | `post_process_xgb.py` | Cross-town IDF penalty on low-confidence predictions |
| 14 | `flag_stop_words.py --ubiquity` | Prune words appearing in 20+ towns |
| 15 | `cluster_reefs.py` | Leiden clustering within towns (per island) |
| 16 | `cluster_bucket_reefs.py` | Cluster non-topical bucket islands |
| 17 | `compute_word_stats.py` | Specificity, IDF, island concentration per word |
| 18 | `compute_hierarchy_stats.py` | Aggregate stats up the hierarchy |
| 19 | `populate_exports.py` | Classify + score + normalize into export tables |
| 20 | `test_battery.py` | 53-query validation suite |

### Quick Start

```bash
python3 -m venv ws && source ws/bin/activate
pip3 install -r requirements.txt

bash v3/load.sh           # ~2-3 hrs (GPU), full pipeline
python v3/export.py --verify   # export to v3/exports/ + verify
```

### Iteration Loop

For weight tuning, only the last two steps need re-running:

```bash
python v3/populate_exports.py   # ~15s, fully idempotent
python v3/test_battery.py       # ~1s, 53-query pass/fail
```

## Current Stats

| Metric | Value |
|--------|-------|
| Vocabulary | 149,691 words |
| Embedding dimensions | 768 (nomic-embed-text-v1.5) |
| Archipelagos | 6 |
| Islands | 44 (41 topical + 3 bucket) |
| Towns | 332 |
| Reefs | 3,919 |
| ReefWords | 449,925 |
| Exported word-reef pairs | 415,616 |
| Unique words with exports | ~128K |
| Compound words | ~67K |
| Database size | ~780 MB (SQLite) |
| Export size | ~20 MB (msgpack) |
| Export format | v3.1 |
| Weight encoding | u8 per-group min-max normalized [0, 255] |
| Test battery | 47/53 pass (89%) |

## Hierarchy Overview

The six archipelagos and their islands:

| Archipelago | Islands |
|-------------|---------|
| Applied Science | Medicine, Engineering, Computer Science, Architecture, Agriculture, Media & Communications, Military, Transportation, Manufacturing, Veterinary Medicine |
| Pure Science | Mathematics, Physics, Chemistry, Biology, Astronomy, Earth Science |
| Social Science | Economics, Finance, Commerce, Law, Politics, Psychology, Sociology, Education, Anthropology |
| Humanities | Art, Music, Literature, Linguistics, History, Philosophy, Performing Arts, Mythology |
| Free Time | Sport, Games, Gastronomy, Hobbies & Crafts |
| Doctrines | Religion, Occultism, Ideology |

Three additional **bucket islands** (Languages, Regional, Miscellaneous) handle non-topical vocabulary that should be identified but excluded from topic scoring.

## File Layout

```
v3/                             # All V3 code lives here
  # Pipeline orchestration
  load.sh                       Full 20-step rebuild script
  schema.sql                    4-tier hierarchy schema

  # Hierarchy population (SQL scripts)
  populate_archipelagos_islands.sql   6 archipelagos + 44 islands
  populate_towns.sql                  332 towns
  populate_bucket_islands.sql         3 bucket islands + 24 towns

  # Vocabulary & embeddings
  load_wordnet_vocab.py         Load ~147K WordNet lemmas
  reembed_words.py              Embed words with nomic-embed-text-v1.5
  compute_dimstats.py           Per-dimension statistics

  # Seed population
  import_wdh.py                 Import FBK WordNet Domains via SKI bridge
  populate_claude_seeds.sql     Claude-generated single-word seeds
  populate_compound_seeds.sql   Claude-generated compound seeds
  link_seed_words.py            Link seeds to vocabulary word_ids
  generate_seeds.py             Generate new seeds via Claude API
  generate_compound_seeds.py    Generate compound seeds via Claude API
  sanity_check_seeds.py         Deduplicate across sibling towns
  detect_island_words.py        Capital triage + island-level detection

  # Classification & clustering
  flag_stop_words.py            Baseline + ubiquity stop word flagging
  train_town_xgboost.py         Per-town binary XGBoost classifiers
  post_process_xgb.py           Cross-town IDF adjustment
  cluster_reefs.py              Leiden clustering within towns
  cluster_bucket_reefs.py       Cluster bucket island towns

  # Stats & export
  compute_word_stats.py         Specificity, IDF, island concentration
  compute_hierarchy_stats.py    Aggregate stats up the hierarchy
  populate_exports.py           Classify + score + normalize exports
  export.py                     Serialize to msgpack for Lagoon (v3.1)
  test_battery.py               53-query validation suite
  sweep.py                      Automated parameter sweep tool

  # Data & artifacts
  data/                         WDH + SKI bridge data files
  models/                       XGBoost model files (per island/town)
  intermediates/                Embedding checkpoints
  exports/                      Exported binary files for Lagoon
  windowsill.db                 SQLite database (~780 MB)

  # Documentation
  structures.md                 Design principles for the schema
  tuning.md                     Export weight tuning guide
  changes.md                    Session changelog
```

## Key Technical Choices

- **nomic-embed-text-v1.5**: 768-dim Matryoshka embeddings with `"clustering: "` prefix (better suited than `"classification: "` for topical word clustering)
- **Top-down hierarchy from WDH**: Three curated levels grounded in the FBK WordNet Domain Hierarchy, extended for modern topics. Only the reef level uses statistical discovery.
- **XGBoost at town level**: Binary classifiers per town with 50/50 hard/easy negative sampling (sibling towns provide hard negatives), producing focused classifiers that distinguish Ball Games from Combat Sports
- **Three-level export promotion**: Words export at exactly one level (reef, town, or island) based on specificity and spread, avoiding double-counting
- **Three-signal effective_sim blend**: `centroid_sim` (50%) + `island_name_cos` (30%) + `group_name_cos` (20%) for reef/town exports
- **Hybrid island normalization**: 80% global + 20% per-island min-max prevents wide-range islands from compressing mid-range word weights
- **Bucket islands**: Non-topical vocabulary (languages, regional terms, linguistic register) is tracked but excluded from topical scoring
- **Leiden algorithm** over Louvain: Better community quality with resolution parameter for tuning
- **SQLite** with views for SQL-queryable validation: `ExportIndex` and `WordSearch` views enable direct search testing without export roundtrip

## Dependencies

Core: `numpy`, `sqlite3` (stdlib), `msgpack`, `tqdm`, `nltk`
ML: `xgboost`, `scikit-learn`, `sentence-transformers`
Clustering: `leidenalg`, `python-igraph`
NLP: `nltk` (WordNet), `snowballstemmer`
API: `anthropic` (Claude, for seed generation only)

## Documentation

- **[DETAILS.md](DETAILS.md)** -- Technical reference: database schema, scoring formulas, feature engineering, clustering algorithms, export format. Optimized for getting an LLM up to speed on the codebase.
- **[ECOSYSTEM.md](ECOSYSTEM.md)** -- Stack separation of concerns: Windowsill / Lagoon / Shoal.
- **v3/tuning.md** -- Export weight tuning guide: formula parameters, normalization strategies, diagnostic queries.
