# Ecosystem: Windowsill / Lagoon / Shoal

This document defines the architecture, responsibilities, and data contracts across the three-layer stack. Read this before working on any cross-cutting concern.

## The Stack

```
                         +-----------+
                         |   Shoal   |    Application layer
                         |  (RAG)    |    Specific use case: document retrieval
                         +-----+-----+
                               |
                               | uses lagoon as a library
                               |
                         +-----+-----+
                         |  Lagoon   |    Scoring library
                         | (scorer)  |    General-purpose text scoring engine
                         +-----+-----+
                               |
                               | consumes exported binary data
                               |
                         +-----+-----+
                         | Windowsill|    Data pipeline
                         | (weights) |    Builds and exports the weight database
                         +-----------+
```

## Separation of Concerns

### Windowsill (data pipeline)

**Responsibility:** Calculate THE weight for every (word, reef) association.

A single weight may incorporate 10+ signals during its calculation (source quality, IDF, centroid similarity, domain-name cosine, XGBoost confidence, ubiquity penalty, etc.), but all of that intelligence is distilled into one final number per (word, reef) pair. That number is the single point of truth.

Windowsill owns:
- Vocabulary construction (WordNet + Claude + XGBoost expansion)
- All weight computation logic (scoring formulas, blending, thresholds)
- Domain taxonomy (which domains exist, how they cluster into reefs/islands/archs)
- Data quality (stop word filtering, ubiquity pruning, DNC floor, consolidation)
- Binary export to the format lagoon consumes

Windowsill does NOT:
- Score arbitrary text (that's lagoon)
- Know what text will be scored at runtime
- Make contextual decisions that depend on the interaction between words in a specific input

### Lagoon (scoring library)

**Responsibility:** Given pre-computed weights, score arbitrary text contextually.

Lagoon is a general-purpose library. It could be used by shoal, a CLI tool, a web service, or anything else. Its intelligence is about runtime context: understanding what the words in a specific piece of text mean *together*, not what any single word-reef weight should be.

Lagoon owns:
- Tokenization (compound detection, stemming, stop word handling)
- Contextual scoring (how words in a specific text interact to produce reef rankings)
- Background normalization (z-scores, alpha scaling)
- Vocabulary extension API (custom words injected at runtime)
- Document analysis (topic segmentation, boundary detection)

Lagoon does NOT:
- Compute or second-guess the weights (they come from windowsill, use them as-is)
- Duplicate windowsill's logic (no re-deriving IDF, specificity, source quality, etc.)
- Store documents or make routing decisions (that's a downstream consumer)
- Know how the weights were calculated (it's a black box -- the weight IS the answer)

### Shoal (application)

**Responsibility:** A specific retrieval engine (the "R" in RAG) built on lagoon.

Shoal owns:
- Document ingestion, parsing, chunking
- Retrieval strategy (reef overlap scoring, section hierarchy)
- Corpus management (SQLite storage, tags, deduplication)
- Query interface (CLI, HTTP daemon, Python API)

Shoal does NOT:
- Modify lagoon's scoring behavior
- Compute weights or manage the taxonomy

## The Data Contract: Windowsill -> Lagoon

The interface between windowsill and lagoon is a set of binary files (msgpack, format v6.0). The critical contract:

**Each (word, reef) association is expressed as a single `weight_q` (u16, quantized).** This is the ONLY value lagoon uses to determine how important a word is to a reef. All of windowsill's intelligence -- IDF, source quality, centroid similarity, domain-name cosine, ubiquity adjustments -- is baked into this one number.

Lagoon also receives structural metadata:
- **Reef hierarchy** (reef -> island -> arch grouping) for coherence-based scoring
- **Background model** (per-reef mean/std) for z-score normalization
- **Compound phrases** for multi-word detection
- **Domainless word set** for coverage tracking

This metadata describes the *shape* of the data, not the weights themselves. Lagoon may use it to inform contextual scoring strategy (e.g., using the hierarchy to detect coherent reef activations), but it never re-derives or adjusts the underlying weights.

**Why a single weight?** This separation means:
- Windowsill can evolve its weight formula (add signals, change blending) without lagoon changing
- Lagoon can evolve its contextual scoring (add passes, change normalization) without needing new data from windowsill
- The two systems are testable independently: windowsill validates weight quality via test queries, lagoon validates scoring quality via sentence correlation

## What Lagoon Should and Should Not Do

**Lagoon's contextual intelligence** is about the interaction between words in a specific text. This cannot be pre-computed by windowsill because it depends on what text is being scored at runtime.

Examples of legitimate lagoon-level logic:
- "This reef was activated by only 1 word out of 15 matched -- that's likely noise" (corroboration)
- "Astronomy is already strongly activated; cloud's astronomy weight should be favored over cloud_computing" (progressive context)
- "These 3 activated reefs share an island -- that's a coherent signal" (hierarchical coherence)

Examples of logic that belongs in windowsill, NOT lagoon:
- "This word has low IDF, so dampen its contribution" (bake it into the weight)
- "This word is from an xgboost source with low confidence" (bake it into the weight)
- "This word appears in 20+ domains, so penalize it" (ubiquity pruning during weight calculation)

**The test:** If a scoring decision depends only on static properties of a word-reef pair (properties that are the same regardless of what other text is present), it belongs in windowsill. If it depends on what OTHER words are in the current text, it belongs in lagoon.
