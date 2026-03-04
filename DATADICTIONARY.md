# Windowsill Data Dictionary

Complete reference for every table, view, column, and index in the `v3/windowsill.db` SQLite database. All row counts, value ranges, and distributions are from the current production build.

**Database size:** 746 MB (191,080 pages x 4,096 bytes)
**Total exported word-reef pairs:** 415,616

---

## Table of Contents

1. [Hierarchy Tables](#1-hierarchy-tables) — Archipelagos, Islands, Towns, Reefs
2. [Dictionary Tables](#2-dictionary-tables) — Words, ReefWords
3. [Core Export Tables](#3-core-export-tables) — ReefWordExports, TownWordExports, IslandWordExports
4. [Optional Export Tables](#4-optional-export-tables) — WordsExport, NamesExport, EquivalencesExport, AcronymsExport
5. [Pipeline Support Tables](#5-pipeline-support-tables) — DimStats, IslandWords, SeedWords, AugmentedTowns, WordIslandStats, WordVariants
6. [Views](#6-views) — ExportIndex, WordSearch, Compounds, HierarchyPath
7. [Indexes](#7-indexes)
8. [Reference: Specificity Scale](#8-reference-specificity-scale)
9. [Reference: Export Promotion Rules](#9-reference-export-promotion-rules)
10. [Reference: Weight Formulas](#10-reference-weight-formulas)

---

## 1. Hierarchy Tables

Four-tier semantic hierarchy: Archipelago > Island > Town > Reef. The top three levels are curated; reefs are statistically discovered via Leiden clustering.

### Archipelagos

Broadest grouping. Six knowledge families.

**Row count:** 6

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `archipelago_id` | INTEGER | PK | Auto-increment primary key |
| `name` | TEXT | NOT NULL, UNIQUE | Archipelago name: Applied Science, Pure Science, Social Science, Humanities, Free Time, Doctrines |
| `island_count` | INTEGER | DEFAULT 0 | Cached count of child islands. Range: 4–10 |
| `town_count` | INTEGER | DEFAULT 0 | Cached count of child towns. Range: 30–87 |
| `reef_count` | INTEGER | DEFAULT 0 | Cached count of child reefs. Range: 265–1,141 |
| `word_count` | INTEGER | DEFAULT 0 | Cached count of unique words across all child reefs. Range: 21,324–69,597 |

**Current data:**

| ID | Name | Islands | Towns | Reefs | Words |
|----|------|---------|-------|-------|-------|
| 1 | Applied Science | 10 | 87 | 1,141 | 69,597 |
| 2 | Pure Science | 6 | 47 | 698 | 64,365 |
| 3 | Social Science | 10 | 55 | 726 | 48,043 |
| 4 | Humanities | 10 | 81 | 642 | 43,852 |
| 5 | Free Time | 4 | 32 | 447 | 33,120 |
| 6 | Doctrines | 4 | 30 | 265 | 21,324 |

**Populated by:** `populate_archipelagos_islands.sql` (step 2)
**Stats updated by:** `compute_hierarchy_stats.py` (step 18)

---

### Islands

Major disciplines within an archipelago. 41 topical + 3 bucket = 44 total.

**Row count:** 44

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `island_id` | INTEGER | PK | Auto-increment primary key. Range: 1–44 |
| `archipelago_id` | INTEGER | NOT NULL, FK | Parent archipelago |
| `name` | TEXT | NOT NULL | Island name. Unique within archipelago |
| `is_bucket` | INTEGER | NOT NULL, DEFAULT 0 | **0** = topical island (40 islands). **1** = non-topical bucket (4 islands: Linguistic Register, Languages, Regional, Miscellaneous). Bucket islands are tracked but excluded from topic scoring |
| `town_count` | INTEGER | DEFAULT 0 | Cached child town count. Range: 1–17 |
| `reef_count` | INTEGER | DEFAULT 0 | Cached child reef count. Range: 3–277 |
| `word_count` | INTEGER | DEFAULT 0 | Cached unique word count. Range: 112–34,892 |
| `noun_frac` | REAL | nullable | Fraction of words with POS = noun. Range: 0.348–0.888 |
| `verb_frac` | REAL | nullable | Fraction of words with POS = verb. Range: 0.000–0.157 |
| `adj_frac` | REAL | nullable | Fraction of words with POS = adjective. Range: 0.009–0.240 |
| `adv_frac` | REAL | nullable | Fraction of words with POS = adverb. Range: 0.000–0.074 |
| `avg_specificity` | REAL | nullable | Mean specificity of words in this island. Range: 0.41–2.77 |

**Bucket islands** (is_bucket = 1):

| ID | Name | Towns | Reefs | Words |
|----|------|-------|-------|-------|
| 33 | Linguistic Register | 11 | 11 | 244 |
| 42 | Languages | 13 | 13 | 638 |
| 43 | Regional | 3 | 3 | 112 |
| 44 | Miscellaneous | 7 | 7 | 256 |

**Populated by:** `populate_archipelagos_islands.sql` (step 2) + `populate_bucket_islands.sql` (step 4)
**Stats updated by:** `compute_hierarchy_stats.py` (step 18)

---

### Towns

Specific subfields within an island.

**Row count:** 332

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `town_id` | INTEGER | PK | Auto-increment primary key |
| `island_id` | INTEGER | NOT NULL, FK | Parent island |
| `name` | TEXT | NOT NULL | Town name. Unique within island. Examples: Ophthalmology, Quantum Mechanics, Ball Games |
| `is_capital` | INTEGER | DEFAULT 0 | **1** = catch-all capital town for island. Currently **0 capital towns** exist (triage fully processed) |
| `model_f1` | REAL | nullable | XGBoost validation F1 score. Range: 0.400–1.000, mean 0.896. NULL for 46 towns (bucket towns + towns without enough training data) |
| `reef_count` | INTEGER | DEFAULT 0 | Cached child reef count. Range: 1–42 |
| `word_count` | INTEGER | DEFAULT 0 | Cached unique word count |
| `noun_frac` | REAL | nullable | Fraction of words with POS = noun |
| `verb_frac` | REAL | nullable | Fraction of words with POS = verb |
| `adj_frac` | REAL | nullable | Fraction of words with POS = adjective |
| `adv_frac` | REAL | nullable | Fraction of words with POS = adverb |
| `avg_specificity` | REAL | nullable | Mean specificity of words in this town |

**Distribution:** 1–17 towns per island (mean 7.5). 1–42 reefs per town (mean 12.2).

**Populated by:** `populate_towns.sql` (step 3) + `populate_bucket_islands.sql` (step 4)
**Stats updated by:** `compute_hierarchy_stats.py` (step 18)
**model_f1 set by:** `train_town_xgboost.py` (step 12)

---

### Reefs

Leiden-discovered clusters within towns. The bottom level of the hierarchy.

**Row count:** 3,919

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `reef_id` | INTEGER | PK | Auto-increment primary key |
| `town_id` | INTEGER | NOT NULL, FK | Parent town |
| `name` | TEXT | nullable | Auto-generated from top core words. Format: `word1_word2_word3`. All 3,919 reefs have names |
| `centroid` | BLOB | nullable | L2-normalized float32 x 768 = 3,072 bytes. Mean embedding of core members. All 3,919 reefs have centroids |
| `word_count` | INTEGER | DEFAULT 0 | Total words assigned to this reef. Range: 7–1,284 (mean 114.8) |
| `core_word_count` | INTEGER | DEFAULT 0 | Leiden core members only. Range: 7–1,143 (mean 98.6) |
| `avg_specificity` | REAL | nullable | Mean specificity of reef members. Range: -0.79–2.98 (mean 0.83) |
| `noun_frac` | REAL | nullable | Fraction of words with POS = noun. Range: 0.000–1.000 (mean 0.740) |
| `verb_frac` | REAL | nullable | Fraction of words with POS = verb |
| `adj_frac` | REAL | nullable | Fraction of words with POS = adjective |
| `adv_frac` | REAL | nullable | Fraction of words with POS = adverb |

**Name examples:**
- `camelia_kahikatea_carissa` (Botany, 1,284 words)
- `balsam capivi_copaiba balsam_crepe marocain` (Organic Chemistry, 858 words)
- `embitterment_embitter_embrasure` (Social Psychology, 22 words)

**Populated by:** `cluster_reefs.py` (step 15) + `cluster_bucket_reefs.py` (step 16)
**Stats updated by:** `compute_hierarchy_stats.py` (step 18)

---

## 2. Dictionary Tables

### Words

Master vocabulary table. One row per unique word form.

**Row count:** 149,691

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `word_id` | INTEGER | PK | Auto-increment primary key |
| `word` | TEXT | NOT NULL | Canonical word form (lowercase). Includes single words, compounds, and phrasal verbs |
| `word_hash` | INTEGER | NOT NULL | FNV-1a u64 hash, stored as signed i64 (SQLite constraint). ~50% are negative when read as i64. Use `hash & 0xFFFFFFFFFFFFFFFF` to recover the unsigned value. Indexed for O(1) lookup by Lagoon |
| `pos` | TEXT | nullable | Dominant part of speech: **n** (noun), **v** (verb), **a** (adjective), **r** (adverb). NULL for 2,385 words (added via Claude compounds, no WordNet POS) |
| `specificity` | INTEGER | nullable | Global specificity based on reef count. Range: -2 to 3. See [Specificity Scale](#8-reference-specificity-scale). NULL for 22,725 words not in any reef |
| `cosine_sim` | REAL | nullable | Best cosine similarity to any assigned reef centroid. Range: 0.708–0.985 (mean 0.876). NULL for 22,725 unassigned words |
| `idf` | REAL | nullable | Global IDF: `log2(total_reefs / reef_count)`. Range: 0.000–11.936 (mean 8.876). 0.0 = appears in all reefs |
| `embedding` | BLOB | nullable | float32 x 768 = 3,072 bytes. nomic-embed-text-v1.5 embedding with `"clustering: "` prefix. All 149,691 words have embeddings |
| `word_count` | INTEGER | DEFAULT 1 | Number of space-separated tokens. 1 = single word, 2+ = compound |
| `category` | TEXT | nullable | Word type: **single** (83,118), **compound** (62,221), **phrasal_verb** (1,967). NULL for 2,385 |
| `is_stop` | INTEGER | DEFAULT 0 | **1** = function word (the, and, ...) or ubiquity-flagged. 224 stop words total |
| `reef_count` | INTEGER | DEFAULT 0 | Number of reefs this word appears in. Range: 0–36. 22,725 words have 0 (not in any reef) |
| `town_count` | INTEGER | DEFAULT 0 | Number of distinct towns. Range: 0–36 |
| `island_count` | INTEGER | DEFAULT 0 | Number of distinct islands. Range: 0–25 |

**POS distribution:**

| POS | Count | Percent |
|-----|-------|---------|
| n (noun) | 115,858 | 77.4% |
| a (adjective) | 18,915 | 12.6% |
| v (verb) | 8,562 | 5.7% |
| r (adverb) | 3,971 | 2.7% |
| NULL | 2,385 | 1.6% |

**Category distribution:**

| Category | Count | Percent |
|----------|-------|---------|
| single | 83,118 | 55.5% |
| compound | 62,221 | 41.6% |
| phrasal_verb | 1,967 | 1.3% |
| NULL | 2,385 | 1.6% |

**Token count distribution:**

| Tokens | Count |
|--------|-------|
| 1 | 85,503 |
| 2 | 54,533 |
| 3 | 7,766 |
| 4 | 1,454 |
| 5+ | 435 |

**Most spread words** (highest reef_count):

| Word | Reefs | Towns | Islands |
|------|-------|-------|---------|
| cut | 36 | 36 | 25 |
| beat | 30 | 30 | 18 |
| line | 27 | 27 | 18 |
| head | 26 | 26 | 23 |
| light | 26 | 26 | 19 |

**Populated by:** `load_wordnet_vocab.py` (step 5) + `link_seed_words.py` (step 9c, adds compounds)
**Embeddings by:** `reembed_words.py` (steps 6, 9d)
**Stats by:** `compute_word_stats.py` (step 17)

---

### ReefWords

Working dictionary. One row per (reef, word) assignment.

**Row count:** 449,925

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `reef_id` | INTEGER | NOT NULL, PK | FK to Reefs |
| `word_id` | INTEGER | NOT NULL, PK | FK to Words |
| `pos` | TEXT | nullable | Contextual POS within this reef. Overrides `Words.pos` when set. Distribution: noun 75.9%, adj 12.5%, verb 8.1%, adv 2.1%, NULL 1.5% |
| `specificity` | INTEGER | nullable | Contextual specificity. Always matches `Words.specificity` for the same word (global, not per-reef). Range: -2 to 3 |
| `cosine_sim` | REAL | nullable | Cosine to this reef's centroid embedding. Range: 0.708–0.985 (mean 0.863). Never NULL |
| `idf` | REAL | nullable | Global IDF (same as `Words.idf`). Range: 6.766–11.936 (mean 9.776). Never NULL |
| `island_idf` | REAL | nullable | Island-scoped IDF: `log2(island_total_reefs / word_reefs_in_island)`. Range: 1.585–8.114 (mean 6.507). Measures rarity within the island |
| `source` | TEXT | NOT NULL | How this word was assigned: **curated** (269,659 = 59.9%) or **xgboost** (180,266 = 40.1%) |
| `source_quality` | REAL | NOT NULL, DEFAULT 1.0 | Confidence signal. Range: 0.7–1.0 (mean 0.932). Values: 1.0 (curated/seed words), 0.9 (xgboost core), 0.7 (xgboost non-core) |
| `is_core` | INTEGER | DEFAULT 0 | **1** = Leiden core member (386,237 = 85.8%). **0** = periphery member (63,688 = 14.2%) |

**Source x is_core crosstab:**

| Source | Core (1) | Non-core (0) | Total |
|--------|----------|--------------|-------|
| curated | 269,659 | 0 | 269,659 |
| xgboost | 116,578 | 63,688 | 180,266 |

All curated words are core. XGBoost words are core if they were placed in a Leiden community; non-core if they fell outside communities but were still assigned to the nearest reef.

**Cardinality:**
- Words per reef: min 7, mean 114.8, max 1,284
- Reefs per word: min 1, mean 3.5, max 36

**Populated by:** `cluster_reefs.py` (step 15) + `cluster_bucket_reefs.py` (step 16)

---

## 3. Core Export Tables

Three tables implementing the export promotion chain. A word appears at **exactly one** level (verified: 0 words appear at multiple levels). 22,725 words (15.2%) are not exported at all (no reef assignment, mostly stop words and unclassified vocabulary).

### ReefWordExports

Words that are specific to a single reef within a town.

**Row count:** 30,079 (28,713 unique words)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `reef_id` | INTEGER | NOT NULL, PK | FK to Reefs |
| `word_id` | INTEGER | NOT NULL, PK | FK to Words |
| `idf` | REAL | nullable | Global IDF used in weight calculation. Range: 7.849–11.936 (mean 11.759) |
| `centroid_sim` | REAL | nullable | Cosine to reef centroid. Range: 0.708–0.976 (mean 0.852) |
| `name_cos` | REAL | nullable | Blended hierarchy name cosine (see [Weight Formulas](#10-reference-weight-formulas)). Range: 0.561–0.978 (mean 0.685) |
| `effective_sim` | REAL | nullable | `0.5 * centroid_sim + 0.2 * group_name_cos + 0.3 * island_name_cos`. Range: 0.655–0.944 (mean 0.784) |
| `specificity` | INTEGER | nullable | Only values **-1** (1,492 singleton rescues) and **3** (28,587 single-reef words) |
| `source_quality` | REAL | nullable | Range: 0.9–1.0 (mean 0.938) |
| `export_weight` | INTEGER | NOT NULL | u8 [0, 255]. Per-reef min-max normalized. Mean 135.8 |

**Export weight distribution:**

| Range | Count | Percent |
|-------|-------|---------|
| 0–49 | 5,622 | 18.7% |
| 50–99 | 4,710 | 15.7% |
| 100–149 | 5,543 | 18.4% |
| 150–199 | 6,290 | 20.9% |
| 200–255 | 7,914 | 26.3% |

**Populated by:** `populate_exports.py` (step 19)

---

### TownWordExports

Words that appear in multiple reefs within a town but don't spread across towns within the island.

**Row count:** 240,520 (75,542 unique words)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `town_id` | INTEGER | NOT NULL, PK | FK to Towns |
| `word_id` | INTEGER | NOT NULL, PK | FK to Words |
| `idf` | REAL | nullable | Global IDF |
| `centroid_sim` | REAL | nullable | Cosine to town centroid (max across member reefs) |
| `name_cos` | REAL | nullable | Blended hierarchy name cosine |
| `effective_sim` | REAL | nullable | Range: 0.655–0.930 (mean 0.774) |
| `specificity` | INTEGER | nullable | Only values **1** (123,946) and **2** (116,574). Spec 1 = 4–5 reefs, spec 2 = 2–3 reefs |
| `source_quality` | REAL | nullable | Max source quality across member reefs |
| `export_weight` | INTEGER | NOT NULL | u8 [0, 255]. Per-town min-max normalized. Mean 113.5 |
| `export_town_weight` | INTEGER | NOT NULL, DEFAULT 128 | Reserved for future per-town weighting. Currently all values = **128** |

**Export weight distribution:**

| Range | Count | Percent |
|-------|-------|---------|
| 0–49 | 25,404 | 10.6% |
| 50–99 | 70,228 | 29.2% |
| 100–149 | 88,237 | 36.7% |
| 150–199 | 45,681 | 19.0% |
| 200–255 | 10,970 | 4.6% |

**Populated by:** `populate_exports.py` (step 19)

---

### IslandWordExports

Words that spread across multiple towns within an island, or are too generic for town-level discrimination.

**Row count:** 145,017 (22,711 unique words)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `island_id` | INTEGER | NOT NULL, PK | FK to Islands |
| `word_id` | INTEGER | NOT NULL, PK | FK to Words |
| `idf` | REAL | nullable | Island-level IDF: `log2(N_islands / n_islands_word)` |
| `centroid_sim` | REAL | nullable | Cosine to island centroid (max across member reefs) |
| `name_cos` | REAL | nullable | Cosine to island name embedding |
| `effective_sim` | REAL | nullable | `0.5 * centroid_sim + 0.5 * island_name_cos`. Range: 0.665–0.980 (mean 0.783) |
| `specificity` | INTEGER | nullable | Values: **-2** (777), **-1** (25,613), **0** (118,627). Generic/moderate words |
| `source_quality` | REAL | nullable | Max source quality across member reefs |
| `export_weight` | INTEGER | NOT NULL | u8 [0, 255]. Hybrid normalized (80% global + 20% per-island), then scaled by exclusivity factor. Mean 59.0 |
| `export_island_weight` | INTEGER | NOT NULL, DEFAULT 128 | Reserved for future per-island weighting. Currently all values = **128** |

**Export weight distribution:**

| Range | Count | Percent |
|-------|-------|---------|
| 0–49 | 45,743 | 31.5% |
| 50–99 | 95,055 | 65.5% |
| 100–149 | 4,136 | 2.9% |
| 150–199 | 82 | 0.1% |
| 200–255 | 1 | <0.01% |

Weights are heavily compressed toward the low end because the exclusivity factor (`1 / n_islands^0.33`) penalizes words that appear in many islands — which is the common case at this level.

**Populated by:** `populate_exports.py` (step 19)

---

## 4. Optional Export Tables

Exported but only optionally loaded by Lagoon to keep runtime memory low.

### WordsExport

Debugging aid for reverse lookups (word_id -> text).

**Row count:** 0 (not yet populated)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `word_id` | INTEGER | PK | FK to Words |
| `word` | TEXT | NOT NULL | Word text |

**Populated by:** `export.py`

---

### NamesExport

First/last names for person detection and pronoun collapsing.

**Row count:** 0 (planned, not yet built)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `name_id` | INTEGER | PK | Auto-increment |
| `name` | TEXT | NOT NULL | Name text |
| `type` | TEXT | NOT NULL | `'first'` or `'last'` (CHECK constraint) |

---

### EquivalencesExport

Alternate word hashes that resolve to a canonical word. Covers morphological variants.

**Row count:** 0 (not yet populated)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `variant_hash` | INTEGER | NOT NULL, PK | FNV-1a u64 of the variant form (stored as i64) |
| `word_id` | INTEGER | NOT NULL, PK | FK to Words (canonical form) |
| `variant` | TEXT | NOT NULL | Variant surface form (e.g., "running" for canonical "run") |
| `source` | TEXT | NOT NULL | Origin: morphy, snowball, nickname, typo, plural |

**Populated from:** WordVariants table during export phase

---

### AcronymsExport

Acronym expansions, optionally scoped to a domain.

**Row count:** 0 (planned, not yet built)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `acronym_id` | INTEGER | PK | Auto-increment |
| `acronym` | TEXT | NOT NULL | Acronym text (e.g., "CPU") |
| `expansion` | TEXT | NOT NULL | Full expansion (e.g., "central processing unit") |
| `word_id` | INTEGER | nullable, FK | FK to Words if expansion exists in vocabulary |
| `island_id` | INTEGER | nullable, FK | Optional domain scope |

---

## 5. Pipeline Support Tables

Working tables used during the build pipeline. Persist across stages.

### DimStats

Per-embedding-dimension statistics for z-score feature engineering in XGBoost.

**Row count:** 768 (one per embedding dimension)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `dim_id` | INTEGER | PK | Dimension index, 0–767 |
| `mean` | REAL | nullable | Mean of this dimension across all word embeddings. Range: -4.824 to 2.515 (overall mean ~0.008) |
| `std` | REAL | nullable | Standard deviation. Range: 0.342–0.669 (mean 0.485) |
| `threshold` | REAL | nullable | Activation threshold: `mean + 2.0 * std`. Words above this are considered "active" on this dimension |
| `member_count` | INTEGER | nullable | Words above threshold. Range: 2,712–4,219 (mean 3,346) |
| `selectivity` | REAL | nullable | `1.0 - (member_count / total_words)`. Range: 0.018–0.029 (mean 0.023). Higher = more selective |

**Populated by:** `compute_dimstats.py` (step 7)

---

### IslandWords

Words that belong to an island but are not discriminative for any specific town within it. These are detected pre-XGBoost (via cosine-std across town centroids) and post-XGBoost (predicted by >= 80% of an island's towns).

**Row count:** 244

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `island_id` | INTEGER | NOT NULL, PK | FK to Islands |
| `word_id` | INTEGER | NOT NULL, PK | FK to Words |
| `word` | TEXT | NOT NULL | Denormalized word text for convenience |
| `source` | TEXT | NOT NULL | **seed_cosine_std** (13) = pre-XGBoost detection, low variance across town centroids. **xgboost_filter** (231) = post-XGBoost detection, predicted by >= 80% of towns |
| `cosine_std` | REAL | nullable | Std of cosine similarities to town centroids. Lower = more generic. Range: 0.006–0.010 (mean 0.008). Only set for seed_cosine_std source |
| `avg_cosine` | REAL | nullable | Mean cosine to town centroids. Range: 0.789–0.851 |

**Top islands by island-word count:** Gastronomy (56), Astronomy (24), Finance (17), Transportation (15), Hobbies & Crafts (15)

**Populated by:** `detect_island_words.py` (step 11) + `train_town_xgboost.py` (step 12, post-filter)

---

### SeedWords

Starting vocabulary for each town before XGBoost expansion and Leiden clustering. Three sources feed seeds.

**Row count:** 116,130

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `town_id` | INTEGER | NOT NULL, PK | FK to Towns |
| `word` | TEXT | NOT NULL, PK | Seed word text (primary key with town_id) |
| `word_id` | INTEGER | nullable, FK | FK to Words. NULL for 13,105 seeds (11.3%) not in vocabulary (multi-word phrases not yet linked, or out-of-vocabulary) |
| `source` | TEXT | NOT NULL | **wordnet** (90,184 = 77.7%) = WDH via SKI bridge. **claude_augmented** (23,518 = 20.3%) = Claude-generated single words. **claude_compound** (2,428 = 2.1%) = Claude-generated multi-word terms |
| `confidence` | TEXT | nullable | **core** (105,632 = 90.9%) or **peripheral** (10,498 = 9.0%) |
| `score` | REAL | nullable | Source-specific confidence score. Currently all NULL (reserved) |

**Seeds per town:** min 12 (Physical Chemistry), mean 361.8, max 7,912 (Botany)

**Populated by:** `import_wdh.py` (step 8) + `populate_claude_seeds.sql` (step 9) + `populate_compound_seeds.sql` (step 9b)
**Linked by:** `link_seed_words.py` (step 9c)
**Deduplicated by:** `sanity_check_seeds.py` (step 10)

---

### AugmentedTowns

XGBoost predictions above threshold, per town. The expansion vocabulary before Leiden clustering.

**Row count:** 349,811

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `town_id` | INTEGER | NOT NULL, PK | FK to Towns |
| `word_id` | INTEGER | NOT NULL, PK | FK to Words |
| `score` | REAL | NOT NULL | XGBoost prediction probability. Range: 0.400–1.000 (mean 0.810). Floor of 0.4 is the prediction threshold |
| `source` | TEXT | NOT NULL | Always **xgboost** |

**Predictions per town:** min 46, mean 1,223.1, max 5,140

**Populated by:** `train_town_xgboost.py` (step 12)
**Refined by:** `post_process_xgb.py` (step 13, removes low-confidence cross-town predictions)

---

### WordIslandStats

Per (word, island) concentration statistics. One row for every (word, island) pair where the word appears in at least one reef of that island.

**Row count:** 391,534

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `word_id` | INTEGER | NOT NULL, PK | FK to Words |
| `island_id` | INTEGER | NOT NULL, PK | FK to Islands |
| `reef_count` | INTEGER | NOT NULL | Number of reefs the word appears in within this island. Range: 1–9 (mean 1.1) |
| `town_count` | INTEGER | NOT NULL | Number of towns the word appears in within this island. Range: 1–9 (mean 1.1) |
| `concentration` | REAL | NOT NULL | Fraction of the word's total reefs that fall in this island. Range: 0.028–1.000 (mean 0.324). 1.0 = word appears in only one island |
| `avg_cosine` | REAL | nullable | Mean cosine to reef centroids within this island |

**Islands per word:** min 1, mean 3.1, max 25

**Populated by:** `compute_word_stats.py` (step 17)

---

### WordVariants

Morphological and stemming variants of vocabulary words.

**Row count:** 0 (not yet populated)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `variant_hash` | INTEGER | NOT NULL, PK | FNV-1a u64 of variant text (stored as i64) |
| `variant` | TEXT | NOT NULL | Variant surface form |
| `word_id` | INTEGER | NOT NULL, PK, FK | Canonical word in Words table |
| `source` | TEXT | NOT NULL | Origin: **base**, **morphy**, **snowball** |

**Will feed into:** EquivalencesExport during export phase

---

## 6. Views

### ExportIndex

Unified export view resolving every exported word to its full hierarchy path. Union of all three export tables joined to hierarchy.

**Virtual row count:** 415,616 (30,079 reef + 240,520 town + 145,017 island)

| Column | Source | Description |
|--------|--------|-------------|
| `archipelago_id` | Archipelagos | Archipelago ID |
| `archipelago` | Archipelagos.name | Archipelago name |
| `island_id` | Islands | Island ID |
| `island` | Islands.name | Island name |
| `town_id` | Towns / NULL | Town ID. NULL for island-level exports |
| `town` | Towns.name / NULL | Town name. NULL for island-level exports |
| `reef_id` | Reefs / NULL | Reef ID. NULL for town and island-level exports |
| `reef` | Reefs.name / NULL | Reef name. NULL for town and island-level exports |
| `word_id` | Export tables | Word ID |
| `export_weight` | Export tables | u8 weight [0, 255] |
| `export_level` | Literal | `'reef'`, `'town'`, or `'island'` |

**Example query — single word:**
```sql
SELECT * FROM ExportIndex
WHERE word_id = (SELECT word_id FROM Words WHERE word = 'violin')
ORDER BY export_weight DESC;
```

**Example query — multi-word search:**
```sql
SELECT town, island, archipelago, SUM(export_weight) AS score
FROM ExportIndex
WHERE word_id IN (SELECT word_id FROM Words
                  WHERE word IN ('hockey','stick','ice','rink'))
GROUP BY town_id
ORDER BY score DESC LIMIT 10;
```

---

### WordSearch

Convenience view joining ExportIndex with word text for human-readable results.

| Column | Source | Description |
|--------|--------|-------------|
| `word` | Words.word | Word text |
| *(all ExportIndex columns)* | ExportIndex | Full hierarchy path + weight |

**Example:**
```sql
SELECT * FROM WordSearch WHERE word = 'photosynthesis' ORDER BY export_weight DESC;
```

---

### Compounds

Words with multiple tokens, for Aho-Corasick multi-word tokenization.

**Virtual row count:** 64,182

| Column | Source | Description |
|--------|--------|-------------|
| `word` | Words.word | Compound word text (e.g., "machine learning") |
| `word_id` | Words.word_id | Word ID |

**Filter:** `word_count > 1 AND is_stop = 0`

---

### HierarchyPath

Full hierarchy path for any reef, useful for reporting.

**Virtual row count:** 3,919 (one per reef)

| Column | Source | Description |
|--------|--------|-------------|
| `archipelago_id` | Archipelagos | Archipelago ID |
| `archipelago` | Archipelagos.name | Archipelago name |
| `island_id` | Islands | Island ID |
| `island` | Islands.name | Island name |
| `town_id` | Towns | Town ID |
| `town` | Towns.name | Town name |
| `reef_id` | Reefs | Reef ID |
| `reef` | Reefs.name | Reef name |
| `reef_words` | Reefs.word_count | Words in this reef |
| `town_words` | Towns.word_count | Words in this town |
| `island_words` | Islands.word_count | Words in this island |

---

## 7. Indexes

| Index | Table | Column(s) | Purpose |
|-------|-------|-----------|---------|
| `idx_islands_arch` | Islands | archipelago_id | Parent lookup |
| `idx_towns_island` | Towns | island_id | Parent lookup |
| `idx_reefs_town` | Reefs | town_id | Parent lookup |
| `idx_words_word` | Words | word | Text lookup |
| `idx_words_hash` | Words | word_hash | Hash lookup (Lagoon tokenizer) |
| `idx_rw_word` | ReefWords | word_id | Word-centric queries across all reefs |
| `idx_rwe_word` | ReefWordExports | word_id | Export search by word |
| `idx_twe_word` | TownWordExports | word_id | Export search by word |
| `idx_iwe_word` | IslandWordExports | word_id | Export search by word |
| `idx_equiv_hash` | EquivalencesExport | variant_hash | Hash lookup for tokenization |
| `idx_iw_word` | IslandWords | word_id | Word lookup |
| `idx_seed_town` | SeedWords | town_id | Seeds by town |
| `idx_seed_word` | SeedWords | word_id | Seeds by word |
| `idx_at_town` | AugmentedTowns | town_id | Predictions by town |
| `idx_at_word` | AugmentedTowns | word_id | Predictions by word |
| `idx_wis_island` | WordIslandStats | island_id | Stats by island |
| `idx_wv_hash` | WordVariants | variant_hash | Variant hash lookup |
| `idx_wv_word` | WordVariants | word_id | Variants by canonical word |

---

## 8. Reference: Specificity Scale

Specificity maps a word's reef count to a discrete category indicating how discriminative the word is. Computed by `compute_word_stats.py`.

| Specificity | Reef Count | Meaning | Words | In ReefWordExports | In TownWordExports | In IslandWordExports |
|-------------|-----------|---------|-------|--------------------|--------------------|----------------------|
| 3 | 1 | Highly specific | 28,587 | 28,587 | — | — |
| 2 | 2–3 | Very specific | 47,508 | — | 116,574 | — |
| 1 | 4–5 | Specific | 28,034 | — | 123,946 | — |
| 0 | 6–10 | Moderate | 20,074 | — | — | 118,627 |
| -1 | 11–20 | Generic | 2,715 | 1,492 (singletons) | — | 25,613 |
| -2 | 21+ | Very generic | 48 | — | — | 777 |

The "singleton rescue" rule allows spec -1 words that appear in only 1 reef within an island to export at reef level instead of island level.

---

## 9. Reference: Export Promotion Rules

Each word is classified into exactly one export level based on specificity and within-island spread:

| Condition | Export Level | Rationale |
|-----------|-------------|-----------|
| spec >= 0, 1 town in island | reef | Specific word in a single town |
| spec >= 1, 2+ towns in island | town | Specific word spanning multiple towns |
| spec == 0, 2+ towns in island | island | Moderate spread across towns |
| spec == -1, 1 reef in island | reef | Singleton rescue: generic but localized |
| spec == -1, 2+ reefs in island | island | Generic and spread |
| spec <= -2 | island | Very generic |

**Result distribution:**

| Level | Rows | Unique Words | Percent of Exports |
|-------|------|-------------|-------------------|
| Reef | 30,079 | 28,713 | 7.2% |
| Town | 240,520 | 75,542 | 57.9% |
| Island | 145,017 | 22,711 | 34.9% |

---

## 10. Reference: Weight Formulas

### Reef and Town Level

```
effective_sim = 0.5 * centroid_sim + 0.2 * group_name_cos + 0.3 * island_name_cos
raw_score     = global_idf * source_quality * effective_sim
export_weight = per_group_minmax_normalize(raw_score) → [0, 255]
```

- `centroid_sim`: cosine similarity to the reef/town centroid embedding
- `group_name_cos`: cosine to the reef or town name embedding (local topic signal)
- `island_name_cos`: cosine to the island name embedding (broad domain signal)
- `global_idf`: `log2(total_reefs / word_reef_count)` — reef-level rarity
- `source_quality`: `max(raw_quality, 0.9)` where raw is 1.0 (curated), 0.9 (xgboost core), 0.7 (xgboost non-core)
- Normalization: per-reef (or per-town) min-max to [0, 255]

### Island Level

```
effective_sim    = 0.5 * centroid_sim + 0.5 * island_name_cos
raw_score        = island_idf * source_quality * effective_sim
hybrid_weight    = 0.8 * global_minmax(raw) + 0.2 * per_island_minmax(raw)  → [0, 255]
export_weight    = round(hybrid_weight * exclusivity_factor)
exclusivity_factor = 1.0 / n_islands^0.33
```

- `island_idf`: `log2(N_islands / n_islands_word)` — island-level rarity
- Hybrid normalization prevents wide-range islands from compressing mid-range weights
- Exclusivity factor (cube-root dampening): 1 island → 1.00, 2 → 0.79, 4 → 0.63, 7 → 0.52

### Source Quality Values

| Source | is_core | Raw Quality | After Floor (0.9) |
|--------|---------|-------------|-------------------|
| curated (seed/wordnet) | 1 | 1.0 | 1.0 |
| xgboost | 1 (core) | 0.9 | 0.9 |
| xgboost | 0 (non-core) | 0.7 | 0.9 |

The floor of 0.9 softens the xgboost non-core penalty, since non-core words may still be highly relevant.

---

## Notes

- **SQLite integer storage:** All integers are stored as signed i64. FNV-1a u64 hashes must be masked with `& 0xFFFFFFFFFFFFFFFF` to recover the unsigned value.
- **Embedding format:** float32 x 768 dimensions = 3,072 bytes per BLOB. Generated with nomic-embed-text-v1.5 using the `"clustering: "` prefix.
- **PRAGMA settings:** `journal_mode = WAL`, `foreign_keys = ON`.
- **Idempotency:** Export tables are fully cleared and repopulated by `populate_exports.py`. The tuning loop (`populate_exports.py` + `test_battery.py`) is safe to run repeatedly.
