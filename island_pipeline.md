# Island Pipeline: Archipelagos, Islands, and Reefs

This document is a deep-dive technical reference for the three-generation island hierarchy in Windowsill. It describes every step of the pipeline that discovers, refines, and names semantic clusters within the 768 embedding dimensions of nomic-embed-text-v1.5. 

---

## 1. What the Hierarchy Represents

Each of the 768 embedding dimensions has a set of ~1,000-8,000 "member" words (those whose activation exceeds a z-score threshold). Many dimensions share significant word overlap — e.g., multiple dimensions might all activate for musical instrument words. The island pipeline discovers these natural clusters of co-activating dimensions and organizes them into a three-level hierarchy:

- **Gen-0: Archipelagos** (4 total) — The broadest semantic domains. Named things like "natural sciences and taxonomy", "social order and assessment". Each contains ~120-250 dimensions.
- **Gen-1: Islands** (52 total) — Mid-level topics within an archipelago. Named things like "musical instruments", "skin disease pathology". Each contains ~5-65 dimensions.
- **Gen-2: Reefs** (207 total) — The tightest semantic clusters. Named things like "string instruments", "woodwind instruments". Each contains ~2-20 dimensions.

The hierarchy is discovered bottom-up via community detection, not imposed top-down. Dimensions that don't cluster with anything become **noise** (`island_id = -1`).

---

## 2. Foundation: How Dimensions Get Members

Before island detection can run, phases 4-5 must have populated `dim_memberships`. The membership rule is simple:

```
A word is a member of dimension D if its raw activation value >= mean_D + ZSCORE_THRESHOLD * std_D
```

`ZSCORE_THRESHOLD` is currently **2.0** (stored in `config.py`). This threshold was deliberately lowered from 2.45 to 2.0 to provide enough overlap density for meaningful clustering:

| Threshold | Avg dims/word | Total memberships | Effect on island detection |
|-----------|---------------|-------------------|---------------------------|
| 2.45σ | ~6 | ~800K | Jaccard between related dims < 0.01; too sparse for coherent reefs |
| 2.0σ | ~17.5 | ~2.56M | Jaccard in range 0.01-0.18; dense enough for meaningful clustering |

The tradeoff: 2.0σ introduces more noise memberships (words weakly activating in a dimension they don't truly belong to). This noise is managed downstream by **depth filtering** — requiring a word to appear in multiple dimensions of a reef before considering it meaningfully present (see section 10).

**Key detail:** Every membership stores a `z_score` (how many standard deviations above the mean) and a raw `value`. These are used later for affinity scoring.

---

## 3. Jaccard Similarity Matrix (Phase 9, Step 1)

**Code:** `islands.compute_jaccard_matrix(con)`

The first step computes pairwise Jaccard similarity between all 768 dimensions based on their member word sets. This produces a 768x768 similarity matrix.

### 3.1 Single-token filtering

Only single-token words (no spaces) are used for Jaccard computation. Multi-word expressions like "heart attack" are excluded because they introduce compound contamination — a word like "heart" and "attack" would both be members due to the compound, inflating the apparent overlap between unrelated dimensions. The filter is `word NOT LIKE '% %'`.

### 3.2 Sparse matrix computation

The membership data is loaded into a sparse binary matrix `M` of shape `(768, N_words)`. The intersection matrix is computed as `I = M @ M.T`, giving the number of shared words for every dimension pair. This is O(768^2) but uses sparse operations for efficiency.

### 3.3 Jaccard and hypergeometric z-score

For each pair `(i, j)` where `i < j` and `intersection > 0`:

```
jaccard = intersection_size / union_size
expected_intersection = n_i * n_j / N      (hypergeometric expected value)
variance = n_i * n_j * (N - n_i) * (N - n_j) / (N^2 * (N - 1))
z_score = (intersection_size - expected_intersection) / sqrt(variance)
```

Where `n_i` and `n_j` are the member counts of each dimension, and `N` is the total single-token word count.

**Why hypergeometric z-score matters:** Raw Jaccard is misleading for dimensions with very different sizes. A dimension with 5,000 members will naturally have higher Jaccard overlap with everything than a dimension with 1,000 members, simply due to base rates. The hypergeometric z-score normalizes for this: it asks "how many more words do these dimensions share than we'd expect by chance, given their sizes?" This is the critical edge filter for graph construction.

### 3.4 Storage

Results go into `dim_jaccard` table. Only pairs with `intersection_size > 0` are stored (~294K pairs out of 768*767/2 = 294,528 possible pairs). The z-score is stored alongside the raw Jaccard for flexible graph construction.

**Typical value ranges:**
- Jaccard: 0.0001 to ~0.5 (most pairs < 0.05)
- Z-score: -1.2 to ~50+ (pairs above 3.0 are statistically significant)

---

## 4. Gen-0: Archipelago Detection (Phase 9, Step 2)

**Code:** `islands.detect_islands(con)`

### 4.1 Graph construction

An igraph `Graph` is created with 768 vertices (one per dimension). Edges are added for dimension pairs where `z_score >= ISLAND_JACCARD_ZSCORE` (default: **3.0**). Edge weights are set to the raw Jaccard value (not the z-score — Jaccard is the similarity metric, z-score is just the filter).

This threshold of 3.0 is conservative: it means a pair must have intersection at least 3 standard deviations above what random chance would predict. This filters out ~90% of dimension pairs, keeping only the statistically significant ones.

### 4.2 Leiden community detection

```python
partition = leidenalg.find_partition(
    g,
    leidenalg.RBConfigurationVertexPartition,
    weights="weight",
    resolution_parameter=ISLAND_LEIDEN_RESOLUTION,  # 1.0
    seed=42,
)
```

The Leiden algorithm (an improvement over Louvain) finds communities that maximize modularity. Key parameters:

- **Resolution = 1.0** (`ISLAND_LEIDEN_RESOLUTION`) — Standard resolution. Higher values produce more/smaller communities; lower values merge communities together. 1.0 is the "natural" scale.
- **Seed = 42** — Deterministic results across runs.
- **Weights = Jaccard values** — Communities form around groups of dimensions with high pairwise Jaccard overlap.

### 4.3 Noise assignment

Communities smaller than `ISLAND_MIN_COMMUNITY_SIZE` (default: **2**) are assigned `island_id = -1` (noise). With current parameters, this typically produces **4 archipelagos** covering the vast majority of dimensions, with a handful of noise dimensions.

### 4.4 Stats and characteristic words

After detection:
- `compute_island_stats(con, generation=0)` computes per-archipelago stats: n_dims, n_words (union of all member word sets), internal Jaccard statistics, core word counts, median word depth.
- `compute_characteristic_words(con, generation=0)` computes PMI-ranked words per archipelago.

**Core words:** Words appearing in `>= max(2, ceil(n_dims * 0.10))` of the island's dimensions. These are words deeply embedded in the island's semantic theme.

**PMI (Pointwise Mutual Information):** `log2(island_freq / corpus_freq)` where `island_freq = n_dims_in_island / n_island_dims` and `corpus_freq = total_dims / 768`. High PMI = much more common in this island than in the corpus overall. Only words appearing in >= 2 island dims are considered. Top 100 per island are stored.

---

## 5. Gen-1: Island Detection (Phase 9, Step 3)

**Code:** `islands.detect_sub_islands(con, parent_generation=0)`

Each gen-0 archipelago is subdivided into gen-1 islands using the same Leiden approach, but with a **higher resolution** to produce more granular clusters.

### 5.1 Per-parent subdivision

For each gen-0 archipelago with `>= ISLAND_MIN_DIMS_FOR_SUBDIVISION` (default: **10**) dimensions:

1. Extract the dims belonging to this archipelago
2. Build a local subgraph using only edges between those dims (still filtered at z >= 3.0)
3. Run Leiden with `ISLAND_SUB_LEIDEN_RESOLUTION` (**1.5**, higher than gen-0's 1.0)
4. Assign globally-sequential island IDs to the resulting communities
5. Communities < 2 dims become noise (`island_id = -1`)

### 5.2 Why higher resolution

The resolution parameter controls the granularity/size tradeoff. At gen-0, resolution 1.0 produces 4 broad archipelagos. At gen-1, resolution 1.5 breaks each archipelago into ~10-15 islands. The same z-score edge filter (>= 3.0) is used — only the resolution changes.

### 5.3 Parent-child relationship

Each gen-1 island stores `parent_island_id` pointing to its gen-0 archipelago. This is how the hierarchy is maintained. A dimension's full lineage is: archipelago -> island -> reef.

After subdivision: `compute_island_stats(con, generation=1)` and `compute_characteristic_words(con, generation=1)`.

---

## 6. Gen-2: Reef Detection (Phase 9, Step 4)

**Code:** `islands.detect_sub_islands(con, parent_generation=1)`

The exact same process as gen-1, but applied to gen-1 islands to produce gen-2 reefs. Each gen-1 island with >= 10 dims is subdivided using the same Leiden parameters (resolution 1.5, z >= 3.0 edge filter).

This produces **207 reefs** from the 52 islands. Reefs are the finest-grained semantic clusters — e.g., an "instruments" island might split into "string instruments", "woodwind instruments", "percussion instruments" reefs.

After subdivision: `compute_island_stats(con, generation=2)` and `compute_characteristic_words(con, generation=2)`.

**Important:** Gen-2 reef assignments can change during Phase 10 (reef refinement). All other generations are fixed after initial detection.

---

## 7. Denormalization: Backfilling onto dim_memberships

**Code:** `islands.backfill_membership_islands(con)`

After the hierarchy is established, the island/reef assignments are denormalized from `dim_islands` onto `dim_memberships` for query convenience:

```sql
-- For each dim_membership row, set:
-- archipelago_id = the gen-0 island_id for this dim_id
-- island_id      = the gen-1 island_id for this dim_id
-- reef_id        = the gen-2 island_id for this dim_id
```

Noise dimensions (`island_id = -1` in `dim_islands`) are set to `NULL` in the denormalized columns for cleaner queries.

**Why this matters:** Without denormalization, answering "which reefs does word X belong to?" requires joining `dim_memberships -> dim_islands` and filtering by generation. With denormalization, it's just `SELECT DISTINCT reef_id FROM dim_memberships WHERE word_id = ? AND reef_id IS NOT NULL`. This is used extensively by the explorer commands (`relationship`, `archipelago`, `evaluate`, etc.).

**When it runs:** After initial island detection (phase 9), after re-backfill (phase 9b), and after reef refinement (phase 10).

---

## 8. Word-Reef Affinity Scores

**Code:** `islands.compute_word_reef_affinity(con)`

The `word_reef_affinity` table stores continuous affinity metrics for every word-reef pair where the word activates in at least one of the reef's dimensions. This is computed by joining `dim_memberships` with `dim_islands` (gen=2) and `dim_stats`:

```sql
INSERT INTO word_reef_affinity
SELECT m.word_id,
       di.island_id AS reef_id,
       COUNT(*) AS n_dims,             -- depth: how many reef dims this word activates
       MAX(m.z_score) AS max_z,        -- strongest single-dim z-score
       SUM(m.z_score) AS sum_z,        -- total z-score mass
       MAX(m.z_score * ds.dim_weight) AS max_weighted_z,   -- strongest information-weighted score
       SUM(m.z_score * ds.dim_weight) AS sum_weighted_z    -- total information-weighted mass
FROM dim_memberships m
JOIN dim_islands di ON m.dim_id = di.dim_id
JOIN dim_stats ds ON m.dim_id = ds.dim_id
WHERE di.generation = 2 AND di.island_id >= 0
GROUP BY m.word_id, di.island_id
```

### 8.1 Key columns

- **`n_dims`** — The "depth" of a word in a reef. Depth 1 means the word activates in only one of the reef's dimensions (may be noise). Depth 2+ means stronger evidence of genuine membership. This is the primary signal used by `REEF_MIN_DEPTH`.
- **`max_z`** — The highest z-score across the word's activations in this reef's dims. Higher z-scores indicate stronger, more confident membership.
- **`max_weighted_z`** — `z_score * dim_weight` where `dim_weight = -log2(max(universal_pct, 0.01))`. This combines activation strength with dimension informativeness. Concrete/specific dimensions (low universal_pct) have high dim_weight, so a word activating strongly in a concrete reef dimension gets a very high weighted_z. This was found to be the strongest single discriminator for meaningful depth-1 memberships.
- **`sum_weighted_z`** — Total information-weighted mass. Useful for ranking a word's reef affinities.

### 8.2 Scale

Typically produces ~1.97M rows (every word × every reef it touches, even at depth 1).

### 8.3 Usage

- **Explorer `affinity <word>` command** — Shows all reefs a word has affinity for, ranked by max_weighted_z.
- **Explorer `evaluate` battery** — Uses `_pair_similarity()` which queries shared reefs/islands/archipelagos via denormalized `dim_memberships` columns to classify word relationships.
- **Analysis** — Enables questions like "which words almost belong to this reef?" or "which reefs does this word have the strongest affinity for?"

---

## 9. Depth Filtering: REEF_MIN_DEPTH

**Config:** `REEF_MIN_DEPTH = 2`

A word's "depth" in a reef is the number of the reef's dimensions that the word activates in (i.e., `n_dims` in `word_reef_affinity`). Depth filtering is the primary mechanism for managing noise introduced by the 2.0σ z-score threshold.

### 9.1 The problem

At z = 2.0σ, ~2.3% of the population exceeds the threshold by chance alone (one-tailed normal). With ~146K words and ~4-20 dims per reef, many words will be "members" of a reef dimension purely by statistical noise. These noise memberships are almost always depth-1: the word activates in exactly one of the reef's dims.

### 9.2 The solution

Genuine semantic membership manifests as multi-dimension overlap. If "guitar" truly belongs to a "string instruments" reef, it should activate in multiple (e.g., 7 out of 17) of that reef's dimensions, not just one. Requiring depth >= 2 prunes the vast majority of noise while retaining genuine memberships.

### 9.3 Where depth filtering is applied

Depth filtering is **not** baked into the data — `dim_memberships` and `word_reef_affinity` store everything including depth-1 pairs. Instead, it's applied at query time:

- **`explore.archipelago()`** — Prunes the display tree to only show structures where the word has depth >= `REEF_MIN_DEPTH`
- **`explore.exclusion()`** — Only considers reefs where a word has depth >= `REEF_MIN_DEPTH`
- **`explore.bridge_profile()`** — Same depth filter
- **Direct SQL** — Analysts can apply their own threshold via `WHERE n_dims >= 2` on `word_reef_affinity`

The raw data is preserved for analysis flexibility — you might want to examine depth-1 relationships with high `max_weighted_z` scores, which represent strong single-dim activations that didn't quite reach multi-dim depth.

---

## 10. Reef Refinement (Phase 10)

**Code:** `reef_refine.run_reef_refinement(con)`

Leiden community detection is a good first pass, but not perfect. Some dimensions end up in a reef where they have higher average Jaccard similarity to a sibling reef's dimensions than to their own. Phase 10 detects and corrects these misplacements through iterative refinement.

### 10.1 Loyalty metric

For each dimension `d` in a reef `R`:

1. **Own affinity:** Mean Jaccard between `d` and all other dims in `R`
2. **Best sibling affinity:** For each sibling reef `S` (same parent island), compute mean Jaccard between `d` and all dims in `S`. Take the maximum.
3. **Loyalty ratio:** `own_affinity / best_sibling_affinity`

If `loyalty_ratio < REEF_REFINE_LOYALTY_THRESHOLD` (default: **1.0**), the dimension is considered misplaced — it's more similar to a sibling reef than to its own.

### 10.2 Iteration

The refinement loop:

1. Load current reef assignments from `dim_islands` (gen=2)
2. For each reef with `>= REEF_REFINE_MIN_DIMS` (default: **4**) dimensions:
   - Compute loyalty for every dim in the reef
   - Collect dims with `loyalty_ratio < 1.0` and a valid best sibling
3. Apply all moves (UPDATE dim_islands)
4. Recompute `island_stats` and `island_characteristic_words` for gen-2
5. Repeat until convergence (no moves) or `REEF_REFINE_MAX_ITERATIONS` (default: **5**)

Typically converges in 2-3 iterations.

### 10.3 Jaccard lookup

The full Jaccard lookup is loaded into a dictionary at the start (symmetric: both `(a,b)` and `(b,a)` entries). This is reused across all iterations since the underlying Jaccard values don't change — only the reef assignments change.

### 10.4 Post-convergence

After refinement converges:
1. `backfill_membership_islands(con)` — Re-denormalize the updated reef assignments onto `dim_memberships`
2. `compute_word_reef_affinity(con)` — Recompute affinity scores with the new reef boundaries

### 10.5 Why only gen-2

Refinement only operates on gen-2 reefs (not gen-0 or gen-1). The rationale: reefs are the tightest clusters and most sensitive to individual dim placement. Moving a dim between archipelagos would be a much larger structural change with unclear benefit.

---

## 11. Naming (Phase 9c)

**Code:** `islands.generate_island_names(con)`

Every entity in the hierarchy gets a human-readable name via a bottom-up LLM-assisted pipeline using the Claude API.

### 11.1 Bottom-up order

1. **Reefs first** (gen-2, most specific)
2. **Islands next** (gen-1, synthesized from child reef names)
3. **Archipelagos last** (gen-0, synthesized from child island names)

### 11.2 Reef naming

For each gen-1 island that has child reefs:

1. Compute **exclusive words** per reef: words that appear in this reef's dims but in no sibling reef's dims. These are the strongest differentiator.
2. Get **PMI-ranked words** per reef (top 20).
3. Send a prompt to Claude with both exclusive and PMI words for all sibling reefs together. The prompt emphasizes contrast — names should differentiate siblings.
4. Claude returns a JSON `{reef_id: "name"}` mapping.

### 11.3 Island and archipelago naming

For gen-1 islands and gen-0 archipelagos, Claude is given the names of child entities and asked to find the overarching theme. For entities without children (too small to subdivide), it falls back to PMI-ranked characteristic words.

### 11.4 Naming constraints

- 2-4 words, lowercase
- Distinct from sibling names
- More general than any child name, more specific than any parent name
- Readable and informative over obscure technical terms

### 11.5 Phase ordering

Phase 9c runs **after** phase 10 (reef refinement) because refinement recomputes reef stats and characteristic words, which would wipe out previously-generated names. The phase order is: `9 -> 9b -> 9d -> 10 -> 9c`.

---

## 12. Pipeline Execution Flow

### Full pipeline (phase 9)

`islands.run_island_detection(con)` executes:

```
1. compute_jaccard_matrix(con)           -- 768x768 pairwise Jaccard + hypergeometric z
2. detect_islands(con)                   -- Gen-0 Leiden (resolution 1.0)
3. compute_island_stats(con, gen=0)      -- Stats for archipelagos
4. compute_characteristic_words(con, gen=0) -- PMI for archipelagos
5. detect_sub_islands(con, parent_gen=0) -- Gen-1 Leiden (resolution 1.5)
6. compute_island_stats(con, gen=1)      -- Stats for islands
7. compute_characteristic_words(con, gen=1) -- PMI for islands
8. detect_sub_islands(con, parent_gen=1) -- Gen-2 Leiden (resolution 1.5)
9. compute_island_stats(con, gen=2)      -- Stats for reefs
10. compute_characteristic_words(con, gen=2) -- PMI for reefs
11. backfill_membership_islands(con)     -- Denormalize onto dim_memberships
12. compute_word_reef_affinity(con)      -- Compute affinity scores
13. print_archipelago_summary(con)       -- Display results
```

### Standalone re-backfill + affinity (phase 9b)

`run_phase9b()` in main.py:
```
1. migrate_schema(con)
2. backfill_membership_islands(con)
3. compute_word_reef_affinity(con)
```

Use this after manually modifying island assignments — it refreshes the denormalized columns and affinity scores without recomputing the Jaccard matrix or re-running Leiden.

### Reef refinement (phase 10)

`reef_refine.run_reef_refinement(con)` (see section 10), followed by backfill + affinity.

### Naming (phase 9c)

`islands.generate_island_names(con)` (see section 11). Requires `ANTHROPIC_API_KEY`.

---

## 13. Configuration Reference

All thresholds controlling the island pipeline, in order of influence:

| Constant | Value | Where Used | Effect |
|----------|-------|------------|--------|
| `ZSCORE_THRESHOLD` | 2.0 | Phase 5 (analyzer) | Controls membership density in `dim_memberships`. Lower = more members per dim = denser Jaccard overlap. This is the single most impactful parameter. |
| `ISLAND_JACCARD_ZSCORE` | 3.0 | `compute_jaccard_matrix`, `detect_islands`, `detect_sub_islands` | Edge filter for the community detection graph. Only dimension pairs with hypergeometric z >= 3.0 form edges. Higher = sparser graph = fewer/larger communities. |
| `ISLAND_LEIDEN_RESOLUTION` | 1.0 | `detect_islands` (gen-0 only) | Leiden resolution for archipelago detection. Standard scale. |
| `ISLAND_SUB_LEIDEN_RESOLUTION` | 1.5 | `detect_sub_islands` (gen-1 and gen-2) | Higher resolution produces more/smaller sub-communities. |
| `ISLAND_MIN_COMMUNITY_SIZE` | 2 | `detect_islands`, `detect_sub_islands` | Singleton dimensions become noise. |
| `ISLAND_MIN_DIMS_FOR_SUBDIVISION` | 10 | `detect_sub_islands` | Islands/archipelagos with fewer dims aren't subdivided. |
| `ISLAND_CHARACTERISTIC_WORDS_N` | 100 | `compute_characteristic_words` | How many PMI-ranked words are stored per island. |
| `REEF_MIN_DEPTH` | 2 | Explorer commands (`archipelago`, `exclusion`, `bridge_profile`) | Min dims a word must activate in a reef to count as meaningfully present. Applied at query time, not baked into data. |
| `REEF_REFINE_MIN_DIMS` | 4 | `reef_refine` | Reefs smaller than this are skipped during refinement (not enough data for meaningful loyalty analysis). |
| `REEF_REFINE_LOYALTY_THRESHOLD` | 1.0 | `reef_refine` | Dims with loyalty_ratio < 1.0 are reassigned. Value of 1.0 means "if you're more similar to ANY sibling reef than to your own, you move." |
| `REEF_REFINE_MAX_ITERATIONS` | 5 | `reef_refine` | Safety valve. Typically converges in 2-3 iterations. |

---

## 14. Database Tables

Quick reference for tables involved in the island pipeline:

| Table | Role | Row Count |
|-------|------|-----------|
| `dim_jaccard` | Pairwise dimension similarity (Jaccard + hypergeometric z) | ~294K |
| `dim_islands` | Dimension -> island assignments (one row per dim per generation) | ~2,304 (768 x 3) |
| `island_stats` | Per-island aggregate statistics + names | ~263 (4 + 52 + 207) |
| `island_characteristic_words` | PMI-ranked words per island (up to 100 each) | ~9.2K |
| `word_reef_affinity` | Continuous word-reef affinity scores | ~1.97M |
| `dim_memberships` (columns) | `archipelago_id`, `island_id`, `reef_id` — denormalized from `dim_islands` | (part of existing ~2.56M row table) |

---

## 15. Explorer Commands for the Hierarchy

These explorer commands query the island hierarchy:

| Command | What It Does | Key Tables |
|---------|-------------|------------|
| `archipelago <word>` | Nested tree view: archipelago -> island -> reef -> dims, depth-filtered | `dim_memberships` + `dim_islands` + `island_stats` |
| `relationship <w1> <w2>` | Classify pair as same-reef / reef-neighbors / island-neighbors / different, with named shared structures | `dim_memberships` (denormalized columns) + `island_stats` |
| `exclusion <w1> <w2>` | Shared reef exclusions between universal words (Jaccard of avoided reefs) | `dim_memberships` + `dim_islands` + `island_stats` |
| `bridge_profile <word>` | Reef distribution by archipelago + cross-archipelago bridge pairs | `dim_memberships` + `dim_islands` + `island_stats` |
| `affinity <word>` | All reefs ranked by max_weighted_z | `word_reef_affinity` + `island_stats` |
| `evaluate` | 30-pair semantic battery, graded against expected tier (same-reef / same-island / same-archipelago / different) | `dim_memberships` (denormalized) via `_pair_similarity()` |
| `dim_info <id>` | Shows island hierarchy for a single dimension | `dim_islands` + `island_stats` |

### How `_pair_similarity()` works

The `evaluate` battery and `relationship` command both use `_pair_similarity()` which computes shared structure counts between two words using the denormalized columns on `dim_memberships`:

```python
# Shared reefs = count of distinct reef_ids that both words have
# Shared islands = count of distinct island_ids that both words have
# Shared archs = count of distinct archipelago_ids that both words have

# Level assignment:
# 1 = same reef (shared_reefs > 0)
# 2 = same island (shared_islands > 0, no shared reefs)
# 3 = same archipelago (shared_archs > 0, no shared islands)
# 4 = different (no shared structure)
```

Note that "shared reefs" here counts ANY reef overlap including depth-1 memberships (the denormalized columns don't apply depth filtering). For stricter analysis, use `word_reef_affinity` with `n_dims >= 2` filtering.

---

## 16. Data-Driven Insights: What We Learned

### 16.1 The z-score threshold is the master knob

The `ZSCORE_THRESHOLD` for dimension membership is the single most impactful parameter in the entire system. At 2.45σ, the Jaccard matrix was too sparse for coherent clustering. At 2.0σ, the 3x increase in memberships raised pairwise Jaccard into the significant range, enabling tight, semantically coherent reefs. The "right" threshold is the one that produces enough overlap for community detection while not drowning in noise — and depth filtering handles the noise.

### 16.2 Hypergeometric z-score beats raw Jaccard for edge filtering

Raw Jaccard between two large-membership dimensions will naturally be higher than between two small-membership dimensions, even if neither pair has meaningful semantic overlap. The hypergeometric z-score normalizes for dimension sizes, producing a fair comparison. Filtering at z >= 3.0 ensures only statistically significant overlaps form edges.

### 16.3 Depth is the key discriminator for word-reef membership

A word appearing in 1 out of 17 reef dimensions is likely noise. A word appearing in 7 out of 17 is almost certainly genuine. Depth >= 2 is a surprisingly effective binary filter — it eliminates the vast majority of false positives while retaining nearly all true positives. This is because genuine semantic membership is almost always multi-dimensional: "guitar" doesn't just activate one music dimension, it activates many.

### 16.4 Weighted z-score identifies meaningful depth-1 memberships

For the ~46% of words that never reach depth-2 in any reef (thin members), `max_weighted_z = z_score * dim_weight` is the best single discriminator. A high weighted-z at depth-1 means the word has a strong activation in an informative (concrete) dimension — more meaningful than a weak activation in an abstract dimension. The current evaluation battery achieves 30/30 PASS using the denormalized columns on `dim_memberships` (which include depth-1 memberships), confirming that even shallow reef associations carry useful semantic signal.

### 16.5 Reef coverage is essentially 100%

Of the ~146,697 words with dimension memberships, **100.0%** (146,695) touch at least one reef dimension. Only 1 word exists exclusively in noise dims with no reef association at all. Of the 768 dimensions, 629 (90.0%) are assigned to reefs; the remaining 70 are in Leiden's noise cluster at gen-2. The depth-2 filter narrows strong membership to 54.1% of words (79,395), but even the remaining "thin" depth-1 members carry useful semantic signal — as evidenced by the 30/30 evaluation battery PASS rate using denormalized columns that include depth-1 memberships.

### 16.6 Resolution controls the level of detail

The same Leiden algorithm with the same edge filter produces completely different results at different resolutions. Resolution 1.0 gives 4 archipelagos (broadest themes). Resolution 1.5 on the same data (but per-archipelago) gives 52 islands and 207 reefs. The resolution parameter is the primary control for how fine-grained the clustering is.

### 16.7 Refinement converges fast

Reef refinement typically converges in 2-3 iterations, moving a relatively small number of dims (typically < 30 across all reefs). This suggests Leiden does a good job on the first pass, with refinement catching edge cases where a dimension has ambiguous affinity.

---

## 17. Diagram: Data Flow

```
  dim_memberships (2.56M rows)
  (word activations per dimension, z-scores)
         │
         ▼
  ┌─────────────────────────────┐
  │  compute_jaccard_matrix()   │  Only single-token words
  │  768x768 pairwise Jaccard   │  Hypergeometric z-score
  │  → dim_jaccard (~295K rows) │
  └──────────────┬──────────────┘
                 │ edges where z >= 3.0
                 ▼
  ┌─────────────────────────────┐
  │  detect_islands()           │  Leiden, resolution 1.0
  │  → 4 gen-0 archipelagos     │  → dim_islands (gen=0)
  └──────────────┬──────────────┘
                 │ per-archipelago subgraph
                 ▼
  ┌─────────────────────────────┐
  │  detect_sub_islands(gen=0)  │  Leiden, resolution 1.5
  │  → 52 gen-1 islands         │  → dim_islands (gen=1)
  └──────────────┬──────────────┘
                 │ per-island subgraph
                 ▼
  ┌─────────────────────────────┐
  │  detect_sub_islands(gen=1)  │  Leiden, resolution 1.5
  │  → 207 gen-2 reefs          │  → dim_islands (gen=2)
  └──────────────┬──────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  ┌──────────┐    ┌──────────────┐
  │ stats +  │    │ backfill +   │
  │ PMI per  │    │ affinity     │
  │ gen      │    │              │
  └──────────┘    └──────┬───────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  reef_refine (10)   │  Iterative loyalty analysis
              │  moves misplaced    │  Recomputes stats/PMI/backfill/
              │  dims between       │  affinity after convergence
              │  sibling reefs      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  generate_names     │  Bottom-up via Claude API
              │  (9c)               │  Reefs → Islands → Archipelagos
              └─────────────────────┘
```

---

## 18. Common Modification Scenarios

### "I want to change the number of reefs"

Adjust `ISLAND_SUB_LEIDEN_RESOLUTION` (currently 1.5). Higher = more/smaller reefs. Lower = fewer/larger reefs. Then re-run phase 9.

### "I want to change which words count as reef members"

Adjust `REEF_MIN_DEPTH` (currently 2). This is a query-time filter, not a data-time one — no recomputation needed, just change the config and re-run explorer commands.

### "I moved some dims between reefs manually"

Run phase 9b to re-backfill denormalized columns and recompute affinity scores. Then run phase 9c to re-name reefs if desired.

### "I want to re-run refinement with different thresholds"

Adjust `REEF_REFINE_LOYALTY_THRESHOLD` or `REEF_REFINE_MIN_DIMS` in config.py, then run phase 10. This will recompute reef assignments and refresh everything downstream.

### "I want to add a new level to the hierarchy"

The system is designed for 3 generations (0, 1, 2). Adding a gen-3 would require: (1) calling `detect_sub_islands(con, parent_generation=2)`, (2) adding gen-3 to stats/PMI/backfill logic, (3) adding a denormalized column to dim_memberships. The Leiden-based subdivision is already parametric on parent_generation, so the detection code would work with minimal changes.
