# Update Plan: Chess Convergence Fix + Pipeline Consolidation

## Problem Summary

The 207-reef taxonomy has no reef for games/strategy/competition. Chess vocabulary (9 words, 9 different reefs, zero convergence) is structurally invisible to reef-based retrieval. Root cause: four chess-coherent dimensions (52, 417, 58, 585) are all classified as noise — two in small islands below the subdivision threshold, two as Leiden singletons in large islands. Meanwhile, the UNION ALL sense enrichment inflates polysemous words' reef counts 3-7x without improving convergence.

## Solution: Three Combined Approaches

### Approach 1: Lower subdivision threshold + noise dim recovery

**Threshold change:** `ISLAND_MIN_DIMS_FOR_SUBDIVISION` from 10 to 2. Data-justified: all 10 small islands (4-9 dims) have internal Jaccard (avg 0.023) comparable to existing reefs (2-dim reefs avg 0.038, 3-dim reefs avg 0.028). `ISLAND_MIN_COMMUNITY_SIZE=2` already prevents single-dim reefs. This captures chess dims 52+585 (in 8-dim island 38).

**Noise recovery:** After Leiden partitioning, orphan dims (singleton communities) in large islands get assigned to their nearest sibling reef by Jaccard affinity, if above a minimum threshold. This captures chess dims 417+58 (noise singletons in islands 19 and 8). Currently 70 noise dims in large islands; many are meaningful signals stranded by Leiden.

### Approach 2: Domain-anchored sense enrichment

Use WordNet `topic_domains` to create short synthetic compound embeddings. For words with domain-linked senses (2,642 ambiguous words, 438 domains), generate compounds like `"chess rook"`, `"chess pawn"`, `"chess knight"`. These produce sharper activations than full-gloss sense embeddings — tested: `"chess piece"` hits 4/4 chess dims (z 2.0-3.3) vs rook's gloss embedding hitting 3/4 with lower z-scores.

Store domain-anchored sense activations alongside (not replacing) existing full-gloss sense data. The full-gloss senses remain useful for exploration and disambiguation; the domain compounds are specifically for scoring enrichment.

### Approach 3: Replace UNION ALL with domain-aware affinity

Replace the naive `UNION ALL` in `compute_word_reef_affinity()` with domain-aware logic:
- For words WITH domain-anchored senses: use domain-anchored sense dims only (not all WordNet senses)
- For words WITHOUT domain-anchored senses: use word-level dims only (no sense enrichment)

This eliminates the 3-7x reef inflation (rook: 7 reefs → 44, queen: 13 → 96) while providing targeted enrichment where domain data exists.

---

## Pipeline Consolidation

### Current ordering (messy)

```
2, 3, 4, 4b, 4c, 5, 5b, 5c, 6, 6b, 9, 9b, 9d, 9e, 10, 9f, 9g, 9c, 11, 7, explore
```

22 entries with inconsistent numbering, sub-letter phases, out-of-order execution (9→9b→9d→9e→10→9f→9g→9c→11→7), and redundant work (9b duplicates the tail of 9; phase 10 recomputes backfill/affinity/valence/pos/edges that 9d-9g also compute).

### New ordering

| Phase | Name | What it does | Old phases |
|-------|------|-------------|------------|
| 1 | Vocabulary | Word list curation | 2 |
| 2 | Embeddings | Word embedding generation | 3 |
| 3 | Database | Schema + bulk insert + POS/components backfill + word hashes | 4, 4b, 4c |
| 4 | Analysis | Dimension thresholds + word counts + specificity + pair overlap | 5, 6 |
| 5 | Senses | Sense embeddings + sense dim analysis + **domain-anchored compounds (NEW)** | 5b, 5c |
| 6 | Enrichment | POS enrichment, compound contamination, compositionality, dim abstractness/specificity, sense spread, negation, valence | 6b |
| 7 | Islands | Jaccard matrix, 3-generation Leiden detection, **noise recovery (NEW)** | 9 |
| 8 | Refinement | Reef loyalty refinement (iterative dim reassignment) | 10 |
| 9 | Reef Analytics | Backfill hierarchy, **domain-aware affinity (CHANGED)**, reef IDF, arch concentration, reef valence, POS composition, reef edges, composite weight | 9b, 9d, 9e, 9f, 9g |
| 10 | Naming | LLM-based naming (reefs → islands → archipelagos) | 9c |
| 11 | Finalization | Morphy variant expansion + DB maintenance (integrity, indexes, optimization) | 11, 7 |

`explore` becomes a standalone command, not a pipeline phase. `export` remains a separate script.

### Key dependency changes

- Phase 5 (senses) now comes before phase 6 (enrichment), which is correct: sense_spread in phase 6 needs word_senses from phase 5.
- Phase 6 must complete before phase 7 (islands) because `dim_weight` (from `compute_dimension_abstractness`) is used by `compute_word_reef_affinity` downstream.
- Phase 8 (refinement) subsumes the redundant recomputation that was split across 9b/9d/9f/9g. After refinement converges, all post-island analytics run once in phase 9.
- Phase 9b is eliminated — its work (backfill + affinity) now runs once in phase 9 (after refinement), not twice.

---

## Detailed Changes by File

### `config.py`

```python
# Change
ISLAND_MIN_DIMS_FOR_SUBDIVISION = 2          # was 10

# Add
NOISE_RECOVERY_MIN_JACCARD = 0.01            # min avg Jaccard to assign orphan dim to sibling reef

# Remove hardcoded counts (make dynamic)
# N_REEFS = 207   → computed from DB
# N_ISLANDS = 52  → computed from DB
# N_ARCHS = 4     → computed from DB
```

`N_REEFS`, `N_ISLANDS`, `N_ARCHS` become computed at runtime from `island_stats` rather than hardcoded. The reef count will change with the threshold lowering and noise recovery (expect ~220-240 reefs, well within the 256 limit of the export's 8-bit reef ID).

### `islands.py`

**`detect_sub_islands()`** — add noise dim recovery after Leiden:

After the Leiden partitioning loop assigns communities, add a recovery step for noise dims (those in singleton communities → `island_id = -1`):

1. For each noise dim in the current parent island, compute its average Jaccard to each non-noise sibling community's dims
2. If the best sibling's average Jaccard exceeds `NOISE_RECOVERY_MIN_JACCARD`, assign the noise dim to that community
3. Otherwise, leave it as noise

This reuses the same Jaccard data already loaded for the Leiden graph construction. The logic is similar to `reef_refine._compute_dim_loyalty()` but simpler (one-shot assignment, no iteration).

**`compute_word_reef_affinity()`** — replace UNION ALL:

```sql
-- OLD: naive UNION ALL of all senses
WITH all_activations AS (
    SELECT word_id, dim_id, z_score FROM dim_memberships
    UNION ALL
    SELECT ws.word_id, sdm.dim_id, sdm.z_score
    FROM sense_dim_memberships sdm
    JOIN word_senses ws ON sdm.sense_id = ws.sense_id
)

-- NEW: domain-anchored senses only
WITH all_activations AS (
    SELECT word_id, dim_id, z_score FROM dim_memberships
    UNION ALL
    SELECT ws.word_id, sdm.dim_id, sdm.z_score
    FROM sense_dim_memberships sdm
    JOIN word_senses ws ON sdm.sense_id = ws.sense_id
    WHERE ws.is_domain_anchored = TRUE
)
```

Words without domain-anchored senses get no sense enrichment (just their base dim_memberships). Words with domain-anchored senses get focused enrichment from the synthetic compound embeddings.

**`run_island_detection()`** — remove the trailing `backfill_membership_islands()` and `compute_word_reef_affinity()` calls. These now run in phase 9 (after refinement), not in phase 7 (during island detection). This eliminates the current double-computation.

### `embedder.py`

**New function: `build_domain_compound_texts(con)`**

1. Load all words that have entries in `word_senses`
2. For each word, query WordNet synsets and check `topic_domains()` (and `usage_domains()`)
3. For senses with a domain link, extract the domain lemma (e.g., `chess.n.02` → `"chess"`)
4. Build synthetic compound: `f"{config.SENSE_EMBEDDING_PREFIX}{domain_word} {word}"`
5. Return list of dicts: `{sense_id, word_id, pos, synset_name, domain, text}`

**Updated `embed_senses()` or new `embed_domain_compounds()`:**

Embed the synthetic compounds through the same model. Store in `word_senses` with `is_domain_anchored = TRUE`.

### `database.py`

**Schema change to `word_senses`:**

Add column `is_domain_anchored BOOLEAN DEFAULT FALSE`. Domain-anchored senses created from synthetic compounds get `TRUE`; existing full-gloss senses keep `FALSE`.

No other schema changes needed — `sense_dim_memberships` works as-is since domain-anchored senses are just a new type of sense entry.

### `reef_refine.py`

**Remove redundant post-convergence recomputation.** Currently `run_reef_refinement()` calls `backfill_membership_islands`, `compute_word_reef_affinity`, `compute_reef_valence`, `compute_hierarchy_pos_composition`, `compute_hierarchy_specificity`, `compute_reef_edges` at the end. In the consolidated pipeline, all of these run once in phase 9 (reef analytics) after refinement completes. The refinement module should only handle the iterative dim reassignment + recompute island_stats/characteristic_words per iteration (which it already does).

### `export.py`

**Remove hardcoded count assertions.** Replace:

```python
assert len(reefs) == config.N_REEFS, ...
assert len(islands) == config.N_ISLANDS, ...
assert len(archs) == config.N_ARCHS, ...
```

With dynamic counts read from the DB. Verify the reef count fits in 8 bits (max 255) and island count fits in 6 bits (max 63) for the `pack_hierarchy_addr` encoding.

### `main.py`

**Complete rewrite of phase dispatch.** Replace the current 22-entry `PHASE_ORDER` and `run_phase*` functions with 11 clean phases. Each phase function consolidates the work of its predecessors. The `--phase` and `--from` arguments work with the new numbering. `explore` becomes a separate entry point (not a pipeline phase).

### `meta_rel.py`

No changes needed. `MetaRelScorer` reads from `word_reef_affinity` and `reef_edges`, which will be populated correctly by the updated pipeline.

---

## Implementation Order

1. **`database.py`** — Add `is_domain_anchored` column to `word_senses` schema. Remove hardcoded count references if any.
2. **`config.py`** — Lower threshold, add noise recovery constant, remove hardcoded N_REEFS/N_ISLANDS/N_ARCHS.
3. **`embedder.py`** — Add `build_domain_compound_texts()` function.
4. **`islands.py`** — Noise recovery in `detect_sub_islands()`. Domain-aware affinity in `compute_word_reef_affinity()`. Remove trailing backfill/affinity from `run_island_detection()`.
5. **`reef_refine.py`** — Remove redundant post-convergence recomputation.
6. **`export.py`** — Dynamic counts, remove hardcoded assertions.
7. **`main.py`** — Consolidate into 11 clean phases.

Steps 1-6 are functional changes. Step 7 is restructuring. Each step is independently testable — the DB rebuild from scratch means no migration concerns.

---

## Verification

After rebuild, re-run the queries from `data_issue.md`:

- [ ] "rook" BM25 spread should be wider than 0.25 (chess reef should dominate)
- [ ] "chess rook pawn knight bishop" should converge on a chess-relevant reef
- [ ] "rook movement" confidence should be >> 0.0001
- [ ] 9 chess words should share at least one common top-2 reef
- [ ] "tower" should have some chess-related signal via domain enrichment
- [ ] Polysemous word reef counts should NOT inflate 3-7x (rook should be ~10-15, not 44)
