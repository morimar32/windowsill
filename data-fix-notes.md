# Data Fix Investigation Notes

Investigation of the chess/polysemy retrieval gap reported in `data-fix-req.md`.
All queries run against the 233-reef build (`vector_distillery.duckdb`).

---

## 1. The Export ID Remapping

The downstream system (lagoon/shoal) uses **export IDs**, not database IDs. The
export process (`export.py` Phase 1) remaps reef IDs to contiguous 0-based values
sorted by archipelago → island → db_reef_id.

Key mapping:
- **Export reef 207** = **Database reef 208** ("deliberate behavior and cunning")
- **Export reef 117** = **Database reef 117** ("casual culture and recreation")
- **Export reef 180** = **Database reef 181** ("deception and obscurity")

All analysis below uses database IDs unless marked `(export)`.

---

## 2. Domain-Anchored Enrichment IS Working

The pipeline correctly creates domain-anchored senses for chess terms:

| Word | Chess sense embedding | Dims activated |
|------|----------------------|----------------|
| rook | `"classification: chess rook"` (castle.n.03) | 16 dims |
| knight | `"classification: chess knight"` (knight.n.02) | 19 dims |
| castle | `"classification: chess castle"` (castle.n.03 + castle.v.01) | 22 dims |
| castling | `"classification: chess castling"` (castle.v.01) | 19 dims |
| pawn | `"classification: chess pawn"` (pawn.n.03) | 14 dims |
| bishop | (no domain-anchored sense — unambiguous POS) | — |

These senses activate a coherent chess-cluster of dimensions: **dims 52, 417, 585,
656, 453, 374** recur across multiple chess senses.

The senses flow into `word_reef_affinity` via the UNION ALL in
`islands.py:compute_word_reef_affinity()`:

```sql
-- dim_memberships (base word)
UNION ALL
-- sense_dim_memberships WHERE is_domain_anchored = TRUE
→ deduplicate by MAX(z_score) per (word_id, dim_id)
→ aggregate by (word_id, reef_id)
```

Example: "rook" goes from 11 reef entries (base only) to 24 entries (with chess
sense enrichment). The chess sense adds reef 181 "deception and obscurity" at
n_dims=2 as rook's strongest reef connection.

---

## 3. The Chess Dimensions Are NOT in Reef 208

Reef 208 ("deliberate behavior and cunning") contains **dims 172, 193, 365, 486,
611, 621**.

The chess domain-anchored senses activate dims like 52, 417, 585, 656 — **zero
overlap** with reef 208's dimension set.

The chess-cluster dims were Leiden-clustered into reef 181 ("deception and
obscurity", dims 52, 101, 197, 585) and scattered across 13+ other reefs.

---

## 4. Critical Reframing: Reef 208 Is a LAW Reef

Querying each domain's collective reef affinity reveals:

| Domain | #1 Reef | total_swz | n_words touching |
|--------|---------|-----------|-----------------|
| **law** | **208 (deliberate behavior and cunning)** | **825.54** | **120** |
| chess | 181 (deception and obscurity) | 278.42 | 32 |
| music | 49 (geometric and musical) | 714.72 | — |
| botany | 12 (flowering plants gardening) | — | — |

**Reef 208 is fundamentally a law/deliberation reef.** Chess content scores there
in the downstream system because chess articles use legal/strategic discourse
register ("deliberate", "cunning", "calculated") — not because chess is
semantically close to reef 208.

The chess domain's actual home reef in the embedding space is **reef 181
("deception and obscurity")**, and the domain-anchored enrichment correctly
routes there.

---

## 5. Reef 181 vs Reef 208: Structural Distance

| | Reef 181 | Reef 208 |
|---|---|---|
| Name | deception and obscurity | deliberate behavior and cunning |
| Dims | 52, 101, 197, 585 | 172, 193, 365, 486, 611, 621 |
| Island | 38 — "flawed and deceptive" | 43 — "intentional conduct demeanor" |
| Archipelago | 2 — "negative states and abstract relations" | 3 — "qualities and evaluative attributes" |
| Edge weight | 0.074 (181→208) | 0.024 (208→181) |

Different archipelagos. Weak cross-reef edge.

---

## 6. Chess Words Are Extremely Sparse Across Reefs

| Word | Total reef entries | Depth-1 | Depth-2 | Max depth |
|------|-------------------|---------|---------|-----------|
| bishop | 11 | 11 | 0 | 1 |
| rook | 24 | 23 | 1 | 2 |
| knight | 26 | 22 | 4 | 2 |
| chess | 19 | 17 | 2 | 2 |
| chessboard | 27 | 22 | 5 | 2 |
| castling | 18 | 16 | 2 | 2 |
| pawn | 26 | 22 | 4 | 2 |

Compare to README example: "guitar activates 7/17 music-related dims." Chess
terms max out at depth 2 in any single reef.

---

## 7. Domain Concentration Analysis

Domains with high concentration map tightly to a single reef (good for
retrieval). Diffuse domains spread across many reefs (vulnerable to the class of
problem described here).

### Most concentrated (natural reef homes exist)

| Domain | Concentration | Top Reef |
|--------|-------------|----------|
| botany | 15.6% | flowering plants gardening |
| gardening | 13.3% | flowering plants gardening |
| plant | 13.0% | flowering plants gardening |
| acoustics | 12.2% | sound noise instruments |

### Most diffuse (highest gap risk)

| Domain | Concentration | n_reefs touched |
|--------|-------------|-----------------|
| plural | 3.1% | 185 |
| music | 3.2% | 191 |
| recording | 3.4% | 167 |
| surgery | 3.5% | 180 |
| government | 3.6% | 176 |
| boxing | 3.7% | 157 |
| chess | ~4% | ~160 |

---

## 8. Embedding Similarity Experiments

### Chess centroid vs reef centroids (cosine similarity)

The chess domain centroid (mean of all 42 chess domain-anchored sense embeddings)
was compared to centroids of all 233 reefs:

- Top reefs: animal traits/deception (0.8524), historical warfare (0.8519),
  cooking/food (0.8507)
- **Reef 181** (deception and obscurity): rank 10, sim=0.8447
- **Reef 208** (deliberate behavior and cunning): **rank 209**, sim=0.8038

The similarity range is very compressed (0.80–0.85), and reef 208 is near the
bottom.

### Individual chess senses vs reef centroids

| Word | Reef 208 rank | Reef 208 sim | Top reef |
|------|--------------|-------------|----------|
| rook (chess sense) | 205 | 0.7358 | historical warfare (0.7964) |
| knight (chess sense) | 189 | 0.7477 | discomfort/oppression (0.8028) |
| castling (chess sense) | 202 | 0.7529 | historical warfare (0.8121) |

### Base vs domain-anchored vs standard sense → reef 208 centroid

| Word | Base embedding | Domain "chess X" | Standard "X: gloss" |
|------|---------------|-----------------|---------------------|
| rook | 0.7809 | 0.7360 | 0.7107 |
| knight | 0.7987 | 0.7483 | 0.7043 |
| pawn | 0.8107 | 0.7593 | 0.7048 |

**Adding MORE context moves FURTHER from reef 208**, not closer. The base word
is closest, domain-anchored is second, gloss-enriched is furthest. This confirms
reef 208 is not a natural home for chess semantics regardless of enrichment
strategy.

### Chess centroid dimension activation

The chess domain centroid only exceeds the activation threshold in **8
dimensions** (out of 768). Those 8 map to:
- Reef 181 (deception and obscurity): 2 dims
- 6 other reefs: 1 dim each
- **Reef 208: 0 dims**

### Embedding arithmetic (domain direction vector)

Domain direction = mean(chess sense embeddings) - mean(base word embeddings for
same words). Norm: 10.64.

| Alpha | New dims activated | Reef 208 dims hit |
|-------|-------------------|-------------------|
| 1.0 | 60 | 0 |
| 2.0 | 113 | 0 |
| 3.0 | 166 | 0 |
| 5.0 | 283 | 2 (dims 486, 611) |
| 10.0 | 493 | 3 |

The direction is diffuse — it generates massive noise before hitting reef 208.
Not a viable targeting mechanism.

---

## 9. Orphaned Standard Sense Embeddings

**61,436 standard sense embeddings** exist in `word_senses` with
`is_domain_anchored = FALSE`. All have `sense_embedding` vectors but **zero**
have dim memberships (`total_dims = 0` for all).

Cause: Phase 5 pipeline ordering in `main.py`:
1. Line 168: `run_sense_analysis()` on standard senses → populates
   `sense_dim_memberships`
2. Line 201: `run_sense_analysis()` on domain-anchored senses → **overwrites**
   the table (clears and re-inserts)

The second call wipes the first call's results. Domain-anchored senses: 4,409
with 89,124 dim membership rows. Standard senses: 61,436 with **zero** dim
membership rows.

These are gloss-based sense embeddings like
`"classification: rook: one of two chess pieces that can move any number of
unoccupied squares in a direction parallel to the sides of the chessboard"` —
rich contextual embeddings that are currently dark.

---

## 10. Chess Domain Vocabulary

42 chess-domain senses across 34 unique words (avg ~20.9 dims per sense):

Includes: rook, knight, castle, castling, pawn, check, checkmate, queen, fork,
pin, stalemate, en passant, opening, etc.

Additionally, 21 vocabulary words contain "chess": chess, chessboard, chessman,
chess board, chess club, chess game, chess master, chess match, chess move, chess
opening, chess piece, chess player, chess set, etc. (Also catches archduchess,
duchess, grand duchess.)

All 279 domain labels exist as vocabulary words (none missing).

---

## 11. The Class of Problem

**Domain vocabulary routes to reef X (where the domain's concepts live in
embedding space). Domain content scores on reef Y (where the domain's discourse
register lives). When X ≠ Y, word-level BM25 can't find the content.**

This affects domains where:
1. Vocabulary is polysemous (chess pieces borrowed from birds/clergy/buildings)
2. Domain concentration is low (spread across many reefs)
3. Discourse register differs from lexical semantics (chess articles use
   legal/strategic language)

Music partially avoids this because musical vocabulary is less polysemous and
music content uses the same register as music vocabulary.

---

## 12. Brainstormed Approaches

### Approach 1: Domain Constellation Profiles (new export structure)

For each of the 279 WordNet domains, precompute an aggregated reef profile from
all domain members' reef affinities. Export as a new data structure.

Chess domain collective profile:
- Reef 181 (deception/obscurity): 278 total_swz, 32/34 words touch it
- Reef 49 (geometric/musical): 139 total_swz
- ...reef 208 at rank 65 with swz=36

The downstream scorer could use domain tags: when it sees "rook" has a chess
domain sense, overlay the chess domain's reef profile onto the query.

**Pro**: Works for ALL domains. Uses existing data. Clean architectural addition.
**Con**: Requires downstream scorer changes.

### Approach 2: Restore Standard Sense Dim Memberships

Fix Phase 5 pipeline ordering so standard sense analysis survives domain-anchored
analysis. This unlocks 61K gloss-based sense embeddings for dim analysis.

Then selectively include standard senses in `word_reef_affinity`:
- Only include senses that activate reefs the base word does NOT (novel signal)
- Filter to senses whose gloss matches a known domain (prevent polysemy inflation)

**Pro**: Massive increase in sense-level data. Low implementation cost (pipeline fix).
**Con**: Standard senses move further from reef 208 than domain-anchored senses
(cosine sim data). Adds breadth, not necessarily in the right direction.

### Approach 3: Gloss-Derived Contextual Reef Profiles

For each domain-anchored sense, extract content words from the gloss, look up
those words' reef affinities, aggregate into a "contextual reef profile."

Chess gloss content words (86 unique) aggregate to reef 208 at rank 18
(swz=72.0) — weak but present. Signal comes from gloss words like "attack",
"defend", "capture" whose individual reef 208 affinities reflect strategic/
deliberate contexts.

**Pro**: Bridges lexical↔discourse gap using definitional context.
**Con**: Signal is weak and noisy. Needs careful thresholding.

### Approach 4: Enriched Domain Compound Format

Change domain-anchored embedding from `"classification: {domain} {word}"` to
`"classification: {domain} {word}: {gloss}"`. Combines domain label + word +
definitional context.

**Pro**: Gives embedding model maximum disambiguation signal.
**Con**: Cosine sim data shows more context moves further from reef 208. May
improve reef 181 signal though (untested).

### Approach 5: Domain Direction Amplification (Embedding Arithmetic)

Compute domain shift vector = mean(domain sense embs) - mean(base word embs).
Apply at higher alpha to push further into domain space.

**Pro**: Systematic, applies to all domains.
**Con**: Chess domain direction is diffuse (norm 10.64). Needs alpha=5+ to
activate 2/6 reef 208 dims while generating 283 noise dims. Not efficient.

### Approach 6: Reef Edge Propagation

Chess terms' reef 181 affinity could propagate to reef 208 through the reef edge
(weight 0.074). For each word with domain senses, propagate affinity through
reef edges to reach neighboring reefs.

**Pro**: Uses existing reef edge infrastructure.
**Con**: Edge weight 0.074 is very low. Propagated signal would be negligible.

---

## 13. Key Insight

The fundamental gap is between **lexical semantics** (what a word means →
dimension activations → reef affinity) and **discourse semantics** (what
register/style a domain's content uses → chunk-level reef scoring).

No amount of word-level enrichment will make "rook" activate reef 208's
dimensions (172, 193, 365, 486, 611, 621) because those dimensions encode
behavioral/attitudinal concepts (shrewdness, cautiousness, furtiveness) that have
nothing to do with chess pieces. Chess articles score there because of HOW chess
is discussed, not WHAT chess is.

The most promising path is providing the downstream system with **domain-level
context** (Approach 1) so it can bridge the gap itself, combined with
**restoring the 61K orphaned sense embeddings** (Approach 2) for broader
sense coverage.

---

## 14. Artificial Reefs: The Concept

### The Idea

For each of the 279 WordNet domains, create a dedicated **artificial reef**
containing just that domain's words. Hang it off the parent island of the
domain's best-matching natural reef (making it a sibling). Mark it as artificial
in the database. This gives every domain a dedicated routing channel without
polluting natural reefs.

### Why Not Just Add Words to Existing Reefs?

Initial analysis considered adding missing domain words to their best-matching
natural reef directly. This was rejected because:

1. **Containment is already high.** 210/280 domains (75%) already have ≥90%
   containment in their best reef. Chess: 32/34 words (94%) already in reef 181.
   Very few words would actually be "added."

2. **False positives.** Adding chess words to reef 181 (deception and obscurity)
   means deception/obscurity queries would surface chess vocabulary. Reef 181's
   characteristic words are "achlamydeous", "hermaphroditus", "abstrusity",
   "disingenuously" — nothing chess-related. Strengthening a conceptually wrong
   connection creates noise.

3. **BM25 score explosion for new tiny reefs.** A new 34-word chess reef produces
   BM25 scores of 29,000–39,000 (3–4x higher than any natural entry) because
   BM25's length normalization (`n_words / avg_reef_words` = 34/5733 = 0.006)
   vanishes for tiny reefs.

4. **The real gap isn't missing words — it's weak connections.** Chess words ARE
   in reef 181 already. The problem is depth (n_dims=1–2) producing weak BM25
   scores that don't survive background subtraction.

### The Sibling Approach (Refined)

Instead of augmenting existing reefs:

1. For each domain, find the natural reef with the highest word overlap
2. Create a NEW artificial reef containing JUST the domain's words
3. Hang it off the same parent island as the best-matching natural reef
4. Mark the reef as artificial in the database (`is_artificial` column)

This gives:
- **Clean separation**: No false positives. The artificial reef is its own entity.
- **Structural kinship**: Shares a parent island with the semantically closest
  natural reef. Sibling reef edges are computable.
- **Proper hierarchy address**: arch → island → artificial reef, fits the
  existing 3-generation model.
- **Automatic downstream activation**: If "rook" has a BM25 entry for the chess
  artificial reef, then chess article chunks (containing chess vocabulary) would
  naturally activate that reef through the existing BM25 scoring pipeline. No
  special domain-detection logic needed.

---

## 15. Artificial Reef Placement Plan

### Domain → Sibling Reef → Parent Island → Archipelago

Key examples:

| Domain | Words | Sibling Reef | Parent Island | Archipelago |
|--------|-------|-------------|---------------|-------------|
| chess | 34 | 181 (deception/obscurity) | 38 (flawed and deceptive) | 2 (negative states) |
| music | 181 | 70 (physical motion/oscillation) | 15 (physical world navigation) | 1 (human activities) |
| cooking | 172 | 2 (biochemical properties) | 0 (life sciences/chemistry) | 0 (scientific terminology) |
| law | 151 | 98 (livestock/animal husbandry) | 19 (traditional labor domains) | 1 (human activities) |
| sport | 143 | 96 (athleticism/physical strength) | 19 (traditional labor domains) | 1 (human activities) |
| medicine | 79 | 200 (literary artistic forms) | 41 (professional domains) | 3 (evaluative attributes) |
| botany | 39 | 12 (flowering plants/gardening) | 1 (biological classification) | 0 (scientific terminology) |
| architecture | 47 | 68 (historical/monumental themes) | 13 (natural environments) | 0 (scientific terminology) |

### Parent Island Load Distribution

Some parent islands would accumulate many artificial reef children:

| Parent Island | Natural Reefs | +Artificial | Total |
|--------------|--------------|-------------|-------|
| traditional labor domains | 6 | 14 | 20 |
| physical world navigation | 6 | 13 | 19 |
| life sciences and chemistry | 7 | 9 | 16 |
| professional domains artifacts | 8 | 8 | 16 |
| visual perception and appearance | 5 | 7 | 12 |

---

## 16. Export Feasibility: The u8 Ceiling

The export packs `arch(2)|island(6)|reef(8)` into a u16 hierarchy_addr.
8 bits for reef = **256 max**. With 233 natural reefs, only **23 slots** remain
for artificial reefs.

### Domain counts by minimum word threshold

| Min words | Domains qualifying | Total reefs | Status |
|-----------|-------------------|-------------|--------|
| 1 | 280 | 513 | OVER |
| 5 | 143 | 376 | OVER |
| 10 | 85 | 318 | OVER |
| 15 | 62 | 295 | OVER |
| 20 | 49 | 282 | OVER |
| 30 | ~23 | ~256 | TIGHT FIT |

### Domain size distribution

| Threshold | Domains |
|-----------|---------|
| ≥ 1 word | 280 |
| ≥ 5 words | 143 |
| ≥ 10 words | 85 |
| ≥ 20 words | 49 |
| ≥ 50 words | 19 |
| ≥ 100 words | 6 |

### Options for the u8 constraint

1. **Be selective**: Pick ~23 most impactful domains. Selection criteria could
   combine size and diffuseness: `size × (1 - concentration)`. Chess (34 words,
   ~4% concentration) would score well on this metric.

2. **Widen the format**: Move to a separate u16 reef_id field instead of the
   packed hierarchy_addr. Breaking change to the downstream v2 binary format.

3. **Two-tier**: Keep natural reefs in the u8 space, put artificial reefs in a
   parallel namespace that the scorer handles as an overlay.

---

## 17. BM25 Score Calibration for Artificial Reefs

Artificial reefs have no natural dimensions. The BM25 formula
`tf = n_dims / reef_total_dims` needs synthetic values.

### The length normalization problem

| Scenario | n_words | n_dims | tf | BM25 for rook | vs top natural |
|----------|---------|--------|-----|---------------|---------------|
| Tiny reef (raw) | 34 | 1 | 1.0 | 39,000 | **3.9x** |
| Neutralized length | avg (5733) | 1 | 1.0 | 18,486 | **1.83x** |
| Neutralized + synthetic dims | avg (5733) | 3 (median) | 0.33 | ~8,800 | **0.87x** |

### Recommended calibration

Set artificial reef `n_words = avg_reef_words` (neutralizes length normalization)
and `reef_total_dims = 3` (median across natural reefs). This produces BM25
scores **comparable to top natural entries** (~8,800 vs ~10,000 for rook).

The artificial reef entry would be **strong but not dominant** — ranking among
the top few BM25 entries for each word rather than overwhelming everything.

Full simulated scores with `n_words=avg, n_dims=1, reef_total_dims=3`:

| Word | Artificial BM25 | Top Natural BM25 | Ratio |
|------|----------------|-----------------|-------|
| rook | ~8,800 | 10,096 (deception/obscurity) | 0.87 |
| knight | ~8,200 | 6,466 (dinosaurs/neurons) | 1.27 |
| chess | ~11,100 | 12,762 (deception/obscurity) | 0.87 |
| checkmate | ~8,500 | 9,745 (deception/obscurity) | 0.87 |

---

## 18. How Artificial Reefs Solve the Chess Problem (End-to-End)

### Query side
User searches "rook movements":
1. Scorer looks up "rook" → finds BM25 entries including chess artificial reef
2. Scorer looks up "movements" → no chess artificial reef entry
3. Combined query profile includes chess artificial reef from "rook"'s signal
4. After background subtraction, chess artificial reef has positive z-score

### Chunk side (at ingestion)
Chess article chunk "The rook is the most powerful piece after the queen":
1. Scorer processes chunk text through BM25
2. Words "rook", "piece", "queen" all have chess artificial reef BM25 entries
3. Combined chunk profile naturally includes chess artificial reef activation
4. Stored alongside natural reef profile (reef 208, reef 181, etc.)

### Retrieval
Query reef profile (has chess artificial reef) overlaps with chunk reef profile
(also has chess artificial reef) → chess article surfaces in results.

**Key insight**: This works through the existing BM25 pipeline with no special
domain-detection logic. The BM25 entry IS the domain signal. Both queries and
chunks independently activate the same artificial reef through shared vocabulary.

---

## 19. Open Questions

1. **Selection criteria**: If limited to ~23 slots, which domains get artificial
   reefs? Pure size favors colloquialism (222 words) over chess (34). A combined
   `size × (1 - concentration)` metric would balance size with need (diffuse
   domains benefit more).

2. **IDF interaction**: Adding artificial reefs to the corpus increases total
   reef count, which slightly increases IDF for all words. Adding one reef entry
   per word barely moves individual IDF (delta -0.03 to -0.08). If 50 new reefs
   are added, IDF actually *increases* for most words (+0.11 to +0.16) because
   the denominator (total reefs) grows faster than any word's reef count.

3. **Reef edges**: Should artificial reefs participate in the reef edge graph?
   Computing containment/lift between artificial reefs and natural reefs would
   require defining "member words" for the artificial reef — which is just the
   domain word list. This is straightforward. Edges between artificial reefs and
   their sibling natural reefs would be especially meaningful.

4. **Background model**: The export's background model (phase 10) samples random
   word sets to compute per-reef z-score normalization. Artificial reefs need to
   be included in this sampling. Since artificial reefs are small and domain-
   specific, random word samples would mostly NOT activate them, producing low
   background means — which means domain vocabulary activation would produce
   high z-scores after subtraction. This is desirable behavior.

5. **Naming**: Each artificial reef would be named after its WordNet domain:
   "chess", "music", "cooking", "law", etc. Simple and interpretable.

---

## 20. Agreed Next Steps

1. **Restore orphaned standard sense dim memberships** (§9). Fix Phase 5
   pipeline ordering so `run_sense_analysis()` on standard senses isn't wiped
   by the subsequent domain-anchored analysis. If the data doesn't play nice,
   find a different solution rather than re-orphaning.

2. **Implement artificial reefs** (§14–18). Create domain-specific artificial
   reefs as siblings of best-matching natural reefs. Mark as artificial in the
   database. Calibrate BM25 with synthetic `n_words=avg, reef_total_dims=3`.

3. **Resolve the u8 constraint** (§16). Either select top ~23 domains, widen
   the export format, or use a two-tier approach.

---

## 21. The Pivot: Artificial Dimensions, Not Artificial Reefs

### The Scale Mismatch Problem

Artificial reefs with < 100 words would compete against natural reefs with
several thousand words. This is a fundamental scale mismatch:

- Natural reefs: 3,000–18,000 words
- Artificial chess reef: 34 words
- BM25 needs synthetic calibration (§17) to avoid score explosion
- Reef edge computation, background model, and IDF all need special-casing
- The u8 ceiling (§16) limits us to ~23 artificial reefs

### The Insight: Inject Dimensions, Not Reefs

Instead of creating artificial reefs that bolt on AFTER clustering, create
**artificial dimensions** that participate in clustering BEFORE reef creation.

For each qualifying domain, create a new `dim_id ≥ 768` containing just the
domain's words as members. Inject these into the pipeline **before Phase 7**
(Leiden community detection), so they participate in organic clustering:

1. Artificial dim has 34 members (chess) vs natural dims with 2,400–4,300
2. Hypergeometric z-score handles the scale mismatch NATURALLY
3. Leiden clustering decides where the artificial dim lands — no manual
   placement needed
4. Noise recovery acts as a safety net (if Leiden doesn't cluster it,
   it gets recovered to the nearest sibling)
5. Everything downstream (affinity, BM25, export) works unchanged
6. No special-casing anywhere in the pipeline

### Why Hypergeometric Z-Scores Handle the Size Difference

The Jaccard matrix between dimensions uses hypergeometric z-scoring:
- Expected intersection of 34-word dim with 3,000-word dim: < 1 word
- If 24 of the 34 chess words appear in dim 52 (3,008 members), that's
  a z-score of **28.2** (expected overlap: 0.7 words)
- Clustering threshold: ISLAND_JACCARD_ZSCORE = 3.0
- The z-score normalizes for dimension size, so small artificial dims
  can cluster with large natural dims on equal footing

---

## 22. Empirical Analysis: Chess Artificial Dim vs Natural Dims

### Domain-Anchored Sense Concentrations

The 34 chess domain words (via domain-anchored sense dim memberships) show
dramatic concentration in natural dimensions:

| Natural Dim | Chess Words | Dim Size | Hyper-Z | Current Reef |
|-------------|-------------|----------|---------|-------------|
| 52 | 24/34 (71%) | 3,008 | **28.2** | 181 (deception/obscurity) |
| 457 | 23/34 (68%) | 3,216 | **27.0** | 0 (medical conditions) |
| 585 | 22/34 (65%) | 3,381 | **25.7** | 181 (deception/obscurity) |
| 453 | 21/34 (62%) | 3,529 | **24.4** | 73 (vehicles/riding) |
| 465 | 20/34 (59%) | 3,386 | **23.4** | 187 (obsession/instinct) |
| 58 | 19/34 (56%) | 2,619 | **22.5** | 49 (geometric/musical) |
| 417 | 19/34 (56%) | 3,976 | **21.8** | 99 (warfare/transport) |
| 374 | 17/34 (50%) | 3,142 | **19.7** | 107 (discomfort/oppression) |
| 656 | 17/34 (50%) | 3,480 | **19.6** | 37 (medical suffixes) |
| 229 | 15/34 (44%) | 3,234 | **17.3** | 103 (cooking/food prep) |
| 317 | 15/34 (44%) | 3,332 | **17.2** | 77 (formed structures) |
| 616 | 14/34 (41%) | 3,630 | **15.9** | 70 (physical motion) |

ALL z-scores are 4-9x above the clustering threshold of 3.0.

### Base Membership Comparison (for reference)

Without domain-anchored enrichment, chess words scatter weakly:

| Natural Dim | Chess Words | Dim Size | Hyper-Z | Current Reef |
|-------------|-------------|----------|---------|-------------|
| 616 | 8/34 (24%) | 3,630 | **8.6** | 70 (physical motion) |
| 651 | 6/34 (18%) | 3,313 | **6.4** | 122 (texture/tactile) |
| 466 | 6/34 (18%) | 3,201 | **6.4** | 154 (flow/conductivity) |

Domain-anchored senses produce **3x more concentration** than base
embeddings. Max 24/34 words (z=28.2) vs max 8/34 words (z=8.6).

### The Scattered Signal Problem

The artificial chess dim connects to **17 natural dims across 16 different
reefs**. The chess signal is real but shattered across unrelated semantic
neighborhoods. This is exactly why chess words can't find chess content —
no single reef accumulates enough signal.

Aggregate signal by reef:

| Reef | Name | # Linked Dims | Total Chess Overlaps |
|------|------|--------------|---------------------|
| 181 | deception and obscurity | 2 (dims 52, 585) | 46 |
| 0 | medical conditions | 1 (dim 457) | 23 |
| 73 | vehicles and riding | 1 (dim 453) | 21 |
| 187 | obsession and instinct | 1 (dim 465) | 20 |
| 49 | geometric and musical | 1 (dim 58) | 19 |
| 99 | warfare and transport | 1 (dim 417) | 19 |
| 107 | discomfort/oppression | 1 (dim 374) | 17 |
| 37 | medical suffixes | 1 (dim 656) | 17 |

Reef 181 has the strongest aggregate pull (2 dims, 46 overlaps).

### What Leiden Would Do

The artificial chess dim would form edges (z >> 3) with dims in 16 reefs.
Its strongest connections are to reef 181's dims (52 and 585). Leiden would
either:

1. Assign it to reef 181 (most connected community) — chess words gain
   additional n_dims in that reef
2. Keep it as a noise dim — noise recovery assigns it to the nearest sibling
   of reef 181, which is still in the chess neighborhood

Either outcome routes chess words to the correct semantic area.

---

## 23. Cross-Domain Clustering: The Exciting Discovery

### Natural Hub Dimensions

Natural dimensions already serve as convergence points for multiple game/sport
domains. These are the "hub dims" where game semantics concentrates:

| Dim | # Domains | Total Words | Top Domains |
|-----|----------|-------------|-------------|
| 551 | 10 | 128 | sport(44), baseball(28), football(23), basketball(7), chess(5), golf(5) |
| 58 | 9 | 111 | golf(26), baseball(25), chess(19), tennis(10), sport(9), cricket(7) |
| 478 | 9 | 91 | baseball(24), sport(20), golf(12), game(9), football(8) |
| 417 | 8 | 110 | baseball(53), chess(19), sport(16) |

### The Organic Reef Hypothesis

If we create artificial dims for chess (34 words), baseball (89), football
(~40), golf (~30), basketball (~15), etc., these artificial dims would ALL
have strong Jaccard with the same natural hub dims (551, 58, 478, 417).

Leiden could discover that these artificial dims form a community together —
creating a **new organic "games and competition" reef** that never existed
because the signal was too scattered across individual domain words.

This is the most elegant possible outcome:
- Multiple domain artificial dims cluster with each other and with natural
  hub dims
- Leiden creates a reef that represents "games/sport/competition" as a
  genuine semantic community
- Chess words, baseball words, football words all route to this reef
- The downstream system can distinguish chess from baseball through the
  OTHER reef entries (chess also activates strategy reefs, baseball
  activates physical motion reefs, etc.)

### Cross-Domain Word Overlap

Related domains share words that would strengthen inter-domain edges:

| Domain A | Domain B | Shared Words |
|----------|----------|-------------|
| baseball | sport | 7 |
| football | sport | 6 |
| golf | sport | 6 |
| baseball | football | 4 |
| game | sport | 3 |

---

## 24. Scale and Feasibility

### Domain Size Distribution

| Size Bucket | # Domains | Total Words |
|-------------|----------|-------------|
| < 3 words | 100 | 137 |
| 3–4 words | 37 | 126 |
| 5–9 words | 58 | 381 |
| 10–19 words | 36 | 492 |
| 20–49 words | 30 | 940 |
| 50–99 words | 13 | 928 |
| 100+ words | 6 | 995 |

The distribution is heavily right-skewed: 137/280 domains (49%) have < 5
words. A minimum threshold is needed to avoid degenerate artificial dims.

Recommended minimum: **5 words** (143 qualifying domains). Could go to 10
(83 domains) for a more conservative approach.

### Jaccard Matrix Growth

Current matrix: 768 × 768 = ~295K pairs.

| Min Words | New Dims | Total Dims | Matrix Pairs | Growth |
|-----------|---------|------------|-------------|--------|
| 5 | 143 | 911 | ~415K | +41% |
| 10 | 83 | 851 | ~362K | +23% |
| 20 | 49 | 817 | ~333K | +13% |

All manageable. The Jaccard computation is already the fastest part of Phase 7.

### Natural Dimension Size Context

| Stat | Natural Dims |
|------|-------------|
| Min | 2,422 |
| P10 | 2,985 |
| P25 | 3,156 |
| Median | 3,347 |
| P75 | 3,508 |
| P90 | 3,683 |
| Max | 4,320 |

Artificial dims (5–222 members) are 10–100x smaller than natural dims.
Hypergeometric z-scoring handles this naturally.

### The u8 Ceiling Dissolves

Artificial dimensions are NOT artificial reefs. They participate in organic
clustering and land in existing or new reefs through Leiden. The number of
REEFS after clustering may increase modestly (some new communities may form)
but not by 143. The u8 ceiling is no longer the binding constraint. Per user
confirmation, u8→u16 is acceptable anyway since nothing is in production.

---

## 25. Implementation Details

### z-Score for Artificial Dim Members

Natural dim memberships use z ≥ 2.0 (threshold). Options for artificial dims:

| Approach | z-Score | Rationale |
|----------|---------|-----------|
| Floor | 2.0 | Conservative, minimum membership |
| Median | 2.28 | Middle of natural distribution |
| Fixed per-word | varies | Use domain-anchored sense z-score as proxy |

Recommendation: **z = 2.0 (floor)**. Artificial dim provides floor-level
domain routing. Natural dim memberships (up to 5.7) dominate the affinity
calculation. The artificial dim acts as a tiebreaker/bootstrap.

### dim_weight for Artificial Dims

Formula: `dim_weight = -log2(max(universal_pct, 0.01))`

For chess (34 words, ~2 universal words like "open", "black"):
- universal_pct = 2/34 = 0.059
- dim_weight = -log2(0.059) ≈ 4.08
- Natural range: [1.04, 4.11] — chess lands at the high end (correct:
  domain-specific dims are highly discriminative)

For domains with zero universal words:
- Clipped to 0.01 → dim_weight = 6.64 (above natural max of 4.11)
- May need a cap at max natural dim_weight to prevent overweighting

### Pipeline Insertion Point

New **Phase 6.5** between enrichment and islands:

```
Phase 6:   Enrichment (existing)
Phase 6.5: Artificial Dimensions (NEW)
  1. Query domains with ≥ N words from word_senses (is_domain_anchored)
  2. For each: create dim_stats entry (dim_id ≥ 768, is_artificial = TRUE)
  3. Create dim_memberships rows (word_id, dim_id, z_score = 2.0)
  4. Compute dim_stats fields (n_members, universal_pct, dim_weight)
Phase 7:   Islands (Leiden sees 768+M dims, clusters organically)
```

### Database Schema Additions

```sql
-- dim_stats: add column
ALTER TABLE dim_stats ADD COLUMN is_artificial BOOLEAN DEFAULT FALSE;

-- dim_memberships: add column
ALTER TABLE dim_memberships ADD COLUMN is_artificial BOOLEAN DEFAULT FALSE;

-- New: artificial dim metadata
-- (domain source tracked via dim_stats or new table)
```

---

## 26. Updated Next Steps (Supersedes §20)

1. **Restore orphaned standard sense dim memberships** (§9). Fix Phase 5
   pipeline ordering. If data doesn't play nice, find alternative solution.

2. **Implement artificial dimensions** (§21–25). Create domain-defined dims
   (dim_id ≥ 768) before Phase 7 so they participate in organic Leiden
   clustering. Minimum domain size threshold of 5–10 words.

3. **No special-casing needed**: No BM25 calibration hacks, no u8 ceiling
   worries, no manual reef placement. The existing pipeline handles
   everything through hypergeometric z-scoring and community detection.

4. **u8→u16 confirmed acceptable**: Export format can widen if new reefs
   are created through organic clustering. Nothing is in production.

5. **Validate after implementation**: Run the full pipeline, check whether
   game-domain artificial dims cluster together into a new reef, verify
   chess words gain meaningful reef affinity, test retrieval in shoal.
