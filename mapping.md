# Taxonomy Redesign: Towns Layer via Wikipedia Categories

**Date:** 2026-02-25
**Status:** Exploration / pre-planning
**Context:** The current 258-reef taxonomy is too coarse. Sub-reefs exist but have garbage labels and are ignored by Lagoon's scorer. We need a named, pre-defined subdivision layer ("towns") between islands and reefs.

---

## The Problem We're Solving

### Current hierarchy (broken)
```
Archipelagos (5-8, Leiden-clustered)
  └── Islands (258, pre-defined domains) ← scoring happens here
        └── Sub-reefs (2,534, Leiden-clustered) ← exported but IGNORED by Lagoon
```

### Why it's broken
1. **258 buckets is too coarse.** "Sport" contains 3,809 words spanning 24 original domains (hockey, tennis, boxing, baseball...). The scorer can only say "this is sport" — it can't distinguish hockey from tennis.

2. **Sub-reefs have no meaningful names.** Labels like `classification_people_animal` and `nonsensicality_inconclusiveness_unexpansive` are auto-generated from top characteristic words. Useless for display, reasoning, or debugging.

3. **Sub-reefs are ignored at scoring time.** Lagoon accumulates weights at the reef (= island = domain) level. The `_sub_reef_id` field in `word_reefs.bin` is literally prefixed with an underscore in the scoring loop. All the Leiden clustering work produces information that does nothing.

4. **XGBoost trains at the island level.** When 24 sport domains are consolidated into one before XGBoost training, the classifier learns an absurdly broad decision boundary. Words like "action", "combat", "furniture", and "judicial" get classified as sport because the positive training set is so diverse.

5. **Test results are flattering.** The Shoal test suite only needs to pick the right island out of 258 — a trivially easy classification task. The system looks good because the corpus doesn't challenge it at the granularity where it's blind.

### Proposed hierarchy
```
Archipelagos (5-8, Leiden-clustered from islands)
  └── Islands (258, pre-defined — keep the consolidated domains)
        └── Towns (NEW, ~1,000-2,500, pre-defined from Wikipedia categories)
              └── Reefs (Leiden-clustered within towns)
```

Towns are the new primary scoring unit. XGBoost trains at the island+town level (e.g., "sport:ice hockey" not just "sport"). Lagoon scores at the town level.

---

## Wikipedia Categories as Town Source

### Why Wikipedia
- **Free and open.** No copyright issues (unlike DDC which is owned by OCLC).
- **Comprehensive.** Covers everything from ornithology to esports to nanotechnology.
- **Hierarchical.** Categories have parent-child relationships we can walk programmatically.
- **Named.** Every category has a human-readable name — no auto-generated labels.
- **Crowd-curated.** Continuously maintained by thousands of editors.
- **Modern.** Covers internet culture, gaming, cryptocurrency, AI — topics that DDC and LCSH miss or barely touch.

### Why not other sources
- **Dewey Decimal (DDC):** Copyrighted by OCLC. Also weak on modern topics, uneven granularity (CS gets 3 numbers, religion gets 100).
- **Library of Congress Subject Headings (LCSH):** Freely available but flat thesaurus rather than clean tree. Extracting hierarchy is messy.
- **Generate our own via Claude:** Possible but less systematic. No external validation. Risk of gaps and inconsistencies.
- **DDC as comparison:** Still worth comparing our domains against DDC structure to find blind spots, even if we don't use it as the primary source.

### Coverage results (from `wiki_categories.md`)

We queried the MediaWiki API for all 258 domains:

| Status | Count |
|--------|-------|
| Wikipedia match found | 203 |
| Still missing (network/mapping) | 25 |
| Skipped (non-topical linguistic domains) | 16 |
| Skipped (too vague/ambiguous) | ~14 |

**Key domains with rich subdivision trees:**

| Domain | Wikipedia subcategories | Notable subdivisions |
|--------|------------------------|---------------------|
| sport | 30+ | Combat sports, Ball games, Winter sports, Equestrian, Esports, Cycling, Motorsport |
| zoology | 34 | Ornithology, Herpetology, Ichthyology, Mammalogy, Ethology |
| biology | 28 + 37 branches | Genetics, Ecology, Cell biology, Marine biology, Molecular biology, Neuroscience |
| music | 35 + genres | Music by genre drills into dozens of genre categories |
| religion | 25 | (would need deeper drill into specific religions) |
| mathematics | 20 | (would need "Fields of mathematics" drill) |
| cooking | 17 | Techniques, Cuisine, Culinary arts |

### What the mapping solves directly

| Current problem | How towns fix it |
|----------------|-----------------|
| Sport reef at 75.5% of chunks | "Ice hockey" town only fires for hockey words, not all sport words |
| Bird words → astrology (Issue 10) | "Ornithology" town under zoology island captures bird vocabulary specifically |
| "pie" → fairytale (Issue 12) | "Cuisine" or "Baking" town under cooking island gets food vocabulary |
| XGBoost too broad | Classifier trains on "sport:combat sports" — tight positive set, no "furniture" leaking in |
| Sub-reef names are garbage | Wikipedia gives us "Ornithology", "Ice hockey", "Baroque music" — real names |

---

## Mapping Challenges

### 1. Granularity selection
Wikipedia categories vary wildly in depth. "Sports by type" has 40+ immediate children. "Cooking" has 17. Some are too broad ("Music by genre" contains dozens of sub-genres, each with further sub-genres). Others are too narrow ("Jewish sports" probably doesn't need its own town).

**Approach:** Start with level-1 subcategories, drill into "Branches of X" / "X by type" patterns. Then apply size-based pruning: towns with fewer than N words after XGBoost training get merged back into their parent island or into a catch-all town.

### 2. Non-topical Wikipedia cruft
Wikipedia categories are full of administrative/organizational categories that aren't semantic topics:
- "X by location", "X by country", "X by year" (geographic/temporal)
- "X-related lists", "History of X" (meta)
- "People in X", "X organizations" (entities, not topics)
- "X in popular culture", "Films about X" (cross-references)

**Approach:** Pattern-based filtering (already implemented in `explore_wiki_categories.py`). The filter catches most of this. Manual review needed for edge cases.

### 3. Domains with no clear Wikipedia mapping
~41 domains don't map cleanly:

**Linguistic/stylistic domains (16):** abbreviation, acronym, archaism, blend, combining form, comparative, disparagement, dialect, ethnic slur, euphemism, formality, intensifier, irony, plural, regionalism, trope. These are word-property categories, not topical domains. They probably don't need towns — they're already specialized enough. Keep as flat islands with no subdivision.

**Too vague/abstract (6):** genesis, growth, medium, vent, wit, ovid. Some of these might be candidates for removal from the taxonomy entirely.

**Geographic (3):** canada, jamaica, west indies. These are cultural/regional markers, not topical domains. Probably don't need towns.

**Genuinely missing (network failures, ~16):** military, statistics, nanotechnology, etc. These definitely have Wikipedia categories — just need retry or manual mapping.

### 4. Overlap between towns across islands
The same Wikipedia category might appear under multiple parent categories. "Marine biology" appears under both Biology and Zoology. "Forensic science" could be under both Law and Chemistry.

**Approach:** Allow overlap at the town definition level (a town can appear under multiple islands). At scoring time, the island+town combination is the unique identifier. "biology:marine biology" and "zoology:marine biology" could be separate towns that happen to share vocabulary, or they could be deduplicated.

### 5. Quality of Wikipedia subcategories varies
Some categories have beautiful, clean subdivisions (Zoology → Subfields of zoology is perfect). Others are messy:
- "Music by genre" contains "Youth music" alongside "Ballet music" — inconsistent granularity
- "Cooking" has "Creamy dishes" as a subcategory — too specific and weird
- Some categories are social/cultural rather than topical ("Gender and sport", "Race and sports")

**Approach:** This requires curation. The Wikipedia pull gives us a starting point, not a finished taxonomy. We'll need to review and prune the results, possibly merging some categories and splitting others.

---

## Pipeline Implications

### What changes in Windowsill
1. **New data source:** Town definitions (from curated Wikipedia categories) stored alongside domain definitions
2. **XGBoost training:** Train per island+town pair, not per island. ~1,000-2,500 classifiers instead of 258. Training time scales linearly.
3. **Scoring:** `domain_word_scores` gets a `town` column. Weights are computed at the town level.
4. **Export:** `word_reefs.bin` entries reference towns (the new primary scoring unit), not islands. Island membership is derived from town→island mapping.
5. **Consolidation:** May need rethinking. Currently merges 444→258 before XGBoost. With towns, we might keep the 258 islands but train XGBoost at the town level within each island.

### What changes in Lagoon
1. **Scoring unit:** Accumulate weights per town, not per island. Town count goes from 258 to ~1,000-2,500.
2. **Hierarchy:** Island coherence scoring now uses the town→island mapping. Multiple towns in the same island = coherent signal.
3. **Background model:** Per-town background means/stds instead of per-island. More towns = sparser background sampling, may need adjustment.
4. **Memory:** More scoring units = larger arrays. 2,500 towns × 4 bytes = 10KB per array. Negligible.

### What changes in Shoal
1. **Display:** Can show town names ("Ornithology") instead of island names ("Zoology") for more specific result explanations.
2. **Scoring granularity:** Better discrimination between documents in the same broad domain.
3. **Tests:** Need to update expected reef IDs / names. Test expectations become more specific (and more meaningful).

---

## Open Questions

1. **How many towns do we target?** 1,000? 2,000? 2,500? More towns = more discrimination but also more classifiers to train and more sparse data per classifier.

2. **Minimum town size?** If a town has fewer than N words after XGBoost training, what do we do? Merge into parent island? Merge into a sibling town? Keep it small?

3. **Do we keep the current 258 islands, or let Wikipedia restructure them too?** The user's instinct is to keep the 258 for now. But some domains are arguably wrong at the island level (e.g., "dressage" is a sport, "logic" is philosophy/math).

4. **Training order:** Do we train XGBoost at the town level from scratch, or do we train at the island level first and then subdivide? The former is cleaner but requires town-level seed words. The latter reuses existing infrastructure.

5. **Town seed words:** XGBoost needs positive examples to train. Where do they come from for towns? Options:
   - Pull from Wikipedia article text (articles in each category)
   - Use Claude to generate town-specific vocabulary (like we do for island augmentation)
   - Use the existing island vocabulary + Leiden clustering to assign words to towns, then retrain
   - Some combination

6. **Curation effort:** How much manual work to turn the raw Wikipedia category dump into a clean town taxonomy? The `wiki_categories.md` file is a starting point but needs human review.

---

## Next Steps (when ready to proceed)

1. **Curate town definitions.** Review `wiki_categories.md`, select appropriate subcategories for each island, prune junk, fill gaps. Output: `town_definitions.json` mapping island→[town1, town2, ...].

2. **Generate town seed words.** Use Claude and/or Wikipedia article scraping to produce 50-200 seed words per town.

3. **Prototype with one island.** Pick "sport" (worst offender), define its towns, generate seeds, train XGBoost at the town level, and measure the impact on Shoal retrieval tests.

4. **Assess scale.** Based on the prototype, estimate total town count, training time, export size, and Lagoon memory impact.

5. **Full pipeline rebuild.** If the prototype works, redesign the pipeline stages for the new hierarchy.
