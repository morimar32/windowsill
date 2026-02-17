# Data Fix Request: Chess & Polysemy Problems

This document describes issues found while testing the shoal retrieval engine
against lagoon's 233-reef data build (build timestamp `2026-02-16T20:55:49Z`).
All observations are from the shoal test corpus: 38 documents (36 Wikipedia
articles + 2 Gutenberg books) producing ~3,500 chunks.

---

## 1. Background: How Word Data Drives Retrieval

For retrieval to work, there must be reef overlap between the query's scored
profile and the stored chunk profiles. The chain is:

1. User query text → `scorer.score(query)` → query reef profile (top reefs + z-scores)
2. Each stored chunk has a reef profile computed at ingestion via `scorer.score(chunk_text)`
3. Retrieval ranks chunks by `SUM(chunk_z * query_z)` over shared reefs

If a query word's BM25 entries don't overlap with the reefs present in the target
document's chunks, retrieval fails — even if the word appears hundreds of times in
the document text. The word is "in the dictionary" but points to the wrong reefs.

---

## 2. The Chess Article Problem

The chess Wikipedia article (193 chunks after ingestion) is dominated by a single
top reef:

    deliberate behavior and cunning (reef 207, island 43, arch 3): 193/193 chunks

This reef is correct — chess content is strategic, deliberate, tactical. The
problem is that **individual chess terms don't have BM25 entries pointing to
reef 207**, so queries using those terms can't find the chess chunks via reef
overlap.

### 2.1 Word-Level Analysis

Each word below is in lagoon's vocabulary but its BM25 entries miss reef 207
entirely. The reef associations reflect non-chess senses of these polysemous words.

#### "rook" (word_id=112064, spec=+1, idf_q=115)

Top BM25 entries:
```
reef 229  bm25_q=16391  mortality and ephemeral existence
reef 103  bm25_q=15283  cooking and food prep
reef 117  bm25_q=14407  casual culture and recreation
reef  73  bm25_q=10205  vehicles and riding
reef  77  bm25_q=10197  formed structures and launching
```

Full reef 207 position: rank 43 out of 233, z=-0.691 (anti-correlated).

The word "rook" appears in 49 of the chess article's 193 chunks. Its BM25 entries
reflect the bird sense (corvid family) and possibly the verb sense ("to rook"
meaning to cheat/swindle). There is no chess piece signal. The top entry — mortality
and ephemeral existence — likely comes from the bird's association with death
symbolism in literature.

Notably, "rook" does have a weak entry for reef 117 (casual culture and recreation,
bm25_q=14407), which is the closest to a games/recreation signal. But this reef
doesn't appear in the chess article's chunk profiles, so it doesn't help retrieval.

#### "castle" (word_id=21226, spec=+0, idf_q=104)

Top BM25 entries:
```
reef 229  bm25_q=14800  mortality and ephemeral existence
reef  81  bm25_q=13662  light and radiation
reef  69  bm25_q=10496  physical objects and depictions
reef  99  bm25_q= 9112  historical warfare and transport
reef  98  bm25_q= 9011  livestock and animal husbandry
```

Full reef 207 position: rank 0 (top), z=0.630 — but as a single-word query with
24 scattered BM25 entries across unrelated reefs, the z-score of 0.63 after
background subtraction produces confidence of only 0.10. It barely passes the
retrieval quality gate.

No entry for reef 207 in BM25 data. The word maps to building/fortification
senses (warfare, physical objects, livestock — rural estate). The chess meaning
(castling — a specific move involving the king and rook) is absent.

34 chess chunks contain "castle" or "castling".

#### "knight" (word_id=73683, spec=+0, idf_q=107)

Top BM25 entries:
```
reef  81  bm25_q=14117  light and radiation
reef 107  bm25_q=13790  discomfort and oppression
reef 122  bm25_q=13744  texture and tactile
reef 132  bm25_q= 9739  maritime occupations places
reef 180  bm25_q= 9419  deception and obscurity
```

Full reef 207 position: rank 229 out of 233, z=-1.250 (strongly anti-correlated).

The word maps to medieval/feudal senses (nobility, warfare, historical). No chess
piece signal at all. The anti-correlation with reef 207 means "knight" in a query
actively pushes results AWAY from chess content.

#### "castling" (word_id=21230, spec=+1, idf_q=113)

Top BM25 entries:
```
reef 197  bm25_q=15065  claims and assertions
reef 103  bm25_q=15012  cooking and food prep
reef 173  bm25_q=13748  neural and biological systems
reef  28  bm25_q=12421  natural species and habitats
reef  73  bm25_q=10024  vehicles and riding
```

Full reef 207 position: rank 9, z=0.050 (negligible positive signal).

This word is chess-specific — "castling" has no other common meaning. Yet its BM25
entries map to cooking, biology, and natural habitats. This suggests the word was
mapped from a WordNet synset chain that doesn't reflect actual usage (possibly via
the etymology of "castle" → building-related synsets → those reef clusters).

#### "pawn" (word_id=96917, spec=+0, idf_q=101)

Top BM25 entries:
```
reef 229  bm25_q=14338  mortality and ephemeral existence
reef 101  bm25_q=14112  compression and alphabetization
reef 224  bm25_q=13585  commerce and competition
reef 150  bm25_q=13153  contamination and discoloration
reef 134  bm25_q=12176  proper names and heritage
```

Full reef 207 position: rank 2, z=1.020 — this one actually does have some signal
toward "deliberate behavior and cunning" via the commerce/manipulation sense. But
the dominant entries are in unrelated reefs.

#### "bishop" (word_id=13868, spec=+1, idf_q=154)

Top BM25 entries:
```
reef 128  bm25_q=22136  coastal and frontier
reef 206  bm25_q=17493  acceptance and dedication
reef  52  bm25_q=13271  medical conditions and clergy
```

Full reef 207 position: rank 1, z=0.722.

Maps to the clergy/religious sense. "Bishop" as a chess piece is absent.

### 2.2 Words That DO Work

For comparison, these chess terms were substantially improved in the 233-reef build:

#### "chess" (word_id=23347, spec=+0, idf_q=145)

After background subtraction, produces z=1.70 on reef 207 (STRONG, conf 1.16).
This works because the combined BM25 signal from its entries — "theoretical
compliance" (19085), "cooking and food prep" (19319), "vehicles and riding" (12900),
"deception and obscurity" (12762) — after background subtraction, converges on
"deliberate behavior and cunning" as the top z-scored reef. The raw BM25 entries
don't directly target reef 207, but the background subtraction math happens to
produce the right result.

#### "checkmate" (word_id=23096, spec=+0, idf_q=111)

After background subtraction, produces z=2.67 on reef 207 (STRONG, conf 1.11).
Similar mechanism to "chess" — no direct BM25 entry for reef 207, but the combined
signal after subtraction lands there. "Checkmate" is chess-specific with no
competing senses, which helps.

#### "game" (word_id=52015, spec=+0, idf_q=91)

Has a strong direct entry for reef 117 (casual culture and recreation, bm25_q=16774).
This is the clearest games/recreation signal in the vocabulary. However, this reef
doesn't appear in chess chunk profiles (the chess article scores as "deliberate
behavior and cunning", not "casual culture and recreation"), so "game" doesn't
actually help find chess content either.

### 2.3 The Compound Effect

The real problem emerges in multi-word queries. When a user searches "rook
movements", the scorer combines the BM25 entries for both words, then subtracts
background. Since "rook" points to bird/death/cooking reefs and "movements" points
to building/physical motion reefs, the combined profile lands on "building exteriors
and surfaces" (z=2.40) — nowhere near reef 207. The chess article is invisible.

Query-to-retrieval trace for "rook movements":
```
scorer.score("rook movements") → conf 0.82, top reef: building exteriors and surfaces (z=2.40)
→ reef overlap query joins on: building exteriors, casual culture, physical motion, vehicles
→ chess chunks have reef 207 (deliberate behavior and cunning) stored
→ ZERO overlap between query reefs and chess chunk reefs
→ chess not in results
```

---

## 3. The Reef Breadth Problem

Even when a query does activate reef 207 (e.g., "chess strategy", "checkmate"),
the chess article competes with many non-chess documents that also score heavily on
"deliberate behavior and cunning":

| Document | Chunks on reef 207 | Why |
|----------|-------------------|-----|
| Chess | 193 | Strategic gameplay |
| Monarchy of the UK | 170 | Political maneuvering |
| Crane Machine | 190 | Deliberate engineering/operation |
| Apple Inc | 147 | Corporate strategy |
| Supreme Court | 106 | Legal strategy |
| Huckleberry Finn | 371 | Con artists, deception |
| French Revolution | 82 | Political cunning |

Reef 207 has 11,869 base words (n_words) and 18,044 word entries — it's one of
the larger reefs. It covers a broad semantic territory: strategic thinking,
deliberate action, cunning, deception, planning. This breadth means chess content
doesn't stand out from political or legal content that shares the same reef.

The "casual culture and recreation" reef (reef 117, 3,895 words) exists and covers
games/recreation, but chess chunks don't score on it. Only the word "game" directly
maps there (bm25_q=16774). Chess-specific terms like "chess", "checkmate", "rook",
"bishop" have no entries for reef 117, or only very weak ones ("rook" has
bm25_q=14407 for reef 117, but that's the bird/recreation sense, not chess).

---

## 4. Specific Requests

### 4.1 Chess Piece Words Need Reef 207 Signal

The following words need BM25 entries for reef 207 ("deliberate behavior and
cunning") or whatever reef chess content should map to:

| Word | Current reef 207 position | Request |
|------|---------------------------|---------|
| rook | rank 43, z=-0.69 | Needs positive signal; currently anti-correlated |
| castle | rank 0, z=+0.63 | Already top reef after subtraction but BM25 entry missing; weak conf (0.10) |
| knight | rank 229, z=-1.25 | Strongly anti-correlated; needs substantial positive signal |
| castling | rank 9, z=+0.05 | Negligible; needs meaningful positive signal |
| pawn | rank 2, z=+1.02 | Has some signal via commerce sense; could be strengthened |
| bishop | rank 1, z=+0.72 | Has some signal; could be strengthened |

### 4.2 "castling" Has No Valid Sense Mapping

"Castling" is a chess-specific term with no common non-chess meaning. Its current
top BM25 entries (claims and assertions, cooking and food prep, neural and
biological systems, natural species and habitats) don't reflect any real usage
of the word. This suggests a WordNet synset chain problem — the word was likely
reached through the "castle" lemma and inherited building/structure-related
associations that have nothing to do with the chess move.

This is probably the most clear-cut data bug in this set — unlike "rook" or
"bishop" which genuinely have non-chess senses, "castling" has only one meaning.

### 4.3 Consider Reef Granularity for Games/Strategy

The 233-reef build added "casual culture and recreation" (reef 117) which is
a step in the right direction. But chess content doesn't activate this reef —
it activates "deliberate behavior and cunning" (reef 207) instead, because
chess is about strategy and calculation, not casual recreation.

Options to consider:
- A sub-reef or new reef for structured competitive games/strategy (chess,
  go, bridge, poker, etc.) — distinct from casual recreation and from
  political/legal strategy
- Splitting reef 207 to separate game strategy from political/criminal cunning
- Ensuring chess piece terms have entries in reef 117 so that "game" + "rook"
  queries can find chess content via reef 117 overlap

### 4.4 "chessboard" and "grandmaster" Have No Chess Signal

Additional chess vocabulary that's in the dictionary but maps to wrong reefs:

**"chessboard"** (word_id=23358, spec=+0, idf_q=124):
```
Top: light and radiation (16326), discomfort and oppression (15948),
     formed structures and launching (11002)
```
Another chess-specific compound word that should unambiguously map to chess/game
reefs but doesn't.

**"grandmaster"** (word_id=58841, spec=+1, idf_q=149):
```
Top: official roles status (17587), sudden acquisition events (17186),
     natural species and habitats (16416)
```
Maps to official/authority senses. The "grandmaster" title is primarily a chess
term (FIDE title), though it's used in other contexts (martial arts, freemasonry).

---

## 5. Other Polysemous Words (For Reference)

These follow the same pattern as the chess vocabulary — polysemous words where
lagoon has learned a single dominant sense that misses the test corpus's usage.

### "python" (word_id=106376, spec=+0, idf_q=129)

```
Top: hair and bodily oddities (18438), skeletal anatomy (10987)
```
Neither the snake sense (which should map to reptile/zoology reefs) nor the
programming language sense (which should map to technology/computing reefs).
Searching "python" returns French Revolution and Jupiter articles.

### "mercury" (word_id=83293, spec=+0, idf_q=135)

```
Top: spec- technical terms (29501), visual art (18476), fragments and outlaws (18436)
```
Partially works — the "spec-" technical prefix reef gives it some chemistry
signal, and retrieval does return Mercury Planet and Mercury Element articles
(though at low confidence 0.48). The planet sense and element sense both have
some signal via indirect reef activation.

### "crane" (word_id=30674, spec=+0, idf_q=132)

```
Top: aquatic creatures and wildlife (18754), livestock and animal husbandry (18416)
```
Maps only to the bird sense. The machine sense is absent. However, the crane
machine article scores on physical/mechanical reefs that partially overlap with
the bird-associated reefs, so retrieval coincidentally returns some machine
results alongside bird results.

---

## 6. Summary

The core issue is that chess terminology is polysemous and the current data build
maps these words exclusively to their non-chess senses. The 207→233 reef expansion
helped substantially (creating reef 207 "deliberate behavior and cunning" which
gives chess content a home, and making "chess" and "checkmate" produce STRONG
signal). But the chess piece names — which are the most natural query terms for
chess content — still point to the wrong reefs.

Priority ranking:
1. **castling** — chess-specific word with no valid non-chess sense; clearly a data bug
2. **rook** — strongest test case; appears in 49/193 chess chunks; currently anti-correlated with chess reef
3. **knight** — strongly anti-correlated (rank 229/233) with chess reef
4. **chessboard** — chess-specific compound; no valid non-chess sense
5. **bishop, pawn, castle** — polysemous but should have chess sense alongside existing senses
6. **grandmaster** — primarily chess term; maps to authority/official roles instead
