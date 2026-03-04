-- ============================================================
-- Windowsill V3 Schema
-- ============================================================
--
-- Four-tier hierarchy:  Archipelago > Island > Town > Reef
--   - Three named/curated levels (Archipelago, Island, Town)
--   - One discovered level (Reef, via Leiden clustering)
--
-- Design principles (from structures.md):
--   1. Everything explicit in the database
--   2. Search/validate directly via SQL queries
--   3. INSERT scripts as milestones for manual review
--   4. Subselects with readable labels, not hardcoded IDs
--
-- Notes:
--   - SQLite stores all integers as signed i64.
--     FNV-1a u64 hashes need: hash & 0xFFFFFFFFFFFFFFFF on read.
--   - export_weight is u8 [0..255], per-reef min-max normalized.
--   - Embedding BLOBs are float32 x 768 = 3072 bytes each.
--
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;


-- ============================================================
-- 1. HIERARCHY TABLES
-- ============================================================

CREATE TABLE Archipelagos (
    archipelago_id  INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL UNIQUE,

    -- cached stats (updated after population)
    island_count    INTEGER DEFAULT 0,
    town_count      INTEGER DEFAULT 0,
    reef_count      INTEGER DEFAULT 0,
    word_count      INTEGER DEFAULT 0
);


CREATE TABLE Islands (
    island_id       INTEGER PRIMARY KEY,
    archipelago_id  INTEGER NOT NULL
                    REFERENCES Archipelagos(archipelago_id),
    name            TEXT    NOT NULL,
    is_bucket       INTEGER NOT NULL DEFAULT 0, -- 1 = non-topical bucket (e.g. Linguistic Register)
                                                -- words are identified but excluded from topic scoring

    -- cached stats
    town_count      INTEGER DEFAULT 0,
    reef_count      INTEGER DEFAULT 0,
    word_count      INTEGER DEFAULT 0,
    noun_frac       REAL,
    verb_frac       REAL,
    adj_frac        REAL,
    adv_frac        REAL,
    avg_specificity REAL,

    UNIQUE (archipelago_id, name)
);


CREATE TABLE Towns (
    town_id         INTEGER PRIMARY KEY,
    island_id       INTEGER NOT NULL
                    REFERENCES Islands(island_id),
    name            TEXT    NOT NULL,
    is_capital      INTEGER DEFAULT 0,     -- 1 = catch-all capital town for island
    model_f1        REAL,                  -- XGBoost validation F1 score

    -- cached stats
    reef_count      INTEGER DEFAULT 0,
    word_count      INTEGER DEFAULT 0,
    noun_frac       REAL,
    verb_frac       REAL,
    adj_frac        REAL,
    adv_frac        REAL,
    avg_specificity REAL,

    UNIQUE (island_id, name)
);


CREATE TABLE Reefs (
    reef_id         INTEGER PRIMARY KEY,
    town_id         INTEGER NOT NULL
                    REFERENCES Towns(town_id),
    name            TEXT,                   -- set after Leiden clustering
    centroid        BLOB,                   -- L2-normalized float32 embedding

    -- cached stats
    word_count      INTEGER DEFAULT 0,
    core_word_count INTEGER DEFAULT 0,
    avg_specificity REAL,
    noun_frac       REAL,
    verb_frac       REAL,
    adj_frac        REAL,
    adv_frac        REAL
);


-- ============================================================
-- 2. DICTIONARY TABLES
-- ============================================================

CREATE TABLE Words (
    word_id         INTEGER PRIMARY KEY,
    word            TEXT    NOT NULL,
    word_hash       INTEGER NOT NULL,       -- FNV-1a u64 (stored as i64)
    pos             TEXT,                   -- dominant POS: n, v, a, r
    specificity     INTEGER,               -- global: based on total reef count
    cosine_sim      REAL,                  -- best cosine to any assigned reef centroid
    idf             REAL,                  -- log2(N_total_reefs / reef_count)
    embedding       BLOB,                  -- float32 x 768 (pipeline use, not exported)
    word_count      INTEGER DEFAULT 1,     -- number of space-separated tokens
    category        TEXT,                  -- single, compound, phrasal_verb, etc.
    is_stop         INTEGER DEFAULT 0,

    -- computed by compute_word_stats.py (populated after clustering)
    reef_count      INTEGER DEFAULT 0,     -- number of reefs this word appears in
    town_count      INTEGER DEFAULT 0,     -- number of towns
    island_count    INTEGER DEFAULT 0      -- number of islands
);


-- ReefWords: the working dictionary, one row per (reef, word) pair.
-- POS, specificity, cosine_sim, and idf here are contextual —
-- they override the general values on Words when set.
CREATE TABLE ReefWords (
    reef_id         INTEGER NOT NULL
                    REFERENCES Reefs(reef_id),
    word_id         INTEGER NOT NULL
                    REFERENCES Words(word_id),
    pos             TEXT,                   -- contextual POS; overrides Words.pos
    specificity     INTEGER,               -- specificity within this reef
    cosine_sim      REAL,                  -- cosine to this reef's centroid
    idf             REAL,                  -- IDF within this reef's context
    island_idf      REAL,                  -- log2(island_total_reefs / word_reefs_in_island)
    source          TEXT    NOT NULL,       -- wordnet, claude_augmented, xgboost
    source_quality  REAL    NOT NULL DEFAULT 1.0,
    is_core         INTEGER DEFAULT 0,     -- 1 = Leiden core member

    PRIMARY KEY (reef_id, word_id)
);


-- ============================================================
-- 3. EXPORT TABLES — CORE
-- ============================================================
--
-- Promotion chain:
--   ReefWords  -->  ReefWordExports
--     words with no cross-reef overlap in their town  -->  TownWordExports
--       words with no cross-town overlap in their island  -->  IslandWordExports
--
-- After promotion, a word exists at exactly ONE export level.
-- All three tables share the same calculation columns so the
-- weight derivation is transparent and auditable at every level.
--
-- export_weight is u8 [0..255], per-reef min-max normalized.
-- ============================================================

CREATE TABLE ReefWordExports (
    reef_id         INTEGER NOT NULL
                    REFERENCES Reefs(reef_id),
    word_id         INTEGER NOT NULL
                    REFERENCES Words(word_id),
    idf             REAL,
    centroid_sim    REAL,               -- cosine to reef centroid
    name_cos        REAL,               -- cosine to hierarchy name embedding
    effective_sim   REAL,               -- blended similarity
    specificity     INTEGER,
    source_quality  REAL,
    export_weight   INTEGER NOT NULL
                    CHECK (export_weight BETWEEN 0 AND 255),

    PRIMARY KEY (reef_id, word_id)
);


CREATE TABLE TownWordExports (
    town_id         INTEGER NOT NULL
                    REFERENCES Towns(town_id),
    word_id         INTEGER NOT NULL
                    REFERENCES Words(word_id),
    idf             REAL,
    centroid_sim    REAL,
    name_cos        REAL,
    effective_sim   REAL,
    specificity     INTEGER,
    source_quality  REAL,
    export_weight   INTEGER NOT NULL
                    CHECK (export_weight BETWEEN 0 AND 255),
    export_town_weight  INTEGER NOT NULL DEFAULT 128
                    CHECK (export_town_weight BETWEEN 0 AND 255),

    PRIMARY KEY (town_id, word_id)
);


CREATE TABLE IslandWordExports (
    island_id       INTEGER NOT NULL
                    REFERENCES Islands(island_id),
    word_id         INTEGER NOT NULL
                    REFERENCES Words(word_id),
    idf             REAL,
    centroid_sim    REAL,
    name_cos        REAL,
    effective_sim   REAL,
    specificity     INTEGER,
    source_quality  REAL,
    export_weight   INTEGER NOT NULL
                    CHECK (export_weight BETWEEN 0 AND 255),
    export_island_weight INTEGER NOT NULL DEFAULT 128
                    CHECK (export_island_weight BETWEEN 0 AND 255),

    PRIMARY KEY (island_id, word_id)
);


-- ============================================================
-- 4. EXPORT TABLES — OPTIONAL
-- ============================================================
--
-- These tables are exported but only optionally loaded by Lagoon,
-- to keep the runtime memory footprint down.
-- ============================================================

-- Debugging aid: just word_id + text, for reverse lookups.
CREATE TABLE WordsExport (
    word_id         INTEGER PRIMARY KEY,
    word            TEXT    NOT NULL
);


-- First/last names for person detection and pronoun collapsing.
-- Dataset to be built later; table defined now to plan ahead.
CREATE TABLE NamesExport (
    name_id         INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL,
    type            TEXT    NOT NULL
                    CHECK (type IN ('first', 'last'))
);


-- Alternate word hashes that resolve to a canonical word.
-- Covers: plurals, morphy forms, snowball stems, nicknames,
-- common typos, etc.
CREATE TABLE EquivalencesExport (
    variant_hash    INTEGER NOT NULL,       -- FNV-1a u64 of variant form
    word_id         INTEGER NOT NULL
                    REFERENCES Words(word_id),
    variant         TEXT    NOT NULL,        -- the variant surface form
    source          TEXT    NOT NULL,        -- morphy, snowball, nickname, typo, plural

    PRIMARY KEY (variant_hash, word_id)
);


-- Acronym expansions, optionally scoped to a domain.
CREATE TABLE AcronymsExport (
    acronym_id      INTEGER PRIMARY KEY,
    acronym         TEXT    NOT NULL,
    expansion       TEXT    NOT NULL,
    word_id         INTEGER                 -- FK to Words if expansion is in vocab
                    REFERENCES Words(word_id),
    island_id       INTEGER                 -- optional domain context
                    REFERENCES Islands(island_id)
);


-- ============================================================
-- 5. PIPELINE SUPPORT TABLES
-- ============================================================
--
-- Working tables used during the build pipeline.
-- Temporary tables may supplement these, but these persist
-- as essential references across pipeline stages.
-- ============================================================

-- Per-embedding-dimension statistics for z-score features.
CREATE TABLE DimStats (
    dim_id          INTEGER PRIMARY KEY,    -- 0..767
    mean            REAL,
    std             REAL,
    threshold       REAL,                   -- mean + 2.0 * std
    member_count    INTEGER,
    selectivity     REAL                    -- 1.0 - (member_count / total_words)
);


-- Island-level words: vocabulary that belongs to an island but is
-- not discriminative for any specific town within it.
-- Detected pre-XGBoost via cosine-std across town centroids,
-- and post-XGBoost via the island-word filter (predicted by >=80%
-- of towns).  These feed into IslandWordExports during export.
CREATE TABLE IslandWords (
    island_id       INTEGER NOT NULL
                    REFERENCES Islands(island_id),
    word_id         INTEGER NOT NULL
                    REFERENCES Words(word_id),
    word            TEXT    NOT NULL,        -- denormalized for convenience
    source          TEXT    NOT NULL,        -- seed_cosine_std, xgboost_filter
    cosine_std      REAL,                   -- std of cosine sims to town centroids (lower = more generic)
    avg_cosine      REAL,                   -- mean cosine to town centroids

    PRIMARY KEY (island_id, word_id)
);


-- Town-level seed words: the starting vocabulary before XGBoost
-- expansion and Leiden clustering.  Populated from WordNet ground
-- truth, Claude augmentation, and Wikipedia article pulls.
CREATE TABLE SeedWords (
    town_id         INTEGER NOT NULL
                    REFERENCES Towns(town_id),
    word            TEXT    NOT NULL,        -- seed word text
    word_id         INTEGER                 -- FK to Words; NULL if not yet in vocab
                    REFERENCES Words(word_id),
    source          TEXT    NOT NULL,        -- wordnet, claude_augmented, wikipedia
    confidence      TEXT,                   -- core, peripheral
    score           REAL,                   -- source-specific confidence score

    PRIMARY KEY (town_id, word)
);


-- XGBoost predictions above threshold, per town.
-- Populated by train_town_xgboost.py.  Feeds into cluster_reefs.py.
CREATE TABLE AugmentedTowns (
    town_id         INTEGER NOT NULL
                    REFERENCES Towns(town_id),
    word_id         INTEGER NOT NULL
                    REFERENCES Words(word_id),
    score           REAL    NOT NULL,
    source          TEXT    NOT NULL,        -- xgboost

    PRIMARY KEY (town_id, word_id)
);


-- Per (word, island) concentration stats.
-- Populated by compute_word_stats.py after clustering.
CREATE TABLE WordIslandStats (
    word_id         INTEGER NOT NULL
                    REFERENCES Words(word_id),
    island_id       INTEGER NOT NULL
                    REFERENCES Islands(island_id),
    reef_count      INTEGER NOT NULL,
    town_count      INTEGER NOT NULL,
    concentration   REAL    NOT NULL,       -- fraction of word's reefs in this island
    avg_cosine      REAL,

    PRIMARY KEY (word_id, island_id)
);


-- Morphological and stemming variants of vocabulary words.
-- Feeds into EquivalencesExport during the export phase.
CREATE TABLE WordVariants (
    variant_hash    INTEGER NOT NULL,       -- FNV-1a u64 of variant text
    variant         TEXT    NOT NULL,
    word_id         INTEGER NOT NULL
                    REFERENCES Words(word_id),
    source          TEXT    NOT NULL,        -- base, morphy, snowball

    PRIMARY KEY (variant_hash, word_id)
);


-- ============================================================
-- 6. VIEWS
-- ============================================================

-- Unified export view: resolves every exported word to its full
-- hierarchy path.  Use this for direct search validation.
--
-- Example — single word lookup:
--   SELECT * FROM ExportIndex
--   WHERE word_id = (SELECT word_id FROM Words WHERE word = 'violin')
--   ORDER BY export_weight DESC;
--
-- Example — multi-word search (accumulate per town):
--   SELECT town, island, archipelago, SUM(export_weight) AS score
--   FROM ExportIndex
--   WHERE word_id IN (SELECT word_id FROM Words
--                     WHERE word IN ('hockey','stick','ice','rink'))
--   GROUP BY town_id
--   ORDER BY score DESC
--   LIMIT 10;
--
CREATE VIEW ExportIndex AS
SELECT
    a.archipelago_id,   a.name  AS archipelago,
    i.island_id,        i.name  AS island,
    t.town_id,          t.name  AS town,
    r.reef_id,          r.name  AS reef,
    e.word_id,
    e.export_weight,
    'reef' AS export_level
FROM ReefWordExports   e
JOIN Reefs             r ON e.reef_id         = r.reef_id
JOIN Towns             t ON r.town_id         = t.town_id
JOIN Islands           i ON t.island_id       = i.island_id
JOIN Archipelagos      a ON i.archipelago_id  = a.archipelago_id

UNION ALL

SELECT
    a.archipelago_id,   a.name,
    i.island_id,        i.name,
    t.town_id,          t.name,
    NULL,               NULL,
    e.word_id,
    e.export_weight,
    'town'
FROM TownWordExports   e
JOIN Towns             t ON e.town_id         = t.town_id
JOIN Islands           i ON t.island_id       = i.island_id
JOIN Archipelagos      a ON i.archipelago_id  = a.archipelago_id

UNION ALL

SELECT
    a.archipelago_id,   a.name,
    i.island_id,        i.name,
    NULL,               NULL,
    NULL,               NULL,
    e.word_id,
    e.export_weight,
    'island'
FROM IslandWordExports e
JOIN Islands           i ON e.island_id       = i.island_id
JOIN Archipelagos      a ON i.archipelago_id  = a.archipelago_id;


-- Convenience view: joins ExportIndex with word text for
-- human-readable search results.
CREATE VIEW WordSearch AS
SELECT
    w.word,
    ei.*
FROM ExportIndex ei
JOIN Words w ON ei.word_id = w.word_id;


-- Compound words for Aho-Corasick tokenization.
-- Derived from Words where word_count > 1.
CREATE VIEW Compounds AS
SELECT word, word_id
FROM Words
WHERE word_count > 1
  AND is_stop = 0;


-- Full hierarchy path for any reef, useful for reporting.
CREATE VIEW HierarchyPath AS
SELECT
    a.archipelago_id,   a.name  AS archipelago,
    i.island_id,        i.name  AS island,
    t.town_id,          t.name  AS town,
    r.reef_id,          r.name  AS reef,
    r.word_count        AS reef_words,
    t.word_count        AS town_words,
    i.word_count        AS island_words
FROM Reefs        r
JOIN Towns        t ON r.town_id         = t.town_id
JOIN Islands      i ON t.island_id       = i.island_id
JOIN Archipelagos a ON i.archipelago_id  = a.archipelago_id;


-- ============================================================
-- 7. INDEXES
-- ============================================================

-- Hierarchy (parent lookups)
CREATE INDEX idx_islands_arch    ON Islands(archipelago_id);
CREATE INDEX idx_towns_island    ON Towns(island_id);
CREATE INDEX idx_reefs_town      ON Reefs(town_id);

-- Words (lookup by text and hash)
CREATE INDEX idx_words_word      ON Words(word);
CREATE INDEX idx_words_hash      ON Words(word_hash);

-- ReefWords (lookup by word across all reefs)
CREATE INDEX idx_rw_word         ON ReefWords(word_id);

-- Export tables (word_id lookups for search)
CREATE INDEX idx_rwe_word        ON ReefWordExports(word_id);
CREATE INDEX idx_twe_word        ON TownWordExports(word_id);
CREATE INDEX idx_iwe_word        ON IslandWordExports(word_id);

-- Equivalences (hash lookup for tokenization)
CREATE INDEX idx_equiv_hash      ON EquivalencesExport(variant_hash);

-- Pipeline
CREATE INDEX idx_iw_word         ON IslandWords(word_id);
CREATE INDEX idx_seed_town       ON SeedWords(town_id);
CREATE INDEX idx_seed_word       ON SeedWords(word_id);
CREATE INDEX idx_at_town         ON AugmentedTowns(town_id);
CREATE INDEX idx_at_word         ON AugmentedTowns(word_id);
CREATE INDEX idx_wis_island      ON WordIslandStats(island_id);
CREATE INDEX idx_wv_hash         ON WordVariants(variant_hash);
CREATE INDEX idx_wv_word         ON WordVariants(word_id);
