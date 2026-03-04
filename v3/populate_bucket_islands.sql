-- ============================================================
-- Windowsill V3 — Populate Bucket Islands for Dropped Domains
-- ============================================================
--
-- Brings back ALL 24 WordNet domains that were dropped in the
-- v2→v3 transition.  Every domain becomes a town under one of
-- three new bucket islands (is_bucket = 1).
--
-- Bucket islands are identified but excluded from topic scoring.
-- This lets us distinguish "genuinely orphaned" words from words
-- that have a non-topical classification.
--
-- 3 new islands, 24 new towns.
--
-- Run against an existing v3 database (after populate_archipelagos_islands.sql
-- and populate_towns.sql).
--
-- ============================================================

BEGIN TRANSACTION;

-- ============================================================
-- BUCKET ISLANDS (3)
-- ============================================================

-- Languages: words tagged by their language of origin.
-- Non-topical — knowing a word is from French doesn't indicate subject matter.
INSERT INTO Islands (archipelago_id, name, is_bucket)
SELECT a.archipelago_id, 'Languages', 1
FROM Archipelagos a
WHERE a.name = 'Humanities';

-- Regional: geographic/cultural region markers from WordNet.
-- Non-topical — "brazil", "canada", "haiti" tag provenance, not subject.
INSERT INTO Islands (archipelago_id, name, is_bucket)
SELECT a.archipelago_id, 'Regional', 1
FROM Archipelagos a
WHERE a.name = 'Social Science';

-- Miscellaneous: grab-bag of WordNet categories that don't form
-- coherent topical groups (aristotle, genesis, growth, etc.).
-- Kept as bucket so their words aren't counted as orphans.
INSERT INTO Islands (archipelago_id, name, is_bucket)
SELECT a.archipelago_id, 'Miscellaneous', 1
FROM Archipelagos a
WHERE a.name = 'Humanities';


-- ============================================================
-- TOWNS (24) — one per dropped v2 domain
-- ============================================================

-- Languages (13 towns)
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Arabic'   AS name UNION ALL
    SELECT 'French'           UNION ALL
    SELECT 'German'           UNION ALL
    SELECT 'Hebrew'           UNION ALL
    SELECT 'Italian'          UNION ALL
    SELECT 'Japanese'         UNION ALL
    SELECT 'Latin'            UNION ALL
    SELECT 'Persian'          UNION ALL
    SELECT 'Pashto'           UNION ALL
    SELECT 'Sanskrit'         UNION ALL
    SELECT 'Spanish'          UNION ALL
    SELECT 'Swahili'          UNION ALL
    SELECT 'Yiddish'
) v
WHERE i.name = 'Languages';

-- Regional (3 towns)
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Brazil'  AS name UNION ALL
    SELECT 'Canada'          UNION ALL
    SELECT 'Haiti'
) v
WHERE i.name = 'Regional';

-- Miscellaneous (7 towns)
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Aristotle' AS name UNION ALL
    SELECT 'Genesis'           UNION ALL
    SELECT 'Growth'            UNION ALL
    SELECT 'Holism'            UNION ALL
    SELECT 'Vent'              UNION ALL
    SELECT 'Wit'               UNION ALL
    SELECT 'Wordnet'
) v
WHERE i.name = 'Miscellaneous';


COMMIT;


-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Expected: 4 (Linguistic Register + 3 new)
SELECT 'Bucket island count: ' || COUNT(*)
FROM Islands WHERE is_bucket = 1;

-- Expected: 44 (41 + 3 new)
SELECT 'Total island count: ' || COUNT(*)
FROM Islands;

-- Expected: 332 (308 + 24 new)
SELECT 'Total town count: ' || COUNT(*)
FROM Towns;

-- New bucket islands with their towns
SELECT i.name AS island, i.is_bucket, COUNT(t.town_id) AS town_count
FROM Islands i
LEFT JOIN Towns t USING (island_id)
WHERE i.is_bucket = 1
GROUP BY i.island_id
ORDER BY i.island_id;
