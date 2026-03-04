-- ============================================================
-- Windowsill V3 — Populate Towns
-- ============================================================
--
-- Populates the third tier of the hierarchy: towns within islands.
-- Run against a v3 database after populate_archipelagos_islands.sql
-- and BEFORE populate_bucket_islands.sql.
--
-- Sources:
--   - Contextual Wikipedia pulls per island
--   - WDH level-3 categories where available
--   - Extended for modern topics and coverage gaps
--
-- 308 towns across 41 islands:
--   - 297 topical towns across 40 topical islands
--   - 11 Linguistic Register towns (bucket island, created in
--     populate_archipelagos_islands.sql)
--
-- Each INSERT uses CROSS JOIN from the Islands table by name,
-- so island IDs are resolved at runtime.
--
-- ============================================================

BEGIN TRANSACTION;

-- ============================================================
-- APPLIED SCIENCE (10 islands, 87 towns)
-- ============================================================

-- ------------------------------------------------------------
-- Agriculture — Applied Science (6 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Aquaculture'  AS name UNION ALL
    SELECT 'Farming'              UNION ALL
    SELECT 'Forestry'             UNION ALL
    SELECT 'Horticulture'         UNION ALL
    SELECT 'Livestock'            UNION ALL
    SELECT 'Viticulture'
) v
WHERE i.name = 'Agriculture';

-- ------------------------------------------------------------
-- Architecture — Applied Science (5 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Classical Architecture' AS name UNION ALL
    SELECT 'Construction'                   UNION ALL
    SELECT 'Interior Design'                UNION ALL
    SELECT 'Landscape Architecture'         UNION ALL
    SELECT 'Urban Planning'
) v
WHERE i.name = 'Architecture';

-- ------------------------------------------------------------
-- Computer Science — Applied Science (16 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Algorithms'            AS name UNION ALL
    SELECT 'Artificial Intelligence'       UNION ALL
    SELECT 'Bioinformatics'                UNION ALL
    SELECT 'Cloud Computing'               UNION ALL
    SELECT 'Computer Graphics'             UNION ALL
    SELECT 'Cybersecurity'                 UNION ALL
    SELECT 'Data Science'                  UNION ALL
    SELECT 'Databases'                     UNION ALL
    SELECT 'Distributed Systems'           UNION ALL
    SELECT 'Mobile Computing'              UNION ALL
    SELECT 'Network Engineering'           UNION ALL
    SELECT 'Operating Systems'             UNION ALL
    SELECT 'Programming Languages'         UNION ALL
    SELECT 'Quantum Computing'             UNION ALL
    SELECT 'Software Engineering'          UNION ALL
    SELECT 'User Experience'
) v
WHERE i.name = 'Computer Science';

-- ------------------------------------------------------------
-- Engineering — Applied Science (9 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT '3D Printing'          AS name UNION ALL
    SELECT 'Chemical Engineering'         UNION ALL
    SELECT 'Civil Engineering'            UNION ALL
    SELECT 'Electrical Engineering'       UNION ALL
    SELECT 'Materials Science'            UNION ALL
    SELECT 'Mechanical Engineering'       UNION ALL
    SELECT 'Nanotechnology'               UNION ALL
    SELECT 'Renewable Energy'             UNION ALL
    SELECT 'Robotics'
) v
WHERE i.name = 'Engineering';

-- ------------------------------------------------------------
-- Manufacturing — Applied Science (9 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Carpentry'              AS name UNION ALL
    SELECT 'Ceramics'                       UNION ALL
    SELECT 'Containers & Packaging'         UNION ALL
    SELECT 'Factory Production'             UNION ALL
    SELECT 'Furniture Making'               UNION ALL
    SELECT 'Masonry'                        UNION ALL
    SELECT 'Metallurgy'                     UNION ALL
    SELECT 'Mining'                         UNION ALL
    SELECT 'Textiles'
) v
WHERE i.name = 'Manufacturing';

-- ------------------------------------------------------------
-- Media & Communications — Applied Science (7 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Advertising'      AS name UNION ALL
    SELECT 'Broadcasting'             UNION ALL
    SELECT 'Journalism'               UNION ALL
    SELECT 'Publishing'               UNION ALL
    SELECT 'Social Media'             UNION ALL
    SELECT 'Streaming Media'          UNION ALL
    SELECT 'Telephony'
) v
WHERE i.name = 'Media & Communications';

-- ------------------------------------------------------------
-- Medicine — Applied Science (21 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Anatomy'        AS name UNION ALL
    SELECT 'Bacteriology'           UNION ALL
    SELECT 'Cardiology'             UNION ALL
    SELECT 'Dentistry'              UNION ALL
    SELECT 'Dermatology'            UNION ALL
    SELECT 'Embryology'             UNION ALL
    SELECT 'Epidemiology'           UNION ALL
    SELECT 'Health & Nutrition'     UNION ALL
    SELECT 'Immunology'             UNION ALL
    SELECT 'Neurology'              UNION ALL
    SELECT 'Obstetrics'             UNION ALL
    SELECT 'Oncology'               UNION ALL
    SELECT 'Ophthalmology'          UNION ALL
    SELECT 'Otology'                UNION ALL
    SELECT 'Pathology'              UNION ALL
    SELECT 'Pharmacology'           UNION ALL
    SELECT 'Physiology'             UNION ALL
    SELECT 'Psychiatry'             UNION ALL
    SELECT 'Radiology'              UNION ALL
    SELECT 'Surgery'                UNION ALL
    SELECT 'Toxicology'
) v
WHERE i.name = 'Medicine';

-- ------------------------------------------------------------
-- Military — Applied Science (7 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Intelligence'         AS name UNION ALL
    SELECT 'Military History'             UNION ALL
    SELECT 'Military Law'                 UNION ALL
    SELECT 'Military Strategy'            UNION ALL
    SELECT 'Navy'                         UNION ALL
    SELECT 'Terrorism'                    UNION ALL
    SELECT 'Weapons & Ordnance'
) v
WHERE i.name = 'Military';

-- ------------------------------------------------------------
-- Transportation — Applied Science (6 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Aviation'           AS name UNION ALL
    SELECT 'Maritime Transport'         UNION ALL
    SELECT 'Motor Vehicles'             UNION ALL
    SELECT 'Navigation'                 UNION ALL
    SELECT 'Rail Transport'             UNION ALL
    SELECT 'Spaceflight'
) v
WHERE i.name = 'Transportation';

-- ------------------------------------------------------------
-- Veterinary Medicine — Applied Science (1 town)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Veterinary Medicine' AS name
) v
WHERE i.name = 'Veterinary Medicine';


-- ============================================================
-- PURE SCIENCE (6 islands, 47 towns)
-- ============================================================

-- ------------------------------------------------------------
-- Astronomy — Pure Science (5 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Astrophysics'              AS name UNION ALL
    SELECT 'Cosmology'                         UNION ALL
    SELECT 'Observational Astronomy'           UNION ALL
    SELECT 'Planetary Science'                 UNION ALL
    SELECT 'Stellar Astronomy'
) v
WHERE i.name = 'Astronomy';

-- ------------------------------------------------------------
-- Biology — Pure Science (9 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Botany'               AS name UNION ALL
    SELECT 'Cell Biology'                 UNION ALL
    SELECT 'Ecology'                      UNION ALL
    SELECT 'Evolutionary Biology'         UNION ALL
    SELECT 'Genetics'                     UNION ALL
    SELECT 'Marine Biology'               UNION ALL
    SELECT 'Microbiology'                 UNION ALL
    SELECT 'Paleontology'                 UNION ALL
    SELECT 'Zoology'
) v
WHERE i.name = 'Biology';

-- ------------------------------------------------------------
-- Chemistry — Pure Science (6 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Analytical Chemistry'  AS name UNION ALL
    SELECT 'Biochemistry'                  UNION ALL
    SELECT 'Crystallography'               UNION ALL
    SELECT 'Inorganic Chemistry'           UNION ALL
    SELECT 'Organic Chemistry'             UNION ALL
    SELECT 'Physical Chemistry'
) v
WHERE i.name = 'Chemistry';

-- ------------------------------------------------------------
-- Earth Science — Pure Science (7 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Geology'       AS name UNION ALL
    SELECT 'Glaciology'            UNION ALL
    SELECT 'Hydrology'             UNION ALL
    SELECT 'Meteorology'           UNION ALL
    SELECT 'Oceanography'          UNION ALL
    SELECT 'Tectonics'             UNION ALL
    SELECT 'Volcanology'
) v
WHERE i.name = 'Earth Science';

-- ------------------------------------------------------------
-- Mathematics — Pure Science (10 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Algebra'             AS name UNION ALL
    SELECT 'Arithmetic'                  UNION ALL
    SELECT 'Calculus'                    UNION ALL
    SELECT 'Combinatorics'               UNION ALL
    SELECT 'Geometry'                    UNION ALL
    SELECT 'Logic'                       UNION ALL
    SELECT 'Number Theory'               UNION ALL
    SELECT 'Numeration Systems'          UNION ALL
    SELECT 'Statistics'                  UNION ALL
    SELECT 'Topology'
) v
WHERE i.name = 'Mathematics';

-- ------------------------------------------------------------
-- Physics — Pure Science (10 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Acoustics'        AS name UNION ALL
    SELECT 'Electromagnetism'         UNION ALL
    SELECT 'Fluid Dynamics'           UNION ALL
    SELECT 'Mechanics'                UNION ALL
    SELECT 'Nuclear Physics'          UNION ALL
    SELECT 'Optics'                   UNION ALL
    SELECT 'Particle Physics'         UNION ALL
    SELECT 'Quantum Mechanics'        UNION ALL
    SELECT 'Relativity'               UNION ALL
    SELECT 'Thermodynamics'
) v
WHERE i.name = 'Physics';


-- ============================================================
-- SOCIAL SCIENCE (9 islands, 52 towns)
-- ============================================================

-- ------------------------------------------------------------
-- Anthropology — Social Science (5 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Archaeology'              AS name UNION ALL
    SELECT 'Cultural Anthropology'            UNION ALL
    SELECT 'Ethnography'                      UNION ALL
    SELECT 'Linguistic Anthropology'          UNION ALL
    SELECT 'Physical Anthropology'
) v
WHERE i.name = 'Anthropology';

-- ------------------------------------------------------------
-- Commerce — Social Science (7 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Auction'       AS name UNION ALL
    SELECT 'Business'              UNION ALL
    SELECT 'Marketing'             UNION ALL
    SELECT 'Real Estate'           UNION ALL
    SELECT 'Retail'                UNION ALL
    SELECT 'Supply Chain'          UNION ALL
    SELECT 'Trade'
) v
WHERE i.name = 'Commerce';

-- ------------------------------------------------------------
-- Economics — Social Science (5 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Development Economics' AS name UNION ALL
    SELECT 'Econometrics'                  UNION ALL
    SELECT 'Labor Economics'               UNION ALL
    SELECT 'Macroeconomics'                UNION ALL
    SELECT 'Microeconomics'
) v
WHERE i.name = 'Economics';

-- ------------------------------------------------------------
-- Education — Social Science (5 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Academia'        AS name UNION ALL
    SELECT 'Curriculum'              UNION ALL
    SELECT 'E-Learning'              UNION ALL
    SELECT 'Library Science'         UNION ALL
    SELECT 'Pedagogy'
) v
WHERE i.name = 'Education';

-- ------------------------------------------------------------
-- Finance — Social Science (6 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Accounting'      AS name UNION ALL
    SELECT 'Banking'                 UNION ALL
    SELECT 'Cryptocurrency'          UNION ALL
    SELECT 'Insurance'               UNION ALL
    SELECT 'Investment'              UNION ALL
    SELECT 'Taxation'
) v
WHERE i.name = 'Finance';

-- ------------------------------------------------------------
-- Law — Social Science (7 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Constitutional Law'    AS name UNION ALL
    SELECT 'Contract Law'                  UNION ALL
    SELECT 'Criminal Law'                  UNION ALL
    SELECT 'Intellectual Property'         UNION ALL
    SELECT 'International Law'             UNION ALL
    SELECT 'Organized Crime'               UNION ALL
    SELECT 'Roman Law'
) v
WHERE i.name = 'Law';

-- ------------------------------------------------------------
-- Politics — Social Science (5 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Diplomacy'            AS name UNION ALL
    SELECT 'Elections'                    UNION ALL
    SELECT 'Governance'                   UNION ALL
    SELECT 'Political Theory'             UNION ALL
    SELECT 'Public Administration'
) v
WHERE i.name = 'Politics';

-- ------------------------------------------------------------
-- Psychology — Social Science (7 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Clinical Psychology'       AS name UNION ALL
    SELECT 'Cognitive Psychology'              UNION ALL
    SELECT 'Developmental Psychology'          UNION ALL
    SELECT 'Neuropsychology'                   UNION ALL
    SELECT 'Psychoanalysis'                    UNION ALL
    SELECT 'Psychophysics'                     UNION ALL
    SELECT 'Social Psychology'
) v
WHERE i.name = 'Psychology';

-- ------------------------------------------------------------
-- Sociology — Social Science (5 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Criminology'            AS name UNION ALL
    SELECT 'Demography'                     UNION ALL
    SELECT 'Gender Studies'                  UNION ALL
    SELECT 'Social Stratification'           UNION ALL
    SELECT 'Urban Sociology'
) v
WHERE i.name = 'Sociology';


-- ============================================================
-- HUMANITIES (8 topical islands, 50 towns)
-- ============================================================

-- ------------------------------------------------------------
-- Art — Humanities (7 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Fashion'         AS name UNION ALL
    SELECT 'Graphic Design'          UNION ALL
    SELECT 'Medium'                  UNION ALL
    SELECT 'Photography'             UNION ALL
    SELECT 'Printmaking'             UNION ALL
    SELECT 'Sculpture'               UNION ALL
    SELECT 'Visual Arts'
) v
WHERE i.name = 'Art';

-- ------------------------------------------------------------
-- History — Humanities (6 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Ancient History'   AS name UNION ALL
    SELECT 'Heraldry'                  UNION ALL
    SELECT 'Historiography'            UNION ALL
    SELECT 'Medieval History'          UNION ALL
    SELECT 'Modern History'            UNION ALL
    SELECT 'Social History'
) v
WHERE i.name = 'History';

-- ------------------------------------------------------------
-- Linguistics — Humanities (8 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Grammar'                AS name UNION ALL
    SELECT 'Historical Linguistics'         UNION ALL
    SELECT 'Morphology'                     UNION ALL
    SELECT 'Phonetics'                      UNION ALL
    SELECT 'Pragmatics'                     UNION ALL
    SELECT 'Semantics'                      UNION ALL
    SELECT 'Sociolinguistics'               UNION ALL
    SELECT 'Syntax'
) v
WHERE i.name = 'Linguistics';

-- ------------------------------------------------------------
-- Literature — Humanities (6 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Fairy Tales'               AS name UNION ALL
    SELECT 'Literary Criticism'                UNION ALL
    SELECT 'Non-Fiction'                       UNION ALL
    SELECT 'Novel'                             UNION ALL
    SELECT 'Poetry'                            UNION ALL
    SELECT 'Science Fiction'
) v
WHERE i.name = 'Literature';

-- ------------------------------------------------------------
-- Music — Humanities (8 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Music Genres'       AS name UNION ALL
    SELECT 'Music History'              UNION ALL
    SELECT 'Music Production'           UNION ALL
    SELECT 'Music Theory'               UNION ALL
    SELECT 'Musical Instruments'        UNION ALL
    SELECT 'Opera'                      UNION ALL
    SELECT 'Singing'                    UNION ALL
    SELECT 'Sound Recording'
) v
WHERE i.name = 'Music';

-- ------------------------------------------------------------
-- Performing Arts — Humanities (7 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Cinema'              AS name UNION ALL
    SELECT 'Circus Arts'                 UNION ALL
    SELECT 'Dance'                       UNION ALL
    SELECT 'Drama & Playwriting'         UNION ALL
    SELECT 'Film & Video Art'            UNION ALL
    SELECT 'Stand-Up Comedy'             UNION ALL
    SELECT 'Theater'
) v
WHERE i.name = 'Performing Arts';

-- ------------------------------------------------------------
-- Philosophy — Humanities (8 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Aesthetics'             AS name UNION ALL
    SELECT 'Epistemology'                   UNION ALL
    SELECT 'Ethics'                         UNION ALL
    SELECT 'Existentialism'                 UNION ALL
    SELECT 'Metaphysics'                    UNION ALL
    SELECT 'Philosophy of Mind'             UNION ALL
    SELECT 'Philosophy of Science'          UNION ALL
    SELECT 'Political Philosophy'
) v
WHERE i.name = 'Philosophy';


-- ============================================================
-- FREE TIME (4 islands, 32 towns)
-- ============================================================

-- ------------------------------------------------------------
-- Games — Free Time (7 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Board Games'        AS name UNION ALL
    SELECT 'Card Games'                 UNION ALL
    SELECT 'Dice Games'                 UNION ALL
    SELECT 'Esports'                    UNION ALL
    SELECT 'Puzzles'                    UNION ALL
    SELECT 'Role-Playing Games'         UNION ALL
    SELECT 'Video Games'
) v
WHERE i.name = 'Games';

-- ------------------------------------------------------------
-- Gastronomy — Free Time (6 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Baking'        AS name UNION ALL
    SELECT 'Beverages'             UNION ALL
    SELECT 'Cooking'               UNION ALL
    SELECT 'Food Science'          UNION ALL
    SELECT 'Tasting'               UNION ALL
    SELECT 'Winemaking'
) v
WHERE i.name = 'Gastronomy';

-- ------------------------------------------------------------
-- Hobbies & Crafts — Free Time (6 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Collecting'    AS name UNION ALL
    SELECT 'Gardening'             UNION ALL
    SELECT 'Handicraft'            UNION ALL
    SELECT 'Model Building'        UNION ALL
    SELECT 'Needlework'            UNION ALL
    SELECT 'Woodworking'
) v
WHERE i.name = 'Hobbies & Crafts';

-- ------------------------------------------------------------
-- Sport — Free Time (13 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Archery & Shooting' AS name UNION ALL
    SELECT 'Athletics'                  UNION ALL
    SELECT 'Ball Games'                 UNION ALL
    SELECT 'Combat Sports'              UNION ALL
    SELECT 'Cycling'                    UNION ALL
    SELECT 'Fishing'                    UNION ALL
    SELECT 'Gymnastics'                 UNION ALL
    SELECT 'Horse Racing'               UNION ALL
    SELECT 'Motorsport'                 UNION ALL
    SELECT 'Mountain Sports'            UNION ALL
    SELECT 'Racquet Sports'             UNION ALL
    SELECT 'Water Sports'               UNION ALL
    SELECT 'Winter Sports'
) v
WHERE i.name = 'Sport';


-- ============================================================
-- DOCTRINES (4 islands, 29 towns)
-- ============================================================

-- ------------------------------------------------------------
-- Ideology — Doctrines (7 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Anarchism'       AS name UNION ALL
    SELECT 'Conservatism'            UNION ALL
    SELECT 'Environmentalism'        UNION ALL
    SELECT 'Feminism'                UNION ALL
    SELECT 'Liberalism'              UNION ALL
    SELECT 'Marxism'                 UNION ALL
    SELECT 'Nationalism'
) v
WHERE i.name = 'Ideology';

-- ------------------------------------------------------------
-- Mythology — Doctrines (8 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Arthurian Legend'       AS name UNION ALL
    SELECT 'Celtic Mythology'               UNION ALL
    SELECT 'Egyptian Mythology'             UNION ALL
    SELECT 'Folklore'                       UNION ALL
    SELECT 'Greek Mythology'                UNION ALL
    SELECT 'Mythological Literature'        UNION ALL
    SELECT 'Norse Mythology'                UNION ALL
    SELECT 'Roman Mythology'
) v
WHERE i.name = 'Mythology';

-- ------------------------------------------------------------
-- Occultism — Doctrines (5 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Alchemy'       AS name UNION ALL
    SELECT 'Astrology'             UNION ALL
    SELECT 'Divination'            UNION ALL
    SELECT 'Hermeticism'           UNION ALL
    SELECT 'Spiritualism'
) v
WHERE i.name = 'Occultism';

-- ------------------------------------------------------------
-- Religion — Doctrines (10 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Buddhism'              AS name UNION ALL
    SELECT 'Christianity'                  UNION ALL
    SELECT 'Comparative Religion'          UNION ALL
    SELECT 'Gnosticism'                    UNION ALL
    SELECT 'Hinduism'                      UNION ALL
    SELECT 'Islam'                         UNION ALL
    SELECT 'Judaism'                       UNION ALL
    SELECT 'Sikhism'                       UNION ALL
    SELECT 'Taoism'                        UNION ALL
    SELECT 'Theology'
) v
WHERE i.name = 'Religion';


-- ============================================================
-- HUMANITIES — LINGUISTIC REGISTER (bucket island, 11 towns)
-- ============================================================
-- Linguistic Register is a bucket island (is_bucket = 1) created
-- in populate_archipelagos_islands.sql.  Its towns classify words
-- by register/style rather than topic — identified but excluded
-- from topic scoring.

-- ------------------------------------------------------------
-- Linguistic Register — Humanities (11 towns)
-- ------------------------------------------------------------
INSERT INTO Towns (island_id, name)
SELECT i.island_id, v.name
FROM Islands i
CROSS JOIN (
    SELECT 'Archaism'            AS name UNION ALL
    SELECT 'Dialect'                     UNION ALL
    SELECT 'Euphemism'                   UNION ALL
    SELECT 'Formality'                   UNION ALL
    SELECT 'Handwriting'                 UNION ALL
    SELECT 'Regionalism'                 UNION ALL
    SELECT 'Rhetorical Devices'          UNION ALL
    SELECT 'Slang'                       UNION ALL
    SELECT 'Street Names'                UNION ALL
    SELECT 'Taboo Language'              UNION ALL
    SELECT 'Word Formation'
) v
WHERE i.name = 'Linguistic Register';


COMMIT;


-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Expected: 308 (297 topical + 11 Linguistic Register)
SELECT 'Total towns created by this script: ' || COUNT(*)
FROM Towns t
JOIN Islands i USING (island_id)
WHERE i.is_bucket = 0 OR i.name = 'Linguistic Register';

-- Expected: 297
SELECT 'Topical towns (is_bucket=0): ' || COUNT(*)
FROM Towns t
JOIN Islands i USING (island_id)
WHERE i.is_bucket = 0;

-- Expected: 11
SELECT 'Linguistic Register towns: ' || COUNT(*)
FROM Towns t
JOIN Islands i USING (island_id)
WHERE i.name = 'Linguistic Register';

-- Per-archipelago breakdown
SELECT a.name AS archipelago,
       COUNT(DISTINCT i.island_id) AS islands,
       COUNT(t.town_id) AS towns
FROM Archipelagos a
JOIN Islands i USING (archipelago_id)
LEFT JOIN Towns t USING (island_id)
WHERE i.is_bucket = 0 OR i.name = 'Linguistic Register'
GROUP BY a.name
ORDER BY a.archipelago_id;

-- Per-island breakdown (all islands covered by this script)
SELECT a.name AS archipelago,
       i.name AS island,
       i.is_bucket,
       COUNT(t.town_id) AS towns
FROM Archipelagos a
JOIN Islands i USING (archipelago_id)
LEFT JOIN Towns t USING (island_id)
WHERE i.is_bucket = 0 OR i.name = 'Linguistic Register'
GROUP BY i.island_id
ORDER BY a.name, i.name;
