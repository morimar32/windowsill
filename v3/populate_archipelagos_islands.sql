-- ============================================================
-- Windowsill V3 — Populate Archipelagos & Islands
-- ============================================================
--
-- Populates the first two tiers of the hierarchy.
-- Run against a fresh v3 database (after schema.sql).
--
-- Sources:
--   - FBK WordNet Domain Hierarchy (WDH) top two levels
--   - Extended for modern topics (AI/ML, streaming, crypto, etc.)
--   - Existing 258-domain coverage preserved
--
-- 6 archipelagos, 41 islands (40 topical + 1 bucket).
--
-- Comments list expected town-level domains per island —
-- these are populated in a separate script.
--
-- ============================================================

BEGIN TRANSACTION;

-- ============================================================
-- ARCHIPELAGOS (6)
-- ============================================================

INSERT INTO Archipelagos (name) VALUES
    ('Applied Science'),
    ('Pure Science'),
    ('Social Science'),
    ('Humanities'),
    ('Free Time'),
    ('Doctrines');


-- ============================================================
-- ISLANDS
-- ============================================================

-- ------------------------------------------------------------
-- Applied Science (10 islands)
-- ------------------------------------------------------------

INSERT INTO Islands (archipelago_id, name)
SELECT a.archipelago_id, v.name
FROM Archipelagos a
CROSS JOIN (
    -- Medicine: ophthalmology, otology, neurology, radiology, pharmacology,
    --   obstetrics, dentistry, psychiatry, immunology, epidemiology,
    --   embryology, bacteriology, anatomy, physiology, toxicology,
    --   neuroscience, health food
    -- Also rescued non-domain: muscle (1921 words, anatomy/physiology)
    SELECT 'Medicine' AS name
    UNION ALL
    -- Engineering: electricity, electronics, jet engine, materials_science,
    --   3d_printing, nanotechnology, renewable_energy, robotics
    SELECT 'Engineering'
    UNION ALL
    -- Computer Science: artificial_intelligence, machine_learning,
    --   cloud_computing, cybersecurity, distributed_systems,
    --   mobile_computing, network_engineering, software_testing,
    --   version_control, bioinformatics, cryptography,
    --   augmented_reality, virtual_reality, quantum_computing,
    --   user_experience, agile_methodology
    -- Stays as one island; data-driven split later if disproportionate
    SELECT 'Computer Science'
    UNION ALL
    -- Architecture: classical architecture, construction
    SELECT 'Architecture'
    UNION ALL
    -- Agriculture: farming, livestock, vegetation, forestry, viticulture
    SELECT 'Agriculture'
    UNION ALL
    -- Media & Communications: broadcasting, telephone, recording,
    --   streaming_media, social_media, news article, publication
    SELECT 'Media & Communications'
    UNION ALL
    -- Military: navy, military law, terrorism
    SELECT 'Military'
    UNION ALL
    -- Veterinary Medicine: narrow but WDH-defined
    SELECT 'Veterinary Medicine'
    UNION ALL
    -- Transportation: aircraft, motor vehicle, navigation, train,
    --   aeronautics, spaceflight (shared with Astronomy)
    SELECT 'Transportation'
    UNION ALL
    -- Manufacturing: carpentry, masonry, ceramics, mining, metallurgy,
    --   factory, industry, furniture, containers
    SELECT 'Manufacturing'
) v
WHERE a.name = 'Applied Science';


-- ------------------------------------------------------------
-- Pure Science (6 islands)
-- ------------------------------------------------------------

INSERT INTO Islands (archipelago_id, name)
SELECT a.archipelago_id, v.name
FROM Archipelagos a
CROSS JOIN (
    -- Mathematics: arithmetic, geometry, statistics, logic,
    --   numeration system
    SELECT 'Mathematics' AS name
    UNION ALL
    -- Physics: mechanics, nuclear physics, particle physics,
    --   thermodynamics, relativity, optics, acoustics,
    --   particle_accelerators
    SELECT 'Physics'
    UNION ALL
    -- Chemistry: physical chemistry, biochemistry, crystallography
    SELECT 'Chemistry'
    UNION ALL
    -- Biology: zoology, botany, ecology, genetics, genomics,
    --   embryology, bacteriology, paleontology
    -- Also rescued non-domains: quadruped (1811), ruminant (1703)
    SELECT 'Biology'
    UNION ALL
    -- Astronomy: spaceflight (shared with Transportation)
    SELECT 'Astronomy'
    UNION ALL
    -- Earth Science: geology, meteorology, tectonics, mineralogy,
    --   climate_science, ocean
    -- Also rescued non-domains: lake (1193), river (1144), ocean (2042)
    SELECT 'Earth Science'
) v
WHERE a.name = 'Pure Science';


-- ------------------------------------------------------------
-- Social Science (9 islands)
-- ------------------------------------------------------------

INSERT INTO Islands (archipelago_id, name)
SELECT a.archipelago_id, v.name
FROM Archipelagos a
CROSS JOIN (
    -- Economics: pure economic theory
    SELECT 'Economics' AS name
    UNION ALL
    -- Finance: cryptocurrency, commerce-adjacent financial topics
    SELECT 'Finance'
    UNION ALL
    -- Commerce: business, trade, corporation, auction, supply_chain
    SELECT 'Commerce'
    UNION ALL
    -- Law: contract law, roman law, military law, crime, trademark
    SELECT 'Law'
    UNION ALL
    -- Politics: diplomacy
    SELECT 'Politics'
    UNION ALL
    -- Psychology: psychoanalysis, psychophysics
    -- Also rescued non-domain: jung (1447 words)
    SELECT 'Psychology'
    UNION ALL
    -- Sociology: social phenomena
    SELECT 'Sociology'
    UNION ALL
    -- Education: academia, university, library science, information science
    SELECT 'Education'
    UNION ALL
    -- Anthropology: archeology
    -- Also rescued non-domains: homo (1482), ethiopia (1401),
    --   pakistan (1386), west indies (1572), jamaica (1024)
    SELECT 'Anthropology'
) v
WHERE a.name = 'Social Science';


-- ------------------------------------------------------------
-- Humanities (8 topical islands)
-- ------------------------------------------------------------

INSERT INTO Islands (archipelago_id, name)
SELECT a.archipelago_id, v.name
FROM Archipelagos a
CROSS JOIN (
    -- Art: fashion, ceramics (art side), photography
    SELECT 'Art' AS name
    UNION ALL
    -- Music: singing, music genre
    SELECT 'Music'
    UNION ALL
    -- Literature: science fiction, fairytale, rhetoric
    -- Also rescued non-domain: ovid (2478 words, literary/poetic)
    SELECT 'Literature'
    UNION ALL
    -- Linguistics: grammar, phonetics, language
    SELECT 'Linguistics'
    UNION ALL
    -- History: middle ages, heraldry
    SELECT 'History'
    UNION ALL
    -- Philosophy: existentialism
    SELECT 'Philosophy'
    UNION ALL
    -- Performing Arts: movie, drama, dance
    SELECT 'Performing Arts'
    UNION ALL
    -- Linguistic Register: slang, euphemism, archaism, dialect,
    --   formality, regionalism, trope, irony, disparagement, obscenity,
    --   ethnic slur, abbreviation, acronym, blend, combining form,
    --   comparative, plural, synecdoche, metonymy, intensifier,
    --   handwriting, phonology, african american vernacular english,
    --   street name
    -- Bucket island: words identified but excluded from topic scoring
    SELECT 'Linguistic Register'
) v
WHERE a.name = 'Humanities';

-- Linguistic Register is a non-topical bucket
UPDATE Islands
SET is_bucket = 1
WHERE name = 'Linguistic Register';


-- ------------------------------------------------------------
-- Free Time (4 islands)
-- ------------------------------------------------------------

INSERT INTO Islands (archipelago_id, name)
SELECT a.archipelago_id, v.name
FROM Archipelagos a
CROSS JOIN (
    -- Sport: ball game, water sport, gymnastics, horse racing, riding,
    --   dressage, mountain climbing, fishing,
    --   + 24 merged sports from consolidation map
    SELECT 'Sport' AS name
    UNION ALL
    -- Games: craps, esports, video_gaming, + 7 merged games
    SELECT 'Games'
    UNION ALL
    -- Gastronomy: cooking, tasting, winemaking, health food
    SELECT 'Gastronomy'
    UNION ALL
    -- Hobbies & Crafts: handicraft, heraldry (shared), numismatics-type
    SELECT 'Hobbies & Crafts'
) v
WHERE a.name = 'Free Time';


-- ------------------------------------------------------------
-- Doctrines (4 islands)
-- ------------------------------------------------------------

INSERT INTO Islands (archipelago_id, name)
SELECT a.archipelago_id, v.name
FROM Archipelagos a
CROSS JOIN (
    -- Religion: buddhism, hinduism, taoism, zen, gnosticism,
    --   + 21 merged denominations
    SELECT 'Religion' AS name
    UNION ALL
    -- Mythology: norse mythology, arthurian legend, fairytale (shared),
    --   + 6 merged mythologies
    SELECT 'Mythology'
    UNION ALL
    -- Occultism: alchemy, astrology, spiritualism
    SELECT 'Occultism'
    UNION ALL
    -- Ideology: marxism
    SELECT 'Ideology'
) v
WHERE a.name = 'Doctrines';


COMMIT;


-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Expected: 6
SELECT 'Archipelago count: ' || COUNT(*) FROM Archipelagos;

-- Expected: 41
SELECT 'Island count: ' || COUNT(*) FROM Islands;

-- Expected: 1
SELECT 'Bucket count: ' || COUNT(*) FROM Islands WHERE is_bucket = 1;

-- Distribution per archipelago
SELECT a.name AS archipelago, COUNT(i.island_id) AS islands
FROM Archipelagos a
JOIN Islands i USING (archipelago_id)
GROUP BY a.name
ORDER BY a.archipelago_id;
