"""
Transfer seed words from v2 augmented_domains into v3 SeedWords table.

Sources used:
  - wordnet: always included
  - claude_augmented: always included (carries core/peripheral confidence)
  - xgboost: ONLY when old domain name exactly matches new town name
  - pipeline, domain_name: skipped

All WordNet domains are mapped. Non-topical domains (languages, regional,
miscellaneous) are mapped to bucket islands (is_bucket=1).
"""

import sqlite3
from collections import defaultdict

V2_DB = "v2.db"
V3_DB = "v3/windowsill.db"

# ============================================================
# DOMAIN → TOWN MAPPING
# ============================================================
# Format: 'old_domain': ('Island', 'Town')
# Domains not listed here are DROPPED.
#
# When old domain is broad (matches island name), it maps to
# the most general town. XGBoost retraining will sort properly.
# ============================================================

DOMAIN_TO_TOWN = {
    # --- Applied Science: Medicine ---
    "ophthalmology":           ("Medicine", "Ophthalmology"),
    "otology":                 ("Medicine", "Otology"),
    "neurology":               ("Medicine", "Neurology"),
    "neurophysiology":         ("Medicine", "Neurology"),
    "neuroscience":            ("Medicine", "Neurology"),
    "neuroscience_imaging":    ("Medicine", "Neurology"),
    "radiology":               ("Medicine", "Radiology"),
    "pharmacology":            ("Medicine", "Pharmacology"),
    "narcotic":                ("Medicine", "Pharmacology"),
    "obstetrics":              ("Medicine", "Obstetrics"),
    "dentistry":               ("Medicine", "Dentistry"),
    "psychiatry":              ("Medicine", "Psychiatry"),
    "immunology":              ("Medicine", "Immunology"),
    "epidemiology":            ("Medicine", "Epidemiology"),
    "anatomy":                 ("Medicine", "Anatomy"),
    "muscle":                  ("Medicine", "Anatomy"),
    "physiology":              ("Medicine", "Physiology"),
    "toxicology":              ("Medicine", "Toxicology"),
    "embryology":              ("Medicine", "Embryology"),
    "bacteriology":            ("Medicine", "Bacteriology"),
    "health food":             ("Medicine", "Health & Nutrition"),
    "medicine":                ("Medicine", "Pathology"),  # broad → general catch-all

    # --- Applied Science: Engineering ---
    "electricity":             ("Engineering", "Electrical Engineering"),
    "electronics":             ("Engineering", "Electrical Engineering"),
    "jet engine":              ("Engineering", "Mechanical Engineering"),
    "materials_science":       ("Engineering", "Materials Science"),
    "nanotechnology":          ("Engineering", "Nanotechnology"),
    "renewable_energy":        ("Engineering", "Renewable Energy"),
    "robotics":                ("Engineering", "Robotics"),
    "3d_printing":             ("Engineering", "3D Printing"),
    "engineering":             ("Engineering", "Mechanical Engineering"),  # broad

    # --- Applied Science: Computer Science ---
    "artificial_intelligence": ("Computer Science", "Artificial Intelligence"),
    "machine_learning":        ("Computer Science", "Artificial Intelligence"),
    "cloud_computing":         ("Computer Science", "Cloud Computing"),
    "cybersecurity":           ("Computer Science", "Cybersecurity"),
    "cryptography":            ("Computer Science", "Cybersecurity"),
    "distributed_systems":     ("Computer Science", "Distributed Systems"),
    "mobile_computing":        ("Computer Science", "Mobile Computing"),
    "network_engineering":     ("Computer Science", "Network Engineering"),
    "software_testing":        ("Computer Science", "Software Engineering"),
    "version_control":         ("Computer Science", "Software Engineering"),
    "agile_methodology":       ("Computer Science", "Software Engineering"),
    "bioinformatics":          ("Computer Science", "Bioinformatics"),
    "computer graphics":       ("Computer Science", "Computer Graphics"),
    "augmented_reality":       ("Computer Science", "Computer Graphics"),
    "virtual_reality":         ("Computer Science", "Computer Graphics"),
    "quantum_computing":       ("Computer Science", "Quantum Computing"),
    "user_experience":         ("Computer Science", "User Experience"),
    "computer science":        ("Computer Science", "Algorithms"),  # broad
    "computer":                ("Computer Science", "Operating Systems"),  # broad

    # --- Applied Science: Architecture ---
    "classical architecture":  ("Architecture", "Classical Architecture"),
    "construction":            ("Architecture", "Construction"),
    "architecture":            ("Architecture", "Classical Architecture"),  # broad

    # --- Applied Science: Agriculture ---
    "farming":                 ("Agriculture", "Farming"),
    "livestock":               ("Agriculture", "Livestock"),
    "forestry":                ("Agriculture", "Forestry"),
    "viticulture":             ("Agriculture", "Viticulture"),
    "vegetation":              ("Agriculture", "Horticulture"),

    # --- Applied Science: Media & Communications ---
    "broadcasting":            ("Media & Communications", "Broadcasting"),
    "telephone":               ("Media & Communications", "Telephony"),
    "recording":               ("Music", "Sound Recording"),
    "streaming_media":         ("Media & Communications", "Streaming Media"),
    "social_media":            ("Media & Communications", "Social Media"),
    "news article":            ("Media & Communications", "Journalism"),
    "publication":             ("Media & Communications", "Publishing"),

    # --- Applied Science: Military ---
    "navy":                    ("Military", "Navy"),
    "military law":            ("Military", "Military Law"),
    "terrorism":               ("Military", "Terrorism"),
    "military":                ("Military", "Military Strategy"),  # broad

    # --- Applied Science: Veterinary Medicine ---
    "veterinary medicine":     ("Veterinary Medicine", "Veterinary Medicine"),

    # --- Applied Science: Transportation ---
    "aircraft":                ("Transportation", "Aviation"),
    "aeronautics":             ("Transportation", "Aviation"),
    "motor vehicle":           ("Transportation", "Motor Vehicles"),
    "navigation":              ("Transportation", "Navigation"),
    "train":                   ("Transportation", "Rail Transport"),
    "spaceflight":             ("Transportation", "Spaceflight"),
    "transportation":          ("Transportation", "Maritime Transport"),  # broad → catch-all

    # --- Applied Science: Manufacturing ---
    "carpentry":               ("Manufacturing", "Carpentry"),
    "masonry":                 ("Manufacturing", "Masonry"),
    "ceramics":                ("Manufacturing", "Ceramics"),
    "mining":                  ("Manufacturing", "Mining"),
    "metallurgy":              ("Manufacturing", "Metallurgy"),
    "factory":                 ("Manufacturing", "Factory Production"),
    "industry":                ("Manufacturing", "Factory Production"),
    "furniture":               ("Manufacturing", "Furniture Making"),
    "containers":              ("Manufacturing", "Containers & Packaging"),

    # --- Pure Science: Mathematics ---
    "arithmetic":              ("Mathematics", "Arithmetic"),
    "geometry":                ("Mathematics", "Geometry"),
    "statistics":              ("Mathematics", "Statistics"),
    "logic":                   ("Mathematics", "Logic"),
    "numeration system":       ("Mathematics", "Numeration Systems"),
    "mathematics":             ("Mathematics", "Algebra"),  # broad

    # --- Pure Science: Physics ---
    "mechanics":               ("Physics", "Mechanics"),
    "nuclear physics":         ("Physics", "Nuclear Physics"),
    "particle physics":        ("Physics", "Particle Physics"),
    "particle_accelerators":   ("Physics", "Particle Physics"),
    "thermodynamics":          ("Physics", "Thermodynamics"),
    "relativity":              ("Physics", "Relativity"),
    "optics":                  ("Physics", "Optics"),
    "acoustics":               ("Physics", "Acoustics"),
    "physics":                 ("Physics", "Electromagnetism"),  # broad

    # --- Pure Science: Chemistry ---
    "physical chemistry":      ("Chemistry", "Physical Chemistry"),
    "biochemistry":            ("Chemistry", "Biochemistry"),
    "crystallography":         ("Chemistry", "Crystallography"),
    "chemistry":               ("Chemistry", "Organic Chemistry"),  # broad

    # --- Pure Science: Biology ---
    "zoology":                 ("Biology", "Zoology"),
    "quadruped":               ("Biology", "Zoology"),
    "ruminant":                ("Biology", "Zoology"),
    "botany":                  ("Biology", "Botany"),
    "ecology":                 ("Biology", "Ecology"),
    "genetics":                ("Biology", "Genetics"),
    "genomics":                ("Biology", "Genetics"),
    "paleontology":            ("Biology", "Paleontology"),
    "biology":                 ("Biology", "Cell Biology"),  # broad

    # --- Pure Science: Astronomy ---
    "astronomy":               ("Astronomy", "Stellar Astronomy"),  # broad

    # --- Pure Science: Earth Science ---
    "geology":                 ("Earth Science", "Geology"),
    "mineralogy":              ("Earth Science", "Geology"),
    "meteorology":             ("Earth Science", "Meteorology"),
    "climate_science":         ("Earth Science", "Meteorology"),
    "tectonics":               ("Earth Science", "Tectonics"),
    "ocean":                   ("Earth Science", "Oceanography"),
    "lake":                    ("Earth Science", "Hydrology"),
    "river":                   ("Earth Science", "Hydrology"),

    # --- Social Science: Economics ---
    "economics":               ("Economics", "Macroeconomics"),  # broad

    # --- Social Science: Finance ---
    "finance":                 ("Finance", "Banking"),  # broad
    "cryptocurrency":          ("Finance", "Cryptocurrency"),

    # --- Social Science: Commerce ---
    "business":                ("Commerce", "Business"),
    "corporation":             ("Commerce", "Business"),
    "commerce":                ("Commerce", "Trade"),
    "trade":                   ("Commerce", "Trade"),
    "auction":                 ("Commerce", "Auction"),
    "supply_chain":            ("Commerce", "Supply Chain"),

    # --- Social Science: Law ---
    "contract law":            ("Law", "Contract Law"),
    "roman law":               ("Law", "Roman Law"),
    "crime":                   ("Law", "Criminal Law"),
    "trademark":               ("Law", "Intellectual Property"),
    "mafia":                   ("Law", "Organized Crime"),
    "law":                     ("Law", "Constitutional Law"),  # broad

    # --- Social Science: Politics ---
    "diplomacy":               ("Politics", "Diplomacy"),
    "politics":                ("Politics", "Political Theory"),  # broad

    # --- Social Science: Psychology ---
    "psychoanalysis":          ("Psychology", "Psychoanalysis"),
    "jung":                    ("Psychology", "Psychoanalysis"),
    "psychophysics":           ("Psychology", "Psychophysics"),
    "psychology":              ("Psychology", "Clinical Psychology"),  # broad

    # --- Social Science: Sociology ---
    "sociology":               ("Sociology", "Social Stratification"),  # broad

    # --- Social Science: Education ---
    "academia":                ("Education", "Academia"),
    "university":              ("Education", "Academia"),
    "library science":         ("Education", "Library Science"),
    "information science":     ("Education", "Library Science"),
    "education":               ("Education", "Pedagogy"),  # broad

    # --- Social Science: Anthropology ---
    "archeology":              ("Anthropology", "Archaeology"),
    "homo":                    ("Anthropology", "Physical Anthropology"),
    "anthropology":            ("Anthropology", "Cultural Anthropology"),
    "ethiopia":                ("Anthropology", "Cultural Anthropology"),
    "pakistan":                 ("Anthropology", "Cultural Anthropology"),
    "west indies":             ("Anthropology", "Cultural Anthropology"),
    "jamaica":                 ("Anthropology", "Cultural Anthropology"),

    # --- Humanities: Art ---
    "art":                     ("Art", "Visual Arts"),  # broad
    "fashion":                 ("Art", "Fashion"),

    # --- Humanities: Music ---
    "music":                   ("Music", "Music Theory"),  # broad
    "singing":                 ("Music", "Singing"),
    "music genre":             ("Music", "Music Genres"),

    # --- Humanities: Literature ---
    "ovid":                    ("Literature", "Poetry"),
    "rhetoric":                ("Literature", "Poetry"),
    "science fiction":         ("Literature", "Science Fiction"),
    "fairytale":               ("Literature", "Fairy Tales"),
    "literature":              ("Literature", "Literary Criticism"),  # broad

    # --- Humanities: Linguistics ---
    "grammar":                 ("Linguistics", "Grammar"),
    "phonetics":               ("Linguistics", "Phonetics"),
    "phonology":               ("Linguistics", "Phonetics"),
    "language":                ("Linguistics", "Semantics"),
    "linguistics":             ("Linguistics", "Morphology"),  # broad

    # --- Humanities: History ---
    "middle ages":             ("History", "Medieval History"),
    "heraldry":                ("History", "Heraldry"),
    "history":                 ("History", "Ancient History"),  # broad

    # --- Humanities: Philosophy ---
    "existentialism":          ("Philosophy", "Existentialism"),
    "science":                 ("Philosophy", "Philosophy of Science"),
    "philosophy":              ("Philosophy", "Ethics"),  # broad

    # --- Humanities: Performing Arts ---
    "movie":                   ("Performing Arts", "Cinema"),
    "drama":                   ("Performing Arts", "Theater"),
    "dance":                   ("Performing Arts", "Dance"),

    # --- Humanities: Linguistic Register (bucket) ---
    "slang":                   ("Linguistic Register", "Slang"),
    "euphemism":               ("Linguistic Register", "Euphemism"),
    "archaism":                ("Linguistic Register", "Archaism"),
    "dialect":                 ("Linguistic Register", "Dialect"),
    "african american vernacular english": ("Linguistic Register", "Dialect"),
    "formality":               ("Linguistic Register", "Formality"),
    "regionalism":             ("Linguistic Register", "Regionalism"),
    "trope":                   ("Linguistic Register", "Rhetorical Devices"),
    "irony":                   ("Linguistic Register", "Rhetorical Devices"),
    "synecdoche":              ("Linguistic Register", "Rhetorical Devices"),
    "metonymy":                ("Linguistic Register", "Rhetorical Devices"),
    "intensifier":             ("Linguistic Register", "Rhetorical Devices"),
    "disparagement":           ("Linguistic Register", "Taboo Language"),
    "obscenity":               ("Linguistic Register", "Taboo Language"),
    "ethnic slur":             ("Linguistic Register", "Taboo Language"),
    "abbreviation":            ("Linguistic Register", "Word Formation"),
    "acronym":                 ("Linguistic Register", "Word Formation"),
    "blend":                   ("Linguistic Register", "Word Formation"),
    "combining form":          ("Linguistic Register", "Word Formation"),
    "comparative":             ("Linguistic Register", "Word Formation"),
    "plural":                  ("Linguistic Register", "Word Formation"),
    "handwriting":             ("Linguistic Register", "Handwriting"),
    "street name":             ("Linguistic Register", "Street Names"),

    # --- Free Time: Sport ---
    "sport":                   ("Sport", "Ball Games"),  # broad — biggest bucket
    "ball game":               ("Sport", "Ball Games"),
    "water sport":             ("Sport", "Water Sports"),
    "gymnastics":              ("Sport", "Gymnastics"),
    "horse racing":            ("Sport", "Horse Racing"),
    "riding":                  ("Sport", "Horse Racing"),
    "dressage":                ("Sport", "Horse Racing"),
    "mountain climbing":       ("Sport", "Mountain Sports"),
    "fishing":                 ("Sport", "Fishing"),

    # --- Free Time: Games ---
    "game":                    ("Games", "Board Games"),  # broad
    "craps":                   ("Games", "Dice Games"),
    "esports":                 ("Games", "Esports"),
    "video_gaming":            ("Games", "Video Games"),

    # --- Free Time: Gastronomy ---
    "cooking":                 ("Gastronomy", "Cooking"),
    "tasting":                 ("Gastronomy", "Tasting"),
    "winemaking":              ("Gastronomy", "Winemaking"),

    # --- Free Time: Hobbies & Crafts ---
    "handicraft":              ("Hobbies & Crafts", "Handicraft"),

    # --- Doctrines: Religion ---
    "buddhism":                ("Religion", "Buddhism"),
    "hinduism":                ("Religion", "Hinduism"),
    "taoism":                  ("Religion", "Taoism"),
    "zen":                     ("Religion", "Buddhism"),
    "gnosticism":              ("Religion", "Gnosticism"),
    "religion":                ("Religion", "Comparative Religion"),  # broad

    # --- Doctrines: Mythology ---
    "norse mythology":         ("Mythology", "Norse Mythology"),
    "arthurian legend":        ("Mythology", "Arthurian Legend"),
    "mythology":               ("Mythology", "Greek Mythology"),  # broad
    "fairytale":               ("Mythology", "Folklore"),  # note: also in Literature

    # --- Doctrines: Occultism ---
    "alchemy":                 ("Occultism", "Alchemy"),
    "astrology":               ("Occultism", "Astrology"),
    "spiritualism":            ("Occultism", "Spiritualism"),

    # --- Doctrines: Ideology ---
    "marxism":                 ("Ideology", "Marxism"),

    # --- Bucket: Languages (is_bucket=1) ---
    "arabic":                  ("Languages", "Arabic"),
    "french":                  ("Languages", "French"),
    "german":                  ("Languages", "German"),
    "hebrew":                  ("Languages", "Hebrew"),
    "italian":                 ("Languages", "Italian"),
    "japanese":                ("Languages", "Japanese"),
    "latin":                   ("Languages", "Latin"),
    "persian":                 ("Languages", "Persian"),
    "pashto":                  ("Languages", "Pashto"),
    "sanskrit":                ("Languages", "Sanskrit"),
    "spanish":                 ("Languages", "Spanish"),
    "swahili":                 ("Languages", "Swahili"),
    "yiddish":                 ("Languages", "Yiddish"),

    # --- Bucket: Regional (is_bucket=1) ---
    "brazil":                  ("Regional", "Brazil"),
    "canada":                  ("Regional", "Canada"),
    "haiti":                   ("Regional", "Haiti"),

    # --- Bucket: Miscellaneous (is_bucket=1) ---
    "aristotle":               ("Miscellaneous", "Aristotle"),
    "genesis":                 ("Miscellaneous", "Genesis"),
    "growth":                  ("Miscellaneous", "Growth"),
    "holism":                  ("Miscellaneous", "Holism"),
    "medium":                  ("Miscellaneous", "Medium"),
    "vent":                    ("Miscellaneous", "Vent"),
    "wit":                     ("Miscellaneous", "Wit"),
    "wordnet":                 ("Miscellaneous", "Wordnet"),
}

# No domains are dropped — all WordNet groups are mapped.
# Bucket islands (is_bucket=1) handle non-topical domains.
DROPPED = set()


def main():
    v2 = sqlite3.connect(V2_DB)
    v3 = sqlite3.connect(V3_DB)
    v3.execute("PRAGMA foreign_keys = ON")

    # Build town lookup: (island_name, town_name) → town_id
    town_lookup = {}
    for row in v3.execute("""
        SELECT i.name, t.name, t.town_id
        FROM Towns t JOIN Islands i USING (island_id)
    """):
        town_lookup[(row[0], row[1])] = row[2]

    # Build set of exact-match town names (lowercased, underscores→spaces)
    town_names_lower = {name.lower() for _, name in town_lookup}

    # Verify mapping integrity
    errors = []
    for domain, (island, town) in DOMAIN_TO_TOWN.items():
        if (island, town) not in town_lookup:
            errors.append(f"  {domain!r} → ({island!r}, {town!r}) — NOT FOUND")
    if errors:
        print("MAPPING ERRORS:")
        for e in errors:
            print(e)
        return

    # Check for unmapped domains (not in mapping AND not in DROPPED)
    all_v2_domains = {r[0] for r in v2.execute("SELECT DISTINCT domain FROM augmented_domains")}
    mapped = set(DOMAIN_TO_TOWN.keys())
    unmapped = all_v2_domains - mapped - DROPPED
    if unmapped:
        print(f"WARNING: {len(unmapped)} unmapped domains (will be skipped):")
        for d in sorted(unmapped):
            count = v2.execute(
                "SELECT COUNT(*) FROM augmented_domains WHERE domain = ? AND source IN ('wordnet', 'claude_augmented')",
                (d,)
            ).fetchone()[0]
            print(f"  {d} ({count} wordnet+claude words)")
        print()

    # Transfer words
    stats = defaultdict(lambda: defaultdict(int))
    dupes = 0
    inserted = 0

    cur = v3.cursor()
    cur.execute("BEGIN TRANSACTION")

    for domain, (island_name, town_name) in DOMAIN_TO_TOWN.items():
        town_id = town_lookup[(island_name, town_name)]

        # Determine allowed sources for this domain
        # Normalize domain name for comparison: lowercase, underscores→spaces
        domain_normalized = domain.lower().replace("_", " ")
        town_normalized = town_name.lower()
        is_exact_match = domain_normalized == town_normalized

        if is_exact_match:
            source_filter = "('wordnet', 'claude_augmented', 'xgboost')"
        else:
            source_filter = "('wordnet', 'claude_augmented')"

        rows = v2.execute(f"""
            SELECT word, source, confidence, score
            FROM augmented_domains
            WHERE domain = ? AND source IN {source_filter}
        """, (domain,)).fetchall()

        for word, source, confidence, score in rows:
            try:
                cur.execute(
                    "INSERT INTO SeedWords (town_id, word, source, confidence, score) VALUES (?, ?, ?, ?, ?)",
                    (town_id, word, source, confidence, score)
                )
                inserted += 1
                stats[(island_name, town_name)][source] += 1
            except sqlite3.IntegrityError:
                # Duplicate (town_id, word) — multiple domains mapped to same town
                dupes += 1

    cur.execute("COMMIT")

    # Report
    print(f"Inserted: {inserted:,} seed words")
    print(f"Duplicates skipped: {dupes:,}")
    print()

    # Per-town summary
    town_totals = {}
    for (island, town), sources in sorted(stats.items()):
        total = sum(sources.values())
        town_totals[(island, town)] = total

    # Towns with seeds vs without
    all_towns = set(town_lookup.keys())
    seeded_towns = set(stats.keys())
    unseeded = all_towns - seeded_towns

    print(f"Towns with seeds: {len(seeded_towns)}")
    print(f"Towns without seeds: {len(unseeded)}")
    print()

    if unseeded:
        print("UNSEEDED TOWNS (need Claude generation):")
        for island, town in sorted(unseeded):
            print(f"  {island} > {town}")
        print()

    # Distribution
    print("SEED DISTRIBUTION (top 20 by count):")
    for (island, town), total in sorted(town_totals.items(), key=lambda x: -x[1])[:20]:
        sources = stats[(island, town)]
        parts = ", ".join(f"{s}={c}" for s, c in sorted(sources.items()))
        print(f"  {island} > {town}: {total} ({parts})")

    print()
    print("SEED DISTRIBUTION (bottom 20 by count):")
    for (island, town), total in sorted(town_totals.items(), key=lambda x: x[1])[:20]:
        sources = stats[(island, town)]
        parts = ", ".join(f"{s}={c}" for s, c in sorted(sources.items()))
        print(f"  {island} > {town}: {total} ({parts})")

    # Source breakdown
    print()
    total_by_source = defaultdict(int)
    for sources in stats.values():
        for s, c in sources.items():
            total_by_source[s] += c
    print("TOTAL BY SOURCE:")
    for s, c in sorted(total_by_source.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c:,}")

    v2.close()
    v3.close()


if __name__ == "__main__":
    main()
