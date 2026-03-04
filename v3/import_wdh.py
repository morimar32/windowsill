"""Import FBK WordNet Domains (WDH) into the v3 database.

Reads the WDH synset→domain mapping, resolves WN 2.0 synset offsets
to NLTK's WN 3.0 synsets via the SKI (Sense Key Index) bridge, and
populates SeedWords with domain-tagged vocabulary.

Data files required (in v3/data/):
  - wn-domains-3.2.tsv       — FBK WordNet Domains 3.2 (synset→domain)
  - ski-pwn-sets.txt          — SKI cross-version offset mapping

Sources:
  - WDH: https://wndomains.fbk.eu/ (CC BY 3.0)
  - SKI: https://github.com/ekaf/ski (CC BY 4.0)

Pipeline position: runs AFTER schema + hierarchy population,
BEFORE generate_seeds.py.

Usage:
    python v3/import_wdh.py --dry-run
    python v3/import_wdh.py --apply
"""

import argparse
import os
import re
import sqlite3
import time
from collections import defaultdict

from nltk.corpus import wordnet as wn

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V3_DB = os.path.join(_project, "v3/windowsill.db")
WDH_FILE = os.path.join(_project, "v3/data/wn-domains-3.2.tsv")
SKI_FILE = os.path.join(_project, "v3/data/ski-pwn-sets.txt")

# ============================================================
# WDH DOMAIN → V3 TOWN MAPPING
# ============================================================
# Maps 168 WDH domain labels to (Island, Town) pairs.
# Domains not listed here are UNMAPPED and will be reported.
# "factotum" (general/unclassified) is intentionally skipped.
# ============================================================

WDH_TO_TOWN = {
    # --- Medicine ---
    "medicine":             ("Medicine", "Pathology"),
    "anatomy":              ("Medicine", "Anatomy"),
    "dentistry":            ("Medicine", "Dentistry"),
    "pharmacy":             ("Medicine", "Pharmacology"),
    "physiology":           ("Medicine", "Physiology"),
    "psychiatry":           ("Medicine", "Psychiatry"),
    "psychoanalysis":       ("Psychology", "Psychoanalysis"),
    "radiology":            ("Medicine", "Radiology"),
    "surgery":              ("Medicine", "Surgery"),
    "veterinary":           ("Veterinary Medicine", "Veterinary Medicine"),
    "body_care":            ("Medicine", "Health & Nutrition"),
    "health":               ("Medicine", "Health & Nutrition"),
    "sexuality":            ("Medicine", "Physiology"),

    # --- Engineering ---
    "engineering":          ("Engineering", "Mechanical Engineering"),
    "electricity":          ("Engineering", "Electrical Engineering"),
    "electronics":          ("Engineering", "Electrical Engineering"),
    "electrotechnology":    ("Engineering", "Electrical Engineering"),

    # --- Computer Science ---
    "computer_science":     ("Computer Science", "Algorithms"),

    # --- Architecture ---
    "architecture":         ("Architecture", "Classical Architecture"),
    "buildings":            ("Architecture", "Construction"),
    "town_planning":        ("Architecture", "Urban Planning"),

    # --- Agriculture ---
    "agriculture":          ("Agriculture", "Farming"),
    "animal_husbandry":     ("Agriculture", "Livestock"),
    "entomology":           ("Biology", "Zoology"),

    # --- Media & Communications ---
    "telecommunication":    ("Media & Communications", "Telephony"),
    "telegraphy":           ("Media & Communications", "Telephony"),
    "telephony":            ("Media & Communications", "Telephony"),
    "radio+tv":             ("Media & Communications", "Broadcasting"),
    "publishing":           ("Media & Communications", "Publishing"),
    "photography":          ("Art", "Photography"),
    "post":                 ("Media & Communications", "Publishing"),

    # --- Military ---
    "military":             ("Military", "Military Strategy"),

    # --- Transportation ---
    "transport":            ("Transportation", "Maritime Transport"),
    "aviation":             ("Transportation", "Aviation"),
    "astronautics":         ("Transportation", "Spaceflight"),
    "nautical":             ("Transportation", "Navigation"),
    "railway":              ("Transportation", "Rail Transport"),
    "vehicles":             ("Transportation", "Motor Vehicles"),

    # --- Manufacturing ---
    "artisanship":          ("Manufacturing", "Carpentry"),
    "furniture":            ("Manufacturing", "Furniture Making"),
    "industry":             ("Manufacturing", "Factory Production"),
    "jewellery":            ("Hobbies & Crafts", "Collecting"),

    # --- Mathematics ---
    "mathematics":          ("Mathematics", "Algebra"),
    "geometry":             ("Mathematics", "Geometry"),
    "statistics":           ("Mathematics", "Statistics"),
    "number":               ("Mathematics", "Numeration Systems"),

    # --- Physics ---
    "physics":              ("Physics", "Electromagnetism"),
    "acoustics":            ("Physics", "Acoustics"),
    "atomic_physic":        ("Physics", "Nuclear Physics"),
    "mechanics":            ("Physics", "Mechanics"),
    "optics":               ("Physics", "Optics"),
    "thermodynamics":       ("Physics", "Thermodynamics"),

    # --- Chemistry ---
    "chemistry":            ("Chemistry", "Organic Chemistry"),
    "biochemistry":         ("Chemistry", "Biochemistry"),

    # --- Biology ---
    "biology":              ("Biology", "Cell Biology"),
    "animals":              ("Biology", "Zoology"),
    "plants":               ("Biology", "Botany"),
    "genetics":             ("Biology", "Genetics"),
    "paleontology":         ("Biology", "Paleontology"),
    "environment":          ("Biology", "Ecology"),

    # --- Astronomy ---
    "astronomy":            ("Astronomy", "Stellar Astronomy"),

    # --- Earth Science ---
    "geology":              ("Earth Science", "Geology"),
    "meteorology":          ("Earth Science", "Meteorology"),
    "oceanography":         ("Earth Science", "Oceanography"),
    "geography":            ("Earth Science", "Geology"),
    "topography":           ("Earth Science", "Geology"),
    "earth":                ("Earth Science", "Geology"),

    # --- Economics ---
    "economy":              ("Economics", "Macroeconomics"),
    "book_keeping":         ("Finance", "Accounting"),

    # --- Finance ---
    "banking":              ("Finance", "Banking"),
    "exchange":             ("Finance", "Investment"),
    "insurance":            ("Finance", "Insurance"),
    "money":                ("Finance", "Banking"),
    "finance":              ("Finance", "Banking"),
    "tax":                  ("Finance", "Taxation"),

    # --- Commerce ---
    "commerce":             ("Commerce", "Trade"),
    "enterprise":           ("Commerce", "Business"),

    # --- Law ---
    "law":                  ("Law", "Constitutional Law"),
    "administration":       ("Politics", "Public Administration"),

    # --- Politics ---
    "politics":             ("Politics", "Political Theory"),
    "diplomacy":            ("Politics", "Diplomacy"),

    # --- Psychology ---
    "psychology":           ("Psychology", "Clinical Psychology"),
    "psychological_features": ("Psychology", "Social Psychology"),

    # --- Sociology ---
    "sociology":            ("Sociology", "Social Stratification"),

    # --- Education ---
    "pedagogy":             ("Education", "Pedagogy"),
    "school":               ("Education", "Pedagogy"),
    "university":           ("Education", "Academia"),

    # --- Anthropology ---
    "anthropology":         ("Anthropology", "Cultural Anthropology"),
    "archaeology":          ("Anthropology", "Archaeology"),
    "ethnology":            ("Anthropology", "Ethnography"),

    # --- Art ---
    "art":                  ("Art", "Visual Arts"),
    "color":                ("Art", "Visual Arts"),
    "drawing":              ("Art", "Visual Arts"),
    "graphic_arts":         ("Art", "Graphic Design"),
    "painting":             ("Art", "Visual Arts"),
    "plastic_arts":         ("Art", "Sculpture"),
    "sculpture":            ("Art", "Sculpture"),
    "fashion":              ("Art", "Fashion"),

    # --- Music ---
    "music":                ("Music", "Music Theory"),

    # --- Literature ---
    "literature":           ("Literature", "Literary Criticism"),
    "folklore":             ("Mythology", "Folklore"),

    # --- Linguistics ---
    "linguistics":          ("Linguistics", "Morphology"),
    "grammar":              ("Linguistics", "Grammar"),
    "philology":            ("Linguistics", "Historical Linguistics"),

    # --- History ---
    "history":              ("History", "Ancient History"),
    "heraldry":             ("History", "Heraldry"),
    "numismatics":          ("Hobbies & Crafts", "Collecting"),
    "philately":            ("Hobbies & Crafts", "Collecting"),

    # --- Philosophy ---
    "philosophy":           ("Philosophy", "Ethics"),

    # --- Performing Arts ---
    "cinema":               ("Performing Arts", "Cinema"),
    "theatre":              ("Performing Arts", "Theater"),
    "dance":                ("Performing Arts", "Dance"),

    # --- Linguistic Register (bucket) ---
    "paranormal":           ("Occultism", "Spiritualism"),

    # --- Sport ---
    "sport":                ("Sport", "Ball Games"),
    "athletics":            ("Sport", "Athletics"),
    "archery":              ("Sport", "Archery & Shooting"),
    "badminton":            ("Sport", "Racquet Sports"),
    "baseball":             ("Sport", "Ball Games"),
    "basketball":           ("Sport", "Ball Games"),
    "bowling":              ("Sport", "Ball Games"),
    "boxing":               ("Sport", "Combat Sports"),
    "cricket":              ("Sport", "Ball Games"),
    "cycling":              ("Sport", "Cycling"),
    "diving":               ("Sport", "Water Sports"),
    "fencing":              ("Sport", "Combat Sports"),
    "fishing":              ("Sport", "Fishing"),
    "football":             ("Sport", "Ball Games"),
    "golf":                 ("Sport", "Ball Games"),
    "hockey":               ("Sport", "Winter Sports"),
    "hunting":              ("Sport", "Fishing"),
    "mountaineering":       ("Sport", "Mountain Sports"),
    "racing":               ("Sport", "Motorsport"),
    "rowing":               ("Sport", "Water Sports"),
    "rugby":                ("Sport", "Ball Games"),
    "skating":              ("Sport", "Winter Sports"),
    "skiing":               ("Sport", "Winter Sports"),
    "soccer":               ("Sport", "Ball Games"),
    "swimming":             ("Sport", "Water Sports"),
    "table_tennis":         ("Sport", "Racquet Sports"),
    "tennis":               ("Sport", "Racquet Sports"),
    "volleyball":           ("Sport", "Ball Games"),
    "wrestling":            ("Sport", "Combat Sports"),

    # --- Games ---
    "betting":              ("Games", "Dice Games"),
    "card":                 ("Games", "Card Games"),
    "chess":                ("Games", "Board Games"),
    "play":                 ("Games", "Board Games"),

    # --- Gastronomy ---
    "gastronomy":           ("Gastronomy", "Cooking"),
    "food":                 ("Gastronomy", "Food Science"),

    # --- Hobbies ---
    "home":                 ("Hobbies & Crafts", "Gardening"),

    # --- Religion ---
    "religion":             ("Religion", "Comparative Religion"),
    "roman_catholic":       ("Religion", "Christianity"),
    "theology":             ("Religion", "Theology"),

    # --- Mythology ---
    "mythology":            ("Mythology", "Greek Mythology"),

    # --- Occultism ---
    "astrology":            ("Occultism", "Astrology"),
    "occultism":            ("Occultism", "Alchemy"),

    # --- Ideology ---
    # (no direct WDH domain)

    # --- Meta/skip ---
    "tourism":              ("Commerce", "Retail"),
    "metrology":            ("Physics", "Mechanics"),
    "gas":                  ("Chemistry", "Physical Chemistry"),
    "hydraulics":           ("Engineering", "Civil Engineering"),
}

# Domains to skip (meta-categories, not topical)
WDH_SKIP = {
    "factotum",         # unclassified — too generic
    "person",           # people, not topics
    "quality",          # adjective qualities
    "time_period",      # temporal markers
    "applied_science",  # parent category, not leaf
    "pure_science",     # parent category
    "social_science",   # parent category
    "humanities",       # parent category
    "free_time",        # parent category
    "sub",              # unclear/noise
}


def build_wn20_to_wn30_mapping():
    """Build WN 2.0 synset offset → WN 3.0 offset mapping via SKI."""
    mapping = {}
    with open(SKI_FILE) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue

            entries = {}
            for entry in parts[1].split(","):
                m = re.match(r"(\d+\.\d+(?:\.\d+)?):([nvasr]):(\d+)", entry)
                if m:
                    ver, pos, offset = m.group(1), m.group(2), int(m.group(3))
                    entries[ver] = (pos, offset)

            if "2.0" in entries and "3.0" in entries:
                wn20_pos, wn20_off = entries["2.0"]
                wn30_pos, wn30_off = entries["3.0"]
                key20 = f"{wn20_off:08d}-{wn20_pos}"
                mapping[key20] = (wn30_pos, wn30_off)

    return mapping


def parse_wdh():
    """Parse WDH TSV into list of (synset_id, [domains])."""
    entries = []
    with open(WDH_FILE) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entries.append((parts[0], parts[1].split()))
    return entries


def resolve_wdh_to_lemmas(wdh_entries, offset_mapping):
    """Resolve WDH synsets → WN 3.0 synsets → lemmas.

    Returns:
        domain_lemmas: {domain: {lemma: pos}}
        stats: dict of resolution statistics
    """
    domain_lemmas = defaultdict(dict)  # domain → {lemma: pos}
    resolved = 0
    no_mapping = 0

    for sid, domains in wdh_entries:
        mapping = offset_mapping.get(sid)
        if not mapping:
            no_mapping += 1
            continue

        pos30, off30 = mapping
        try:
            s = wn.synset_from_pos_and_offset(pos30, off30)
            if not s:
                no_mapping += 1
                continue
        except Exception:
            no_mapping += 1
            continue

        # Extract lemmas
        wn_pos = s.pos()
        pos_map = {"n": "noun", "v": "verb", "a": "adj", "s": "adj", "r": "adv"}
        pos_label = pos_map.get(wn_pos, wn_pos)

        lemmas = [l.name().replace("_", " ") for l in s.lemmas()]
        for d in domains:
            if d in WDH_SKIP:
                continue
            for lemma in lemmas:
                domain_lemmas[d][lemma] = pos_label

        resolved += 1

    return domain_lemmas, {"resolved": resolved, "no_mapping": no_mapping}


def main():
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Import WDH domains into v3")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show analysis without modifying database")
    parser.add_argument("--apply", action="store_true",
                        help="Import WDH data and populate SeedWords")
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        parser.print_help()
        print("\nSpecify --dry-run or --apply")
        return

    # Step 1: Build offset mapping
    print("Building WN 2.0 → 3.0 offset mapping from SKI...")
    t0 = time.time()
    offset_mapping = build_wn20_to_wn30_mapping()
    print(f"  {len(offset_mapping):,} mappings ({time.time()-t0:.1f}s)")

    # Step 2: Parse WDH
    print("Parsing WDH domains...")
    wdh_entries = parse_wdh()
    print(f"  {len(wdh_entries):,} synset entries")

    # Step 3: Resolve to lemmas
    print("Resolving synsets → lemmas...")
    t0 = time.time()
    domain_lemmas, stats = resolve_wdh_to_lemmas(wdh_entries, offset_mapping)
    print(f"  Resolved: {stats['resolved']:,} synsets ({time.time()-t0:.1f}s)")
    print(f"  No WN3.0 mapping: {stats['no_mapping']:,}")

    # Step 4: Check mapping coverage
    unmapped_domains = set(domain_lemmas.keys()) - set(WDH_TO_TOWN.keys()) - WDH_SKIP
    if unmapped_domains:
        print(f"\nWARNING: {len(unmapped_domains)} WDH domains have no town mapping:")
        for d in sorted(unmapped_domains):
            print(f"  {d}: {len(domain_lemmas[d]):,} lemmas")

    # Step 5: Map to towns
    town_seeds = defaultdict(dict)  # (island, town) → {lemma: pos}
    mapped_count = 0
    unmapped_count = 0

    for domain, lemma_pos in domain_lemmas.items():
        target = WDH_TO_TOWN.get(domain)
        if not target:
            unmapped_count += len(lemma_pos)
            continue
        for lemma, pos in lemma_pos.items():
            town_seeds[target][lemma] = pos
            mapped_count += 1

    print(f"\nMapped to towns: {mapped_count:,} lemma-domain pairs")
    print(f"Unmapped (skipped domains): {unmapped_count:,}")
    print(f"Target towns: {len(town_seeds)}")

    # Distribution
    print(f"\nTop 20 towns by seed count:")
    for (isl, town), seeds in sorted(town_seeds.items(),
                                      key=lambda x: len(x[1]),
                                      reverse=True)[:20]:
        print(f"  {isl} > {town}: {len(seeds):,}")

    if args.dry_run:
        # Show total stats
        total_lemmas = set()
        for seeds in town_seeds.values():
            total_lemmas.update(seeds.keys())
        print(f"\nTotal unique lemmas across all towns: {len(total_lemmas):,}")
        print(f"\nDry run — no changes made.")
        return

    # Step 6: Apply — populate database
    print("\nPopulating database...")
    con = sqlite3.connect(V3_DB)
    con.execute("PRAGMA foreign_keys = ON")

    # Build town lookup
    town_lookup = {}
    for row in con.execute("""
        SELECT i.name, t.name, t.town_id
        FROM Towns t JOIN Islands i USING (island_id)
    """):
        town_lookup[(row[0], row[1])] = row[2]

    # Verify all target towns exist
    errors = []
    for (island, town) in town_seeds:
        if (island, town) not in town_lookup:
            errors.append(f"  ({island}, {town}) — NOT FOUND")
    if errors:
        print("TOWN MAPPING ERRORS:")
        for e in errors:
            print(e)
        con.close()
        return

    # Build word lookup
    word_to_id = {}
    for wid, word in con.execute("SELECT word_id, word FROM Words"):
        word_to_id[word] = wid

    # Insert SeedWords
    cur = con.cursor()
    cur.execute("BEGIN TRANSACTION")

    inserted = 0
    dupes = 0
    no_word_id = 0

    for (island, town), seeds in town_seeds.items():
        town_id = town_lookup[(island, town)]
        for lemma, pos in seeds.items():
            word_id = word_to_id.get(lemma)
            if not word_id:
                no_word_id += 1
                continue
            try:
                cur.execute("""
                    INSERT INTO SeedWords (town_id, word, word_id, source, confidence)
                    VALUES (?, ?, ?, 'wordnet', 'core')
                """, (town_id, lemma, word_id))
                inserted += 1
            except sqlite3.IntegrityError:
                dupes += 1

    cur.execute("COMMIT")
    con.close()

    print(f"  Inserted: {inserted:,}")
    print(f"  Duplicates skipped: {dupes:,}")
    print(f"  No word_id (not in vocab): {no_word_id:,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
