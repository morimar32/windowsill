"""
Retry Wikipedia category fetch for domains that failed due to network errors.
Reads wiki_categories.md to find failures, retries with backoff, appends results.

Usage:
    python explore_wiki_retry.py >> wiki_categories.md
"""

import urllib.request
import urllib.parse
import json
import time
import re
import sys

SKIP_PATTERNS = [
    r"\bby (location|country|city|continent|region|state|province|territory|year|decade|century|period|month|date|gender|ethnicity|nationality|sport)\b",
    r"\bby country\b",
    r"\blists? of\b",
    r"\bhistory of\b",
    r"\bpeople in\b",
    r"\bpeople by\b",
    r"\bbiographers?\b",
    r"\borganizations?\b",
    r"\bassociations?\b",
    r"\bcompetitions?\b",
    r"\bevents?\b",
    r"\bawards?\b",
    r"\bmedia\b",
    r"\bjournalism\b",
    r"\bphotography of\b",
    r"\bfilms? (about|set)\b",
    r"\bfiction\b",
    r"\bnovels?\b",
    r"\btelevision\b",
    r"\bvideo games?\b",
    r"\bmuseums?\b",
    r"\beducation\b",
    r"\bstubs?\b",
    r"\btemplates?\b",
    r"\bwikiproject\b",
    r"\bwikipedia\b",
    r"\bcommons\b",
    r"\bcategories\b",
    r"\bnavboxes\b",
    r"\binfoboxes\b",
    r"\barticles\b",
    r"\bimages\b",
    r"\bworks about\b",
    r"\bdepictions\b",
    r"\bin popular culture\b",
    r"\bculture and\b",
    r"\band culture\b",
    r"\blaw and\b",
    r"\blegislation\b",
]
SKIP_RE = re.compile("|".join(SKIP_PATTERNS), re.IGNORECASE)

# Extra mappings for domains that need different Wikipedia names
DOMAIN_TO_WIKI = {
    "african american vernacular english": "African-American Vernacular English",
    "agile_methodology": "Agile software development",
    "archeology": "Archaeology",
    "arithmetic": "Arithmetic",
    "arthurian legend": "Arthurian legend",
    "artificial_intelligence": "Artificial intelligence",
    "climate_science": "Climatology",
    "commerce": "Commerce",
    "cryptocurrency": "Cryptocurrencies",
    "cybersecurity": "Computer security",
    "distributed_systems": "Distributed computing",
    "economics": "Economics",
    "existentialism": "Existentialism",
    "factory": "Factories",
    "fairytale": "Fairy tales",
    "farming": "Agriculture",
    "forestry": "Forestry",
    "furniture": "Furniture",
    "genesis": None,  # too ambiguous
    "growth": None,  # too abstract
    "handicraft": "Handicrafts",
    "handwriting": "Penmanship",
    "health food": "Health foods",
    "industry": "Industry",
    "information science": "Information science",
    "intensifier": None,  # linguistic
    "irony": None,  # abstract
    "jet engine": "Jet engines",
    "jung": "Jungian psychology",
    "lake": "Lakes",
    "language": "Languages",
    "literature": "Literature",
    "livestock": "Livestock",
    "logic": "Logic",
    "machine_learning": "Machine learning",
    "mathematics": "Mathematics",
    "mechanics": "Mechanics",
    "medicine": "Medicine",
    "medium": None,  # ambiguous
    "metallurgy": "Metallurgy",
    "meteorology": "Meteorology",
    "military": "Military",
    "mineralogy": "Mineralogy",
    "mobile_computing": "Mobile technology",
    "motor vehicle": "Motor vehicles",
    "mountain climbing": "Mountaineering",
    "movie": "Film",
    "muscle": "Muscular system",
    "music": "Music",
    "music genre": "Music genres",
    "nanotechnology": "Nanotechnology",
    "narcotic": "Narcotics",
    "navy": "Navies",
    "network_engineering": "Computer networking",
    "neurology": "Neurology",
    "neuroscience_imaging": "Neuroimaging",
    "news article": "News media",
    "norse mythology": "Norse mythology",
    "nuclear physics": "Nuclear physics",
    "numeration system": "Numeral systems",
    "obscenity": "Obscenity",
    "ophthalmology": "Ophthalmology",
    "otology": "Otology",
    "paleontology": "Paleontology",
    "particle physics": "Particle physics",
    "particle_accelerators": "Particle accelerators",
    "pharmacology": "Pharmacology",
    "philosophy": "Philosophy",
    "phonology": "Phonology",
    "physical chemistry": "Physical chemistry",
    "physiology": "Physiology",
    "psychoanalysis": "Psychoanalysis",
    "psychology": "Psychology",
    "psychophysics": "Psychophysics",
    "publication": "Publishing",
    "quadruped": None,  # not a useful wiki category
    "quantum_computing": "Quantum computing",
    "regionalism": None,
    "relativity": "Theory of relativity",
    "religion": "Religion",
    "rhetoric": "Rhetoric",
    "riding": "Equestrianism",
    "ruminant": "Ruminants",
    "semiconductor": "Semiconductors",
    "sexology": "Sexology",
    "singing": "Singing",
    "socialism": "Socialism",
    "spiritualism": "Spiritualism",
    "sport": "Sports",
    "telecommunication": "Telecommunications",
    "terrorism": "Terrorism",
    "thermodynamics": "Thermodynamics",
    "topography": "Topography",
    "trademark": "Trademarks",
    "train": "Rail transport",
    "transportation": "Transport",
    "trope": None,
    "university": "Universities and colleges",
    "user_experience": "User experience design",
    "vegetation": "Vegetation",
    "vent": None,  # too vague
    "veterinary medicine": "Veterinary medicine",
    "virtual_reality": "Virtual reality",
    "water sport": "Water sports",
    "wordnet": None,
    "yiddish": "Yiddish",
    "zoology": "Zoology",
    "aeronautics": "Aeronautics",
    "alchemy": "Alchemy",
    "arabic": "Arabic language",
    "architecture": "Architecture",
    "aristotle": None,  # person
    "art": "The arts",
    "astrology": "Astrology",
    "astronomy": "Astronomy",
    "auction": "Auctions",
    "canada": None,  # geography
    "dialect": "Dialects",
    "euphemism": None,
    "formality": None,
    "french": "French language",
    "german": "German language",
    "hebrew": "Hebrew language",
    "italian": "Italian language",
    "jamaica": None,
    "japanese": "Japanese language",
    "latin": "Latin language",
    "persian": "Persian language",
    "phonetics": "Phonetics",
    "spanish": "Spanish language",
    "scottish": None,
    "portuguese": None,
}


def get_subcategories(category: str, limit: int = 50, retries: int = 3) -> list[str]:
    url = "https://en.wikipedia.org/w/api.php"
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "subcat",
        "cmlimit": limit,
        "format": "json",
    })
    full_url = f"{url}?{params}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(full_url, headers={"User-Agent": "WindowsillBot/1.0 (research)"})
            resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
            return [m["title"].replace("Category:", "") for m in resp.get("query", {}).get("categorymembers", [])]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    FAILED after {retries} retries: {e}", file=sys.stderr)
                return []


def filter_categories(cats):
    return [c for c in cats if not SKIP_RE.search(c)]


def main():
    # Parse the existing markdown to find failed domains
    failed_domains = []
    with open("wiki_categories.md") as f:
        in_not_found = False
        for line in f:
            if "No Wikipedia Category Match" in line:
                in_not_found = True
                continue
            if in_not_found and line.startswith("- **"):
                domain = line.split("**")[1]
                tried = line.split("`")[1] if "`" in line else domain
                failed_domains.append((domain, tried))
            if in_not_found and line.startswith("##") and "No Wikipedia" not in line:
                in_not_found = False

    print(f"Retrying {len(failed_domains)} failed domains...", file=sys.stderr)

    drill_re = re.compile(
        r"^branches of|^types of|^subfields of|^forms of|by type$|by discipline$|by genre$|by instrument$|by style$|by medium$",
        re.IGNORECASE,
    )

    print()
    print("## Retried Domains")
    print()

    found = 0
    still_missing = []

    for i, (domain, _tried) in enumerate(failed_domains):
        wiki_name = DOMAIN_TO_WIKI.get(domain, domain.replace("_", " "))

        if wiki_name is None:
            still_missing.append(domain)
            print(f"  [{i+1:3d}/{len(failed_domains)}] {domain:35s} -> SKIPPED (non-topical)", file=sys.stderr)
            continue

        level1 = get_subcategories(wiki_name, limit=50, retries=3)
        time.sleep(0.2)

        if not level1 and not wiki_name[0].isupper():
            alt = wiki_name.title()
            level1 = get_subcategories(alt, limit=50, retries=3)
            time.sleep(0.2)
            if level1:
                wiki_name = alt

        if not level1:
            still_missing.append(domain)
            print(f"  [{i+1:3d}/{len(failed_domains)}] {domain:35s} -> NOT FOUND ({wiki_name})", file=sys.stderr)
            continue

        filtered = filter_categories(level1)
        found += 1
        print(f"  [{i+1:3d}/{len(failed_domains)}] {domain:35s} -> FOUND ({len(filtered)} subcats)", file=sys.stderr)

        # Drill into "Branches of" etc.
        expanded = []
        for cat in filtered:
            if drill_re.search(cat):
                children = get_subcategories(cat, limit=40, retries=3)
                time.sleep(0.2)
                children_filtered = filter_categories(children)
                expanded.append({"name": cat, "children": children_filtered})
            else:
                expanded.append({"name": cat, "children": []})

        print(f"### {domain}")
        print(f"**Wikipedia:** `Category:{wiki_name}` — "
              f"{len(level1)} raw → {len(filtered)} filtered subcategories")
        print()
        for sub in expanded:
            if sub["children"]:
                print(f"- **{sub['name']}**")
                for child in sub["children"]:
                    print(f"  - {child}")
            else:
                print(f"- {sub['name']}")
        print()

    print(f"\nRetry complete: {found} newly found, {len(still_missing)} still missing", file=sys.stderr)

    if still_missing:
        print(f"## Still Missing ({len(still_missing)} domains)")
        print()
        for d in still_missing:
            wiki = DOMAIN_TO_WIKI.get(d, d)
            note = "(skipped)" if wiki is None else f"(tried: `{wiki}`)"
            print(f"- {d} {note}")


if __name__ == "__main__":
    main()
