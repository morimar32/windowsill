"""
Explore Wikipedia category hierarchy for each of our 258 domains.
Pulls subcategories 1-2 levels deep and outputs a markdown report.

Usage:
    python explore_wiki_categories.py > wiki_categories.md
"""

import sqlite3
import urllib.request
import urllib.parse
import json
import time
import re
import sys

# Patterns to filter out non-topical Wikipedia categories
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

# Some of our domains need a different Wikipedia category name
# (our domain name != Wikipedia category name)
DOMAIN_TO_WIKI = {
    "3d_printing": "3D printing",
    "african american vernacular english": "African-American Vernacular English",
    "ball game": "Ball games",
    "combining form": None,  # linguistic concept, no wiki category
    "street name": None,  # not a topical domain
    "ethnic slur": None,
    "virtual_reality": "Virtual reality",
    "climate_science": "Climate science",
    "video_gaming": "Video games",
    "user_experience": "User experience",
    "nuclear physics": "Nuclear physics",
    "mountain climbing": "Climbing",
    "water sport": "Water sports",
    "horse racing": "Horse racing",
    "martial art": "Martial arts",
    "ball game": "Ball games",
    "ice hockey": "Ice hockey",
    "science fiction": "Science fiction",
    "computer science": "Computer science",
    "west indies": None,
    "ovid": None,  # specific author, not a domain
    "wit": None,  # abstract concept
    "disparagement": None,
    "archaism": None,
    "comparative": None,
    "blend": None,  # linguistic term
    "plural": None,
    "acronym": None,
    "abbreviation": None,
    "dressage": "Dressage",
    "existentialism": "Existentialism",
    "psychoanalysis": "Psychoanalysis",
}


def get_subcategories(category: str, limit: int = 50) -> list[str]:
    """Fetch subcategories of a Wikipedia category via the MediaWiki API."""
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
    req = urllib.request.Request(full_url, headers={"User-Agent": "WindowsillBot/1.0 (research)"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        members = resp.get("query", {}).get("categorymembers", [])
        return [m["title"].replace("Category:", "") for m in members]
    except Exception as e:
        return []


def filter_categories(cats: list[str]) -> list[str]:
    """Remove non-topical categories (geographic, temporal, meta, etc.)."""
    filtered = []
    for c in cats:
        if not SKIP_RE.search(c):
            filtered.append(c)
    return filtered


def explore_domain(domain: str, wiki_name: str | None) -> dict:
    """Explore a single domain's Wikipedia category tree."""
    if wiki_name is None:
        return {"domain": domain, "wiki_category": None, "status": "skipped", "subcats": []}

    # Try the wiki name directly
    level1 = get_subcategories(wiki_name, limit=50)
    time.sleep(0.1)  # be nice to the API

    if not level1:
        # Try with "s" suffix (e.g., "Sport" -> "Sports")
        alt = wiki_name + "s" if not wiki_name.endswith("s") else wiki_name[:-1]
        level1 = get_subcategories(alt, limit=50)
        time.sleep(0.1)
        if level1:
            wiki_name = alt

    if not level1:
        # Try title case
        alt = wiki_name.title()
        level1 = get_subcategories(alt, limit=50)
        time.sleep(0.1)
        if level1:
            wiki_name = alt

    filtered = filter_categories(level1)

    # For subcategories that look like "Branches of X" or "Types of X" or "X by type",
    # go one level deeper to get the actual subdivisions
    expanded = []
    drill_patterns = [
        r"^branches of",
        r"^types of",
        r"^subfields of",
        r"^forms of",
        r"by type$",
        r"by discipline$",
        r"by genre$",
        r"by instrument$",
        r"by style$",
        r"by medium$",
    ]
    drill_re = re.compile("|".join(drill_patterns), re.IGNORECASE)

    for cat in filtered:
        if drill_re.search(cat):
            children = get_subcategories(cat, limit=40)
            time.sleep(0.1)
            children_filtered = filter_categories(children)
            expanded.append({"name": cat, "children": children_filtered})
        else:
            expanded.append({"name": cat, "children": []})

    return {
        "domain": domain,
        "wiki_category": wiki_name,
        "status": "found" if level1 else "not_found",
        "raw_count": len(level1),
        "filtered_count": len(filtered),
        "subcats": expanded,
    }


def main():
    conn = sqlite3.connect("v2.db")
    domains = [r[0] for r in conn.execute(
        "SELECT DISTINCT domain FROM domain_word_scores ORDER BY domain"
    ).fetchall()]
    conn.close()

    print(f"# Wikipedia Category Mapping for {len(domains)} Windowsill Domains")
    print()
    print(f"Auto-generated exploration of Wikipedia's category hierarchy.")
    print(f"For each domain, shows filtered subcategories (non-topical categories removed).")
    print()

    stats = {"found": 0, "not_found": 0, "skipped": 0, "total_subcats": 0}

    results = []
    for i, domain in enumerate(domains):
        wiki_name = DOMAIN_TO_WIKI.get(domain, domain.replace("_", " "))
        result = explore_domain(domain, wiki_name)
        results.append(result)
        stats[result["status"]] = stats.get(result["status"], 0) + 1
        if result["subcats"]:
            stats["total_subcats"] += sum(
                1 + len(s["children"]) for s in result["subcats"]
            )

        # Progress to stderr
        status_char = "." if result["status"] == "found" else ("S" if result["status"] == "skipped" else "X")
        n_subs = result.get("filtered_count", 0)
        print(f"  [{i+1:3d}/{len(domains)}] {domain:30s} -> {status_char} ({n_subs} subcats)", file=sys.stderr)

    # Summary
    print(f"## Summary")
    print()
    print(f"| Metric | Count |")
    print(f"|--------|-------|")
    print(f"| Domains queried | {len(domains)} |")
    print(f"| Wikipedia match found | {stats['found']} |")
    print(f"| No match found | {stats['not_found']} |")
    print(f"| Skipped (non-topical) | {stats['skipped']} |")
    print(f"| Total subcategories found | {stats['total_subcats']} |")
    print()

    # Domains with no match
    not_found = [r for r in results if r["status"] == "not_found"]
    if not_found:
        print(f"## Domains With No Wikipedia Category Match ({len(not_found)})")
        print()
        for r in not_found:
            print(f"- **{r['domain']}** (tried: `{r['wiki_category']}`)")
        print()

    # Skipped domains
    skipped = [r for r in results if r["status"] == "skipped"]
    if skipped:
        print(f"## Skipped Domains — Non-Topical ({len(skipped)})")
        print()
        for r in skipped:
            print(f"- {r['domain']}")
        print()

    # Main results
    print(f"## Domain → Wikipedia Subcategories")
    print()

    for r in results:
        if r["status"] != "found" or not r["subcats"]:
            continue

        print(f"### {r['domain']}")
        print(f"**Wikipedia:** `Category:{r['wiki_category']}` — "
              f"{r['raw_count']} raw → {r['filtered_count']} filtered subcategories")
        print()

        for sub in r["subcats"]:
            if sub["children"]:
                print(f"- **{sub['name']}**")
                for child in sub["children"]:
                    print(f"  - {child}")
            else:
                print(f"- {sub['name']}")
        print()


if __name__ == "__main__":
    main()
