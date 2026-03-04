"""
Fetch Wikipedia's category hierarchy starting from "Main topic classifications".
Two levels deep: level 1 = islands, level 2 = towns (subcategories).
Categories only — no articles, no filtering.

Output: wiki_categories.xml

Usage:
    python fetch_wiki_xml.py
"""

import urllib.request
import urllib.parse
import json
import time
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WindowsillBot/1.0 (semantic-domain-research)"
REQUEST_DELAY = 0.12
ROOT_CATEGORY = "Main topic classifications"


def _api_get(params: dict, retries: int = 3) -> dict:
    qs = urllib.parse.urlencode(params)
    url = f"{API_URL}?{qs}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"      retry {attempt+1} in {wait}s: {e}", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"      FAILED: {e}", file=sys.stderr)
                return {}
    return {}


def fetch_subcategories(category: str) -> list[str]:
    """Fetch ALL subcategories (handles continuation). No limit, no filtering."""
    all_cats = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "subcat",
        "cmlimit": "500",
        "format": "json",
    }
    while True:
        data = _api_get(params)
        members = data.get("query", {}).get("categorymembers", [])
        all_cats.extend(m["title"].replace("Category:", "") for m in members)
        cont = data.get("continue")
        if cont:
            params["cmcontinue"] = cont["cmcontinue"]
            time.sleep(REQUEST_DELAY)
        else:
            break
    return all_cats


def pretty_xml(root: ET.Element) -> str:
    rough = ET.tostring(root, encoding="unicode", xml_declaration=False)
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def main():
    print(f"Fetching category tree from Category:{ROOT_CATEGORY}...", file=sys.stderr)

    # Level 0: root
    root_el = ET.Element("wikipedia_categories")
    root_el.set("root", ROOT_CATEGORY)

    islands = fetch_subcategories(ROOT_CATEGORY)
    print(f"  Level 1 (islands): {len(islands)} categories", file=sys.stderr)

    total_towns = 0

    for i, island in enumerate(islands):
        island_el = ET.SubElement(root_el, "island")
        island_el.set("name", island)

        time.sleep(REQUEST_DELAY)
        towns = fetch_subcategories(island)
        island_el.set("town_count", str(len(towns)))
        total_towns += len(towns)

        for town in towns:
            town_el = ET.SubElement(island_el, "town")
            town_el.set("name", town)

        print(
            f"  [{i+1:3d}/{len(islands)}] {island:40s} → {len(towns)} towns",
            file=sys.stderr,
        )

    # Write
    out_path = Path("wiki_categories.xml")
    xml_str = pretty_xml(root_el)
    out_path.write_text(xml_str, encoding="utf-8")

    print(f"\nDone. Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)", file=sys.stderr)
    print(f"  Islands: {len(islands)}", file=sys.stderr)
    print(f"  Total towns: {total_towns:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
