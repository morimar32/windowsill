"""
Generate seed words for v3 towns via Claude API.

Uses the archipelago > island > town hierarchy context to produce
discriminative vocabularies. Writes directly into the SeedWords table
with word_id linking at insert time.

Usage:
    python v3/generate_seeds.py              # generate all unseeded
    python v3/generate_seeds.py --dry-run    # show what would be generated
    python v3/generate_seeds.py --dead-towns # regenerate for towns with 0 words
    python v3/generate_seeds.py --low-f1     # regenerate for towns with model_f1 < 0.75
    python v3/generate_seeds.py --island "Sport"
    python v3/generate_seeds.py --town "Chess"
    python v3/generate_seeds.py --batch-size 3
"""

import argparse
import json
import sqlite3
import sys
import time

# -- config --
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
BATCH_SIZE = 5          # towns per API call
API_DELAY = 0.5         # seconds between calls
V3_DB = "v3/windowsill.db"
LOW_F1_THRESHOLD = 0.75
MIN_LINKED_WARN = 15    # warn if fewer linked words than this


def get_unseeded_towns(db):
    """Return towns with zero seed words, grouped by island/archipelago."""
    rows = db.execute("""
        SELECT t.town_id, t.name AS town, i.name AS island, a.name AS archipelago
        FROM Towns t
        JOIN Islands i USING (island_id)
        JOIN Archipelagos a USING (archipelago_id)
        WHERE t.town_id NOT IN (SELECT DISTINCT town_id FROM SeedWords)
        ORDER BY a.archipelago_id, i.island_id, t.town_id
    """).fetchall()
    return [{"town_id": r[0], "town": r[1], "island": r[2], "archipelago": r[3]} for r in rows]


def get_dead_towns(db):
    """Return non-bucket towns with word_count = 0 or NULL."""
    rows = db.execute("""
        SELECT t.town_id, t.name AS town, i.name AS island, a.name AS archipelago
        FROM Towns t
        JOIN Islands i USING (island_id)
        JOIN Archipelagos a USING (archipelago_id)
        WHERE i.is_bucket = 0
          AND (t.word_count = 0 OR t.word_count IS NULL)
        ORDER BY a.archipelago_id, i.island_id, t.town_id
    """).fetchall()
    return [{"town_id": r[0], "town": r[1], "island": r[2], "archipelago": r[3]} for r in rows]


def get_low_f1_towns(db):
    """Return non-bucket towns with model_f1 < threshold."""
    rows = db.execute("""
        SELECT t.town_id, t.name AS town, i.name AS island, a.name AS archipelago,
               t.model_f1
        FROM Towns t
        JOIN Islands i USING (island_id)
        JOIN Archipelagos a USING (archipelago_id)
        WHERE i.is_bucket = 0
          AND t.model_f1 IS NOT NULL
          AND t.model_f1 < ?
        ORDER BY t.model_f1, t.town_id
    """, (LOW_F1_THRESHOLD,)).fetchall()
    return [{"town_id": r[0], "town": r[1], "island": r[2], "archipelago": r[3],
             "model_f1": r[4]} for r in rows]


def get_towns_by_island(db, island_name):
    """Return all towns in a specific island."""
    rows = db.execute("""
        SELECT t.town_id, t.name AS town, i.name AS island, a.name AS archipelago
        FROM Towns t
        JOIN Islands i USING (island_id)
        JOIN Archipelagos a USING (archipelago_id)
        WHERE i.name = ?
        ORDER BY t.town_id
    """, (island_name,)).fetchall()
    return [{"town_id": r[0], "town": r[1], "island": r[2], "archipelago": r[3]} for r in rows]


def get_towns_by_name(db, town_name):
    """Return towns matching a specific name."""
    rows = db.execute("""
        SELECT t.town_id, t.name AS town, i.name AS island, a.name AS archipelago
        FROM Towns t
        JOIN Islands i USING (island_id)
        JOIN Archipelagos a USING (archipelago_id)
        WHERE t.name = ?
        ORDER BY t.town_id
    """, (town_name,)).fetchall()
    return [{"town_id": r[0], "town": r[1], "island": r[2], "archipelago": r[3]} for r in rows]


def get_sibling_towns(db, island_name):
    """Get all towns in the same island (for context)."""
    rows = db.execute("""
        SELECT t.name FROM Towns t
        JOIN Islands i USING (island_id)
        WHERE i.name = ?
        ORDER BY t.name
    """, (island_name,)).fetchall()
    return [r[0] for r in rows]


def get_seeded_siblings_sample(db, island_name, limit=5):
    """Get sample seed words from sibling towns for contrast."""
    rows = db.execute("""
        SELECT t.name, GROUP_CONCAT(s.word, ', ')
        FROM SeedWords s
        JOIN Towns t USING (town_id)
        JOIN Islands i USING (island_id)
        WHERE i.name = ? AND s.confidence = 'core'
        GROUP BY t.town_id
        ORDER BY t.name
        LIMIT ?
    """, (island_name, limit)).fetchall()
    # Truncate word lists to first 15 words
    return [(name, ", ".join(words.split(", ")[:15])) for name, words in rows]


def build_vocab_lookup(db):
    """Build a word -> word_id lookup from the Words table."""
    rows = db.execute("SELECT word_id, word FROM Words").fetchall()
    return {word.lower(): word_id for word_id, word in rows}


def build_prompt(batch, sibling_context):
    """Build generation prompt for a batch of towns."""
    town_sections = []
    for item in batch:
        siblings = item["_siblings"]
        section = f"### {item['town']}\n"
        section += f"Island: {item['island']} (Archipelago: {item['archipelago']})\n"
        section += f"Sibling towns in {item['island']}: {', '.join(siblings)}\n"
        town_sections.append(section)

    towns_text = "\n".join(town_sections)
    town_names = [item["town"] for item in batch]

    # Add sibling context if available
    context_text = ""
    if sibling_context:
        context_lines = []
        for town_name, words in sibling_context:
            context_lines.append(f"  - {town_name}: {words}")
        context_text = f"""
## Existing sibling vocabulary (for contrast — do NOT repeat these words)

{chr(10).join(context_lines)}
"""

    return f"""You are generating training data for a hierarchical domain classifier. For each town below, produce 80-150 single words that are DISCRIMINATIVE for that specific town within its island.

## Hierarchy

The classifier uses a 4-tier hierarchy: Archipelago > Island > Town > Reef.
Words should distinguish THIS town from its sibling towns within the same island.

## CRITICAL: Single words only

Every entry MUST be a single dictionary word. The words will be matched against a WordNet vocabulary, so they must be real English words that appear in a standard dictionary.

GOOD examples: "checkmate", "castling", "bishop", "pawn", "endgame"
BAD examples: "binary tree", "lateral pass", "bhakti devotion", "en passant"

Do NOT produce:
- Multi-word phrases (no spaces)
- Hyphenated compounds ("well-known")
- Proper nouns or names ("Fischer", "Kasparov")
- Abbreviations ("DNA", "CPU")
- Non-English words unless adopted into English dictionaries

## Other rules

1. DO NOT include generic/function words like: create, system, run, process, method, function, data, type, value, set, list, make, use, change, work, build, test, check, see, win, lord, form, kind, order
2. DO NOT include words that belong equally to sibling towns — be SPECIFIC to this town
3. Sort alphabetically within each town
4. Classify into two tiers:
   - "core": unmistakably belongs to this town (e.g., "checkmate" for Chess)
   - "peripheral": strongly associated but could appear in related towns (e.g., "tournament" for Chess)
5. Aim for ~70% core, ~30% peripheral
6. Include technical terms, jargon, tools, methods, and objects specific to the field
7. Prefer concrete, specific words over abstract ones
{context_text}
## Towns to generate

{towns_text}

## Response format

Return ONLY a JSON object mapping town names to their word lists:

{{
  "{town_names[0]}": {{
    "core": ["word1", "word2", ...],
    "peripheral": ["word1", "word2", ...]
  }}
}}

No markdown, no commentary — just the JSON object."""


def call_claude(prompt, max_tokens=8192, max_retries=5):
    """Call Claude API and return parsed JSON. Retries on overload."""
    import anthropic
    client = anthropic.Anthropic()

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except anthropic.OverloadedError:
            wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
            print(f"  API overloaded, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    else:
        print(f"  ERROR: API overloaded after {max_retries} retries")
        return None

    text = response.content[0].text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse response: {text[:500]}")
        return None


def insert_seeds(db, town_id, words, confidence, vocab_lookup):
    """Insert seed words with word_id linking. Returns (inserted, linked, skipped_multi)."""
    cur = db.cursor()
    inserted = 0
    linked = 0
    skipped_multi = 0

    for word in words:
        word = word.lower().strip()
        if not word:
            continue
        # Skip multi-word entries (defensive)
        if " " in word or "-" in word:
            skipped_multi += 1
            continue

        word_id = vocab_lookup.get(word)
        try:
            cur.execute(
                "INSERT OR IGNORE INTO SeedWords (town_id, word, word_id, source, confidence) "
                "VALUES (?, ?, ?, 'claude_augmented', ?)",
                (town_id, word, word_id, confidence)
            )
            if cur.rowcount > 0:
                inserted += 1
                if word_id is not None:
                    linked += 1
        except sqlite3.IntegrityError:
            pass  # duplicate

    return inserted, linked, skipped_multi


def main():
    parser = argparse.ArgumentParser(description="Generate seed words for v3 towns via Claude API")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Towns per API call")
    parser.add_argument("--dead-towns", action="store_true",
                        help="Regenerate for towns with word_count = 0")
    parser.add_argument("--low-f1", action="store_true",
                        help=f"Regenerate for towns with model_f1 < {LOW_F1_THRESHOLD}")
    parser.add_argument("--island", type=str, help="Generate for all towns in this island")
    parser.add_argument("--town", type=str, help="Generate for a specific town")
    args = parser.parse_args()

    db = sqlite3.connect(V3_DB)
    db.execute("PRAGMA foreign_keys = ON")

    # Determine target towns based on flags
    if args.dead_towns:
        targets = get_dead_towns(db)
        mode = "dead towns (word_count = 0)"
    elif args.low_f1:
        targets = get_low_f1_towns(db)
        mode = f"low-F1 towns (model_f1 < {LOW_F1_THRESHOLD})"
    elif args.island:
        targets = get_towns_by_island(db, args.island)
        mode = f"island '{args.island}'"
    elif args.town:
        targets = get_towns_by_name(db, args.town)
        mode = f"town '{args.town}'"
    else:
        targets = get_unseeded_towns(db)
        mode = "unseeded towns"

    print(f"Mode: {mode}")
    print(f"Target towns: {len(targets)}")

    if not targets:
        print("No towns to process!")
        return

    if args.dry_run:
        current_island = None
        for item in targets:
            if item["island"] != current_island:
                current_island = item["island"]
                print(f"\n  {item['archipelago']} > {item['island']}:")
            f1_info = f" (F1={item['model_f1']:.3f})" if "model_f1" in item else ""
            print(f"    - {item['town']}{f1_info}")
        print(f"\nTotal: {len(targets)} towns in {len(set(i['island'] for i in targets))} islands")
        return

    # Build vocab lookup for word_id linking
    print("Building vocabulary lookup...")
    vocab_lookup = build_vocab_lookup(db)
    print(f"  {len(vocab_lookup):,} words in vocabulary")

    # Enrich with sibling info
    island_siblings_cache = {}
    for item in targets:
        island = item["island"]
        if island not in island_siblings_cache:
            island_siblings_cache[island] = get_sibling_towns(db, island)
        item["_siblings"] = island_siblings_cache[island]

    # Process in batches, grouped by island when possible
    batch_size = args.batch_size
    n_batches = (len(targets) + batch_size - 1) // batch_size
    total_inserted = 0
    total_linked = 0
    total_skipped = 0
    total_failed = 0

    print(f"Generating in {n_batches} batches of up to {batch_size}...")
    print()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(targets))
        batch = targets[batch_start:batch_end]
        batch_names = [item["town"] for item in batch]

        print(f"Batch {batch_idx + 1}/{n_batches}: {', '.join(batch_names)}")

        # Get sibling context for contrast
        islands_in_batch = set(item["island"] for item in batch)
        sibling_context = []
        for island in islands_in_batch:
            sibling_context.extend(get_seeded_siblings_sample(db, island))

        prompt = build_prompt(batch, sibling_context)
        result = call_claude(prompt)

        if result is None:
            print(f"  FAILED — skipping batch")
            total_failed += len(batch)
            time.sleep(API_DELAY)
            continue

        db.execute("BEGIN TRANSACTION")

        for item in batch:
            town_name = item["town"]
            town_id = item["town_id"]

            if town_name not in result:
                print(f"  WARNING: No results for '{town_name}'")
                total_failed += 1
                continue

            town_data = result[town_name]
            core = town_data.get("core", [])
            peripheral = town_data.get("peripheral", [])

            core_ins, core_link, core_skip = insert_seeds(
                db, town_id, core, "core", vocab_lookup
            )
            peri_ins, peri_link, peri_skip = insert_seeds(
                db, town_id, peripheral, "peripheral", vocab_lookup
            )

            inserted = core_ins + peri_ins
            linked = core_link + peri_link
            skipped = core_skip + peri_skip
            total_inserted += inserted
            total_linked += linked
            total_skipped += skipped

            link_pct = (linked / inserted * 100) if inserted > 0 else 0
            warn = ""
            if linked < MIN_LINKED_WARN:
                warn = f" *** LOW LINK COUNT"
            print(f"  {town_name}: {len(core)}+{len(peripheral)} raw → "
                  f"{inserted} inserted ({linked} linked={link_pct:.0f}%, "
                  f"{skipped} multi-word skipped){warn}")

        db.commit()
        time.sleep(API_DELAY)

    print()
    link_pct = (total_linked / total_inserted * 100) if total_inserted > 0 else 0
    print(f"Done. Inserted: {total_inserted:,} seed words "
          f"({total_linked:,} linked={link_pct:.0f}%, "
          f"{total_skipped:,} multi-word skipped). "
          f"Failed towns: {total_failed}")

    # Final check for unseeded
    remaining = get_unseeded_towns(db)
    if remaining:
        print(f"\nStill unseeded: {len(remaining)} towns")
        for item in remaining[:10]:
            print(f"  {item['archipelago']} > {item['island']} > {item['town']}")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")
    else:
        print("\nAll towns now have seeds!")

    db.close()


if __name__ == "__main__":
    main()
