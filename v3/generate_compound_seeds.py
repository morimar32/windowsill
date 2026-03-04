"""
Generate compound seed words for v3 towns via Claude API.

Augments the single-word seeds from generate_seeds.py with multi-word
compound terms (2-4 words) that will be matched via Aho-Corasick in Lagoon.

Targets low-health islands by default (computed from seed word sharing
statistics), or accepts --island / --all overrides.

Pipeline position: step 9b — runs after generate_seeds.py, before
sanity_check_seeds.py.

Usage:
    python v3/generate_compound_seeds.py                     # dry-run, low-health islands
    python v3/generate_compound_seeds.py --apply             # insert for low-health islands
    python v3/generate_compound_seeds.py --island "Games" --apply
    python v3/generate_compound_seeds.py --all --apply
    python v3/generate_compound_seeds.py --health-threshold 130 --apply
"""

import argparse
import json
import os
import sqlite3
import struct
import sys
import time

import numpy as np

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project)

# -- config --
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
BATCH_SIZE = 3            # towns per API call (smaller than single-word due to longer output)
API_DELAY = 0.5           # seconds between calls
V3_DB = os.path.join(_project, "v3/windowsill.db")
HEALTH_THRESHOLD = 115    # islands below this are targets

# Embedding config (must match reembed_words.py)
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_PREFIX = "clustering: "
EMBEDDING_DIM = 768

# FNV-1a constants (must match load_wordnet_vocab.py / config.py)
FNV1A_OFFSET = 14695981039346656037
FNV1A_PRIME = 1099511628211


def fnv1a_u64(s):
    """FNV-1a u64 hash of a string."""
    h = FNV1A_OFFSET
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * FNV1A_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def fnv1a_i64(s):
    """FNV-1a hash stored as signed i64 (SQLite convention)."""
    h = fnv1a_u64(s)
    if h >= (1 << 63):
        h -= (1 << 64)
    return h


def pack_embedding(arr):
    """Pack a numpy float32 array into a BLOB."""
    return struct.pack(f"{EMBEDDING_DIM}f", *arr.astype(np.float32))


# ---------------------------------------------------------------------------
# Health score
# ---------------------------------------------------------------------------

def compute_island_health(db):
    """Compute health scores for non-bucket islands based on seed word sharing.

    Health = 2*exclusive% + seeds_per_town/10 + (100 - shared5+%)
    where:
      exclusive% = fraction of island's seed words appearing in ≤2 islands
      shared5+%  = fraction appearing in 5+ islands
      seeds_per_town = average seed count per town in the island
    """
    # Count how many islands each seed word appears in
    word_island_counts = {}
    for word, n_islands in db.execute("""
        SELECT s.word, COUNT(DISTINCT i.island_id)
        FROM SeedWords s
        JOIN Towns t USING (town_id)
        JOIN Islands i USING (island_id)
        WHERE i.is_bucket = 0
        GROUP BY s.word
    """).fetchall():
        word_island_counts[word] = n_islands

    # Per-island aggregates
    islands = db.execute("""
        SELECT i.island_id, i.name,
               COUNT(DISTINCT s.rowid) AS seed_count,
               COUNT(DISTINCT t.town_id) AS town_count
        FROM Islands i
        JOIN Towns t USING (island_id)
        LEFT JOIN SeedWords s USING (town_id)
        WHERE i.is_bucket = 0
        GROUP BY i.island_id
        ORDER BY i.name
    """).fetchall()

    results = []
    for island_id, name, seed_count, town_count in islands:
        if seed_count == 0:
            results.append({
                "island_id": island_id, "name": name, "health": 0,
                "seed_count": seed_count, "town_count": town_count,
                "exclusive_pct": 0, "shared5_pct": 0, "seeds_per_town": 0,
            })
            continue

        # Distinct seed words in this island
        island_words = [r[0] for r in db.execute("""
            SELECT DISTINCT s.word
            FROM SeedWords s
            JOIN Towns t USING (town_id)
            WHERE t.island_id = ?
        """, (island_id,)).fetchall()]

        n_words = len(island_words)
        exclusive = sum(1 for w in island_words if word_island_counts.get(w, 0) <= 2)
        shared5 = sum(1 for w in island_words if word_island_counts.get(w, 0) >= 5)

        exclusive_pct = 100 * exclusive / n_words
        shared5_pct = 100 * shared5 / n_words
        seeds_per_town = seed_count / town_count if town_count > 0 else 0

        health = 2 * exclusive_pct + seeds_per_town / 10 + (100 - shared5_pct)

        results.append({
            "island_id": island_id, "name": name, "health": health,
            "seed_count": seed_count, "town_count": town_count,
            "exclusive_pct": exclusive_pct, "shared5_pct": shared5_pct,
            "seeds_per_town": seeds_per_town,
        })

    return sorted(results, key=lambda x: x["health"])


# ---------------------------------------------------------------------------
# Town / island queries
# ---------------------------------------------------------------------------

def get_towns_for_island(db, island_id):
    """Get all towns for an island."""
    rows = db.execute("""
        SELECT t.town_id, t.name AS town, i.name AS island, a.name AS archipelago
        FROM Towns t
        JOIN Islands i USING (island_id)
        JOIN Archipelagos a USING (archipelago_id)
        WHERE t.island_id = ?
        ORDER BY t.town_id
    """, (island_id,)).fetchall()
    return [{"town_id": r[0], "town": r[1], "island": r[2], "archipelago": r[3]}
            for r in rows]


def get_sibling_towns(db, island_name):
    """Get all town names in the same island (for prompt context)."""
    rows = db.execute("""
        SELECT t.name FROM Towns t
        JOIN Islands i USING (island_id)
        WHERE i.name = ?
        ORDER BY t.name
    """, (island_name,)).fetchall()
    return [r[0] for r in rows]


def get_existing_seeds_sample(db, island_name, limit=5):
    """Get sample single-word seeds from towns for contrast/context."""
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
    return [(name, ", ".join(words.split(", ")[:15])) for name, words in rows]


def build_vocab_lookup(db):
    """Build a word -> word_id lookup from the Words table."""
    rows = db.execute("SELECT word_id, word FROM Words").fetchall()
    return {word.lower(): word_id for word_id, word in rows}


# ---------------------------------------------------------------------------
# Claude prompt
# ---------------------------------------------------------------------------

def build_compound_prompt(batch, sibling_context):
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

    context_text = ""
    if sibling_context:
        context_lines = [f"  - {name}: {words}" for name, words in sibling_context]
        context_text = f"""
## Existing single-word seeds (for context — produce COMPOUND versions and related terms)

{chr(10).join(context_lines)}
"""

    return f"""You are generating compound training data for a hierarchical domain classifier.
For each town below, produce 30-60 MULTI-WORD compound terms that are
DISCRIMINATIVE for that specific town within its island.

## Hierarchy

The classifier uses a 4-tier hierarchy: Archipelago > Island > Town > Reef.
Compounds should distinguish THIS town from its sibling towns within the same island.

## CRITICAL: Multi-word compounds only

Every entry MUST be a multi-word compound term (2-4 words). These will be
matched via Aho-Corasick in running text, so they must be natural phrases
that appear in real English text.

GOOD examples: "machine learning", "neural network", "board game",
               "claw machine", "free throw", "blood pressure"
BAD examples: "algorithm" (single word), "the game" (too generic),
              "Fischer random chess" (proper noun), "very fast car" (not a term)

## Rules

1. Only natural compound terms — noun phrases, technical terms, field-specific jargon
2. NO proper nouns or named entities ("Monte Carlo", "Turing machine" → skip)
3. NO generic phrases ("good thing", "big problem")
4. Compounds should be DISCRIMINATIVE — they should point to THIS town, not siblings
5. Classify into "core" and "peripheral" tiers (same as single-word seeds)
6. Aim for ~70% core, ~30% peripheral
7. Sort alphabetically within each tier
8. Include: technical terms, tools, methods, apparatus, processes, phenomena
{context_text}
## Towns to generate

{towns_text}

## Response format

Return ONLY a JSON object mapping town names to their word lists:

{{
  "{town_names[0]}": {{
    "core": ["compound term 1", "compound term 2", ...],
    "peripheral": ["compound term 1", "compound term 2", ...]
  }}
}}

No markdown, no commentary — just the JSON object."""


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

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
            print(f"  API overloaded, retrying in {wait}s "
                  f"(attempt {attempt + 1}/{max_retries})...")
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


# ---------------------------------------------------------------------------
# Words table insertion
# ---------------------------------------------------------------------------

def insert_compound_words(db, new_compounds):
    """Insert new compound words into the Words table.

    Returns dict of {word: word_id} for all inserted words.
    """
    cur = db.cursor()
    new_ids = {}

    for word in new_compounds:
        word_hash = fnv1a_i64(word)
        word_count = len(word.split())
        try:
            cur.execute("""
                INSERT INTO Words (word, word_hash, pos, word_count, category, is_stop)
                VALUES (?, ?, NULL, ?, 'compound', 0)
            """, (word, word_hash, word_count))
            new_ids[word] = cur.lastrowid
        except sqlite3.IntegrityError:
            # Shouldn't happen (no unique constraint), but be defensive
            row = cur.execute(
                "SELECT word_id FROM Words WHERE word = ?", (word,)
            ).fetchone()
            if row:
                new_ids[word] = row[0]

    return new_ids


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_words(model, words_with_ids):
    """Embed compound words inline. Returns list of (word_id, embedding_blob)."""
    if not words_with_ids:
        return []

    words = [w for w, _ in words_with_ids]
    texts = [f"{EMBEDDING_PREFIX}{w}" for w in words]
    embeddings = model.encode(texts, show_progress_bar=False)

    results = []
    for (_, word_id), emb in zip(words_with_ids, embeddings):
        results.append((word_id, pack_embedding(emb)))
    return results


# ---------------------------------------------------------------------------
# Seed insertion
# ---------------------------------------------------------------------------

def insert_seeds(db, town_id, words, confidence, vocab_lookup):
    """Insert compound seed words into SeedWords. Returns (inserted, linked)."""
    cur = db.cursor()
    inserted = 0
    linked = 0

    for word in words:
        word = word.lower().strip()
        if not word or " " not in word:
            continue

        word_id = vocab_lookup.get(word)
        try:
            cur.execute(
                "INSERT OR IGNORE INTO SeedWords "
                "(town_id, word, word_id, source, confidence) "
                "VALUES (?, ?, ?, 'claude_compound', ?)",
                (town_id, word, word_id, confidence)
            )
            if cur.rowcount > 0:
                inserted += 1
                if word_id is not None:
                    linked += 1
        except sqlite3.IntegrityError:
            pass  # duplicate (town_id, word)

    return inserted, linked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate compound seed words for v3 towns via Claude API")
    parser.add_argument("--apply", action="store_true",
                        help="Actually generate and insert (default is dry-run)")
    parser.add_argument("--island", type=str,
                        help="Target a specific island by name")
    parser.add_argument("--all", action="store_true",
                        help="Target all non-bucket islands")
    parser.add_argument("--health-threshold", type=float, default=HEALTH_THRESHOLD,
                        help=f"Health score threshold for auto-targeting "
                             f"(default: {HEALTH_THRESHOLD})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Towns per API call (default: {BATCH_SIZE})")
    args = parser.parse_args()

    dry_run = not args.apply

    db = sqlite3.connect(V3_DB)
    db.execute("PRAGMA foreign_keys = ON")

    # ---- Determine target islands ----
    if args.island:
        row = db.execute(
            "SELECT island_id, name FROM Islands WHERE name = ? AND is_bucket = 0",
            (args.island,)
        ).fetchone()
        if not row:
            print(f"ERROR: Island '{args.island}' not found (or is a bucket island)")
            return
        target_islands = [{"island_id": row[0], "name": row[1]}]
        mode = f"island '{args.island}'"

    elif args.all:
        rows = db.execute(
            "SELECT island_id, name FROM Islands WHERE is_bucket = 0 ORDER BY name"
        ).fetchall()
        target_islands = [{"island_id": r[0], "name": r[1]} for r in rows]
        mode = "all non-bucket islands"

    else:
        # Auto-target low-health islands
        print("Computing island health scores...")
        health = compute_island_health(db)

        print(f"\n  {'Island':<30} {'Health':>7} {'Excl%':>6} "
              f"{'Sh5+%':>6} {'S/Town':>7}")
        print(f"  {'-' * 58}")
        for h in health:
            marker = " <<<" if h["health"] < args.health_threshold else ""
            print(f"  {h['name']:<30} {h['health']:>7.1f} "
                  f"{h['exclusive_pct']:>5.1f}% {h['shared5_pct']:>5.1f}% "
                  f"{h['seeds_per_town']:>6.1f}{marker}")

        target_islands = [h for h in health
                          if h["health"] < args.health_threshold]
        mode = f"low-health islands (health < {args.health_threshold})"

    # ---- Collect target towns ----
    targets = []
    for island in target_islands:
        targets.extend(get_towns_for_island(db, island["island_id"]))

    print(f"\nMode: {mode}")
    print(f"Target islands: {len(target_islands)}")
    print(f"Target towns: {len(targets)}")

    if not targets:
        print("No towns to process!")
        return

    # ---- Dry-run: just list targets ----
    if dry_run:
        current_island = None
        for item in targets:
            if item["island"] != current_island:
                current_island = item["island"]
                print(f"\n  {item['archipelago']} > {item['island']}:")
            print(f"    - {item['town']}")
        print(f"\nTotal: {len(targets)} towns in {len(target_islands)} islands")
        print("\nUse --apply to generate and insert compound seeds.")
        return

    # ---- Apply mode ----
    print("Building vocabulary lookup...")
    vocab_lookup = build_vocab_lookup(db)
    print(f"  {len(vocab_lookup):,} words in vocabulary")

    print(f"Loading embedding model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print("  Model loaded")

    # Enrich towns with sibling info
    island_siblings_cache = {}
    for item in targets:
        island = item["island"]
        if island not in island_siblings_cache:
            island_siblings_cache[island] = get_sibling_towns(db, island)
        item["_siblings"] = island_siblings_cache[island]

    # Process in batches
    batch_size = args.batch_size
    n_batches = (len(targets) + batch_size - 1) // batch_size

    total_seeds_inserted = 0
    total_seeds_linked = 0
    total_words_added = 0
    total_words_embedded = 0
    total_failed = 0

    print(f"\nGenerating in {n_batches} batches of up to {batch_size}...\n")

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(targets))
        batch = targets[batch_start:batch_end]
        batch_names = [item["town"] for item in batch]

        print(f"Batch {batch_idx + 1}/{n_batches}: {', '.join(batch_names)}")

        # Get sibling context for the prompt
        islands_in_batch = set(item["island"] for item in batch)
        sibling_context = []
        for island in islands_in_batch:
            sibling_context.extend(get_existing_seeds_sample(db, island))

        prompt = build_compound_prompt(batch, sibling_context)
        result = call_claude(prompt)

        if result is None:
            print("  FAILED — skipping batch")
            total_failed += len(batch)
            time.sleep(API_DELAY)
            continue

        # Collect all compound words from this batch
        all_compounds = set()
        batch_town_compounds = {}

        for item in batch:
            town_name = item["town"]
            town_id = item["town_id"]

            if town_name not in result:
                print(f"  WARNING: No results for '{town_name}'")
                total_failed += 1
                continue

            town_data = result[town_name]
            core = [w.lower().strip() for w in town_data.get("core", [])
                    if " " in w.strip()]
            peripheral = [w.lower().strip() for w in town_data.get("peripheral", [])
                          if " " in w.strip()]

            batch_town_compounds[town_id] = {
                "core": core, "peripheral": peripheral, "name": town_name,
            }
            all_compounds.update(core)
            all_compounds.update(peripheral)

        # Insert new compounds into Words table
        new_compounds = [w for w in all_compounds if w not in vocab_lookup]

        if new_compounds:
            db.execute("BEGIN TRANSACTION")
            new_ids = insert_compound_words(db, new_compounds)
            db.commit()
            total_words_added += len(new_ids)

            # Embed the new words
            words_to_embed = list(new_ids.items())
            emb_results = embed_words(model, words_to_embed)

            db.execute("BEGIN TRANSACTION")
            for word_id, blob in emb_results:
                db.execute(
                    "UPDATE Words SET embedding = ? WHERE word_id = ?",
                    (blob, word_id)
                )
            db.commit()
            total_words_embedded += len(emb_results)

            # Update vocab lookup with newly added words
            for w, wid in new_ids.items():
                vocab_lookup[w] = wid

        # Insert seeds
        db.execute("BEGIN TRANSACTION")
        for town_id, data in batch_town_compounds.items():
            core_ins, core_link = insert_seeds(
                db, town_id, data["core"], "core", vocab_lookup)
            peri_ins, peri_link = insert_seeds(
                db, town_id, data["peripheral"], "peripheral", vocab_lookup)

            inserted = core_ins + peri_ins
            linked = core_link + peri_link
            total_seeds_inserted += inserted
            total_seeds_linked += linked

            link_pct = (linked / inserted * 100) if inserted > 0 else 0
            print(f"  {data['name']}: {len(data['core'])}+{len(data['peripheral'])} "
                  f"raw → {inserted} seeds ({linked} linked={link_pct:.0f}%)")

        db.commit()

        if new_compounds:
            print(f"  +{len(new_compounds)} new words added to vocabulary, "
                  f"{total_words_embedded} embedded so far")

        time.sleep(API_DELAY)

    # ---- Summary ----
    print()
    link_pct = ((total_seeds_linked / total_seeds_inserted * 100)
                if total_seeds_inserted > 0 else 0)
    print(f"Done.")
    print(f"  Seeds inserted:  {total_seeds_inserted:,} "
          f"({total_seeds_linked:,} linked={link_pct:.0f}%)")
    print(f"  Words added:     {total_words_added:,} "
          f"({total_words_embedded:,} embedded)")
    print(f"  Failed towns:    {total_failed}")

    # ---- Verification ----
    print(f"\nVerification:")

    compound_count = db.execute(
        "SELECT COUNT(*) FROM Words WHERE category = 'compound' AND word_count > 1"
    ).fetchone()[0]
    print(f"  Compound words in vocabulary: {compound_count:,}")

    compound_seeds = db.execute(
        "SELECT COUNT(*) FROM SeedWords WHERE source = 'claude_compound'"
    ).fetchone()[0]
    print(f"  Compound seeds in SeedWords:  {compound_seeds:,}")

    null_emb = db.execute("""
        SELECT COUNT(*) FROM SeedWords s
        JOIN Words w USING (word_id)
        WHERE s.source = 'claude_compound' AND w.embedding IS NULL
    """).fetchone()[0]
    print(f"  Compound seeds missing embeddings: {null_emb}")

    # Per-island breakdown
    print(f"\n  Per-island compound seeds:")
    for row in db.execute("""
        SELECT i.name, COUNT(*) AS n
        FROM SeedWords s
        JOIN Towns t USING (town_id)
        JOIN Islands i USING (island_id)
        WHERE s.source = 'claude_compound'
        GROUP BY i.island_id
        ORDER BY n DESC
    """).fetchall():
        print(f"    {row[0]:<30} {row[1]:>5}")

    db.close()


if __name__ == "__main__":
    main()
