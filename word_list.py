import re

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from tqdm import tqdm

import config


def ensure_nltk_data():
    for resource in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def extract_wordnet_lemmas():
    ensure_nltk_data()
    return list(wn.all_lemma_names())


def clean_word(word):
    word = word.replace("_", " ").lower().strip()
    if not word:
        return None
    return word


def filter_words(words):
    keep_single = {"a", "i"}
    seen = set()
    filtered = []
    for w in words:
        if w in seen:
            continue
        if re.search(r"\d", w):
            continue
        if len(w) == 1 and w not in keep_single:
            continue
        seen.add(w)
        filtered.append(w)
    filtered.sort()
    return filtered


def get_word_list():
    raw = extract_wordnet_lemmas()
    cleaned = []
    for w in raw:
        c = clean_word(w)
        if c is not None:
            cleaned.append(c)
    words = filter_words(cleaned)
    return [(i + 1, w) for i, w in enumerate(words)]


POS_MAP = {"n": "noun", "v": "verb", "a": "adj", "s": "adj", "r": "adv"}

TAXONOMIC_PREFIXES = (
    "genus ", "family ", "order ", "class ", "phylum ",
    "division ", "kingdom ", "suborder ", "subclass ", "subfamily ",
)


def classify_word(word, lemma_name=None):
    if lemma_name is None:
        lemma_name = word.replace(" ", "_")
    synsets = wn.synsets(lemma_name)

    # Collect POS set
    pos_counts = {}
    has_instance_hypernym = False
    for s in synsets:
        p = POS_MAP.get(s.pos(), s.pos())
        pos_counts[p] = pos_counts.get(p, 0) + 1
        if s.instance_hypernyms():
            has_instance_hypernym = True

    all_pos = list(pos_counts.keys())
    pos = all_pos[0] if len(all_pos) == 1 else None

    # Category detection
    if " " not in word:
        category = "single"
    elif word.lower().startswith(TAXONOMIC_PREFIXES):
        category = "taxonomic"
    elif "verb" in pos_counts:
        category = "phrasal_verb"
    elif has_instance_hypernym:
        category = "named_entity"
    else:
        category = "compound"

    return pos, category, all_pos, pos_counts


def get_word_list_enriched():
    words = get_word_list()
    enriched = []
    for word_id, word in words:
        lemma_name = word.replace(" ", "_")
        pos, category, all_pos, _ = classify_word(word, lemma_name)
        enriched.append((word_id, word, pos, category, all_pos))
    return enriched


def add_words(existing, new_words):
    existing_words = {w for _, w in existing}
    combined = [w for _, w in existing]
    for w in new_words:
        c = clean_word(w)
        if c is not None and c not in existing_words:
            combined.append(c)
            existing_words.add(c)
    combined.sort()
    return [(i + 1, w) for i, w in enumerate(combined)]


def fnv1a_u64(s):
    """FNV-1a u64 hash of a string."""
    h = config.FNV1A_OFFSET
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * config.FNV1A_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def compute_word_hashes(con):
    """Compute FNV-1a u64 hashes for all words and store in words.word_hash."""
    rows = con.execute("SELECT word_id, word FROM words ORDER BY word_id").fetchall()
    print(f"  Hashing {len(rows)} words...")

    word_ids = []
    hashes = []
    for word_id, word in tqdm(rows, desc="Hashing"):
        word_ids.append(word_id)
        hashes.append(fnv1a_u64(word))

    df = pd.DataFrame({"word_id": word_ids, "word_hash": hashes})
    con.execute("""
        UPDATE words SET word_hash = df.word_hash
        FROM df WHERE words.word_id = df.word_id
    """)

    # Verify no collisions
    total = con.execute("SELECT COUNT(*) FROM words WHERE word_hash IS NOT NULL").fetchone()[0]
    distinct = con.execute("SELECT COUNT(DISTINCT word_hash) FROM words WHERE word_hash IS NOT NULL").fetchone()[0]
    nulls = con.execute("SELECT COUNT(*) FROM words WHERE word_hash IS NULL").fetchone()[0]
    print(f"  Hashed {total} words, {distinct} distinct hashes, {nulls} nulls")
    if total != distinct:
        print(f"  WARNING: {total - distinct} hash collisions detected!")
    else:
        print(f"  Zero collisions confirmed")


def expand_morphy_variants(con):
    """Expand WordNet morphy() variants and populate word_variants table."""
    ensure_nltk_data()

    rows = con.execute("SELECT word_id, word, word_hash FROM words ORDER BY word_id").fetchall()
    print(f"  Expanding morphy variants for {len(rows)} words...")

    # Build base entries and morphy expansions
    variants = []  # (variant_hash, variant, word_id, source)
    seen = set()   # (variant_hash, word_id) dedup

    wn_pos = [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]

    for word_id, word, word_hash in tqdm(rows, desc="Morphy expansion"):
        # Base entry: the word itself
        key = (word_hash, word_id)
        if key not in seen:
            variants.append((word_hash, word, word_id, "base"))
            seen.add(key)

        # Try morphy for each POS — morphy returns the base form for an inflected input,
        # but we want the reverse: given a base word, find inflected forms that map back to it.
        # morphy(inflected) -> base, so we check if morphy(word) returns something different
        lemma_name = word.replace(" ", "_")
        for pos in wn_pos:
            result = wn.morphy(lemma_name, pos)
            if result is not None:
                variant_text = result.replace("_", " ")
                if variant_text != word:
                    vh = fnv1a_u64(variant_text)
                    vkey = (vh, word_id)
                    if vkey not in seen:
                        variants.append((vh, variant_text, word_id, "morphy"))
                        seen.add(vkey)

        # Also try the word as a potential inflected form — find what it reduces to
        for pos in wn_pos:
            for synset in wn.synsets(lemma_name, pos):
                for lemma in synset.lemmas():
                    base_name = lemma.name().replace("_", " ").lower()
                    if base_name != word:
                        # This word's lemma is base_name — check if base_name is in our vocab
                        base_row = con.execute(
                            "SELECT word_id, word_hash FROM words WHERE word = ?", [base_name]
                        ).fetchone()
                        if base_row is not None:
                            base_id, base_hash = base_row
                            # Add current word as a variant of the base
                            vkey = (word_hash, base_id)
                            if vkey not in seen:
                                variants.append((word_hash, word, base_id, "morphy"))
                                seen.add(vkey)

    print(f"  Generated {len(variants)} variant entries")

    # Clear and insert
    con.execute("DELETE FROM word_variants")

    chunk_size = 50000
    for i in range(0, len(variants), chunk_size):
        chunk = variants[i:i + chunk_size]
        df = pd.DataFrame(chunk, columns=["variant_hash", "variant", "word_id", "source"])
        con.execute("INSERT INTO word_variants SELECT * FROM df")

    stats = con.execute("""
        SELECT source, COUNT(*) FROM word_variants GROUP BY source ORDER BY source
    """).fetchall()
    distinct = con.execute("SELECT COUNT(DISTINCT variant_hash) FROM word_variants").fetchone()[0]
    print(f"  Variant stats: {', '.join(f'{s}={c:,}' for s, c in stats)}")
    print(f"  Distinct variant hashes: {distinct:,}")
