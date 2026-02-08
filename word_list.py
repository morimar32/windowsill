import re
import nltk
from nltk.corpus import wordnet as wn


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
