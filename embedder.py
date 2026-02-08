import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


def load_model():
    return SentenceTransformer(
        config.MODEL_NAME,
        trust_remote_code=config.TRUST_REMOTE_CODE,
    )


def prepare_texts(words):
    return [config.EMBEDDING_PREFIX + w for w in words]


def find_existing_intermediates(intermediate_dir):
    if not os.path.exists(intermediate_dir):
        return 0, None

    pattern = re.compile(r"^batch_(\d+)\.npy$")
    chunks = []
    for f in sorted(os.listdir(intermediate_dir)):
        m = pattern.match(f)
        if m:
            batch_num = int(m.group(1))
            data = np.load(os.path.join(intermediate_dir, f))
            chunks.append((batch_num, data))

    if not chunks:
        return 0, None

    chunks.sort(key=lambda x: x[0])
    resume_batch = chunks[-1][0] + 1
    partial_matrix = np.concatenate([c[1] for c in chunks], axis=0)
    return resume_batch, partial_matrix


def embed_words(words, model, batch_size=None, intermediate_dir=None, resume=True, prepend_prefix=True):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if intermediate_dir is None:
        intermediate_dir = config.INTERMEDIATE_DIR

    os.makedirs(intermediate_dir, exist_ok=True)

    final_path = os.path.join(intermediate_dir, "embeddings_final.npy")
    if os.path.exists(final_path):
        print(f"Loading existing final embeddings from {final_path}")
        return np.load(final_path)

    start_batch = 0
    existing_embeddings = None
    if resume:
        start_batch, existing_embeddings = find_existing_intermediates(intermediate_dir)
        if existing_embeddings is not None:
            print(f"Resuming from batch {start_batch}, {existing_embeddings.shape[0]} words already embedded")

    texts = prepare_texts(words) if prepend_prefix else list(words)
    total_batches = (len(texts) + batch_size - 1) // batch_size

    new_chunks = []
    save_interval = config.INTERMEDIATE_SAVE_INTERVAL

    for batch_idx in tqdm(range(start_batch, total_batches), initial=start_batch, total=total_batches, desc="Embedding"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]

        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        new_chunks.append(batch_embeddings)

        if len(new_chunks) % save_interval == 0:
            chunk = np.concatenate(new_chunks, axis=0)
            chunk_path = os.path.join(intermediate_dir, f"batch_{start_batch:05d}.npy")
            np.save(chunk_path, chunk)
            if existing_embeddings is not None:
                existing_embeddings = np.concatenate([existing_embeddings, chunk], axis=0)
            else:
                existing_embeddings = chunk
            new_chunks = []
            start_batch = batch_idx + 1
            print(f"\n  Checkpoint saved: {existing_embeddings.shape[0]} words embedded")

    if new_chunks:
        remaining = np.concatenate(new_chunks, axis=0)
        if existing_embeddings is not None:
            all_embeddings = np.concatenate([existing_embeddings, remaining], axis=0)
        else:
            all_embeddings = remaining
    elif existing_embeddings is not None:
        all_embeddings = existing_embeddings
    else:
        raise RuntimeError("No embeddings generated")

    np.save(final_path, all_embeddings)
    print(f"Final embeddings saved: {all_embeddings.shape}")
    return all_embeddings


def build_sense_texts(con):
    from nltk.corpus import wordnet as wn
    import word_list
    word_list.ensure_nltk_data()

    rows = con.execute(
        "SELECT word_id, word FROM words WHERE pos IS NULL"
    ).fetchall()

    senses = []
    sense_id = 0
    for word_id, word in rows:
        lemma_name = word.replace(" ", "_")
        for synset in wn.synsets(lemma_name):
            pos = word_list.POS_MAP.get(synset.pos(), synset.pos())
            gloss = synset.definition()
            text = f"{config.SENSE_EMBEDDING_PREFIX}{word}: {gloss}"
            senses.append({
                "sense_id": sense_id,
                "word_id": word_id,
                "pos": pos,
                "synset_name": synset.name(),
                "gloss": gloss,
                "embedding_text": text,
            })
            sense_id += 1

    print(f"  Built {len(senses)} sense texts from {len(rows)} ambiguous words")
    return senses


def embed_senses(sense_texts, model, batch_size=None, intermediate_dir=None, resume=True):
    if batch_size is None:
        batch_size = config.SENSE_BATCH_SIZE
    if intermediate_dir is None:
        intermediate_dir = config.SENSE_INTERMEDIATE_DIR

    texts = [s["embedding_text"] for s in sense_texts]
    return embed_words(texts, model, batch_size=batch_size,
                       intermediate_dir=intermediate_dir, resume=resume,
                       prepend_prefix=False)


def cleanup_intermediates(intermediate_dir=None):
    if intermediate_dir is None:
        intermediate_dir = config.INTERMEDIATE_DIR
    if not os.path.exists(intermediate_dir):
        return
    pattern = re.compile(r"^batch_\d+\.npy$")
    for f in os.listdir(intermediate_dir):
        if pattern.match(f):
            os.remove(os.path.join(intermediate_dir, f))
            print(f"  Removed {f}")
