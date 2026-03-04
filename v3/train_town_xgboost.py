"""
V3 Town-Level XGBoost Training.

Trains binary classifiers for each town within a specified island.
Uses the full hierarchy context (archipelago > island > sibling towns)
to build more focused negative sampling.

Usage:
    python v3/train_town_xgboost.py --island "Sport"
    python v3/train_town_xgboost.py --island "Sport" --threshold 0.4
    python v3/train_town_xgboost.py --island "Sport" --dry-run

Outputs:
    - Models saved to v3/models/{island}/{town}.json
    - Predictions written to augmented_towns table in v3 DB
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Import real xgboost (avoid the lib/xgboost.py shadow)
# ---------------------------------------------------------------------------
_here = os.path.abspath(os.path.dirname(__file__))
_project = os.path.abspath(os.path.join(_here, ".."))
sys.path.insert(0, _project)

_stashed = sys.modules.pop("xgboost", None)
_orig_path = sys.path[:]
sys.path = [p for p in sys.path
            if p not in ("", ".") and os.path.abspath(p) not in (_here, _project, os.path.join(_project, "lib"))]
from xgboost import XGBClassifier  # noqa: E402
sys.path = _orig_path
if _stashed is not None:
    sys.modules["xgboost"] = _stashed

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              average_precision_score, accuracy_score)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
V3_DB = os.path.join(_project, "v3/windowsill.db")
MODEL_DIR = os.path.join(_project, "v3/models")

NEG_RATIO = 5          # negatives per positive
SCORE_THRESHOLD = 0.4  # min probability for prediction
SEED = 42
KNN_K = 5

# XGBoost hyperparameters (same as v2 production)
XGB_PARAMS = dict(
    objective="binary:logistic",
    max_depth=8,
    n_estimators=2000,
    learning_rate=0.08,
    subsample=0.82,
    colsample_bytree=0.84,
    reg_alpha=0.12,
    reg_lambda=0.06,
    early_stopping_rounds=200,
    eval_metric="logloss",
    random_state=SEED,
    verbosity=0,
)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def unpack_embedding(blob):
    """Unpack a 768-dim float32 embedding from bytes."""
    return np.array(struct.unpack(f"{768}f", blob), dtype=np.float64)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data(v3_con):
    """Load global data shared across all town classifiers.

    Self-contained: uses only v3 database. POS flags derived from
    Words.pos column. DimStats for z-score normalization.
    """
    print("Loading global data...")
    t0 = time.time()

    # DimStats from v3
    ds = v3_con.execute(
        "SELECT dim_id, mean, std FROM DimStats WHERE dim_id < 768 ORDER BY dim_id"
    ).fetchall()
    means = np.array([r[1] for r in ds], dtype=np.float64)
    stds = np.array([r[2] for r in ds], dtype=np.float64)
    stds[stds == 0] = 1.0

    # Words: embeddings + POS
    rows = v3_con.execute(
        "SELECT word_id, embedding, pos FROM Words WHERE embedding IS NOT NULL"
    ).fetchall()
    emb_map = {}
    pos_map = {}
    for wid, blob, pos in rows:
        emb_map[wid] = unpack_embedding(blob)
        pos_map[wid] = pos or ""
    all_wids = sorted(emb_map.keys())

    # Build global feature matrix (768 z-scores + 4 POS flags = 772 cols)
    wid_to_row = {wid: i for i, wid in enumerate(all_wids)}
    n = len(all_wids)

    raw_embeddings = np.vstack([emb_map[wid] for wid in all_wids])

    # Z-score features
    z_scores = (raw_embeddings - means) / stds

    # POS flags (4 cols) — derived from Words.pos
    pos_features = np.zeros((n, 4), dtype=np.float64)
    for i, wid in enumerate(all_wids):
        p = pos_map[wid]
        pos_features[i, 0] = float(p == "noun")
        pos_features[i, 1] = float(p == "verb")
        pos_features[i, 2] = float(p == "adj")
        pos_features[i, 3] = float(p == "adv")

    X_global = np.hstack([z_scores, pos_features])

    # Pre-normalized embeddings for cosine similarity
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_normalized = raw_embeddings / norms

    elapsed = time.time() - t0
    print(f"  {n:,} words, {X_global.shape[1]} global features ({elapsed:.1f}s)")

    return {
        "all_wids": all_wids,
        "X_global": X_global,
        "raw_embeddings": raw_embeddings,
        "emb_normalized": emb_normalized,
        "wid_to_row": wid_to_row,
    }


def load_stop_wids(v3_con):
    """Load set of word_ids flagged as stop words."""
    rows = v3_con.execute(
        "SELECT word_id FROM Words WHERE is_stop = 1"
    ).fetchall()
    return {r[0] for r in rows}


def load_island_towns(v3_con, island_name, stop_wids=frozenset()):
    """Load town info for an island, including seed word_ids (excluding stops)."""
    rows = v3_con.execute("""
        SELECT t.town_id, t.name, a.name AS archipelago, i.name AS island,
               t.is_capital
        FROM Towns t
        JOIN Islands i USING (island_id)
        JOIN Archipelagos a USING (archipelago_id)
        WHERE i.name = ?
        ORDER BY t.town_id
    """, (island_name,)).fetchall()

    towns = []
    for town_id, town_name, arch, island, is_capital in rows:
        seed_wids = [r[0] for r in v3_con.execute(
            "SELECT word_id FROM SeedWords WHERE town_id = ? AND word_id IS NOT NULL",
            (town_id,)
        ).fetchall()]
        # Exclude stop words from seeds
        seed_wids = [w for w in seed_wids if w not in stop_wids]
        towns.append({
            "town_id": town_id,
            "name": town_name,
            "archipelago": arch,
            "island": island,
            "is_capital": bool(is_capital),
            "seed_wids": set(seed_wids),
        })

    return towns


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_town_features(pos_wids, raw_embeddings, emb_normalized, wid_to_row):
    """Compute town-specific features: centroid_cos and knn_top5."""
    pos_rows = [wid_to_row[wid] for wid in pos_wids if wid in wid_to_row]
    if not pos_rows:
        n = raw_embeddings.shape[0]
        return np.zeros((n, 2), dtype=np.float64)

    # Centroid from raw embeddings
    core_raw = raw_embeddings[pos_rows]
    centroid = core_raw.mean(axis=0)
    cnorm = np.linalg.norm(centroid)
    if cnorm > 0:
        centroid /= cnorm

    # Centroid cosine for all words
    centroid_cos = emb_normalized @ centroid

    # KNN mean cosine to core members
    core_normalized = emb_normalized[pos_rows]
    cos_to_core = emb_normalized @ core_normalized.T
    k = min(KNN_K, cos_to_core.shape[1])
    top_k = np.partition(cos_to_core, -k, axis=1)[:, -k:]
    knn_mean = top_k.mean(axis=1)

    return np.column_stack([centroid_cos, knn_mean])


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def build_negatives(pos_wids, sibling_wids, all_wids_set, rng):
    """Build negative sample set.

    Strategy for focused town classifiers:
      - 50% from sibling towns in same island (hard negatives)
      - 50% from global vocabulary (easy negatives)

    This teaches the model to distinguish Ball Games from Combat Sports,
    not just "sports words" from "chemistry words".
    """
    pos_set = set(pos_wids)
    n_neg_total = len(pos_wids) * NEG_RATIO

    # Hard negatives: words from sibling towns (excluding this town's words)
    hard_pool = list(sibling_wids - pos_set)
    n_hard = min(len(hard_pool), n_neg_total // 2)

    # Easy negatives: words from entire vocabulary (excluding island words)
    island_all = sibling_wids | pos_set
    easy_pool = list(all_wids_set - island_all)
    n_easy = min(len(easy_pool), n_neg_total - n_hard)

    hard_negs = rng.choice(hard_pool, size=n_hard, replace=False).tolist() if n_hard > 0 else []
    easy_negs = rng.choice(easy_pool, size=n_easy, replace=False).tolist() if n_easy > 0 else []

    return hard_negs + easy_negs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_town_model(X, y, groups, town_name, use_gpu=True):
    """Train XGBoost classifier for a single town."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    unique_groups = len(set(groups))
    n_splits = min(7, unique_groups)
    if n_splits < 2:
        print(f"    WARNING: Not enough groups ({unique_groups}) for CV")
        return None

    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(gkf.split(X, y, groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    if y_val.sum() == 0 or y_train.sum() == 0:
        print(f"    WARNING: Empty val/train split for {town_name}")
        return None

    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    if use_gpu:
        params["device"] = "cuda"

    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    metrics = {
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_val, y_pred),
        "auc_pr": average_precision_score(y_val, y_prob),
    }

    return {"clf": clf, "metrics": metrics, "n_pos": n_pos, "n_neg": n_neg}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def ensure_augmented_towns_table(v3_con):
    """Create augmented_towns table if it doesn't exist."""
    v3_con.execute("""
        CREATE TABLE IF NOT EXISTS AugmentedTowns (
            town_id     INTEGER NOT NULL REFERENCES Towns(town_id),
            word_id     INTEGER NOT NULL REFERENCES Words(word_id),
            score       REAL    NOT NULL,
            source      TEXT    NOT NULL,
            PRIMARY KEY (town_id, word_id)
        )
    """)
    v3_con.execute("""
        CREATE INDEX IF NOT EXISTS idx_at_town ON AugmentedTowns(town_id)
    """)
    v3_con.execute("""
        CREATE INDEX IF NOT EXISTS idx_at_word ON AugmentedTowns(word_id)
    """)
    v3_con.commit()


def save_predictions(v3_con, town_id, all_wids, probs, seed_wids, threshold,
                     bucket_wids=frozenset(), stop_wids=frozenset()):
    """Save XGBoost predictions above threshold, excluding seeds, bucket words, and stop words."""
    cur = v3_con.cursor()

    # Delete old xgboost predictions for this town
    cur.execute("DELETE FROM AugmentedTowns WHERE town_id = ? AND source = 'xgboost'",
                (town_id,))

    inserted = 0
    for i, wid in enumerate(all_wids):
        if (probs[i] >= threshold
                and wid not in seed_wids
                and wid not in bucket_wids
                and wid not in stop_wids):
            cur.execute(
                "INSERT OR IGNORE INTO AugmentedTowns (town_id, word_id, score, source) VALUES (?, ?, ?, 'xgboost')",
                (town_id, wid, float(probs[i]))
            )
            inserted += 1

    v3_con.commit()
    return inserted


# ---------------------------------------------------------------------------
# Post-filter: island-level word detection + capital starving
# ---------------------------------------------------------------------------

ISLAND_WORD_THRESHOLD = 0.8  # predicted by ≥80% of towns → island-level

def post_filter_predictions(v3_con, towns):
    """Two-pass filter on AugmentedTowns after all training completes.

    Pass 1 — Island-level words: words predicted by ≥80% of towns are
    non-discriminative (e.g. "athlete" in Sport). Promote to IslandWords
    table, remove from AugmentedTowns.

    Pass 2 — Starve capitals: for capital towns, remove any prediction
    where the word was also predicted by ≥1 sibling. Capital gets only
    words that no specific town claimed.
    """
    town_ids = [t["town_id"] for t in towns]
    n_towns = len(town_ids)
    capital_ids = {t["town_id"] for t in towns if t["is_capital"]}
    non_capital_ids = {t["town_id"] for t in towns if not t["is_capital"]}

    if n_towns < 2:
        return

    # Get island_id for IslandWords insertion
    island_name = towns[0]["island"]
    island_id = v3_con.execute(
        "SELECT island_id FROM Islands WHERE name = ?", (island_name,)
    ).fetchone()[0]

    print("\nPost-filter pipeline:")

    # Load all predictions for this island's towns
    placeholders = ",".join("?" * len(town_ids))
    rows = v3_con.execute(f"""
        SELECT word_id, town_id FROM AugmentedTowns
        WHERE town_id IN ({placeholders}) AND source = 'xgboost'
    """, town_ids).fetchall()

    # Build word → set of town_ids
    from collections import defaultdict
    word_towns = defaultdict(set)
    for wid, tid in rows:
        word_towns[wid].add(tid)

    # Pass 1: Island-level words (predicted by ≥80% of towns)
    min_towns_for_island = max(2, int(n_towns * ISLAND_WORD_THRESHOLD))
    island_words = {wid for wid, tids in word_towns.items()
                    if len(tids) >= min_towns_for_island}

    if island_words:
        cur = v3_con.cursor()
        island_word_list = list(island_words)

        # Promote to IslandWords (skip if already there from seed detection)
        word_texts = {}
        for wid in island_word_list:
            row = v3_con.execute(
                "SELECT word FROM Words WHERE word_id = ?", (wid,)
            ).fetchone()
            if row:
                word_texts[wid] = row[0]

        n_promoted = 0
        for wid in island_word_list:
            if wid not in word_texts:
                continue
            try:
                cur.execute("""
                    INSERT INTO IslandWords (island_id, word_id, word, source)
                    VALUES (?, ?, ?, 'xgboost_filter')
                """, (island_id, wid, word_texts[wid]))
                n_promoted += 1
            except Exception:
                pass  # already exists (from seed detection)

        # Remove from AugmentedTowns
        removed_island = 0
        for i in range(0, len(island_word_list), 500):
            batch = island_word_list[i:i+500]
            ph = ",".join("?" * len(batch))
            tph = ",".join("?" * len(town_ids))
            cur.execute(f"""
                DELETE FROM AugmentedTowns
                WHERE word_id IN ({ph})
                AND town_id IN ({tph})
                AND source = 'xgboost'
            """, batch + town_ids)
            removed_island += cur.rowcount
        v3_con.commit()
        print(f"  Pass 1 — Island-level words: {len(island_words):,} words in ≥{min_towns_for_island}/{n_towns} towns → {n_promoted:,} promoted to IslandWords, {removed_island:,} predictions removed")
    else:
        print(f"  Pass 1 — No island-level words found (threshold: ≥{min_towns_for_island}/{n_towns} towns)")

    # Rebuild word_towns after pass 1 (island words removed)
    rows = v3_con.execute(f"""
        SELECT word_id, town_id FROM AugmentedTowns
        WHERE town_id IN ({placeholders}) AND source = 'xgboost'
    """, town_ids).fetchall()

    word_towns = defaultdict(set)
    for wid, tid in rows:
        word_towns[wid].add(tid)

    # Pass 2: Starve capitals — remove predictions where any sibling also has the word
    if not capital_ids:
        print("  Pass 2 — No capital towns in this island")
        return

    capital_words_to_remove = []
    for wid, tids in word_towns.items():
        capital_tids = tids & capital_ids
        sibling_tids = tids & non_capital_ids
        if capital_tids and sibling_tids:
            # Word is in a capital AND at least one non-capital → remove from capital
            for cap_tid in capital_tids:
                capital_words_to_remove.append((cap_tid, wid))

    if capital_words_to_remove:
        cur = v3_con.cursor()
        for i in range(0, len(capital_words_to_remove), 500):
            batch = capital_words_to_remove[i:i+500]
            for tid, wid in batch:
                cur.execute(
                    "DELETE FROM AugmentedTowns WHERE town_id = ? AND word_id = ? AND source = 'xgboost'",
                    (tid, wid)
                )
        v3_con.commit()

    capital_names = [t["name"] for t in towns if t["is_capital"]]
    print(f"  Pass 2 — Starve capitals ({', '.join(capital_names)}): {len(capital_words_to_remove):,} predictions removed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sqlite3

    parser = argparse.ArgumentParser(description="V3 Town-Level XGBoost Training")
    parser.add_argument("--island", required=True, help="Island name to train")
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD,
                        help=f"Min probability for predictions (default: {SCORE_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (no CUDA)")
    args = parser.parse_args()

    v3_con = sqlite3.connect(V3_DB)
    v3_con.execute("PRAGMA foreign_keys = ON")

    # Check if island is a bucket (non-topical) — skip training
    bucket_check = v3_con.execute(
        "SELECT is_bucket FROM Islands WHERE name = ?", (args.island,)
    ).fetchone()
    if bucket_check and bucket_check[0]:
        print(f"Island '{args.island}' is marked as bucket (is_bucket=1) — skipping XGBoost training")
        v3_con.close()
        return

    # Load stop words
    stop_wids = load_stop_wids(v3_con)
    if stop_wids:
        print(f"Stop words: {len(stop_wids):,} (excluded from seeds + predictions)")

    # Load island towns
    towns = load_island_towns(v3_con, args.island, stop_wids)
    if not towns:
        print(f"No towns found for island '{args.island}'")
        return

    print(f"Island: {towns[0]['archipelago']} > {args.island}")
    print(f"Towns: {len(towns)}")
    for t in towns:
        print(f"  {t['name']}: {len(t['seed_wids'])} seeds with embeddings")

    if args.dry_run:
        return

    # Load global data
    global_data = load_all_data(v3_con)
    all_wids = global_data["all_wids"]
    all_wids_set = set(all_wids)
    X_global = global_data["X_global"]
    raw_embeddings = global_data["raw_embeddings"]
    emb_normalized = global_data["emb_normalized"]
    wid_to_row = global_data["wid_to_row"]

    # Build sibling word sets (all seeds from other towns in island)
    all_island_wids = set()
    for t in towns:
        all_island_wids |= t["seed_wids"]

    # Load bucket word_ids — excluded from predictions for non-bucket islands
    bucket_wids = set(r[0] for r in v3_con.execute("""
        SELECT DISTINCT sw.word_id FROM SeedWords sw
        JOIN Towns t USING (town_id)
        JOIN Islands i USING (island_id)
        WHERE i.is_bucket = 1 AND sw.word_id IS NOT NULL
    """).fetchall())
    if bucket_wids:
        print(f"  Excluding {len(bucket_wids):,} bucket island word_ids from predictions")

    # Ensure output table and model dir
    ensure_augmented_towns_table(v3_con)
    island_model_dir = os.path.join(MODEL_DIR, args.island.replace(" ", "_"))
    os.makedirs(island_model_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    total_predictions = 0

    print(f"\nTraining {len(towns)} town classifiers...")
    print(f"Threshold: {args.threshold}")
    print()

    for t in towns:
        town_name = t["name"]
        town_id = t["town_id"]
        pos_wids = [wid for wid in t["seed_wids"] if wid in wid_to_row]

        if len(pos_wids) < 10:
            print(f"  {town_name}: SKIP — only {len(pos_wids)} seeds with embeddings")
            continue

        t0 = time.time()

        # Sibling words (hard negatives)
        sibling_wids = all_island_wids - t["seed_wids"]

        # Negative sampling
        neg_wids = build_negatives(pos_wids, sibling_wids, all_wids_set, rng)

        # Build dataset
        train_wids = pos_wids + neg_wids
        y = np.array([1] * len(pos_wids) + [0] * len(neg_wids))

        # Build feature matrix: global + town-specific
        town_features = compute_town_features(
            pos_wids, raw_embeddings, emb_normalized, wid_to_row
        )

        train_rows = [wid_to_row[wid] for wid in train_wids]
        X_train_global = X_global[train_rows]
        X_train_town = town_features[train_rows]
        X = np.hstack([X_train_global, X_train_town])

        # Groups for stratified split (use word_id as group)
        groups = np.array(train_wids)

        # Train
        result = train_town_model(X, y, groups, town_name, use_gpu=not args.cpu)

        if result is None:
            print(f"  {town_name}: FAILED")
            continue

        clf = result["clf"]
        m = result["metrics"]

        # Save model
        model_path = os.path.join(island_model_dir, f"{town_name.replace(' ', '_').replace('&', 'and')}.json")
        clf.save_model(model_path)

        # Predict on ALL words
        X_all = np.hstack([X_global, town_features])
        probs = clf.predict_proba(X_all)[:, 1]

        # Save predictions
        n_pred = save_predictions(v3_con, town_id, all_wids, probs, t["seed_wids"], args.threshold,
                                  bucket_wids, stop_wids)
        total_predictions += n_pred

        # Record F1 in Towns table
        v3_con.execute("UPDATE Towns SET model_f1 = ? WHERE town_id = ?",
                       (m["f1"], town_id))
        v3_con.commit()

        elapsed = time.time() - t0
        print(f"  {town_name}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"AUC-PR={m['auc_pr']:.3f} | {result['n_pos']}+/{result['n_neg']}- | "
              f"{n_pred:,} predictions ({elapsed:.1f}s)")

    print(f"\nDone. Total raw predictions: {total_predictions:,}")

    # Post-filter: island-level words + capital starving
    post_filter_predictions(v3_con, towns)

    # Summary
    print("\nFinal prediction counts per town:")
    for t in towns:
        count = v3_con.execute(
            "SELECT COUNT(*) FROM AugmentedTowns WHERE town_id = ?",
            (t["town_id"],)
        ).fetchone()[0]
        seeds = len(t["seed_wids"])
        cap_marker = " [CAPITAL]" if t["is_capital"] else ""
        print(f"  {t['name']}: {seeds} seeds + {count} xgboost = {seeds + count} total{cap_marker}")

    v3_con.close()


if __name__ == "__main__":
    main()
