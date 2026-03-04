"""Hyperparameter sweep for export weight tuning.

Replays the scoring chain from populate_exports.py entirely in memory,
sweeping over parameter combinations and evaluating against the test
battery.  Results are written to a JSONL log for offline analysis.

Usage:
    python v3/sweep.py --samples 5              # quick smoke test
    python v3/sweep.py --phase coarse --samples 500   # coarse sweep (~8 min)
    python v3/sweep.py --phase fine --around sweep_coarse.jsonl --top 10
    python v3/sweep.py --full                   # exhaustive (hours)
"""

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_v3_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project)
sys.path.insert(0, _v3_dir)

from test_battery import QUERIES

V3_DB = os.path.join(_v3_dir, "windowsill.db")
EMBEDDING_DIM = 768
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_PREFIX = "clustering: "


def unpack_embedding(blob):
    return np.frombuffer(blob, dtype=np.float32).copy()


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Parameter space ────────────────────────────────────────────────

@dataclass(frozen=True)
class Params:
    group_name_cos_alpha: float = 0.20
    island_name_cos_alpha: float = 0.30
    island_only_alpha: float = 0.50
    source_quality_floor: float = 0.90
    island_global_blend: float = 0.80
    exclusivity_exp: float = 0.33
    norm_fn: str = "minmax"


PARAM_GRID = {
    "group_name_cos_alpha": [0.10, 0.15, 0.20, 0.25, 0.30],
    "island_name_cos_alpha": [0.15, 0.20, 0.25, 0.30, 0.35],
    "island_only_alpha": [0.30, 0.40, 0.50, 0.60],
    "source_quality_floor": [0.70, 0.80, 0.90, 1.00],
    "island_global_blend": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "exclusivity_exp": [0.25, 0.33, 0.50],
    "norm_fn": ["minmax", "log_minmax", "trimmed_minmax", "zscore"],
}


def generate_grid():
    """Generate all valid parameter combinations."""
    combos = []
    keys = list(PARAM_GRID.keys())
    for vals in itertools.product(*PARAM_GRID.values()):
        d = dict(zip(keys, vals))
        if d["group_name_cos_alpha"] + d["island_name_cos_alpha"] > 0.50 + 1e-9:
            continue
        combos.append(Params(**d))
    return combos


def random_sample(combos, n):
    if n >= len(combos):
        return combos
    return random.sample(combos, n)


def fine_grid(base_params_list, grid=PARAM_GRID):
    """Dense grid around known-good parameter sets, +/-1 step in each dimension."""
    combos = set()
    for base in base_params_list:
        base_d = asdict(base)
        neighbors = {}
        for key, values in grid.items():
            current = base_d[key]
            if key == "norm_fn":
                neighbors[key] = values
            else:
                try:
                    idx = values.index(current)
                except ValueError:
                    neighbors[key] = values
                    continue
                lo = max(0, idx - 1)
                hi = min(len(values), idx + 2)
                neighbors[key] = values[lo:hi]

        nkeys = list(neighbors.keys())
        for vals in itertools.product(*neighbors.values()):
            d = dict(zip(nkeys, vals))
            if d["group_name_cos_alpha"] + d["island_name_cos_alpha"] > 0.50 + 1e-9:
                continue
            combos.add(tuple(sorted(d.items())))

    return [Params(**dict(c)) for c in combos]


# ── Normalization functions ────────────────────────────────────────

def norm_minmax(scores):
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi > lo:
        return [255.0 * (s - lo) / (hi - lo) for s in scores]
    return [128.0] * len(scores)


def norm_log_minmax(scores):
    if not scores:
        return []
    log_scores = [math.log1p(s) for s in scores]
    return norm_minmax(log_scores)


def norm_trimmed_minmax(scores):
    if not scores:
        return []
    arr = sorted(scores)
    n = len(arr)
    p2 = arr[max(0, int(n * 0.02))]
    p98 = arr[min(n - 1, int(n * 0.98))]
    clipped = [max(p2, min(p98, s)) for s in scores]
    return norm_minmax(clipped)


def norm_zscore(scores):
    if not scores:
        return []
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    if std == 0:
        return [128.0] * len(scores)
    zs = [(s - mean) / std for s in scores]
    clamped = [max(-3.0, min(3.0, z)) for z in zs]
    return [255.0 * (z + 3.0) / 6.0 for z in clamped]


NORM_FUNCTIONS = {
    "minmax": norm_minmax,
    "log_minmax": norm_log_minmax,
    "trimmed_minmax": norm_trimmed_minmax,
    "zscore": norm_zscore,
}


# ── Data loading ───────────────────────────────────────────────────

class SweepData:
    """Holds all preloaded data and precomputed cosine components."""

    def __init__(self):
        self.reef_to_town = {}
        self.town_to_island = {}
        self.island_names = {}
        self.total_islands = 0

        self.word_to_id = {}
        self.word_stats = {}
        self.word_island_reefs = defaultdict(dict)
        self.global_idf = {}
        self.word_level = {}

        # Precomputed per-entry components (cosines baked in, params not yet applied)
        # Reef: (reef_id, word_id, centroid_sim, reef_name_cos, island_name_cos, idf, sq)
        self.reef_components = []
        # Town: (town_id, word_id, centroid_sim, town_name_cos, island_name_cos, idf, sq)
        self.town_components = []
        # Island: (island_id, word_id, centroid_sim, island_name_cos, iidf, sq, n_islands)
        self.island_components = []

    def load_from_db(self):
        import sqlite3
        from sentence_transformers import SentenceTransformer

        t_start = time.time()
        con = sqlite3.connect(V3_DB)
        con.execute("PRAGMA foreign_keys = ON")

        # ── Hierarchy ──────────────────────────────────────────
        print("  Loading hierarchy...")
        islands = con.execute("SELECT island_id, name FROM Islands").fetchall()
        self.island_names = {iid: name for iid, name in islands}
        self.total_islands = len(islands)

        for town_id, island_id in con.execute(
            "SELECT town_id, island_id FROM Towns"
        ).fetchall():
            self.town_to_island[town_id] = island_id

        for reef_id, town_id in con.execute(
            "SELECT reef_id, town_id FROM Reefs"
        ).fetchall():
            self.reef_to_town[reef_id] = town_id

        # ── Word text -> id ────────────────────────────────────
        print("  Loading word->id mapping...")
        for word_id, word in con.execute("SELECT word_id, word FROM Words").fetchall():
            self.word_to_id[word] = word_id

        # ── Word stats + classification ────────────────────────
        print("  Loading word stats...")
        for row in con.execute("""
            SELECT rw.word_id, w.specificity,
                COUNT(DISTINCT t.island_id) AS n_islands,
                COUNT(DISTINCT r.town_id)   AS n_towns,
                COUNT(DISTINCT rw.reef_id)  AS n_reefs
            FROM ReefWords rw
            JOIN Reefs r USING (reef_id)
            JOIN Towns t USING (town_id)
            JOIN Words w USING (word_id)
            GROUP BY rw.word_id
        """).fetchall():
            self.word_stats[row[0]] = {
                "specificity": row[1],
                "n_islands": row[2],
                "n_towns": row[3],
                "n_reefs": row[4],
            }

        for word_id, island_id, cnt in con.execute("""
            SELECT rw.word_id, t.island_id,
                COUNT(DISTINCT rw.reef_id) AS reefs_in_island
            FROM ReefWords rw
            JOIN Reefs r USING (reef_id)
            JOIN Towns t USING (town_id)
            GROUP BY rw.word_id, t.island_id
        """).fetchall():
            self.word_island_reefs[word_id][island_id] = cnt

        for wid, idf in con.execute(
            "SELECT word_id, idf FROM Words WHERE idf IS NOT NULL"
        ).fetchall():
            self.global_idf[wid] = idf

        self._classify_words()

        # ── Word embeddings ────────────────────────────────────
        print("  Loading word embeddings...")
        t0 = time.time()
        word_embs = {}
        for word_id, blob in con.execute(
            "SELECT word_id, embedding FROM Words WHERE embedding IS NOT NULL"
        ).fetchall():
            word_embs[word_id] = unpack_embedding(blob)
        print(f"    {len(word_embs):,} embeddings ({time.time()-t0:.1f}s)")

        # ── Embed hierarchy names ──────────────────────────────
        print("  Embedding hierarchy names...")
        t0 = time.time()
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

        reefs = con.execute("""
            SELECT r.reef_id, r.name, t.name
            FROM Reefs r JOIN Towns t USING (town_id)
        """).fetchall()
        reef_texts = [
            f"{EMBEDDING_PREFIX}{name if name else town_name}"
            for _, name, town_name in reefs
        ]
        reef_embs_arr = model.encode(reef_texts, show_progress_bar=False, batch_size=256)
        reef_name_embs = {reefs[i][0]: reef_embs_arr[i] for i in range(len(reefs))}

        towns = con.execute("SELECT town_id, name FROM Towns").fetchall()
        town_texts = [f"{EMBEDDING_PREFIX}{name}" for _, name in towns]
        town_embs_arr = model.encode(town_texts, show_progress_bar=False, batch_size=256)
        town_name_embs = {towns[i][0]: town_embs_arr[i] for i in range(len(towns))}

        island_texts = [f"{EMBEDDING_PREFIX}{name}" for _, name in islands]
        island_embs_arr = model.encode(island_texts, show_progress_bar=False, batch_size=256)
        island_name_embs = {islands[i][0]: island_embs_arr[i] for i in range(len(islands))}

        print(f"    {len(reef_name_embs)} reefs, {len(town_name_embs)} towns, "
              f"{len(island_name_embs)} islands ({time.time()-t0:.1f}s)")

        # ── Precompute cosine components ───────────────────────
        print("  Precomputing cosine components...")
        t0 = time.time()
        zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Reef components
        reef_word_rows = con.execute("""
            SELECT rw.reef_id, rw.word_id, rw.cosine_sim,
                   rw.source_quality, r.town_id
            FROM ReefWords rw
            JOIN Reefs r USING (reef_id)
            WHERE rw.cosine_sim IS NOT NULL
        """).fetchall()

        for reef_id, word_id, centroid_sim, sq, town_id in reef_word_rows:
            if self.word_level.get(word_id) != "reef":
                continue
            if word_id not in word_embs:
                continue
            island_id = self.town_to_island.get(town_id)
            reef_ncos = cosine_sim(word_embs[word_id], reef_name_embs.get(reef_id, zero))
            island_ncos = cosine_sim(word_embs[word_id], island_name_embs.get(island_id, zero))
            self.reef_components.append((
                reef_id, word_id, centroid_sim, reef_ncos, island_ncos,
                self.global_idf.get(word_id, 0.0),
                sq if sq is not None else 1.0,
            ))

        # Town components (aggregate MAX csim and MAX sq per town x word)
        town_word_rows = con.execute("""
            SELECT r.town_id, rw.word_id, rw.cosine_sim, rw.source_quality
            FROM ReefWords rw
            JOIN Reefs r USING (reef_id)
            WHERE rw.cosine_sim IS NOT NULL
        """).fetchall()

        town_word_agg = defaultdict(lambda: {"max_csim": -1.0, "max_sq": 0.0})
        for town_id, word_id, csim, sq in town_word_rows:
            if self.word_level.get(word_id) != "town":
                continue
            agg = town_word_agg[(town_id, word_id)]
            if csim is not None and csim > agg["max_csim"]:
                agg["max_csim"] = csim
            if sq is not None and sq > agg["max_sq"]:
                agg["max_sq"] = sq

        for (town_id, word_id), agg in town_word_agg.items():
            if word_id not in word_embs:
                continue
            island_id = self.town_to_island.get(town_id)
            town_ncos = cosine_sim(word_embs[word_id], town_name_embs.get(town_id, zero))
            island_ncos = cosine_sim(word_embs[word_id], island_name_embs.get(island_id, zero))
            self.town_components.append((
                town_id, word_id, agg["max_csim"], town_ncos, island_ncos,
                self.global_idf.get(word_id, 0.0), agg["max_sq"],
            ))

        # Island components (aggregate across all reefs per island x word)
        island_word_agg = defaultdict(lambda: {"max_csim": -1.0, "max_sq": 0.0})
        for town_id, word_id, csim, sq in town_word_rows:
            if self.word_level.get(word_id) != "island":
                continue
            island_id = self.town_to_island.get(town_id)
            agg = island_word_agg[(island_id, word_id)]
            if csim is not None and csim > agg["max_csim"]:
                agg["max_csim"] = csim
            if sq is not None and sq > agg["max_sq"]:
                agg["max_sq"] = sq

        for (island_id, word_id), agg in island_word_agg.items():
            if word_id not in word_embs:
                continue
            n_isl = self.word_stats.get(word_id, {}).get("n_islands", 1)
            island_ncos = cosine_sim(word_embs[word_id], island_name_embs.get(island_id, zero))
            iidf = math.log2(self.total_islands / max(1, n_isl))
            self.island_components.append((
                island_id, word_id, agg["max_csim"], island_ncos,
                iidf, agg["max_sq"], n_isl,
            ))

        con.close()

        print(f"    Reef: {len(self.reef_components):,}, "
              f"Town: {len(self.town_components):,}, "
              f"Island: {len(self.island_components):,} "
              f"({time.time()-t0:.1f}s)")
        print(f"  Total load time: {time.time()-t_start:.1f}s")

    def _classify_words(self):
        """Classify each word into reef/town/island export level (param-independent)."""
        for word_id, stats in self.word_stats.items():
            spec = stats["specificity"]
            n_towns = stats["n_towns"]

            if spec is None:
                self.word_level[word_id] = "island"
            elif spec >= 1 and n_towns >= 2:
                self.word_level[word_id] = "town"
            elif spec >= 0 and n_towns == 1:
                self.word_level[word_id] = "reef"
            elif spec == 0 and n_towns >= 2:
                self.word_level[word_id] = "island"
            elif spec == -1:
                island_reef_counts = self.word_island_reefs.get(word_id, {})
                max_reefs = max(island_reef_counts.values()) if island_reef_counts else 0
                if max_reefs == 1:
                    self.word_level[word_id] = "reef"
                else:
                    self.word_level[word_id] = "island"
            else:
                self.word_level[word_id] = "island"


# ── Per-iteration scoring ──────────────────────────────────────────

def compute_exports(data, params):
    """Compute export weights, return word_id -> {island_id -> max_weight}."""
    sq_floor = params.source_quality_floor
    a1 = params.group_name_cos_alpha
    a2 = params.island_name_cos_alpha
    a_island = params.island_only_alpha
    blend = params.island_global_blend
    exc_exp = params.exclusivity_exp
    norm_fn = NORM_FUNCTIONS[params.norm_fn]

    word_island_weights = defaultdict(lambda: defaultdict(int))

    # ── Reef exports (per-reef minmax, no exclusivity) ─────
    reef_groups = defaultdict(list)
    reef_word_island = {}

    for reef_id, word_id, csim, reef_ncos, island_ncos, idf, sq_raw in data.reef_components:
        sq = max(sq_raw, sq_floor)
        esim = (1 - a1 - a2) * csim + a1 * reef_ncos + a2 * island_ncos
        raw = idf * sq * esim
        reef_groups[reef_id].append((word_id, raw))
        town_id = data.reef_to_town.get(reef_id)
        reef_word_island[(reef_id, word_id)] = data.town_to_island.get(town_id)

    for reef_id, entries in reef_groups.items():
        raws = [e[1] for e in entries]
        weights = norm_minmax(raws)
        for (word_id, _), w in zip(entries, weights):
            island_id = reef_word_island[(reef_id, word_id)]
            if island_id is not None:
                w_int = max(0, min(255, round(w)))
                cur = word_island_weights[word_id]
                if w_int > cur[island_id]:
                    cur[island_id] = w_int

    # ── Town exports (per-town minmax, no exclusivity) ─────
    town_groups = defaultdict(list)

    for town_id, word_id, csim, town_ncos, island_ncos, idf, sq_raw in data.town_components:
        sq = max(sq_raw, sq_floor)
        esim = (1 - a1 - a2) * csim + a1 * town_ncos + a2 * island_ncos
        raw = idf * sq * esim
        town_groups[town_id].append((word_id, raw))

    for town_id, entries in town_groups.items():
        raws = [e[1] for e in entries]
        weights = norm_minmax(raws)
        island_id = data.town_to_island.get(town_id)
        if island_id is None:
            continue
        for (word_id, _), w in zip(entries, weights):
            w_int = max(0, min(255, round(w)))
            cur = word_island_weights[word_id]
            if w_int > cur[island_id]:
                cur[island_id] = w_int

    # ── Island exports (hybrid norm + exclusivity) ─────────
    island_groups = defaultdict(list)

    for island_id, word_id, csim, island_ncos, iidf, sq_raw, n_isl in data.island_components:
        sq = max(sq_raw, sq_floor)
        esim = (1 - a_island) * csim + a_island * island_ncos
        raw = iidf * sq * esim
        island_groups[island_id].append((word_id, raw, n_isl))

    # Flatten for global normalization
    all_entries = []
    for island_id, entries in island_groups.items():
        for word_id, raw, n_isl in entries:
            all_entries.append((island_id, word_id, raw, n_isl))

    if all_entries:
        all_raws = [e[2] for e in all_entries]

        # Global normalization
        global_normed = norm_fn(all_raws)

        # Per-island normalization
        island_indices = defaultdict(list)
        for i, (island_id, _, _, _) in enumerate(all_entries):
            island_indices[island_id].append(i)

        local_normed = [0.0] * len(all_entries)
        for island_id, indices in island_indices.items():
            local_raws = [all_raws[i] for i in indices]
            local_weights = norm_fn(local_raws)
            for idx, w in zip(indices, local_weights):
                local_normed[idx] = w

        # Blend + exclusivity
        for i, (island_id, word_id, _, n_isl) in enumerate(all_entries):
            w = (1 - blend) * local_normed[i] + blend * global_normed[i]
            exc = 1.0 / (max(1, n_isl) ** exc_exp)
            w_int = max(0, min(255, round(w * exc)))
            cur = word_island_weights[word_id]
            if w_int > cur[island_id]:
                cur[island_id] = w_int

    return word_island_weights


def score_tests(word_island_weights, data):
    """Score test queries. Returns (pass_count, total, failures, margins)."""
    pass_count = 0
    failures = []
    margins = {}

    for q in QUERIES:
        island_scores = defaultdict(int)
        for word in q.words:
            word_id = data.word_to_id.get(word)
            if word_id is None:
                continue
            for island_id, weight in word_island_weights.get(word_id, {}).items():
                island_scores[island_id] += weight

        if not island_scores:
            failures.append(q.name)
            margins[q.name] = 0.0
            continue

        ranked = sorted(island_scores.items(), key=lambda x: -x[1])
        winner_id, w1 = ranked[0]
        winner_name = data.island_names.get(winner_id, "?")
        w2 = ranked[1][1] if len(ranked) > 1 else 0
        margin = w1 / w2 if w2 > 0 else 99.0

        if winner_name == q.expected_island:
            pass_count += 1
        else:
            failures.append(q.name)
        margins[q.name] = margin

    return pass_count, len(QUERIES), failures, margins


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for export weights")
    parser.add_argument("--phase", choices=["coarse", "fine"], default="coarse",
                        help="Sweep phase (default: coarse)")
    parser.add_argument("--full", action="store_true",
                        help="Run full exhaustive grid")
    parser.add_argument("--samples", type=int, default=500,
                        help="Random samples for coarse phase (default: 500)")
    parser.add_argument("--around", type=str,
                        help="JSONL from previous sweep (for fine phase)")
    parser.add_argument("--top", type=int, default=10,
                        help="Top-N results to expand for fine phase (default: 10)")
    parser.add_argument("-o", "--output", type=str,
                        help="Output JSONL file (default: sweep_{phase}.jsonl)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.output:
        phase = "full" if args.full else args.phase
        args.output = f"sweep_{phase}.jsonl"

    # ── Load data ──────────────────────────────────────────
    print("Loading data...")
    data = SweepData()
    data.load_from_db()

    # ── Generate parameter combos ──────────────────────────
    all_combos = generate_grid()
    print(f"Full grid: {len(all_combos)} combos")

    if args.full:
        combos = all_combos
        print(f"Running full sweep: {len(combos)} combos")
    elif args.phase == "fine":
        if not args.around:
            print("ERROR: --around required for fine phase")
            sys.exit(1)
        prev_results = []
        with open(args.around) as f:
            for line in f:
                prev_results.append(json.loads(line))
        prev_results.sort(key=lambda r: (-r["pass_count"], -r["mean_margin"]))
        top_params = [Params(**r["params"]) for r in prev_results[:args.top]]
        combos = fine_grid(top_params)
        print(f"Fine sweep around top {args.top}: {len(combos)} combos")
    else:
        combos = random_sample(all_combos, args.samples)
        print(f"Coarse sweep: {len(combos)} random samples")

    if not combos:
        print("No parameter combinations to sweep.")
        sys.exit(0)

    # ── Sweep ──────────────────────────────────────────────
    print(f"\nSweeping {len(combos)} combos -> {args.output}")

    best_pass = 0
    best_margin = 0.0
    best_id = 0
    best_params = combos[0]
    t_sweep = time.time()

    with open(args.output, "w") as f:
        for i, params in enumerate(combos, 1):
            t0 = time.time()
            weights = compute_exports(data, params)
            pass_count, total, failures, margins = score_tests(weights, data)
            elapsed = time.time() - t0

            passing = {k: v for k, v in margins.items() if k not in failures}
            min_margin = min(passing.values()) if passing else 0.0
            mean_margin = (sum(passing.values()) / len(passing)) if passing else 0.0

            result = {
                "id": i,
                "params": asdict(params),
                "pass_count": pass_count,
                "total": total,
                "pct": round(100 * pass_count / total, 1),
                "failures": failures,
                "margins": {k: round(v, 3) for k, v in margins.items()},
                "min_margin": round(min_margin, 3),
                "mean_margin": round(mean_margin, 3),
                "elapsed_s": round(elapsed, 2),
            }
            f.write(json.dumps(result) + "\n")
            f.flush()

            if (pass_count > best_pass or
                    (pass_count == best_pass and mean_margin > best_margin)):
                best_pass = pass_count
                best_margin = mean_margin
                best_id = i
                best_params = params

            if i % 50 == 0 or i == len(combos):
                elapsed_total = time.time() - t_sweep
                rate = i / elapsed_total
                eta = (len(combos) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(combos)}] best={best_pass}/{total} "
                      f"({rate:.1f} iter/s, ETA {eta/60:.0f}m)")

    print(f"\nDone. {len(combos)} combos in {time.time()-t_sweep:.0f}s")
    print(f"Best: {best_pass}/{total} pass (id={best_id})")
    print(f"  {asdict(best_params)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
