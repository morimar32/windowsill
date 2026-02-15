"""Lightweight scoring harness for meta-relationship testing.

Queries the real DB directly to compute: tokenize → word lookup → BM25 →
primary reef activation → single-hop edge propagation → final ranking.
"""

import math

import config
from word_list import fnv1a_u64


class MetaRelScorer:
    def __init__(self, con):
        self.con = con

        # Word text → word_id
        rows = con.execute("SELECT word, word_id FROM words").fetchall()
        self.word_index = {r[0]: r[1] for r in rows}

        # Variant hash → word_id (prefer highest-specificity word on collision)
        rows = con.execute("""
            SELECT wv.variant_hash, wv.word_id
            FROM word_variants wv
            JOIN words w ON wv.word_id = w.word_id
            ORDER BY w.specificity DESC
        """).fetchall()
        self.variant_index = {}
        for vh, wid in rows:
            if vh not in self.variant_index:
                self.variant_index[vh] = wid

        # Word IDF: word_id → reef_idf
        rows = con.execute(
            "SELECT word_id, reef_idf FROM words WHERE reef_idf IS NOT NULL"
        ).fetchall()
        self.word_idf = {r[0]: r[1] for r in rows}

        # Word-reef affinity: word_id → list[(reef_id, n_dims)]
        rows = con.execute(
            "SELECT word_id, reef_id, n_dims FROM word_reef_affinity"
        ).fetchall()
        self.word_affinity = {}
        for wid, rid, nd in rows:
            self.word_affinity.setdefault(wid, []).append((rid, nd))

        # Reef metadata: reef_id → {n_dims, n_words, name}
        rows = con.execute("""
            SELECT island_id, n_dims, n_words, island_name
            FROM island_stats WHERE generation = 2
        """).fetchall()
        self.reef_meta = {}
        for rid, nd, nw, name in rows:
            self.reef_meta[rid] = {
                "n_dims": nd,
                "n_words": nw,
                "name": name or f"reef_{rid}",
            }

        # Average reef words
        total_words = sum(m["n_words"] for m in self.reef_meta.values())
        self.avg_reef_words = total_words / len(self.reef_meta) if self.reef_meta else 1.0

        # Edge weights: (source, target) → weight
        rows = con.execute(
            "SELECT source_reef_id, target_reef_id, weight FROM reef_edges"
        ).fetchall()
        self.edge_weights = {(r[0], r[1]): r[2] for r in rows}

    def lookup_words(self, sentence: str) -> list[int]:
        """Tokenize (split + lowercase), look up in words table, fallback to word_variants."""
        tokens = sentence.lower().split()
        word_ids = []
        seen = set()
        for token in tokens:
            # Direct lookup
            wid = self.word_index.get(token)
            if wid is None:
                # Fallback: hash and look up variant
                h = fnv1a_u64(token)
                wid = self.variant_index.get(h)
            if wid is not None and wid not in seen:
                word_ids.append(wid)
                seen.add(wid)
        return word_ids

    def primary_reef_scores(self, word_ids: list[int]) -> dict[int, float]:
        """Compute BM25-based primary reef scores from word activations."""
        k1 = config.BM25_K1
        b = config.BM25_B
        scores = {}

        for wid in word_ids:
            idf = self.word_idf.get(wid, 0.0)
            if idf == 0.0:
                continue
            affinities = self.word_affinity.get(wid, [])
            for reef_id, n_dims in affinities:
                meta = self.reef_meta.get(reef_id)
                if meta is None:
                    continue
                reef_total_dims = meta["n_dims"]
                reef_n_words = meta["n_words"]
                if reef_total_dims == 0:
                    continue
                tf = n_dims / reef_total_dims
                norm_len = reef_n_words / self.avg_reef_words
                bm25 = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * norm_len))
                scores[reef_id] = scores.get(reef_id, 0.0) + bm25

        return scores

    def propagate(self, primary: dict[int, float]) -> dict[int, float]:
        """Single-hop additive propagation through edge weights."""
        propagated = {}
        for source_id, source_score in primary.items():
            if source_score <= 0:
                continue
            for (src, tgt), w in self.edge_weights.items():
                if src == source_id and w > 0:
                    propagated[tgt] = propagated.get(tgt, 0.0) + source_score * w
        return propagated

    def score(self, sentence: str) -> list[tuple[int, float, str]]:
        """Full pipeline: tokenize → lookup → BM25 → propagate → merge → rank."""
        word_ids = self.lookup_words(sentence)
        if not word_ids:
            return []

        primary = self.primary_reef_scores(word_ids)
        propagated = self.propagate(primary)

        # Merge: final = primary + propagated
        final = dict(primary)
        for reef_id, boost in propagated.items():
            final[reef_id] = final.get(reef_id, 0.0) + boost

        # Rank descending
        ranked = []
        for reef_id, sc in final.items():
            name = self.reef_meta.get(reef_id, {}).get("name", f"reef_{reef_id}")
            ranked.append((reef_id, sc, name))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def get_edge_weight(self, source: int, target: int) -> float:
        """Direct lookup of pre-computed weight from reef_edges."""
        return self.edge_weights.get((source, target), 0.0)
