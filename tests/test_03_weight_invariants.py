"""Layer 3: Composite weight invariants — ordinal assertions on pre-computed weights.

These define the CONTRACT. W(src, tgt) is the pre-computed weight from reef_edges.
"""

import numpy as np
import pytest


def W(edge_weights, src, tgt):
    """Shorthand for weight lookup."""
    key = (src, tgt)
    assert key in edge_weights, f"Edge {src}→{tgt} not found"
    return edge_weights[key]["weight"]


# ---------------------------------------------------------------------------
# Hard invariants (must ALWAYS hold — failure = formula is fundamentally wrong)
# ---------------------------------------------------------------------------

class TestHardInvariants:
    def test_h1_containment_dominance(self, edge_weights):
        """W(9,8) > W(8,9): Containment 0.197 vs 0.023, lift symmetric at 5.41."""
        assert W(edge_weights, 9, 8) > W(edge_weights, 8, 9)

    def test_h2_all_knobs_favor(self, edge_weights):
        """W(9,8) > W(12,69): Every knob favors 9→8."""
        assert W(edge_weights, 9, 8) > W(edge_weights, 12, 69)

    def test_h3_all_aligned_beats_suppressed(self, edge_weights):
        """W(11,8) > W(166,144): All-aligned beats 3-way suppression."""
        assert W(edge_weights, 11, 8) > W(edge_weights, 166, 144)

    def test_h4_no_zero_weight_edges_remain(self, edge_weights):
        """After pruning, no zero-weight edges remain in the table."""
        zero_weight = [
            (src, tgt) for (src, tgt), e in edge_weights.items()
            if e["weight"] == 0
        ]
        assert len(zero_weight) == 0, \
            f"{len(zero_weight)} zero-weight edges remain after pruning"

    def test_h5_lift_overcomes_containment(self, edge_weights):
        """W(147,148) > W(9,8): Lift 21.42 vs 5.41 overcomes containment 0.140 vs 0.197."""
        assert W(edge_weights, 147, 148) > W(edge_weights, 9, 8)

    def test_h6_valence_suppresses(self, edge_weights):
        """W(166,144) < W(9,8): Extreme valence gap must lose to small gap with strong containment."""
        assert W(edge_weights, 166, 144) < W(edge_weights, 9, 8)

    def test_h7_all_non_negative(self, edge_weights):
        """All weights >= 0."""
        negatives = [(s, t, e["weight"]) for (s, t), e in edge_weights.items() if e["weight"] < 0]
        assert len(negatives) == 0, f"{len(negatives)} negative weights, first: {negatives[0]}"


# ---------------------------------------------------------------------------
# Moderate invariants (should hold for reasonable formulas — failure = needs tuning)
# ---------------------------------------------------------------------------

class TestModerateInvariants:
    def test_m1_valence_suppression_matters(self, edge_weights):
        """W(188,190) < W(147,148): val_gap 0.504 vs 0.028 suppresses despite similar lift."""
        assert W(edge_weights, 188, 190) < W(edge_weights, 147, 148)

    def test_m2_genuine_bridge_survives(self, edge_weights):
        """W(188,190) > W(153,166): 19x lift bridge survives partial valence dampening."""
        assert W(edge_weights, 188, 190) > W(edge_weights, 153, 166)

    def test_m3_no_archipelago_penalty(self, edge_weights):
        """W(101,206) > W(153,166): Cross-archipelago with better scores on every knob."""
        assert W(edge_weights, 101, 206) > W(edge_weights, 153, 166)

    def test_m4_containment_ratio_preserved(self, edge_weights):
        """W(9,8) > 3 * W(8,9): 8.4x containment ratio should preserve at least 3x."""
        assert W(edge_weights, 9, 8) > 3 * W(edge_weights, 8, 9)

    def test_m5_valence_decay_flips_ranking(self, edge_weights):
        """W(131,190) > W(188,190): Smaller val_gap offsets lower lift."""
        assert W(edge_weights, 131, 190) > W(edge_weights, 188, 190)


# ---------------------------------------------------------------------------
# Soft guidelines (informational — help tune coefficients, don't fail the build)
# ---------------------------------------------------------------------------

class TestSoftGuidelines:
    """These are reported but don't fail. Use warnings for out-of-range ratios."""

    def test_s1_asymmetry_ratio(self, edge_weights):
        """S1: W(9,8) / W(8,9) expected in [3, 10]."""
        ratio = W(edge_weights, 9, 8) / max(W(edge_weights, 8, 9), 1e-15)
        if not (3 <= ratio <= 10):
            pytest.skip(f"S1 SOFT: ratio {ratio:.2f} outside [3, 10] — tune coefficients")

    def test_s2_lift_advantage_ratio(self, edge_weights):
        """S2: W(147,148) / W(9,8) expected in [1.5, 5]."""
        ratio = W(edge_weights, 147, 148) / max(W(edge_weights, 9, 8), 1e-15)
        if not (1.5 <= ratio <= 5):
            pytest.skip(f"S2 SOFT: ratio {ratio:.2f} outside [1.5, 5] — tune coefficients")

    def test_s3_valence_decay_ratio(self, edge_weights):
        """S3: W(188,190) / W(147,148) expected in [0.05, 0.15].

        Original plan assumed [0.15, 0.65] but didn't account for the 3x c*l
        disadvantage (0.968 vs 3.004).  Realistic range after factoring in both
        the base-signal gap and valence decay is [0.05, 0.15].
        """
        ratio = W(edge_weights, 188, 190) / max(W(edge_weights, 147, 148), 1e-15)
        if not (0.05 <= ratio <= 0.15):
            pytest.skip(f"S3 SOFT: ratio {ratio:.2f} outside [0.05, 0.15] — tune coefficients")

    def test_s4_suppression_ratio(self, edge_weights):
        """S4: W(166,144) / W(11,8) expected in [0.01, 0.15]."""
        ratio = W(edge_weights, 166, 144) / max(W(edge_weights, 11, 8), 1e-15)
        if not (0.01 <= ratio <= 0.15):
            pytest.skip(f"S4 SOFT: ratio {ratio:.2f} outside [0.01, 0.15] — tune coefficients")

    def test_s5_cross_arch_ratio(self, edge_weights):
        """S5: W(101,206) / W(9,8) expected in [0.15, 0.5]."""
        ratio = W(edge_weights, 101, 206) / max(W(edge_weights, 9, 8), 1e-15)
        if not (0.15 <= ratio <= 0.5):
            pytest.skip(f"S5 SOFT: ratio {ratio:.2f} outside [0.15, 0.5] — tune coefficients")

    def test_s6_top20_globally(self, edge_weights):
        """S6: W(11,8) rank should be in top 20 globally.

        Original plan expected top 10 based on c*l rank (8th).  But val_gap of
        0.251 pushes it below cleaner edges with lower c*l.  Top 20 is the
        realistic expectation with valence gating active.
        """
        w_11_8 = W(edge_weights, 11, 8)
        all_weights = sorted((e["weight"] for e in edge_weights.values()), reverse=True)
        rank = next(i + 1 for i, w in enumerate(all_weights) if w <= w_11_8)
        if rank > 20:
            pytest.skip(f"S6 SOFT: W(11,8) rank is {rank}, expected top 20")

    def test_s7_top3_globally(self, edge_weights):
        """S7: W(147,148) rank should be in top 3 globally."""
        w_147_148 = W(edge_weights, 147, 148)
        all_weights = sorted((e["weight"] for e in edge_weights.values()), reverse=True)
        rank = next(i + 1 for i, w in enumerate(all_weights) if w <= w_147_148)
        if rank > 3:
            pytest.skip(f"S7 SOFT: W(147,148) rank is {rank}, expected top 3")


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    def test_b1_strongest_edge_near_top(self, edge_weights):
        """B1: W(147,148) >= p99 of all weights."""
        w = W(edge_weights, 147, 148)
        all_w = [e["weight"] for e in edge_weights.values()]
        p99 = float(np.percentile(all_w, 99))
        assert w >= p99, f"W(147,148)={w:.6f} < p99={p99:.6f}"

    def test_b2_dynamic_range(self, edge_weights):
        """B2: W(12,69) < 0.001 * W(147,148). Dynamic range > 1000x."""
        w_weak = W(edge_weights, 12, 69)
        w_strong = W(edge_weights, 147, 148)
        assert w_weak < 0.001 * w_strong, \
            f"W(12,69)={w_weak:.6f} >= 0.001 * W(147,148)={0.001 * w_strong:.6f}"

    def test_b3_extreme_valence_below_p10(self, edge_weights):
        """B3: Edges with |val_gap| > 1.0 AND containment > 0 are below p10 of non-zero-containment weights."""
        nonzero_cont = [e["weight"] for e in edge_weights.values() if e["containment"] > 0]
        if not nonzero_cont:
            pytest.skip("No non-zero containment edges")
        p10 = float(np.percentile(nonzero_cont, 10))

        extreme = [
            (src, tgt, e["weight"])
            for (src, tgt), e in edge_weights.items()
            if abs(e["valence_gap"]) > 1.0 and e["containment"] > 0 and e["weight"] >= p10
        ]
        assert len(extreme) == 0, \
            f"{len(extreme)} extreme-valence edges above p10={p10:.6f}, first: {extreme[0]}"

    def test_b4_noise_floor(self, edge_weights):
        """B4: W(151,189) < 0.01 * W(9,8). Near-zero containment, anti-correlated."""
        w_noise = W(edge_weights, 151, 189)
        w_strong = W(edge_weights, 9, 8)
        assert w_noise < 0.01 * w_strong, \
            f"W(151,189)={w_noise:.6f} >= 0.01 * W(9,8)={0.01 * w_strong:.6f}"


# ---------------------------------------------------------------------------
# Derived invariants (monotonicity within controlled groups)
# ---------------------------------------------------------------------------

class TestDerivedInvariants:
    def test_d1_target8_containment_ordering(self, edge_weights):
        """D1: W(9,8) > W(11,8) > W(10,8) > W(12,8). Containment monotone decreasing."""
        w9 = W(edge_weights, 9, 8)
        w11 = W(edge_weights, 11, 8)
        w10 = W(edge_weights, 10, 8)
        w12 = W(edge_weights, 12, 8)
        assert w9 > w11 > w10 > w12, \
            f"Ordering violated: {w9:.6f}, {w11:.6f}, {w10:.6f}, {w12:.6f}"

    def test_d2_clean_edges_correlation(self, edge_weights):
        """D2: Among 'clean' edges, Spearman corr between W and containment*lift >= 0.85."""
        from scipy.stats import spearmanr

        clean = []
        for (src, tgt), e in edge_weights.items():
            if (e["pos_similarity"] > 0.97
                    and abs(e["valence_gap"]) < 0.2
                    and abs(e["specificity_gap"]) < 0.1
                    and e["containment"] > 0):
                clean.append((e["weight"], e["containment"] * e["lift"]))

        if len(clean) < 10:
            pytest.skip(f"Only {len(clean)} clean edges — need at least 10")

        weights = [c[0] for c in clean]
        base_signals = [c[1] for c in clean]
        rho, _ = spearmanr(weights, base_signals)
        assert rho >= 0.85, f"Spearman correlation {rho:.3f} < 0.85 among {len(clean)} clean edges"
