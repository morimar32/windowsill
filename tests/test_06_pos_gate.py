"""Layer 6: POS similarity gate — verify alpha_pos actually constrains rankings.

These invariants FAIL with alpha_pos=0 (no POS gating) but PASS with
alpha_pos >= ~4.  They ensure the POS similarity exponent is doing real work
rather than being decorative.

Methodology: each test picks two edges where the one with LOWER containment*lift
wins on weight because its POS similarity is much higher.  Without the POS gate,
the higher c*l edge would dominate.
"""

import math

import pytest


def W(edge_weights, src, tgt):
    """Shorthand for weight lookup."""
    key = (src, tgt)
    assert key in edge_weights, f"Edge {src}→{tgt} not found"
    return edge_weights[key]["weight"]


def knobs(edge_weights, src, tgt):
    """Return full edge profile."""
    key = (src, tgt)
    assert key in edge_weights, f"Edge {src}→{tgt} not found"
    return edge_weights[key]


# ---------------------------------------------------------------------------
# Hard POS invariants (must hold — failure = POS gate is broken or too weak)
# ---------------------------------------------------------------------------

class TestPosHardInvariants:
    """Ranking flips that require POS gating.  Each would go the wrong way
    with alpha_pos=0."""

    def test_p2_182_176_vs_165_166(self, edge_weights):
        """W(182,176) > W(165,166): POS 0.997 vs 0.947 flips near-identical c*l.

        165→166: c*l=0.498, pos=0.947, |val|=0.005 → W≈0.395
        182→176: c*l=0.502, pos=0.997, |val|=0.013 → W≈0.465
        165→166 involves reef 166 (extreme POS outlier, avg_pos=0.887).
        The 0.05 POS gap compounds to ~0.82x at alpha_pos=4.
        """
        assert W(edge_weights, 182, 176) > W(edge_weights, 165, 166)

    def test_p3_167_146_vs_164_133(self, edge_weights):
        """W(167,146) > W(164,133): POS 0.997 vs 0.964 overcomes 1.22x c*l gap.

        164→133: c*l=0.692, pos=0.964, |val|=0.083 → W≈0.456
        167→146: c*l=0.569, pos=0.997, |val|=0.022 → W≈0.516
        Similar valence, controlled specificity.  POS + valence together flip it.
        """
        assert W(edge_weights, 167, 146) > W(edge_weights, 164, 133)

    def test_p4_181_207_vs_133_164(self, edge_weights):
        """W(181,207) > W(133,164): POS 0.999 vs 0.964 with nearly matched c*l.

        133→164: c*l=0.671, pos=0.964, |val|=0.083 → W≈0.442
        181→207: c*l=0.646, pos=0.999, |val|=0.084 → W≈0.491
        Valence and c*l are essentially equal.  Pure POS test.
        """
        assert W(edge_weights, 181, 207) > W(edge_weights, 133, 164)

    def test_p5_94_100_vs_153_166(self, edge_weights):
        """W(94,100) > W(153,166): POS 0.999 vs 0.912 overcomes 1.84x c*l gap.

        153→166: c*l=0.498, pos=0.912, |val|=0.291 → W≈0.134
         94→100: c*l=0.271, pos=0.999, |val|=0.116 → W≈0.190
        Reef 166 is the system's POS outlier (avg_pos_to_neighbors=0.887).
        This is the strongest test: POS + valence together overcome nearly 2x c*l.
        """
        assert W(edge_weights, 94, 100) > W(edge_weights, 153, 166)


# ---------------------------------------------------------------------------
# POS monotonicity (ranking should follow POS when other knobs are controlled)
# ---------------------------------------------------------------------------

class TestPosMonotonicity:

    def test_p6_reef166_suppression_ordering(self, edge_weights):
        """Edges into reef 166 with similar c*l should rank by POS.

        Reef 166 is the system's POS outlier.  Edges FROM different-POS sources
        into reef 166 test whether POS gating creates the right ordering.
        """
        # 165→166: pos=0.947, c*l=0.498
        # 153→166: pos=0.912, c*l=0.498
        # Nearly identical c*l, but 165 has higher POS to 166
        assert W(edge_weights, 165, 166) > W(edge_weights, 153, 166), \
            "165→166 (pos=0.947) should beat 153→166 (pos=0.912) with same c*l"

    def test_p7_high_pos_preserves_signal(self, edge_weights):
        """Edges with POS > 0.99 should lose < 5% to POS gating.

        At alpha_pos=4, pos=0.99^4 = 0.961 (3.9% loss).
        This means POS gating is nearly transparent for well-matched reefs.
        """
        high_pos_edges = [
            (src, tgt, e) for (src, tgt), e in edge_weights.items()
            if e["pos_similarity"] > 0.99 and e["containment"] > 0
        ]
        assert len(high_pos_edges) > 100, "Need enough high-POS edges to test"

        # For these edges, weight should be >= 95% of what it would be without POS gate
        # weight / (c*l * val_decay * spec_decay) should be >= 0.95
        for src, tgt, e in high_pos_edges[:50]:
            cl = e["containment"] * e["lift"]
            if cl < 0.001:
                continue
            val_spec = math.exp(-2.92 * abs(e["valence_gap"])) * math.exp(-0.5 * abs(e["specificity_gap"]))
            no_pos_weight = cl * val_spec
            if no_pos_weight < 1e-9:
                continue
            pos_retention = e["weight"] / no_pos_weight
            assert pos_retention >= 0.95, \
                f"Edge {src}→{tgt} pos={e['pos_similarity']:.4f} lost {(1-pos_retention)*100:.1f}% to POS gate (max 5%)"


# ---------------------------------------------------------------------------
# POS suppression strength (low-POS edges should be meaningfully penalized)
# ---------------------------------------------------------------------------

class TestPosSuppressionStrength:

    def test_p8_low_pos_loses_measurable_weight(self, edge_weights):
        """Edges with POS < 0.92 should lose >= 10% to POS gating.

        At alpha_pos=2, pos=0.92^2 = 0.846 (15.4% loss).
        This tests that the gate has measurable effect, not a specific strength.
        """
        low_pos_edges = [
            (src, tgt, e) for (src, tgt), e in edge_weights.items()
            if e["pos_similarity"] < 0.92 and e["containment"] > 0
        ]
        assert len(low_pos_edges) > 50, f"Only {len(low_pos_edges)} low-POS edges"

        for src, tgt, e in low_pos_edges[:30]:
            cl = e["containment"] * e["lift"]
            if cl < 0.001:
                continue
            val_spec = math.exp(-2.92 * abs(e["valence_gap"])) * math.exp(-0.5 * abs(e["specificity_gap"]))
            no_pos_weight = cl * val_spec
            if no_pos_weight < 1e-9:
                continue
            pos_retention = e["weight"] / no_pos_weight
            assert pos_retention <= 0.90, \
                f"Edge {src}→{tgt} pos={e['pos_similarity']:.4f} only lost {(1-pos_retention)*100:.1f}% (expected >= 10%)"

    def test_p9_pos_gate_dynamic_range(self, edge_weights):
        """The POS retention factor should differ >= 1.1x between
        high-POS (>0.99) and low-POS (<0.93) edges.

        Retention = W / (c*l * val_decay * spec_decay) = pos^alpha_pos.
        At alpha_pos=2: 0.99^2=0.980 vs 0.92^2=0.846 → ratio 1.16x.
        Tests that POS creates a measurable spread, not a specific magnitude.
        """
        high_ret = []
        low_ret = []
        for (src, tgt), e in edge_weights.items():
            cl = e["containment"] * e["lift"]
            if cl < 0.001:
                continue
            val_spec = math.exp(-2.92 * abs(e["valence_gap"])) * math.exp(-0.5 * abs(e["specificity_gap"]))
            no_pos = cl * val_spec
            if no_pos < 1e-9:
                continue
            ret = e["weight"] / no_pos
            if e["pos_similarity"] > 0.99:
                high_ret.append(ret)
            elif e["pos_similarity"] < 0.93:
                low_ret.append(ret)

        if len(high_ret) < 10 or len(low_ret) < 5:
            pytest.skip(f"Need more edges: {len(high_ret)} high-POS, {len(low_ret)} low-POS")

        avg_high = sum(high_ret) / len(high_ret)
        avg_low = sum(low_ret) / len(low_ret)
        ratio = avg_high / max(avg_low, 1e-15)
        assert ratio >= 1.1, \
            f"POS retention ratio {ratio:.2f}x too small (high={avg_high:.4f} n={len(high_ret)}, low={avg_low:.4f} n={len(low_ret)})"


# ---------------------------------------------------------------------------
# Soft POS guidelines (informational)
# ---------------------------------------------------------------------------

class TestPosSoftGuidelines:

    def test_ps1_reef166_avg_suppression(self, edge_weights):
        """PS1: Reef 166 (lowest avg POS) edges should average 15-30% POS loss.

        Reef 166 avg POS to neighbors = 0.887.  At alpha_pos=2:
        0.887^2 = 0.787, so ~21% loss expected.
        """
        reef166_edges = [
            e for (src, tgt), e in edge_weights.items()
            if src == 166 and e["containment"] > 0
        ]
        if len(reef166_edges) < 5:
            pytest.skip("Not enough reef 166 outgoing edges")

        retentions = []
        for e in reef166_edges:
            cl = e["containment"] * e["lift"]
            if cl < 0.001:
                continue
            val_spec = math.exp(-2.92 * abs(e["valence_gap"])) * math.exp(-0.5 * abs(e["specificity_gap"]))
            no_pos = cl * val_spec
            if no_pos > 1e-9:
                retentions.append(e["weight"] / no_pos)

        avg_retention = sum(retentions) / len(retentions)
        avg_loss = 1 - avg_retention
        if not (0.15 <= avg_loss <= 0.30):
            pytest.skip(
                f"PS1 SOFT: reef 166 avg POS loss {avg_loss:.1%} outside [15%, 30%]"
            )
