"""Layer 7: Specificity gap gate — verify alpha_spec actually constrains rankings.

These invariants FAIL with alpha_spec=0 (no specificity gating) but PASS with
alpha_spec >= ~0.39.  They ensure the specificity decay is doing real work
rather than being decorative.

The specificity gate is intentionally mild (alpha_spec=0.5, max ~24% suppression).
It acts as a tiebreaker, not a veto — penalizing edges between reefs of very
different specificity (broad → narrow or narrow → broad).

Methodology: each hard test picks two edges where the one with LOWER no-spec score
wins on weight because its |specificity_gap| is much smaller.  Without the spec gate,
the higher no-spec edge would dominate.
"""

import math

import pytest


def W(edge_weights, src, tgt):
    """Shorthand for weight lookup."""
    key = (src, tgt)
    assert key in edge_weights, f"Edge {src}→{tgt} not found"
    return edge_weights[key]["weight"]


def _no_spec_weight(e):
    """Weight without the specificity gate: c*l * pos^2 * exp(-2.92*|val|)."""
    cl = e["containment"] * e["lift"]
    return cl * (e["pos_similarity"] ** 2) * math.exp(-2.92 * abs(e["valence_gap"]))


def _spec_retention(e):
    """Ratio of actual weight to no-spec weight.  Should equal exp(-alpha_spec * |spec|)."""
    ns = _no_spec_weight(e)
    if ns < 1e-9:
        return None
    return e["weight"] / ns


# ---------------------------------------------------------------------------
# Hard specificity invariants (must hold — failure = spec gate broken or too weak)
# ---------------------------------------------------------------------------

class TestSpecHardInvariants:
    """Ranking flips that require specificity gating.  Each would go the wrong
    way with alpha_spec=0."""

    def test_sp1_148_205_vs_161_64(self, edge_weights):
        """W(148,205) > W(161,64): |spec| 0.051 vs 0.196 flips near-tied no-spec score.

        161→64:  c*l=0.447, pos=0.991, |val|=0.020, spec=0.196 → W≈0.375
        148→205: c*l=0.501, pos=0.989, |val|=0.061, spec=0.051 → W≈0.400
        Without spec gate, 161→64 edges ahead (0.414 vs 0.410).
        The 0.145 spec gap flips the ranking at crossover alpha_spec=0.07.
        """
        assert W(edge_weights, 148, 205) > W(edge_weights, 161, 64)

    def test_sp2_74_76_vs_152_153(self, edge_weights):
        """W(74,76) > W(152,153): |spec| 0.056 vs 0.188 overcomes 1.08x c*l gap.

        152→153: c*l=0.479, pos=0.990, |val|=0.120, spec=-0.188 → W≈0.302
         74→76:  c*l=0.445, pos=0.999, |val|=0.107, spec=0.056  → W≈0.316
        Without spec gate, 152→153 edges ahead (0.331 vs 0.325).
        Crossover alpha_spec=0.15.
        """
        assert W(edge_weights, 74, 76) > W(edge_weights, 152, 153)

    def test_sp3_125_67_vs_152_153(self, edge_weights):
        """W(125,67) > W(152,153): |spec| 0.047 vs 0.188 overcomes 1.04x c*l and val gap.

        152→153: c*l=0.479, pos=0.990, |val|=0.120, spec=-0.188 → W≈0.302
        125→67:  c*l=0.478, pos=0.994, |val|=0.135, spec=0.047  → W≈0.311
        Without spec gate, 152→153 edges ahead (0.331 vs 0.318).
        Crossover alpha_spec=0.29.
        """
        assert W(edge_weights, 125, 67) > W(edge_weights, 152, 153)

    def test_sp4_91_68_vs_70_66(self, edge_weights):
        """W(91,68) > W(70,66): |spec| 0.004 vs 0.121 overcomes 1.05x no-spec gap.

        70→66:  c*l=0.450, pos=0.999, |val|=0.047, spec=0.121 → W≈0.368
        91→68:  c*l=0.498, pos=0.997, |val|=0.097, spec=0.004 → W≈0.373
        Without spec gate, 70→66 edges ahead (0.391 vs 0.374).
        The 0.117 spec gap and near-zero spec on 91→68 flip at crossover alpha_spec=0.39.
        This is the tightest constraint — the strongest structural test of the spec gate.
        """
        assert W(edge_weights, 91, 68) > W(edge_weights, 70, 66)


# ---------------------------------------------------------------------------
# Spec transparency (low-gap edges should barely notice the gate)
# ---------------------------------------------------------------------------

class TestSpecTransparency:

    def test_sp5_low_spec_preserves_signal(self, edge_weights):
        """Edges with |spec| < 0.03 should lose < 2% to specificity gating.

        At alpha_spec=0.5, |spec|=0.03 → exp(-0.015) = 0.985 (1.5% loss).
        The spec gate should be nearly invisible for well-matched reefs.
        """
        low_spec_edges = [
            (src, tgt, e) for (src, tgt), e in edge_weights.items()
            if abs(e["specificity_gap"]) < 0.03 and e["containment"] > 0
        ]
        assert len(low_spec_edges) > 100, \
            f"Only {len(low_spec_edges)} low-spec edges (need > 100)"

        for src, tgt, e in low_spec_edges[:50]:
            ret = _spec_retention(e)
            if ret is None:
                continue
            assert ret >= 0.98, \
                f"Edge {src}→{tgt} |spec|={abs(e['specificity_gap']):.4f} lost {(1-ret)*100:.1f}% to spec gate (max 2%)"


# ---------------------------------------------------------------------------
# Spec suppression strength (high-gap edges should be meaningfully penalized)
# ---------------------------------------------------------------------------

class TestSpecSuppressionStrength:

    def test_sp6_high_spec_loses_measurable_weight(self, edge_weights):
        """Edges with |spec| > 0.25 should lose >= 8% to specificity gating.

        At alpha_spec=0.5, |spec|=0.25 → exp(-0.125) = 0.882 (11.8% loss).
        Tests that the gate has measurable effect on large spec gaps.
        """
        high_spec_edges = [
            (src, tgt, e) for (src, tgt), e in edge_weights.items()
            if abs(e["specificity_gap"]) > 0.25 and e["containment"] > 0
        ]
        assert len(high_spec_edges) > 100, \
            f"Only {len(high_spec_edges)} high-spec edges (need > 100)"

        for src, tgt, e in high_spec_edges[:50]:
            ret = _spec_retention(e)
            if ret is None:
                continue
            assert ret <= 0.92, \
                f"Edge {src}→{tgt} |spec|={abs(e['specificity_gap']):.4f} only lost {(1-ret)*100:.1f}% (expected >= 8%)"

    def test_sp7_spec_gate_dynamic_range(self, edge_weights):
        """The spec retention factor should differ >= 1.05x between
        low-gap (|spec| < 0.03) and high-gap (|spec| > 0.25) edges.

        Retention = W / (c*l * pos^alpha_pos * val_decay) = exp(-alpha_spec * |spec|).
        At alpha_spec=0.5: exp(-0.015)/exp(-0.125) = 1.12x.
        Tests that specificity creates a measurable spread.
        """
        high_ret = []
        low_ret = []
        for (src, tgt), e in edge_weights.items():
            ret = _spec_retention(e)
            if ret is None:
                continue
            spec = abs(e["specificity_gap"])
            if spec < 0.03:
                high_ret.append(ret)
            elif spec > 0.25:
                low_ret.append(ret)

        if len(high_ret) < 50 or len(low_ret) < 50:
            pytest.skip(f"Need more edges: {len(high_ret)} low-spec, {len(low_ret)} high-spec")

        avg_high = sum(high_ret) / len(high_ret)
        avg_low = sum(low_ret) / len(low_ret)
        ratio = avg_high / max(avg_low, 1e-15)
        assert ratio >= 1.05, \
            f"Spec retention ratio {ratio:.3f}x too small (low-gap={avg_high:.4f} n={len(high_ret)}, high-gap={avg_low:.4f} n={len(low_ret)})"


# ---------------------------------------------------------------------------
# Soft specificity guidelines (informational)
# ---------------------------------------------------------------------------

class TestSpecSoftGuidelines:

    def test_sps1_reef12_avg_suppression(self, edge_weights):
        """SPS1: Reef 12 (highest avg |spec|) edges should average 10-20% spec loss.

        Reef 12 avg |spec_gap| to neighbors = 0.301.  At alpha_spec=0.5:
        exp(-0.5*0.301) = 0.860, so ~14% loss expected.
        """
        reef12_edges = [
            e for (src, tgt), e in edge_weights.items()
            if src == 12 and e["containment"] > 0
        ]
        if len(reef12_edges) < 5:
            pytest.skip("Not enough reef 12 outgoing edges")

        retentions = []
        for e in reef12_edges:
            ret = _spec_retention(e)
            if ret is not None:
                retentions.append(ret)

        if not retentions:
            pytest.skip("No valid retention values for reef 12")

        avg_retention = sum(retentions) / len(retentions)
        avg_loss = 1 - avg_retention
        if not (0.10 <= avg_loss <= 0.20):
            pytest.skip(
                f"SPS1 SOFT: reef 12 avg spec loss {avg_loss:.1%} outside [10%, 20%]"
            )
