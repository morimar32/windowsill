"""Layer 8: Valence gap gate — verify alpha_val actually constrains rankings.

These invariants FAIL with alpha_val=0 (no valence gating) but PASS with
alpha_val >= ~0.13.  They ensure the valence decay is doing real work
rather than being decorative.

The valence gate is the strongest gate in the formula (alpha_val=2.92).
It aggressively suppresses edges between reefs of opposite polarity:
|val|=0.30 loses ~58%, |val|=0.50 loses ~77%, |val|=0.80 loses ~90%.

Character of the gate:  The valence gate is bimodal.  Pairs involving
extreme outlier reefs (166, 189 — |val| > 0.70 to most neighbors) flip at
nearly any alpha > 0, while moderate-gap pairs only flip at the boundary
(alpha_val ≈ 2.92).  The binding constraint is M5 in test_03
(131→190 vs 188→190, crossover alpha_val=2.86).  The hard invariants here
test that the gate EXISTS and SUPPRESSES extreme mismatches; the
transparency/suppression/dynamic-range tests characterize its strength.
"""

import math

import pytest


def W(edge_weights, src, tgt):
    """Shorthand for weight lookup."""
    key = (src, tgt)
    assert key in edge_weights, f"Edge {src}→{tgt} not found"
    return edge_weights[key]["weight"]


def _no_val_weight(e):
    """Weight without the valence gate: c*l * pos^2 * exp(-0.5*|spec|)."""
    cl = e["containment"] * e["lift"]
    return cl * (e["pos_similarity"] ** 2) * math.exp(-0.5 * abs(e["specificity_gap"]))


def _val_retention(e):
    """Ratio of actual weight to no-val weight.  Should equal exp(-alpha_val * |val|)."""
    nv = _no_val_weight(e)
    if nv < 1e-9:
        return None
    return e["weight"] / nv


# ---------------------------------------------------------------------------
# Hard valence invariants (must hold — failure = val gate is broken)
# ---------------------------------------------------------------------------

class TestValHardInvariants:
    """Ranking flips that require valence gating.  Each would go the wrong way
    with alpha_val=0.  All involve extreme-outlier reefs (166/189) because the
    valence gate's structural effect is concentrated there."""

    def test_vg1_137_138_vs_189_166(self, edge_weights):
        """W(137,138) > W(189,166): |val| 0.014 vs 0.791 with near-identical c*l.

        189→166: c*l=0.120, pos=0.963, |val|=0.791, |spec|=0.010 → W≈0.011
        137→138: c*l=0.114, pos=1.000, |val|=0.014, |spec|=0.052 → W≈0.106
        c*l nearly matched (1.06x favoring 189→166).  Without val gate,
        189→166 would edge ahead.  At alpha_val=2.92, the 0.791 gap
        crushes 189→166 to 10% of its no-val score.
        Crossover alpha_val=0.004.
        """
        assert W(edge_weights, 137, 138) > W(edge_weights, 189, 166)

    def test_vg2_103_68_vs_166_189(self, edge_weights):
        """W(103,68) > W(166,189): |val| 0.011 vs 0.791 overcomes 1.10x c*l gap.

        166→189: c*l=0.081, pos=0.963, |val|=0.791, |spec|=0.010 → W≈0.007
        103→68:  c*l=0.074, pos=0.998, |val|=0.011, |spec|=0.021 → W≈0.070
        Reef 166→189 has the system's most extreme valence gap.  Even a
        10% c*l advantage cannot survive 79% polarity mismatch.
        Crossover alpha_val=0.039.
        """
        assert W(edge_weights, 103, 68) > W(edge_weights, 166, 189)

    def test_vg3_38_41_vs_166_189(self, edge_weights):
        """W(38,41) > W(166,189): |val| 0.010 vs 0.791 overcomes 1.14x c*l gap.

        166→189: c*l=0.081, pos=0.963, |val|=0.791, |spec|=0.010 → W≈0.007
         38→41:  c*l=0.071, pos=1.000, |val|=0.010, |spec|=0.092 → W≈0.066
        Larger c*l gap than VG2, plus a 0.09 spec gap penalty on 38→41.
        The val gate still dominates.
        Crossover alpha_val=0.128.
        """
        assert W(edge_weights, 38, 41) > W(edge_weights, 166, 189)


# ---------------------------------------------------------------------------
# Valence transparency (low-gap edges should lose little to the gate)
# ---------------------------------------------------------------------------

class TestValTransparency:

    def test_vg4_low_val_preserves_signal(self, edge_weights):
        """Edges with |val| < 0.02 should lose < 7% to valence gating.

        At alpha_val=2.92, |val|=0.02 → exp(-0.058) = 0.943 (5.7% loss).
        The val gate should be nearly invisible for polarity-matched reefs.
        """
        low_val_edges = [
            (src, tgt, e) for (src, tgt), e in edge_weights.items()
            if abs(e["valence_gap"]) < 0.02 and e["containment"] > 0
        ]
        assert len(low_val_edges) > 100, \
            f"Only {len(low_val_edges)} low-val edges (need > 100)"

        for src, tgt, e in low_val_edges[:50]:
            ret = _val_retention(e)
            if ret is None:
                continue
            assert ret >= 0.93, \
                f"Edge {src}→{tgt} |val|={abs(e['valence_gap']):.4f} lost {(1-ret)*100:.1f}% to val gate (max 7%)"


# ---------------------------------------------------------------------------
# Valence suppression strength (high-gap edges should be heavily penalized)
# ---------------------------------------------------------------------------

class TestValSuppressionStrength:

    def test_vg5_high_val_loses_majority(self, edge_weights):
        """Edges with |val| > 0.30 should lose >= 50% to valence gating.

        At alpha_val=2.92, |val|=0.30 → exp(-0.876) = 0.416 (58.4% loss).
        The valence gate should dramatically suppress cross-polarity edges.
        """
        high_val_edges = [
            (src, tgt, e) for (src, tgt), e in edge_weights.items()
            if abs(e["valence_gap"]) > 0.30 and e["containment"] > 0
        ]
        assert len(high_val_edges) > 100, \
            f"Only {len(high_val_edges)} high-val edges (need > 100)"

        for src, tgt, e in high_val_edges[:50]:
            ret = _val_retention(e)
            if ret is None:
                continue
            assert ret <= 0.50, \
                f"Edge {src}→{tgt} |val|={abs(e['valence_gap']):.4f} only lost {(1-ret)*100:.1f}% (expected >= 50%)"

    def test_vg6_extreme_val_near_zero(self, edge_weights):
        """Edges with |val| > 0.80 should retain < 10% of their no-val weight.

        At alpha_val=2.92, |val|=0.80 → exp(-2.336) = 0.097 (90.3% loss).
        Extreme polarity mismatch should nearly eliminate edge weight.
        """
        extreme_val_edges = [
            (src, tgt, e) for (src, tgt), e in edge_weights.items()
            if abs(e["valence_gap"]) > 0.80 and e["containment"] > 0
        ]
        if len(extreme_val_edges) < 5:
            pytest.skip(f"Only {len(extreme_val_edges)} extreme-val edges")

        for src, tgt, e in extreme_val_edges:
            ret = _val_retention(e)
            if ret is None:
                continue
            assert ret <= 0.10, \
                f"Edge {src}→{tgt} |val|={abs(e['valence_gap']):.4f} retained {ret*100:.1f}% (expected < 10%)"

    def test_vg7_val_gate_dynamic_range(self, edge_weights):
        """The val retention factor should differ >= 2.0x between
        low-gap (|val| < 0.02) and high-gap (|val| > 0.30) edges.

        At alpha_val=2.92: exp(-0.058)/exp(-0.876) = 2.27x.
        The valence gate creates massive separation between matched
        and mismatched reefs.
        """
        high_ret = []
        low_ret = []
        for (src, tgt), e in edge_weights.items():
            ret = _val_retention(e)
            if ret is None:
                continue
            val = abs(e["valence_gap"])
            if val < 0.02:
                high_ret.append(ret)
            elif val > 0.30:
                low_ret.append(ret)

        if len(high_ret) < 50 or len(low_ret) < 50:
            pytest.skip(f"Need more edges: {len(high_ret)} low-val, {len(low_ret)} high-val")

        avg_high = sum(high_ret) / len(high_ret)
        avg_low = sum(low_ret) / len(low_ret)
        ratio = avg_high / max(avg_low, 1e-15)
        assert ratio >= 2.0, \
            f"Val retention ratio {ratio:.2f}x too small (low-gap={avg_high:.4f} n={len(high_ret)}, high-gap={avg_low:.4f} n={len(low_ret)})"


# ---------------------------------------------------------------------------
# Soft valence guidelines (informational)
# ---------------------------------------------------------------------------

class TestValSoftGuidelines:

    def test_vgs1_reef166_avg_suppression(self, edge_weights):
        """VGS1: Reef 166 (highest avg |val|) edges should average 75-95% val loss.

        Reef 166 avg |val_gap| to neighbors = 0.629.  At alpha_val=2.92:
        exp(-2.92*0.629) = 0.159, so ~84% loss expected.
        """
        reef166_edges = [
            e for (src, tgt), e in edge_weights.items()
            if src == 166 and e["containment"] > 0
        ]
        if len(reef166_edges) < 5:
            pytest.skip("Not enough reef 166 outgoing edges")

        retentions = []
        for e in reef166_edges:
            ret = _val_retention(e)
            if ret is not None:
                retentions.append(ret)

        if not retentions:
            pytest.skip("No valid retention values for reef 166")

        avg_retention = sum(retentions) / len(retentions)
        avg_loss = 1 - avg_retention
        if not (0.75 <= avg_loss <= 0.95):
            pytest.skip(
                f"VGS1 SOFT: reef 166 avg val loss {avg_loss:.1%} outside [75%, 95%]"
            )
