"""Layer 2: Knob algebraic properties — verify mathematical invariants."""

import pytest


class TestPosSymmetry:
    def test_pos_similarity_symmetric(self, edge_weights):
        """pos_similarity(A→B) == pos_similarity(B→A) for all pairs."""
        violations = []
        checked = set()
        for (src, tgt), e in edge_weights.items():
            if (tgt, src) in checked:
                continue
            checked.add((src, tgt))
            reverse = edge_weights.get((tgt, src))
            if reverse is None:
                continue
            if abs(e["pos_similarity"] - reverse["pos_similarity"]) > 1e-9:
                violations.append((src, tgt, e["pos_similarity"], reverse["pos_similarity"]))
        assert len(violations) == 0, \
            f"{len(violations)} POS asymmetries, first: {violations[0]}"


class TestLiftSymmetry:
    def test_lift_symmetric(self, edge_weights):
        """lift(A→B) == lift(B→A) for all pairs."""
        violations = []
        checked = set()
        for (src, tgt), e in edge_weights.items():
            if (tgt, src) in checked:
                continue
            checked.add((src, tgt))
            reverse = edge_weights.get((tgt, src))
            if reverse is None:
                continue
            if abs(e["lift"] - reverse["lift"]) > 1e-6:
                violations.append((src, tgt, e["lift"], reverse["lift"]))
        assert len(violations) == 0, \
            f"{len(violations)} lift asymmetries, first: {violations[0]}"


class TestValenceAntisymmetry:
    def test_valence_gap_antisymmetric(self, edge_weights):
        """valence_gap(A→B) == -valence_gap(B→A) for all pairs."""
        violations = []
        checked = set()
        for (src, tgt), e in edge_weights.items():
            if (tgt, src) in checked:
                continue
            checked.add((src, tgt))
            reverse = edge_weights.get((tgt, src))
            if reverse is None:
                continue
            if abs(e["valence_gap"] + reverse["valence_gap"]) > 1e-9:
                violations.append((src, tgt, e["valence_gap"], reverse["valence_gap"]))
        assert len(violations) == 0, \
            f"{len(violations)} valence antisymmetry violations, first: {violations[0]}"


class TestSpecificityAntisymmetry:
    def test_specificity_gap_antisymmetric(self, edge_weights):
        """specificity_gap(A→B) == -specificity_gap(B→A) for all pairs."""
        violations = []
        checked = set()
        for (src, tgt), e in edge_weights.items():
            if (tgt, src) in checked:
                continue
            checked.add((src, tgt))
            reverse = edge_weights.get((tgt, src))
            if reverse is None:
                continue
            if abs(e["specificity_gap"] + reverse["specificity_gap"]) > 1e-9:
                violations.append((src, tgt, e["specificity_gap"], reverse["specificity_gap"]))
        assert len(violations) == 0, \
            f"{len(violations)} specificity antisymmetry violations, first: {violations[0]}"


class TestContainmentAsymmetry:
    def test_containment_asymmetry_exists(self, edge_weights):
        """At least 1000 pairs where |containment(A→B) - containment(B→A)| > 0.01."""
        asymmetric_count = 0
        checked = set()
        for (src, tgt), e in edge_weights.items():
            if (tgt, src) in checked:
                continue
            checked.add((src, tgt))
            reverse = edge_weights.get((tgt, src))
            if reverse is None:
                continue
            if abs(e["containment"] - reverse["containment"]) > 0.01:
                asymmetric_count += 1
        assert asymmetric_count >= 1000, \
            f"Only {asymmetric_count} asymmetric containment pairs (need >= 1000)"


class TestAllEdgesPositive:
    def test_all_containment_positive(self, edge_weights):
        """After pruning, all remaining edges have containment > 0."""
        zeros = [k for k, e in edge_weights.items() if e["containment"] == 0]
        assert len(zeros) == 0, f"{len(zeros)} zero-containment edges remain after pruning"

    def test_all_lift_positive(self, edge_weights):
        """After pruning, all remaining edges have lift > 0."""
        zeros = [k for k, e in edge_weights.items() if e["lift"] == 0]
        assert len(zeros) == 0, f"{len(zeros)} zero-lift edges remain after pruning"


class TestContainmentUpperBound:
    def test_containment_max(self, edge_weights):
        """No containment exceeds 0.20."""
        max_cont = max(e["containment"] for e in edge_weights.values())
        assert max_cont <= 0.20, f"Max containment {max_cont:.4f} exceeds 0.20"


class TestPosSimilarityLowerBound:
    def test_pos_similarity_min(self, edge_weights):
        """No POS similarity below 0.82."""
        min_pos = min(e["pos_similarity"] for e in edge_weights.values())
        assert min_pos >= 0.82, f"Min POS similarity {min_pos:.4f} below 0.82"
