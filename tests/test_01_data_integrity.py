"""Layer 1: Data integrity — verify reef_edges is correctly populated."""

import math

import pytest


class TestRowCount:
    def test_row_count_post_pruning(self, db_con):
        """reef_edges contains only edges with weight > 0 (zero-weight rows pruned).

        Before pruning: 206 * 206 = 42,436 (reef 20 excluded, 0 words).
        After pruning: ~24,460 edges with positive containment and weight.
        """
        count = db_con.execute("SELECT COUNT(*) FROM reef_edges").fetchone()[0]
        assert count > 20000, f"Too few edges: {count}"
        assert count < 30000, f"Too many edges: {count} (zero-weight rows should be pruned)"


class TestNoSelfEdges:
    def test_no_self_edges(self, db_con):
        """No rows where source_reef_id == target_reef_id."""
        count = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE source_reef_id = target_reef_id"
        ).fetchone()[0]
        assert count == 0, f"Found {count} self-edges"


class TestCompleteCoverage:
    def test_all_active_reefs_represented(self, db_con):
        """All reefs with words appear as both sources and targets.

        After pruning, not every directed pair exists — only those with
        positive word overlap. But every active reef (has words at depth >= 2)
        should still appear on both sides of at least one edge.
        """
        active_reefs = db_con.execute("""
            SELECT COUNT(DISTINCT reef_id) FROM word_reef_affinity WHERE n_dims >= 2
        """).fetchone()[0]

        source_reefs = db_con.execute(
            "SELECT COUNT(DISTINCT source_reef_id) FROM reef_edges"
        ).fetchone()[0]
        target_reefs = db_con.execute(
            "SELECT COUNT(DISTINCT target_reef_id) FROM reef_edges"
        ).fetchone()[0]

        assert source_reefs >= active_reefs - 1, \
            f"Only {source_reefs} source reefs, expected >= {active_reefs - 1}"
        assert target_reefs >= active_reefs - 1, \
            f"Only {target_reefs} target reefs, expected >= {active_reefs - 1}"


class TestContainmentRange:
    def test_containment_in_unit_interval(self, db_con):
        """All containment values in [0, 1]."""
        violations = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE containment < 0 OR containment > 1"
        ).fetchone()[0]
        assert violations == 0, f"{violations} containment values outside [0, 1]"


class TestLiftRange:
    def test_lift_non_negative(self, db_con):
        """All lift values >= 0."""
        violations = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE lift < 0"
        ).fetchone()[0]
        assert violations == 0, f"{violations} negative lift values"


class TestPosSimilarityRange:
    def test_pos_similarity_in_unit_interval(self, db_con):
        """All POS similarity values in [0, 1]."""
        violations = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE pos_similarity < 0 OR pos_similarity > 1"
        ).fetchone()[0]
        assert violations == 0, f"{violations} pos_similarity values outside [0, 1]"


class TestValenceGapFinite:
    def test_valence_gap_no_nulls(self, db_con):
        nulls = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE valence_gap IS NULL"
        ).fetchone()[0]
        assert nulls == 0, f"{nulls} NULL valence_gap values"

    def test_valence_gap_finite(self, db_con):
        violations = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE isnan(valence_gap) OR isinf(valence_gap)"
        ).fetchone()[0]
        assert violations == 0, f"{violations} non-finite valence_gap values"


class TestSpecificityGapFinite:
    def test_specificity_gap_no_nulls(self, db_con):
        nulls = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE specificity_gap IS NULL"
        ).fetchone()[0]
        assert nulls == 0, f"{nulls} NULL specificity_gap values"

    def test_specificity_gap_finite(self, db_con):
        violations = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE isnan(specificity_gap) OR isinf(specificity_gap)"
        ).fetchone()[0]
        assert violations == 0, f"{violations} non-finite specificity_gap values"


class TestWeightColumn:
    def test_weight_column_exists(self, db_con):
        """weight column is DOUBLE and non-NULL."""
        cols = db_con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'reef_edges' AND column_name = 'weight'"
        ).fetchall()
        assert len(cols) == 1, "weight column not found in reef_edges"
        assert "DOUBLE" in cols[0][1].upper(), f"weight column type is {cols[0][1]}, expected DOUBLE"

    def test_weight_no_nulls(self, db_con):
        nulls = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE weight IS NULL"
        ).fetchone()[0]
        assert nulls == 0, f"{nulls} NULL weight values"

    def test_weight_non_negative(self, db_con):
        """All weight >= 0."""
        violations = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE weight < 0"
        ).fetchone()[0]
        assert violations == 0, f"{violations} negative weight values"


class TestNoZeroWeightEdges:
    def test_all_edges_have_positive_weight(self, db_con):
        """After pruning, all remaining edges have weight > 0."""
        zeros = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE weight = 0"
        ).fetchone()[0]
        assert zeros == 0, f"{zeros} zero-weight edges remain after pruning"

    def test_all_edges_have_positive_containment(self, db_con):
        """After pruning, all remaining edges have containment > 0."""
        zeros = db_con.execute(
            "SELECT COUNT(*) FROM reef_edges WHERE containment = 0"
        ).fetchone()[0]
        assert zeros == 0, f"{zeros} zero-containment edges remain after pruning"


class TestKnownEdgeValues:
    """Spot-check specific edges against verified values."""

    @pytest.mark.parametrize("src,tgt,expected_containment,expected_lift,expected_pos", [
        (9, 8, 0.1965, 5.41, 0.9993),
        (147, 148, 0.1403, 21.42, 0.9784),
        (188, 190, 0.0500, 19.36, 0.9957),
        (166, 144, 0.0495, 1.36, 0.9328),
    ])
    def test_spot_check(self, edge_weights, src, tgt, expected_containment, expected_lift, expected_pos):
        key = (src, tgt)
        assert key in edge_weights, f"Edge {src}→{tgt} not found"
        e = edge_weights[key]
        assert e["containment"] == pytest.approx(expected_containment, abs=0.001), \
            f"containment {e['containment']:.4f} != {expected_containment}"
        assert e["lift"] == pytest.approx(expected_lift, abs=0.05), \
            f"lift {e['lift']:.2f} != {expected_lift}"
        assert e["pos_similarity"] == pytest.approx(expected_pos, abs=0.001), \
            f"pos_similarity {e['pos_similarity']:.4f} != {expected_pos}"
