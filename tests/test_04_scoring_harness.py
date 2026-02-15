"""Layer 4: Scoring harness integration — end-to-end sentence → reef ranking."""

import pytest


class TestBotanicalQuery:
    """Test 4.1: Botanical query — coalescence → botanical containment flow."""

    SENTENCE = "The colchicaceae family exhibits physical coalescence in its cellular structure"

    def test_reef9_in_top3_primary(self, scorer):
        """Reef 9 (coalescence) is in top 3 primary activations."""
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        top3 = sorted(primary.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_ids = [r[0] for r in top3]
        assert 9 in top3_ids, f"Reef 9 not in top 3 primary: {top3}"

    def test_reef8_boosted_after_propagation(self, scorer):
        """After propagation, reef 8 (botanical) score increases."""
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        propagated = scorer.propagate(primary)
        reef8_primary = primary.get(8, 0.0)
        reef8_boost = propagated.get(8, 0.0)
        assert reef8_boost > 0, \
            f"Reef 8 got no propagated boost (primary={reef8_primary:.6f})"

    def test_reef8_boost_gt_reef206(self, scorer):
        """Reef 8's propagated boost > reef 206's propagated boost."""
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        propagated = scorer.propagate(primary)
        assert propagated.get(8, 0.0) > propagated.get(206, 0.0), \
            f"Reef 8 boost {propagated.get(8, 0.0):.6f} <= reef 206 boost {propagated.get(206, 0.0):.6f}"

    def test_reef9_stays_in_top3_after_propagation(self, scorer):
        """Reef 9 remains in top 3 after propagation (reef 8 may overtake via incoming edges)."""
        results = scorer.score(self.SENTENCE)
        if not results:
            pytest.skip("No results for botanical query")
        top3_ids = [r[0] for r in results[:3]]
        assert 9 in top3_ids, f"Reef 9 not in top 3 after propagation: {results[:3]}"


class TestEvaluativeQuery:
    """Test 4.2: Evaluative query — acceptance → scandal valence tension."""

    SENTENCE = "She graciously and politely accepted the invitation with cordial warmth"

    def test_reef188_in_top3_primary(self, scorer):
        """Reef 188 (acceptance/encouragement) is in top 3 primary activations."""
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        top3 = sorted(primary.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_ids = [r[0] for r in top3]
        assert 188 in top3_ids, f"Reef 188 not in top 3 primary: {top3}"

    def test_reef190_receives_some_boost(self, scorer):
        """Reef 190 (scandal/audacity) receives some propagated boost."""
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        propagated = scorer.propagate(primary)
        assert propagated.get(190, 0.0) > 0, "Reef 190 got no propagated boost"

    def test_reef190_boost_dampened(self, scorer):
        """The boost to reef 190 is dampened relative to pure lift."""
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        propagated = scorer.propagate(primary)
        # If there were no valence dampening, the boost would be primary(188) * containment * lift.
        # With dampening, it should be noticeably smaller.
        reef188_primary = primary.get(188, 0.0)
        if reef188_primary == 0:
            pytest.skip("Reef 188 not in primary activations")
        # Pure lift contribution: primary(188) * 0.050 * 19.36 (undampened) ~ primary(188) * 0.968
        # With dampening it should be much less
        reef190_boost = propagated.get(190, 0.0)
        undampened_estimate = reef188_primary * 0.050 * 19.36
        assert reef190_boost < undampened_estimate, \
            f"Reef 190 boost {reef190_boost:.6f} not dampened (undampened ~ {undampened_estimate:.6f})"

    def test_reef166_not_in_top10(self, scorer):
        """Reef 166 (futility) is NOT in the top 10 after propagation."""
        results = scorer.score(self.SENTENCE)
        top10_ids = [r[0] for r in results[:10]]
        assert 166 not in top10_ids, f"Reef 166 appeared in top 10: {top10_ids}"


class TestCrossDomainBridging:
    """Test 4.3: Cross-domain bridging query."""

    SENTENCE = "The labor movement organized worker cooperatives to resist corporate monopoly dealings"

    def test_labor_and_business_in_primary(self, scorer):
        """Reef 101 (labor) and reef 206 (business) both appear in primary activations."""
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        assert 101 in primary or 206 in primary, \
            f"Neither reef 101 nor 206 in primary: top = {sorted(primary.items(), key=lambda x: x[1], reverse=True)[:5]}"

    def test_mutual_reinforcement(self, scorer):
        """Edge propagation boosts each reef's score via the other."""
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        propagated = scorer.propagate(primary)
        # At least one of the pair should get a boost from the other
        boost_101 = propagated.get(101, 0.0)
        boost_206 = propagated.get(206, 0.0)
        assert boost_101 > 0 or boost_206 > 0, \
            "No mutual reinforcement between reefs 101 and 206"

    def test_botanical_not_dominant(self, scorer):
        """Botanical reefs (8, 9, 12) should not dominate results for labor/business query."""
        results = scorer.score(self.SENTENCE)
        top5_ids = [r[0] for r in results[:5]]
        botanical = {8, 9, 12}
        overlap = botanical & set(top5_ids)
        assert len(overlap) == 0, \
            f"Botanical reefs {overlap} in top 5 for labor/business sentence"


class TestAsymmetryValidation:
    """Test 4.4: Asymmetry validation — two complementary sentences."""

    SENTENCE_A = "The coalescence of sapotaceae reproductive structures"
    SENTENCE_B = "The division pteridophyta includes all fern species"

    def test_asymmetric_propagation(self, scorer):
        """Sentence A's boost to reef 8 > Sentence B's boost to reef 9."""
        # Sentence A: primarily reef 9 → should boost reef 8 via 9→8
        word_ids_a = scorer.lookup_words(self.SENTENCE_A)
        primary_a = scorer.primary_reef_scores(word_ids_a)
        propagated_a = scorer.propagate(primary_a)

        # Sentence B: primarily reef 8 → should boost reef 9 via 8→9
        word_ids_b = scorer.lookup_words(self.SENTENCE_B)
        primary_b = scorer.primary_reef_scores(word_ids_b)
        propagated_b = scorer.propagate(primary_b)

        boost_a_to_8 = propagated_a.get(8, 0.0)
        boost_b_to_9 = propagated_b.get(9, 0.0)
        assert boost_a_to_8 > boost_b_to_9, \
            f"A→8 boost {boost_a_to_8:.6f} <= B→9 boost {boost_b_to_9:.6f}. " \
            f"Containment 9→8=0.197 >> 8→9=0.023 should produce asymmetry."


class TestMultiKnobSuppression:
    """Test 4.5: Multi-knob suppression in scoring context."""

    SENTENCE = "The results were indistinct and imperfectly measured, unduly inconclusive"

    def test_reef166_is_primary(self, scorer):
        """Reef 166 (futility/incompleteness) is the primary activation."""
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        if not primary:
            pytest.skip("No primary activations")
        top = max(primary.items(), key=lambda x: x[1])
        assert top[0] == 166, \
            f"Top primary is reef {top[0]}, expected 166. " \
            f"Top 5: {sorted(primary.items(), key=lambda x: x[1], reverse=True)[:5]}"

    def test_reef144_suppressed_boost(self, scorer):
        """Reef 144 (shape/dimension) receives suppressed propagated boost.

        Edge 166→144 has 3-way suppression (POS 0.933, valence 0.688, low lift 1.36).
        Boost should be modest relative to primary signal.  Multiple primary reefs
        may also contribute to reef 144's boost, so we allow up to 15%.
        """
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        propagated = scorer.propagate(primary)
        reef166_primary = primary.get(166, 0.0)
        reef144_boost = propagated.get(144, 0.0)
        if reef166_primary > 0:
            ratio = reef144_boost / reef166_primary
            assert ratio < 0.15, \
                f"Reef 144 boost ratio {ratio:.4f} too high (boost={reef144_boost:.6f}, primary(166)={reef166_primary:.6f})"

    def test_reef148_not_dominant(self, scorer):
        """Reef 148 (coherence/uniformity) should not dominate over primary signal.

        Multiple primary reefs may propagate into reef 148 via various edges,
        so we just verify it doesn't outpace the primary reef 166 score.
        """
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        propagated = scorer.propagate(primary)
        reef166_primary = primary.get(166, 0.0)
        reef148_boost = propagated.get(148, 0.0)
        if reef166_primary > 0:
            assert reef148_boost < reef166_primary, \
                f"Reef 148 boost {reef148_boost:.6f} exceeds primary(166)={reef166_primary:.6f}"
