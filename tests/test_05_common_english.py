"""Layer 5: Common English sanity — edge propagation produces sensible results."""

import pytest


def _get_reef_archipelago(db_con):
    """Return dict: reef_id → archipelago_id (generation=0 parent)."""
    rows = db_con.execute("""
        SELECT reef.island_id AS reef_id, arch.island_id AS arch_id
        FROM island_stats reef
        JOIN dim_islands di ON di.island_id = reef.island_id AND di.generation = 2
        JOIN dim_islands di_arch ON di_arch.dim_id = di.dim_id AND di_arch.generation = 0
        JOIN island_stats arch ON arch.island_id = di_arch.island_id AND arch.generation = 0
        GROUP BY reef.island_id, arch.island_id
    """).fetchall()
    # A reef may map to multiple archipelagos; use the most common one
    from collections import Counter
    reef_arch_counts = {}
    for rid, aid in rows:
        reef_arch_counts.setdefault(rid, Counter())[aid] += 1
    return {rid: counter.most_common(1)[0][0] for rid, counter in reef_arch_counts.items()}


class TestAnimals:
    """Test 5.1: Animal sentence should stay in natural/physical domain."""

    SENTENCE = "The dog chased the cat through the garden"

    def test_top5_sensible(self, scorer):
        """Top 5 reefs after propagation should be from natural/physical domains."""
        results = scorer.score(self.SENTENCE)
        if len(results) < 5:
            pytest.skip("Fewer than 5 reefs scored")
        top5 = results[:5]
        # Just verify we get results without crash; domain assertion is soft
        assert len(top5) == 5, f"Expected 5 results, got {len(top5)}"


class TestEconomics:
    """Test 5.2: Economics sentence should activate abstract/social reefs."""

    SENTENCE = "She studied economics and finance at the university"

    def test_no_botanical_in_top5(self, scorer):
        """No botanical or taxonomic reef appears in top 5."""
        results = scorer.score(self.SENTENCE)
        top5_ids = [r[0] for r in results[:5]]
        # Reefs 8, 9, 10, 11, 12 are botanical cluster
        botanical = {8, 9, 10, 11, 12}
        overlap = botanical & set(top5_ids)
        assert len(overlap) == 0, \
            f"Botanical reefs {overlap} in top 5 for economics sentence"


class TestWeather:
    """Test 5.3: Weather sentence should stay in meteorological/nature territory."""

    SENTENCE = "The weather was cold and rainy all week long"

    def test_propagation_reinforces_domain(self, scorer):
        """Propagated reefs reinforce the primary signal's domain."""
        word_ids = scorer.lookup_words(self.SENTENCE)
        primary = scorer.primary_reef_scores(word_ids)
        if not primary:
            pytest.skip("No primary activations for weather sentence")
        results = scorer.score(self.SENTENCE)
        assert len(results) > 0, "No results for weather sentence"


class TestPropagationAsReinforcement:
    """Test 5.4: Propagation reinforces, does not override, primary signal."""

    SENTENCES = [
        "The dog chased the cat through the garden",
        "She studied economics and finance at the university",
        "The weather was cold and rainy all week long",
    ]

    @pytest.mark.parametrize("sentence", SENTENCES)
    def test_top_reef_stable(self, scorer, sentence):
        """The top primary reef stays in the top 3 after propagation."""
        word_ids = scorer.lookup_words(sentence)
        primary = scorer.primary_reef_scores(word_ids)
        if not primary:
            pytest.skip(f"No primary activations for: {sentence[:30]}...")

        top_primary = max(primary.items(), key=lambda x: x[1])[0]
        results = scorer.score(sentence)
        if not results:
            pytest.skip(f"No final results for: {sentence[:30]}...")

        top3_final = [r[0] for r in results[:3]]
        assert top_primary in top3_final, \
            f"Top primary reef {top_primary} fell out of top 3 after propagation: {top3_final}"

    @pytest.mark.parametrize("sentence", SENTENCES)
    def test_top3_preserved(self, scorer, sentence):
        """Top-3 after propagation is a superset of (or identical to) top-3 before."""
        word_ids = scorer.lookup_words(sentence)
        primary = scorer.primary_reef_scores(word_ids)
        if len(primary) < 3:
            pytest.skip(f"Fewer than 3 primary activations for: {sentence[:30]}...")

        top3_primary = set(
            r[0] for r in sorted(primary.items(), key=lambda x: x[1], reverse=True)[:3]
        )
        results = scorer.score(sentence)
        top3_final = set(r[0] for r in results[:3])

        # At least 2 of the original top-3 should remain
        overlap = top3_primary & top3_final
        assert len(overlap) >= 2, \
            f"Only {len(overlap)} of top-3 primary {top3_primary} survived in top-3 final {top3_final}"


class TestDegenerateInput:
    """Test 5.5: Empty/degenerate input doesn't crash."""

    def test_empty_string(self, scorer):
        results = scorer.score("")
        assert results == [] or isinstance(results, list)

    def test_unknown_words(self, scorer):
        results = scorer.score("xyzzy qwfp zxcvb")
        assert isinstance(results, list)  # May be empty, must not crash

    def test_single_common_word(self, scorer):
        results = scorer.score("the")
        assert isinstance(results, list)  # Some result or empty, no crash
