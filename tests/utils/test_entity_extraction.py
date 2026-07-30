import pytest


@pytest.fixture(autouse=True)
def _ensure_spacy():
    """Skip tests if spaCy model is not available."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except Exception:
        pytest.skip("spaCy en_core_web_sm model not available")


class TestExtractEntities:
    def test_proper_nouns(self):
        from mem0.utils.entity_extraction import extract_entities

        entities = extract_entities("John Smith works at Google on machine learning projects")
        entity_texts = [e[1] for e in entities]
        # Should extract proper nouns
        found_proper = any("John" in t or "Google" in t for t in entity_texts)
        assert found_proper, f"Expected proper nouns, got {entities}"

    def test_quoted_text(self):
        from mem0.utils.entity_extraction import extract_entities

        entities = extract_entities('She is reading "The Great Gatsby" this week')
        entity_texts = [e[1] for e in entities]
        assert any("Great Gatsby" in t for t in entity_texts), f"Expected quoted text, got {entities}"

    def test_compound_nouns(self):
        from mem0.utils.entity_extraction import extract_entities

        entities = extract_entities("The machine learning engineer built a neural network")
        entity_texts = [e[1].lower() for e in entities]
        has_compound = any("machine" in t and "learning" in t for t in entity_texts) or \
                       any("neural" in t and "network" in t for t in entity_texts)
        assert has_compound, f"Expected compound nouns, got {entities}"

    def test_empty_string(self):
        from mem0.utils.entity_extraction import extract_entities

        entities = extract_entities("")
        assert entities == []

    def test_no_entities(self):
        from mem0.utils.entity_extraction import extract_entities

        entities = extract_entities("I like things and stuff")
        # Generic words should be filtered out
        entity_texts = [e[1].lower() for e in entities]
        assert "things" not in entity_texts
        assert "stuff" not in entity_texts

    def test_deduplication(self):
        from mem0.utils.entity_extraction import extract_entities

        entities = extract_entities("Google is great. I love working at Google.")
        google_count = sum(1 for _, t in entities if "Google" in t)
        assert google_count <= 1, f"Expected dedup, got {entities}"

    def test_substring_dedup_respects_word_boundaries(self):
        from mem0.utils.entity_extraction import extract_entities

        # "Sam" is a mid-word substring of "Samsung", not a separate token, so it
        # must not be dropped as a substring of the longer entity.
        entities = extract_entities("At Samsung, Sam leads design.")
        entity_texts = [e[1] for e in entities]
        assert "Sam" in entity_texts, f"Expected 'Sam' to survive alongside 'Samsung', got {entities}"
        assert any("Samsung" in t for t in entity_texts), f"Expected 'Samsung', got {entities}"

    def test_returns_tuples(self):
        from mem0.utils.entity_extraction import extract_entities

        entities = extract_entities("John Smith lives in New York City")
        for entity in entities:
            assert isinstance(entity, tuple)
            assert len(entity) == 2
            assert entity[0] in ("PROPER", "QUOTED", "COMPOUND", "NOUN")
            assert isinstance(entity[1], str)


class TestExtractEntitiesBatch:
    def test_batch_processing(self):
        from mem0.utils.entity_extraction import extract_entities_batch

        texts = [
            "John works at Google",
            "Mary lives in Paris",
            "The cat sat on the mat",
        ]
        results = extract_entities_batch(texts)
        assert len(results) == 3
        assert isinstance(results[0], list)
        assert isinstance(results[1], list)
        assert isinstance(results[2], list)

    def test_empty_input(self):
        from mem0.utils.entity_extraction import extract_entities_batch

        assert extract_entities_batch([]) == []

    def test_consistency_with_single(self):
        from mem0.utils.entity_extraction import extract_entities, extract_entities_batch

        text = "John Smith works at Google headquarters"
        single = extract_entities(text)
        batch = extract_entities_batch([text])
        assert len(batch) == 1
        # Both should extract the same entities
        assert set(t for _, t in single) == set(t for _, t in batch[0])


class TestSpanHygiene:
    """Span hygiene rules (N1).

    These four rules changed the output of four English cases and the existing
    eleven tests in this file stayed green — the behaviour they cover simply was
    not covered. A change in extractor output with no test in the repo where the
    extractor lives is how the `Northwind` regression survived in production:
    the two memories a human would call the answer had no entity link, and
    nothing failed.
    """

    def test_preposition_does_not_glue_two_entities(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities("Alice worked at Northwind in Sao Paulo.")}
        assert "Northwind" in spans
        assert "Sao Paulo" in spans
        assert not any("Northwind in" in s for s in spans), (
            "`in` back in the connector whitelist re-glues distinct entities")

    def test_proper_survives_a_longer_span_containing_it(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities("Sam bought a Samsung phone.")}
        assert "Samsung" in spans, (
            "substring suppression deleted a PROPER — the short span is the one "
            "people search for")

    def test_mis_tagged_verb_head_no_longer_ends_a_span(self):
        """The fallback branch used to emit `compounds + [verb]`, producing
        `Meridian faz`. It now emits the compounds only."""
        from mem0.utils.entity_extraction import extract_entities

        spans = [t for _, t in extract_entities("The Meridian project faz a FFT.")]
        assert "Meridian" in spans
        assert not any(s.endswith(" faz") for s in spans), spans

    @pytest.mark.xfail(
        reason="Needs a Portuguese model (N3). The English pipeline tags the "
               "Portuguese auxiliary `foi` as NOUN, so it survives the "
               "noun-chunk content filter — every POS-keyed hygiene rule is "
               "inert on Portuguese text. Measured: 'Ontem eu viajei para "
               "Recife' tags ALL five tokens PROPN, including the pronoun and "
               "the verb.",
        strict=True,
    )
    def test_portuguese_verb_inside_noun_chunk_span(self):
        from mem0.utils.entity_extraction import extract_entities

        for _, span in extract_entities("Mem0 foi concluída ontem."):
            assert " foi " not in span, span

    def test_caps_reject_clause_length_spans(self):
        from mem0.utils.entity_extraction import (
            MAX_ENTITY_CHARS,
            MAX_ENTITY_TOKENS,
            extract_entities,
        )

        long_text = ("A definição de harness segundo o artigo de Martin Fowler "
                     "sobre engenharia de software moderna")
        for _, span in extract_entities(long_text):
            assert len(span.split()) <= MAX_ENTITY_TOKENS, span
            assert len(span) <= MAX_ENTITY_CHARS, span

    def test_internal_name_words_are_kept(self):
        """`of`/`the`/`and` are internal to real names and must stay."""
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities("He joined the Bank of America team.")}
        assert any("Bank of America" in s for s in spans), spans
