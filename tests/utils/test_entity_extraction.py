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

    def test_portuguese_verb_no_longer_survives_in_a_span(self):
        """Was an xfail(strict=True) waiting for a Portuguese model (N3).

        The English pipeline tagged the Portuguese auxiliary `foi` as NOUN, so
        it slipped through the noun-chunk content filter and every POS-keyed
        rule was inert. With `pt_core_news_sm` the same token tags AUX and the
        filter finally bites. The strict xfail existed so that landing the model
        would BREAK THE BUILD instead of leaving a silent gap; this is that
        break, resolved.
        """
        from mem0.utils.entity_extraction import extract_entities

        for _, span in extract_entities("Mem0 foi concluída ontem.", language="pt"):
            assert " foi " not in span, span

    def test_english_pipeline_still_cannot_do_it(self):
        """The limitation was the MODEL, not the code: without a Portuguese
        pipeline the same sentence still leaks. Pinning this keeps the reason
        for the language parameter visible."""
        from mem0.utils.entity_extraction import extract_entities

        spans = [t for _, t in extract_entities("Mem0 foi concluída ontem.")]
        assert any(" foi " in s for s in spans), spans

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


class TestAdjudicatedEnglishTransformations:
    """The four English outputs that changed, pinned as assertions.

    The span golden freezes English output to force inspection, and freezing is
    re-done deliberately after adjudication. That leaves a hole: a later change
    could quietly revert one of these and a fresh `--freeze-en` would bless it.
    These assertions are the immutable record — the golden can be re-frozen, this
    cannot be re-frozen.
    """

    def test_google_in_california_splits(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities("John works at Google in California.")}
        assert {"Google", "California"} <= spans, spans
        assert "Google in California" not in spans

    def test_quoted_title_also_yields_the_proper(self):
        from mem0.utils.entity_extraction import extract_entities

        got = extract_entities('She read "The Great Gatsby" last summer.')
        assert ("QUOTED", "The Great Gatsby") in got, got
        assert any(t == "PROPER" and "Great Gatsby" in e for t, e in got), got

    def test_brand_survives_alongside_its_compound(self):
        from mem0.utils.entity_extraction import extract_entities

        got = extract_entities("Sam bought a Samsung phone.")
        assert ("PROPER", "Samsung") in got, got
        assert any(t == "COMPOUND" and "Samsung" in e for t, e in got), got

    def test_employer_and_city_are_separate_entities(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in
                 extract_entities("Alice worked at Northwind in Sao Paulo as a Data Lead.")}
        assert {"Northwind", "Sao Paulo", "Data Lead"} <= spans, spans
        assert not any("Northwind in" in s for s in spans), spans


class TestLanguageAwareLexicon:
    """The lexicon is per-language (N2).

    The word lists were 100% English and ran over Portuguese text — the same
    mistake the fork already fixed for BM25 (`lemmatization.py`: "the English
    lemmatizer is noise, or worse, on non-English text"), never fixed for
    entities. Threading `language` through the batch path broke it on the first
    try and only the pre-existing batch tests caught it; these cover the
    parameter itself.
    """

    def test_generic_capitalized_word_survives_in_english(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities("A Fase 7 do Mem0 terminou.")}
        assert "Fase" in spans, "English lexicon must not know Portuguese words"

    def test_generic_capitalized_word_is_dropped_in_portuguese(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities("A Fase 7 do Mem0 terminou.",
                                                language="pt")}
        assert "Fase" not in spans, spans
        assert "Mem0" in spans, "dropping generics must not drop real names"

    def test_uppercase_emphasis_is_not_a_proper_noun(self):
        from mem0.utils.entity_extraction import extract_entities

        for palavra in ("CONCLUÍDO", "DECISÃO", "REMOVIDO"):
            spans = {t for _, t in
                     extract_entities(f"O item foi {palavra} ontem.", language="pt")}
            assert palavra not in spans, f"{palavra} is emphasis, not an entity"

    def test_uppercase_identifiers_and_acronyms_survive(self):
        """The first rule keyed on LENGTH (>=6) and deleted `PYTHONPATH`. An
        acronym or identifier does not inflect; an emphasised word does."""
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities(
            "O carregamento usa PYTHONPATH, FFT, HNSW e MT5.", language="pt")}
        for termo in ("PYTHONPATH", "FFT", "HNSW", "MT5"):
            assert termo in spans, f"{termo} sumiu: {spans}"

    def test_batch_honours_language(self):
        from mem0.utils.entity_extraction import extract_entities_batch

        texto = "A Fase 7 do Mem0 terminou."
        en = {t for _, t in extract_entities_batch([texto])[0]}
        pt = {t for _, t in extract_entities_batch([texto], language="pt")[0]}
        assert "Fase" in en and "Fase" not in pt, (en, pt)

    def test_batch_matches_single_for_the_same_language(self):
        from mem0.utils.entity_extraction import extract_entities, extract_entities_batch

        texto = "A Fase 7 do Mem0 terminou com DECISÃO tomada."
        assert (set(extract_entities(texto, language="pt"))
                == set(extract_entities_batch([texto], language="pt")[0]))

    def test_unknown_language_falls_back_to_english_and_warns(self, caplog):
        """Silence would reproduce the original defect: an English pipeline
        running over another language with nobody aware."""
        import logging

        from mem0.utils import entity_extraction as ee

        ee._LEXICONS_AVISADOS.discard("xx")
        with caplog.at_level(logging.WARNING, logger=ee.__name__):
            spans = {t for _, t in ee.extract_entities("A Fase 7 terminou.",
                                                       language="xx")}
        assert "Fase" in spans, "fallback must be the English behaviour"
        assert any("no lexicon for language" in r.message for r in caplog.records), \
            [r.message for r in caplog.records]


class TestNamedEntitiesAndIdentifiers:
    """`doc.ents` and technical identifiers (N3/N4).

    The extractor carried four hand-rolled branches and never consulted
    `doc.ents` — the trained recogniser of the very model it was already
    loading. And a technical corpus is full of lowercase identifiers that no
    capitalisation rule can reach.
    """

    def test_named_entities_are_used(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities(
            "Ontem eu viajei para Recife com a Alice.", language="pt")}
        assert {"Recife", "Alice"} <= spans, spans

    def test_sentence_initial_name_survives(self):
        """The `has_mid_cap` guard dropped it, because without a Portuguese
        model a sentence-initial capital is indistinguishable from a name."""
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities(
            "Northwind encerrou o projeto em dezembro.", language="pt")}
        assert "Northwind" in spans, spans

    def test_recogniser_span_containing_a_verb_is_rejected(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities(
            "O Docker Compose sobe o Qdrant e o docker compose derruba.",
            language="pt")}
        assert not any("derruba" in s for s in spans), spans

    def test_lowercase_identifiers_are_extracted(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities(
            "O num_ctx do llama-tiny3 subiu e o entity store guarda "
            "linked_memory_ids com embed-v2.", language="pt")}
        for termo in ("num_ctx", "llama-tiny3", "linked_memory_ids", "embed-v2"):
            assert termo in spans, f"{termo} sumiu: {spans}"

    def test_identifier_with_an_assigned_value(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities(
            "A decisão foi manter event_tie_band=0.002.", language="pt")}
        assert "event_tie_band" in spans, spans

    def test_slashed_path_is_kept_whole(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities(
            "O reranker BAAI/bge-reranker-v2-m3 roda em CPU.", language="pt")}
        assert "BAAI/bge-reranker-v2-m3" in spans, spans

    def test_hyphenated_common_words_are_not_identifiers(self):
        """A separator alone is not enough — a digit or an underscore is
        required, or `guarda-chuva` and `bem-vindo` would become entities."""
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities(
            "Comprei um guarda-chuva e disse bem-vindo.", language="pt")}
        assert "guarda-chuva" not in spans and "bem-vindo" not in spans, spans

    def test_portuguese_genitive_stays_inside_a_place_name(self):
        """N1 removed prepositions from the connector whitelist because in
        English `in`/`at` GLUED distinct entities. In Portuguese the genitive
        does the opposite — it is internal to the name."""
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities(
            "O voo sai de Sao Paulo com destino ao Rio de Janeiro.",
            language="pt")}
        assert "Rio de Janeiro" in spans, spans

    def test_section_label_is_not_an_entity(self):
        from mem0.utils.entity_extraction import extract_entities

        spans = {t for _, t in extract_entities(
            "A Fase 7 do add faz a ligação em lote.", language="pt")}
        assert "Fase 7" not in spans, spans

    def test_wrapper_span_is_dropped_but_legit_compound_survives(self):
        """Only connective tissue authorises dropping the longer span. The
        broad version of this rule ate `Samsung phone` and the verbatim quote
        `The Great Gatsby`."""
        from mem0.utils.entity_extraction import extract_entities

        pt = {t for _, t in extract_entities(
            "Alice trabalhou na Northwind in Sao Paulo.", language="pt")}
        assert "Northwind" in pt and "Northwind in Sao Paulo" not in pt, pt

        en = extract_entities("Sam bought a Samsung phone.")
        assert ("COMPOUND", "Samsung phone") in en, en
