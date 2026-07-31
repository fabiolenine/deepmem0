import logging

import pytest
from unittest.mock import Mock

from mem0.memory.utils import (
    normalize_linked_memory_ids,
    parse_messages,
    parse_vision_messages,
    remove_spaces_from_entities,
    sanitize_relationship_for_cypher,
)


class TestParseMessages:
    def test_skips_message_without_content_key(self):
        # Reproduces #5067: a FunctionCalling assistant message carries
        # `tool_calls` but no `content` key -> used to raise KeyError.
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "search"}}]},
            {"role": "assistant", "content": "done"},
        ]
        result = parse_messages(messages)
        assert result == "user: hi\nassistant: done\n"

    def test_skips_explicit_none_content(self):
        messages = [{"role": "assistant", "content": None}, {"role": "user", "content": "ok"}]
        assert parse_messages(messages) == "user: ok\n"

    def test_plain_roles_pass_through(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        assert parse_messages(messages) == "system: sys\nuser: u\nassistant: a\n"


class TestParseVisionMessages:
    def test_skips_message_without_content_key(self):
        # Reproduces #5067 for the vision parser path.
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "search"}}]},
        ]
        result = parse_vision_messages(messages, llm=None)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hi"}

    def test_multimodal_list_without_llm_extracts_text(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]},
        ]
        result = parse_vision_messages(messages, llm=None)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What is this?"

    def test_image_dict_without_llm_is_skipped(self):
        messages = [
            {"role": "user", "content": {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}},
            {"role": "user", "content": "hello"},
        ]
        result = parse_vision_messages(messages, llm=None)
        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_multimodal_with_llm_calls_generate_response(self):
        mock_llm = Mock()
        mock_llm.generate_response.return_value = "A photo of a cat"
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ]},
        ]
        result = parse_vision_messages(messages, llm=mock_llm, vision_details="auto")
        assert result[0]["content"] == "A photo of a cat"
        mock_llm.generate_response.assert_called_once()

    def test_image_only_list_without_llm_is_skipped(self):
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]},
        ]
        result = parse_vision_messages(messages, llm=None)
        assert result == []

    def test_plain_text_messages_pass_through(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = parse_vision_messages(messages, llm=None)
        assert result == messages

    # ------------------------------------------------------------------
    # Structural guards on image parts (vision ENABLED).
    #
    # Two regimes, deliberately asymmetric:
    #   llm is None -> TOTAL for a well-formed message: never raises, drops
    #                  image parts but says so.
    #   llm given   -> STRICT: a structurally invalid image part raises
    #                  ValueError BEFORE an LLM call is spent, and a provider
    #                  exception propagates with its original type and chain.
    #
    # ValueError (not KeyError/TypeError by accident) because consumers class-
    # ify failures by exception type: a broken payload must be poison, failing
    # once and permanently, never an infra blip worth retrying.
    # ------------------------------------------------------------------

    def test_malformed_image_dict_raises_value_error(self):
        # A malformed image part (missing the nested url) used to raise an
        # uncaught KeyError that aborted add(); it should raise a clear ValueError.
        mock_llm = Mock()
        messages = [{"role": "user", "content": {"type": "image_url", "image_url": {}}}]
        with pytest.raises(ValueError, match=r"missing image_url\.url"):
            parse_vision_messages(messages, llm=mock_llm)
        mock_llm.generate_response.assert_not_called()

    def test_none_image_url_raises_value_error(self):
        # image_url present but None (or any non-dict) must also raise the clear
        # ValueError, not an AttributeError from calling .get() on None.
        mock_llm = Mock()
        messages = [{"role": "user", "content": {"type": "image_url", "image_url": None}}]
        with pytest.raises(ValueError, match=r"missing image_url\.url"):
            parse_vision_messages(messages, llm=mock_llm)
        mock_llm.generate_response.assert_not_called()

    def test_bare_string_image_url_raises_value_error(self):
        # `image_url` as a bare string is not the canonical OpenAI shape; it used
        # to raise `TypeError: string indices must be integers`.
        mock_llm = Mock()
        messages = [{"role": "user", "content": {
            "type": "image_url", "image_url": "data:image/png;base64,AAA"}}]
        with pytest.raises(ValueError, match=r"missing image_url\.url"):
            parse_vision_messages(messages, llm=mock_llm)
        mock_llm.generate_response.assert_not_called()

    def test_malformed_image_part_in_list_raises_value_error(self):
        # The LIST branch is the canonical OpenAI multimodal shape -- and the very
        # shape `get_image_description` builds -- yet it had no guard at all: a
        # malformed part sailed through to the provider.
        mock_llm = Mock()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {}},
        ]}]
        with pytest.raises(ValueError, match=r"missing image_url\.url"):
            parse_vision_messages(messages, llm=mock_llm)
        mock_llm.generate_response.assert_not_called()

    def test_missing_image_url_key_in_list_part_raises_value_error(self):
        mock_llm = Mock()
        messages = [{"role": "user", "content": [{"type": "image_url"}]}]
        with pytest.raises(ValueError, match=r"missing image_url\.url"):
            parse_vision_messages(messages, llm=mock_llm)
        mock_llm.generate_response.assert_not_called()

    def test_second_image_part_is_also_prevalidated(self):
        # Proves EVERY image part is validated, not just the first one: a valid
        # image followed by a malformed one must still raise before any LLM call.
        mock_llm = Mock()
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/ok.png"}},
            {"type": "image_url", "image_url": {"url": ""}},
        ]}]
        with pytest.raises(ValueError, match=r"missing image_url\.url"):
            parse_vision_messages(messages, llm=mock_llm)
        mock_llm.generate_response.assert_not_called()

    def test_provider_valueerror_propagates_unchanged(self):
        # The provider's own exception IS the diagnosis: mem0/llms/ollama.py
        # raises actionable ValueErrors ("only base64 data URIs are supported",
        # "http(s) image URLs are not supported"). Wrapping them in
        # `Exception(f"Error while downloading {image_url}.")` did two harms: it
        # lied (nothing is downloaded for a data URI or a local path) and it
        # erased the exception class consumers use to tell a poison payload from
        # sick infrastructure -- turning a permanent failure into N retried adds.
        # Identity implies type, message and traceback in one assertion.
        mock_llm = Mock()
        boom = ValueError("ollama vision: http(s) image URLs are not supported")
        mock_llm.generate_response.side_effect = boom
        messages = [{"role": "user", "content": {
            "type": "image_url", "image_url": {"url": "https://example.com/img.png"}}}]
        with pytest.raises(ValueError) as excinfo:
            parse_vision_messages(messages, llm=mock_llm, vision_details="auto")
        assert excinfo.value is boom

    def test_infrastructure_error_keeps_its_class(self):
        # Mirror of the test above, and the reason the fix is "stop interfering"
        # rather than "convert everything to ValueError": a transient infra error
        # must NOT be laundered into a poison-classified one either.
        class _ConnError(Exception):
            pass

        mock_llm = Mock()
        mock_llm.generate_response.side_effect = _ConnError("connection refused")
        messages = [{"role": "user", "content": {
            "type": "image_url", "image_url": {"url": "https://example.com/img.png"}}}]
        with pytest.raises(_ConnError):
            parse_vision_messages(messages, llm=mock_llm)

    # ------------------------------------------------------------------
    # Vision DISABLED: dropping an image is a fact the caller sent and we did
    # not store, so it is never silent.
    # ------------------------------------------------------------------

    def test_image_dict_without_llm_warns_about_the_discard(self, caplog):
        messages = [{"role": "user", "content": {
            "type": "image_url", "image_url": {"url": "https://example.com/img.png"}}}]
        with caplog.at_level(logging.WARNING, logger="mem0.memory.utils"):
            result = parse_vision_messages(messages, llm=None)
        assert result == []
        assert "vision is disabled" in caplog.text

    def test_image_only_list_without_llm_warns_the_message_is_gone(self, caplog):
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}]}]
        with caplog.at_level(logging.WARNING, logger="mem0.memory.utils"):
            result = parse_vision_messages(messages, llm=None)
        assert result == []
        assert "discarded the whole" in caplog.text

    def test_multimodal_list_without_llm_warns_but_keeps_text(self, caplog):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]}]
        with caplog.at_level(logging.WARNING, logger="mem0.memory.utils"):
            result = parse_vision_messages(messages, llm=None)
        assert result[0]["content"] == "What is this?"
        assert "dropped 1 image part(s)" in caplog.text

    # ------------------------------------------------------------------
    # Hostile input: the "never raises" invariant holds for WELL-FORMED
    # messages only. A malformed container still surfaces as AttributeError/
    # TypeError -- and that is correct, both are poison-classified.
    # ------------------------------------------------------------------

    def test_non_string_text_part_does_not_crash_the_join(self):
        # `" ".join()` over a non-string text part used to raise TypeError.
        messages = [{"role": "user", "content": [
            {"type": "text", "text": 42},
            {"type": "text", "text": "real text"},
        ]}]
        result = parse_vision_messages(messages, llm=None)
        assert result[0]["content"] == "real text"

    def test_empty_text_part_is_preserved_like_upstream(self):
        # Guards the boundary of the isinstance(text, str) filter: it exists to
        # stop a non-string from blowing up the join, NOT to prune empty strings.
        # Upstream kept them, so a message made only of an empty text part still
        # survives; dropping it would be a silent behavior change beyond scope.
        messages = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        result = parse_vision_messages(messages, llm=None)
        assert result == [{"role": "user", "content": ""}]

    def test_non_dict_message_surfaces_as_poison_classified_error(self):
        # Documents the boundary of the invariant: a non-dict message is a
        # malformed CONTAINER, not a malformed image part. It raises
        # AttributeError, which consumers classify as poison -- fail once,
        # permanently -- which is the right outcome. Never silently skipped.
        with pytest.raises(AttributeError):
            parse_vision_messages(["not a dict"], llm=None)


class TestRemoveSpacesFromEntities:
    """
    Covers behavior used by Neo4j, Memgraph (sanitize_relationship=True),
    Kuzu, and Neptune (sanitize_relationship=False). All backends delegate here.
    """

    @pytest.mark.parametrize(
        "sanitize",
        [True, False],
        ids=["cypher_sanitized", "plain"],
    )
    def test_filters_empty_and_incomplete_dicts(self, sanitize):
        mixed = [
            {},
            {"source": "a"},
            {"source": "a", "relationship": "r"},
            {"source": "x", "relationship": "rel", "destination": "y"},
        ]
        out = remove_spaces_from_entities(mixed, sanitize_relationship=sanitize)
        assert len(out) == 1
        assert out[0]["source"] == "x"
        assert out[0]["destination"] == "y"

    @pytest.mark.parametrize("sanitize", [True, False])
    def test_all_empty_returns_empty(self, sanitize):
        assert remove_spaces_from_entities([{}, {}, {}], sanitize_relationship=sanitize) == []

    def test_skips_non_dict_entries(self):
        assert remove_spaces_from_entities([None, "not-a-dict", 123, {"source": "a", "relationship": "r", "destination": "b"}]) == [
            {"source": "a", "relationship": "r", "destination": "b"}
        ]

    def test_sanitize_true_relationship_uses_sanitizer(self):
        """Neo4j / Memgraph path: special characters mapped via sanitize_relationship_for_cypher."""
        entities = [{"source": "A", "relationship": "x/y", "destination": "B"}]
        out = remove_spaces_from_entities(entities, sanitize_relationship=True)
        assert out[0]["relationship"] == sanitize_relationship_for_cypher("x/y".lower().replace(" ", "_"))

    def test_sanitize_false_relationship_plain_only(self):
        """Kuzu / Neptune path: only lowercase and spaces to underscores."""
        entities = [{"source": "A", "relationship": "Works At", "destination": "B Co"}]
        out = remove_spaces_from_entities(entities, sanitize_relationship=False)
        assert out[0]["relationship"] == "works_at"
        assert out[0]["source"] == "a"
        assert out[0]["destination"] == "b_co"

    def test_sanitize_true_vs_false_slash_in_relationship(self):
        """Slash is rewritten when sanitizing (Cypher path); kept as-is for plain path."""
        base = {"source": "s", "relationship": "a/b", "destination": "d"}
        t = remove_spaces_from_entities([dict(base)], sanitize_relationship=True)[0]["relationship"]
        f = remove_spaces_from_entities([dict(base)], sanitize_relationship=False)[0]["relationship"]
        assert t == sanitize_relationship_for_cypher("a/b")
        assert f == "a/b"


class TestNormalizeLinkedMemoryIds:
    """Entity payload `linked_memory_ids` coercion.

    Regression origin: an ad-hoc repair script stored `str(list)` into the field.
    The next writer ran `set(payload.get("linked_memory_ids", []))`, which iterates
    a string CHARACTER BY CHARACTER instead of raising, permanently replacing an
    entity's real links with punctuation and hex digits.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (["a", "b"], ["a", "b"]),
            ([], []),
            (None, []),
            ("['a', 'b']", ["a", "b"]),          # the corruption, recovered
            ('["a", "b"]', ["a", "b"]),          # JSON-ish repr
            ("('a', 'b')", ["a", "b"]),          # tuple repr
            ("[]", []),
            ("a", ["a"]),                        # bare string == one id
            ("  a  ", ["a"]),
            ("", []),
            (("a", "b"), ["a", "b"]),
            ({"b", "a"}, ["a", "b"]),            # set -> stable order
            (["a", None, "", "b"], ["a", "b"]),  # holes dropped
            (42, []),
            ({"a": 1}, []),
            ([1, 2], ["1", "2"]),                # non-str ids stringified
        ],
    )
    def test_shapes(self, raw, expected):
        assert normalize_linked_memory_ids(raw) == expected

    def test_the_regression_by_name(self):
        """The whole point: normalizing must NOT be character iteration."""
        corrupt = "['a', 'b']"
        assert normalize_linked_memory_ids(corrupt) == ["a", "b"]
        assert normalize_linked_memory_ids(corrupt) != list(corrupt)
        assert normalize_linked_memory_ids(corrupt) != sorted(set(corrupt))

    def test_real_production_payload_is_recovered(self):
        """The exact surviving `Brasília` value from the production corpus."""
        raw = "['847c8849-9feb-4a82-b242-281aecd75ed2', '98831cc8-83d6-452b-b149-e003b008ce11']"
        assert normalize_linked_memory_ids(raw) == [
            "847c8849-9feb-4a82-b242-281aecd75ed2",
            "98831cc8-83d6-452b-b149-e003b008ce11",
        ]

    def test_already_exploded_list_is_left_alone(self):
        """Once a row is char-exploded the damage is done; the helper must not
        invent recovery. This is the shape a real corrupted row takes: character
        shrapnel plus one id that survived whole. Stripping shrapnel is a
        deployment-specific decision (repair script), not this helper's job —
        ids are opaque here."""
        raw = ["'", "-", "0", "a1c4b2ad-727c-45f0-b71a-b30842c4dcd2", "a"]
        assert normalize_linked_memory_ids(raw) == raw

    def test_malformed_string_never_raises(self):
        for bad in ["[not python", "[1,", "['unterminated", "[[[[[[[[", "{'a': 1}"]:
            assert isinstance(normalize_linked_memory_ids(bad), list)

    def test_idempotent(self):
        once = normalize_linked_memory_ids("['a', 'b']")
        assert normalize_linked_memory_ids(once) == once
