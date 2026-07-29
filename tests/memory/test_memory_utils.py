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
        invent recovery. `Mastercard`'s real shape: shrapnel + one surviving id.
        Stripping shrapnel is a deployment-specific decision (repair script), not
        this helper's job — ids are opaque here."""
        raw = ["'", "-", "0", "a1c4b2ad-727c-45f0-b71a-b30842c4dcd2", "a"]
        assert normalize_linked_memory_ids(raw) == raw

    def test_malformed_string_never_raises(self):
        for bad in ["[not python", "[1,", "['unterminated", "[[[[[[[[", "{'a': 1}"]:
            assert isinstance(normalize_linked_memory_ids(bad), list)

    def test_idempotent(self):
        once = normalize_linked_memory_ids("['a', 'b']")
        assert normalize_linked_memory_ids(once) == once
