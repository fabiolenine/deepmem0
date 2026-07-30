import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mem0.configs.base import MemoryConfig
from mem0.memory.main import AsyncMemory, Memory, _validate_and_trim_entity_id
from mem0.memory.utils import normalize_scope_id


@pytest.fixture(autouse=True)
def mock_openai():
    os.environ["OPENAI_API_KEY"] = "123"
    with patch("openai.OpenAI") as mock:
        mock.return_value = Mock()
        yield mock


@pytest.fixture
def memory_instance():
    with (
        patch("mem0.utils.factory.EmbedderFactory") as mock_embedder,
        patch("mem0.memory.main.VectorStoreFactory") as mock_vector_store,
        patch("mem0.utils.factory.LlmFactory") as mock_llm,
        patch("mem0.memory.telemetry.capture_event"),
    ):
        mock_embedder.create.return_value = Mock()
        mock_vector_store.create.return_value = Mock()
        mock_vector_store.create.return_value.search.return_value = []
        mock_llm.create.return_value = Mock()

        config = MemoryConfig(version="v1.1")
        return Memory(config)


@pytest.fixture
def memory_custom_instance():
    with (
        patch("mem0.utils.factory.EmbedderFactory") as mock_embedder,
        patch("mem0.memory.main.VectorStoreFactory") as mock_vector_store,
        patch("mem0.utils.factory.LlmFactory") as mock_llm,
        patch("mem0.memory.telemetry.capture_event"),
    ):
        mock_embedder.create.return_value = Mock()
        mock_vector_store.create.return_value = Mock()
        mock_vector_store.create.return_value.search.return_value = []
        mock_llm.create.return_value = Mock()

        config = MemoryConfig(
            version="v1.1",
            custom_instructions="custom prompt extracting memory in json format",
        )
        return Memory(config)


def test_add(memory_instance):
    memory_instance._add_to_vector_store = Mock(return_value=[{"memory": "Test memory", "event": "ADD"}])

    result = memory_instance.add(messages=[{"role": "user", "content": "Test message"}], user_id="test_user")

    assert "results" in result
    assert result["results"] == [{"memory": "Test memory", "event": "ADD"}]

    memory_instance._add_to_vector_store.assert_called_once_with(
        [{"role": "user", "content": "Test message"}], {"user_id": "test_user"}, {"user_id": "test_user"}, True, prompt=None
    )


def test_get(memory_instance):
    mock_memory = Mock(
        id="test_id",
        payload={
            "data": "Test memory",
            "user_id": "test_user",
            "hash": "test_hash",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-02T00:00:00",
            "extra_field": "extra_value",
        },
    )
    memory_instance.vector_store.get = Mock(return_value=mock_memory)

    result = memory_instance.get("test_id")

    assert result["id"] == "test_id"
    assert result["memory"] == "Test memory"
    assert result["user_id"] == "test_user"
    assert result["hash"] == "test_hash"
    assert result["created_at"] == "2023-01-01T00:00:00"
    assert result["updated_at"] == "2023-01-02T00:00:00"
    assert result["metadata"] == {"extra_field": "extra_value"}


def test_search(memory_instance):
    mock_memories = [
        Mock(id="1", payload={"data": "Memory 1", "user_id": "test_user"}, score=0.9),
        Mock(id="2", payload={"data": "Memory 2", "user_id": "test_user"}, score=0.8),
    ]
    memory_instance.vector_store.search = Mock(return_value=mock_memories)
    memory_instance.vector_store.keyword_search = Mock(return_value=None)  # No BM25
    memory_instance.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])

    with patch("mem0.memory.main.lemmatize_for_bm25", return_value="test query"), \
         patch("mem0.memory.main.extract_entities", return_value=[]):
        result = memory_instance.search("test query", filters={"user_id": "test_user"})

    assert "results" in result
    assert len(result["results"]) == 2
    assert result["results"][0]["id"] == "1"
    assert result["results"][0]["memory"] == "Memory 1"
    assert result["results"][0]["user_id"] == "test_user"
    # Score is now combined score (semantic only since no BM25/entity), still 0.9
    assert result["results"][0]["score"] == pytest.approx(0.9)

    # Hybrid pipeline over-fetches: max(20*4, 60) = 80 (top_k default is now 20)
    memory_instance.vector_store.search.assert_called_once_with(
        query="test query", vectors=[0.1, 0.2, 0.3], top_k=80, filters={"user_id": "test_user"}
    )


def test_update(memory_instance):
    memory_instance.embedding_model = Mock()
    memory_instance.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])

    memory_instance._update_memory = Mock()

    result = memory_instance.update("test_id", "Updated memory")

    memory_instance._update_memory.assert_called_once_with(
        "test_id", "Updated memory", {"Updated memory": [0.1, 0.2, 0.3]}, None
    )

    assert result["message"] == "Memory updated successfully!"


def test_update_with_metadata(memory_instance):
    memory_instance.embedding_model = Mock()
    memory_instance.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])

    memory_instance._update_memory = Mock()
    metadata = {"category": "sports", "priority": "high"}

    result = memory_instance.update("test_id", "Updated memory", metadata=metadata)

    memory_instance._update_memory.assert_called_once_with(
        "test_id", "Updated memory", {"Updated memory": [0.1, 0.2, 0.3]}, metadata
    )

    assert result["message"] == "Memory updated successfully!"


def test_update_with_empty_metadata(memory_instance):
    memory_instance.embedding_model = Mock()
    memory_instance.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])

    memory_instance._update_memory = Mock()

    memory_instance.update("test_id", "Updated memory", metadata={})

    memory_instance._update_memory.assert_called_once_with(
        "test_id", "Updated memory", {"Updated memory": [0.1, 0.2, 0.3]}, {}
    )


def test_delete(memory_instance):
    memory_instance._delete_memory = Mock()

    result = memory_instance.delete("test_id")

    # delete() now fetches the memory first and passes it to _delete_memory
    existing_memory = memory_instance.vector_store.get.return_value
    memory_instance._delete_memory.assert_called_once_with("test_id", existing_memory)
    assert result["message"] == "Memory deleted successfully!"


def test_delete_all(memory_instance):
    mock_memories = [Mock(id="1"), Mock(id="2")]
    memory_instance.vector_store.list = Mock(return_value=(mock_memories, None))
    memory_instance.vector_store.reset = Mock()
    memory_instance._delete_memory = Mock()

    result = memory_instance.delete_all(user_id="test_user")

    assert memory_instance._delete_memory.call_count == 2
    # Ensure the collection is NOT dropped — only matched memories should be removed
    memory_instance.vector_store.reset.assert_not_called()

    assert result["message"] == "Memories deleted successfully!"


def test_get_all(memory_instance):
    mock_memories = [Mock(id="1", payload={"data": "Memory 1", "user_id": "test_user"})]
    memory_instance.vector_store.list = Mock(return_value=(mock_memories, None))

    result = memory_instance.get_all(filters={"user_id": "test_user"})

    assert isinstance(result, dict)
    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["id"] == "1"
    assert result["results"][0]["memory"] == "Memory 1"
    assert result["results"][0]["user_id"] == "test_user"

    memory_instance.vector_store.list.assert_called_once_with(filters={"user_id": "test_user"}, top_k=20)


def test_no_telemetry_vector_store_when_disabled():
    """VectorStoreFactory should only be called once (for user data) when telemetry is disabled."""
    with (
        patch("mem0.memory.main.MEM0_TELEMETRY", False),
        patch("mem0.utils.factory.EmbedderFactory") as mock_embedder,
        patch("mem0.memory.main.VectorStoreFactory") as mock_vector_store,
        patch("mem0.utils.factory.LlmFactory") as mock_llm,
        patch("mem0.memory.telemetry.capture_event"),
    ):
        mock_embedder.create.return_value = Mock()
        mock_vector_store.create.return_value = Mock()
        mock_llm.create.return_value = Mock()

        config = MemoryConfig(version="v1.1")
        Memory(config)

        # VectorStoreFactory.create should be called exactly once — for user data only, not telemetry
        assert mock_vector_store.create.call_count == 1


def test_telemetry_vector_store_created_when_enabled():
    """VectorStoreFactory should be called twice (user data + telemetry) when telemetry is enabled."""
    with (
        patch("mem0.memory.main.MEM0_TELEMETRY", True),
        patch("mem0.utils.factory.EmbedderFactory") as mock_embedder,
        patch("mem0.memory.main.VectorStoreFactory") as mock_vector_store,
        patch("mem0.utils.factory.LlmFactory") as mock_llm,
        patch("mem0.memory.telemetry.capture_event"),
    ):
        mock_embedder.create.return_value = Mock()
        mock_vector_store.create.return_value = Mock()
        mock_llm.create.return_value = Mock()

        config = MemoryConfig(version="v1.1")
        Memory(config)

        # VectorStoreFactory.create should be called twice — user data + telemetry
        assert mock_vector_store.create.call_count == 2


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestEntityIdValidation:
    """Tests for entity ID validation (whitespace rejection and trimming)."""

    def test_search_rejects_whitespace_only_user_id(self, memory_instance):
        """Search should reject whitespace-only user_id in filters."""
        with pytest.raises(ValueError, match="Invalid user_id.*cannot be empty"):
            memory_instance.search("test query", filters={"user_id": "   "})

    def test_search_rejects_internal_whitespace_user_id(self, memory_instance):
        """Search should reject user_id with internal whitespace."""
        with pytest.raises(ValueError, match="Invalid user_id.*cannot contain whitespace"):
            memory_instance.search("test query", filters={"user_id": "user 123"})

    def test_search_rejects_tab_in_user_id(self, memory_instance):
        """Search should reject user_id with tab character."""
        with pytest.raises(ValueError, match="Invalid user_id.*cannot contain whitespace"):
            memory_instance.search("test query", filters={"user_id": "user\t123"})

    def test_get_all_rejects_whitespace_only_user_id(self, memory_instance):
        """get_all should reject whitespace-only user_id in filters."""
        with pytest.raises(ValueError, match="Invalid user_id.*cannot be empty"):
            memory_instance.get_all(filters={"user_id": "   "})

    def test_get_all_rejects_internal_whitespace_user_id(self, memory_instance):
        """get_all should reject user_id with internal whitespace."""
        with pytest.raises(ValueError, match="Invalid user_id.*cannot contain whitespace"):
            memory_instance.get_all(filters={"user_id": "user 123"})

    def test_add_rejects_whitespace_only_user_id(self, memory_instance):
        """add should reject whitespace-only user_id."""
        with pytest.raises(ValueError, match="Invalid user_id.*cannot be empty"):
            memory_instance.add("test message", user_id="   ")

    def test_add_rejects_internal_whitespace_user_id(self, memory_instance):
        """add should reject user_id with internal whitespace."""
        with pytest.raises(ValueError, match="Invalid user_id.*cannot contain whitespace"):
            memory_instance.add("test message", user_id="user 123")


SCOPE_NAMES = ["user_id", "agent_id", "run_id"]


def _async_memory():
    """An AsyncMemory with the factories stubbed, built the same way as the
    sync fixture. The async twin validates on its own — it does not delegate to
    the sync one — so it has to be exercised for real."""
    with (
        patch("mem0.utils.factory.EmbedderFactory"),
        patch("mem0.memory.main.VectorStoreFactory") as mock_vector_store,
        patch("mem0.utils.factory.LlmFactory"),
        patch("mem0.memory.telemetry.capture_event"),
    ):
        mock_vector_store.create.return_value = Mock()
        memory = AsyncMemory(MemoryConfig(version="v1.1"))
    memory.vector_store.list = Mock(return_value=([], None))
    return memory


class TestScopeIdNormalization:
    """``normalize_scope_id`` — the single scope-identity rule.

    The store filters on EXACT value, so writer and reader have to agree byte
    for byte. A scope that does not match yields an empty result, never an
    error, which is why these cases assert the MESSAGE and not merely that
    something was raised.
    """

    def test_import_under_test_is_the_tree_that_contains_this_test(self):
        """Guard: prove WHICH mem0 is under test, subprocess included.

        An editable install can point somewhere other than the checkout the
        tests live in, so a run started from the wrong directory exercises other
        code and reports the result as fact. Co-location of the two modules is
        not enough — both would satisfy it while coming from the wrong tree.

        The repository root is derived from THIS file, so the assertion is
        portable: it holds in any checkout and pins `mem0` to the same tree as
        the tests exercising it.
        """
        import mem0.memory.main as main_mod

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        expected = os.path.join(repo_root, "mem0", "memory", "main.py")

        assert os.path.realpath(main_mod.__file__) == os.path.realpath(expected), (
            f"mem0 under test is {main_mod.__file__}, but the tests live in "
            f"{repo_root} — the run is measuring another checkout"
        )

    @pytest.mark.parametrize("name", SCOPE_NAMES)
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (42, "42"),  # a database primary key is a legitimate id
            (0, "0"),  # falsy, and still a valid scope
            (-7, "-7"),
            ("alice", "alice"),
            ("  alice  ", "alice"),
            ("\talice\n", "alice"),
        ],
    )
    def test_accepts_and_normalizes(self, name, raw, expected):
        assert normalize_scope_id(raw, name) == expected

    @pytest.mark.parametrize("name", SCOPE_NAMES)
    def test_none_passes_through(self, name):
        """Absence of scope is the caller's decision, not this function's."""
        assert normalize_scope_id(None, name) is None

    @pytest.mark.parametrize("name", SCOPE_NAMES)
    @pytest.mark.parametrize(
        "raw,fragment",
        [
            (42.0, "expected str (or int), got float"),
            (True, "got bool"),
            (False, "got bool"),
            ({}, "expected str (or int), got dict"),
            ([], "expected str (or int), got list"),
            (b"x", "expected str (or int), got bytes"),
            ("   ", "cannot be empty or whitespace-only"),
            ("", "cannot be empty or whitespace-only"),
            ("a b", "cannot contain whitespace"),
            ("a\tb", "cannot contain whitespace"),
        ],
    )
    def test_rejects_with_actionable_message(self, name, raw, fragment):
        with pytest.raises(ValueError) as excinfo:
            normalize_scope_id(raw, name)
        message = str(excinfo.value)
        assert fragment in message
        # The message has to name the offending parameter, or the caller cannot
        # tell WHICH of three scope arguments was wrong.
        assert name in message

    def test_float_is_not_coerced_because_it_would_split_the_scope(self):
        """Why float is rejected instead of coerced, stated as a test.

        ``str(42.0)`` is ``"42.0"``, which never matches the scope ``"42"`` that
        the integer 42 writes. Coercing both would produce two scopes that look
        like one — silently.
        """
        assert normalize_scope_id(42, "user_id") == "42"
        with pytest.raises(ValueError, match="got float"):
            normalize_scope_id(42.0, "user_id")

    def test_bool_is_rejected_despite_being_an_int_subclass(self):
        """``isinstance(True, int)`` is True in Python.

        Without an explicit guard, ``True`` would be coerced to the scope
        ``"True"`` — a scope nobody ever wrote to.
        """
        assert isinstance(True, int)  # the trap this guard exists for
        with pytest.raises(ValueError, match="got bool"):
            normalize_scope_id(True, "user_id")

    def test_validator_alias_delegates(self):
        """The private validator has to stay a thin alias, not a second copy.

        Two implementations of one identity rule is how writer and reader drift
        apart.
        """
        for value in (42, "  alice ", None):
            assert _validate_and_trim_entity_id(value, "user_id") == normalize_scope_id(
                value, "user_id"
            )


class TestDeleteAllScopeValidation:
    """``delete_all`` normalizes scope before building the filter.

    A delete against the wrong scope is silent by construction: nothing matches,
    nothing is removed, and the call returns success. These tests assert the
    filter ACTUALLY handed to the store, not merely that no exception escaped.
    """

    @staticmethod
    def _arm(memory_instance):
        memory_instance.vector_store.list = Mock(return_value=([], None))
        return memory_instance

    def test_delete_all_coerces_integer_user_id_before_list(self, memory_instance):
        """Upstream's own assertion, kept so a future rebase cannot lose it."""
        self._arm(memory_instance).delete_all(user_id=42)

        assert memory_instance.vector_store.list.call_args.kwargs["filters"] == {
            "user_id": "42"
        }

    def test_delete_all_accepts_zero_as_a_scope(self, memory_instance):
        """``0`` is falsy and valid.

        Validation has to run BEFORE the truthiness test, or 0 is dropped and
        the call dies with "At least one filter is required" — the wrong error
        for the right defect.
        """
        self._arm(memory_instance).delete_all(user_id=0)

        assert memory_instance.vector_store.list.call_args.kwargs["filters"] == {
            "user_id": "0"
        }

    @pytest.mark.parametrize("name", SCOPE_NAMES)
    def test_delete_all_trims_padded_scope(self, memory_instance, name):
        """The reachable defect: a padded id matches nothing and deletes nothing."""
        self._arm(memory_instance).delete_all(**{name: " alice "})

        assert memory_instance.vector_store.list.call_args.kwargs["filters"] == {
            name: "alice"
        }

    @pytest.mark.parametrize("name", SCOPE_NAMES)
    @pytest.mark.parametrize(
        "raw,fragment",
        [
            ("a b", "cannot contain whitespace"),
            ("   ", "cannot be empty or whitespace-only"),
            (42.0, "expected str (or int), got float"),
            (True, "got bool"),
        ],
    )
    def test_delete_all_rejects_invalid_scope(self, memory_instance, name, raw, fragment):
        self._arm(memory_instance)

        with pytest.raises(ValueError) as excinfo:
            memory_instance.delete_all(**{name: raw})

        assert fragment in str(excinfo.value)
        assert name in str(excinfo.value)
        # It has to fail BEFORE touching the store: a rejected scope that had
        # already listed rows would mean the guard runs too late to protect
        # anything.
        memory_instance.vector_store.list.assert_not_called()

    def test_delete_all_still_requires_at_least_one_scope(self, memory_instance):
        """Normalizing must not swallow the no-scope error."""
        self._arm(memory_instance)

        with pytest.raises(ValueError, match="At least one filter is required"):
            memory_instance.delete_all()

    @pytest.mark.parametrize("name", SCOPE_NAMES)
    def test_async_delete_all_trims_padded_scope(self, name):
        """The async twin validates on its own.

        Parameterized over EVERY scope name on purpose: the twin normalizes
        with three separate lines, so dropping or mistyping one of them is
        invisible to a test that only ever passes ``user_id``.

        Driven through ``asyncio.run`` rather than a marker so the assertion
        does not depend on the plugin's mode configuration.
        """
        memory = _async_memory()
        memory._bulk_clear_entity_store = AsyncMock(return_value=True)

        asyncio.run(memory.delete_all(**{name: " alice "}))

        assert memory.vector_store.list.call_args_list[0].kwargs["filters"] == {
            name: "alice"
        }

    @pytest.mark.parametrize("name", SCOPE_NAMES)
    def test_async_delete_all_coerces_integer_scope(self, name):
        memory = _async_memory()
        memory._bulk_clear_entity_store = AsyncMock(return_value=True)

        asyncio.run(memory.delete_all(**{name: 42}))

        assert memory.vector_store.list.call_args_list[0].kwargs["filters"] == {
            name: "42"
        }

    @pytest.mark.parametrize("name", SCOPE_NAMES)
    @pytest.mark.parametrize(
        "raw,fragment",
        [
            ("a b", "cannot contain whitespace"),
            ("   ", "cannot be empty or whitespace-only"),
            (3.5, "expected str (or int), got float"),
            (True, "got bool"),
        ],
    )
    def test_async_delete_all_rejects_invalid_scope(self, name, raw, fragment):
        memory = _async_memory()

        with pytest.raises(ValueError) as excinfo:
            asyncio.run(memory.delete_all(**{name: raw}))

        assert fragment in str(excinfo.value)
        # Naming the parameter is what tells the caller WHICH of the three
        # arguments was wrong — and what catches a copy-paste that validates
        # agent_id under the label "user_id".
        assert name in str(excinfo.value)
        memory.vector_store.list.assert_not_called()

    def test_async_delete_all_survives_an_empty_scope(self):
        """Regression: the async twin crashed when the FIRST page was empty.

        ``hit_page_cap`` was assigned only at the end of the loop body, so the
        ``break`` on an empty first page skipped it and skipped the ``else``
        too — ``UnboundLocalError`` further down. An empty scope is the common
        case (wrong id, scope already drained), and it went unnoticed because
        nothing in production calls ``delete_all``.
        """
        memory = _async_memory()
        memory._bulk_clear_entity_store = AsyncMock(return_value=True)

        result = asyncio.run(memory.delete_all(user_id="nobody"))

        assert result["message"] == "Memories deleted successfully!"


class TestSearchParamValidation:
    """Tests for search parameter validation (threshold and top_k)."""

    @pytest.mark.parametrize("query", ["", "   ", "\n\t"])
    def test_search_rejects_empty_query(self, memory_instance, query):
        """Search should reject empty or whitespace-only queries before retrieval work."""
        memory_instance.embedding_model.embed = Mock()

        with pytest.raises(ValueError, match="Invalid query.*empty or whitespace-only"):
            memory_instance.search(query, filters={"user_id": "test"})

        memory_instance.embedding_model.embed.assert_not_called()

    def test_search_trims_query_before_embedding(self, memory_instance):
        """Search should normalize leading/trailing whitespace before embedding."""
        mock_memories = []
        memory_instance.vector_store.search = Mock(return_value=mock_memories)
        memory_instance.vector_store.keyword_search = Mock(return_value=None)
        memory_instance.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])

        with patch("mem0.memory.main.lemmatize_for_bm25", return_value="test"), \
             patch("mem0.memory.main.extract_entities", return_value=[]):
            memory_instance.search("  test  ", filters={"user_id": "test"})

        memory_instance.embedding_model.embed.assert_called_once_with("test", "search")

    def test_search_rejects_threshold_above_1(self, memory_instance):
        """Search should reject threshold > 1."""
        with pytest.raises(ValueError, match="Invalid threshold.*Must be between 0 and 1"):
            memory_instance.search("test query", filters={"user_id": "test"}, threshold=1.5)

    def test_search_rejects_negative_threshold(self, memory_instance):
        """Search should reject negative threshold."""
        with pytest.raises(ValueError, match="Invalid threshold.*Must be between 0 and 1"):
            memory_instance.search("test query", filters={"user_id": "test"}, threshold=-0.5)

    def test_search_rejects_negative_top_k(self, memory_instance):
        """Search should reject negative top_k."""
        with pytest.raises(ValueError, match="Invalid top_k.*Must be a non-negative"):
            memory_instance.search("test query", filters={"user_id": "test"}, top_k=-5)

    def test_get_all_rejects_negative_top_k(self, memory_instance):
        """get_all should reject negative top_k."""
        with pytest.raises(ValueError, match="Invalid top_k.*Must be a non-negative"):
            memory_instance.get_all(filters={"user_id": "test"}, top_k=-1)

    def test_search_accepts_threshold_zero(self, memory_instance):
        """Search should accept threshold=0 (edge case)."""
        mock_memories = []
        memory_instance.vector_store.search = Mock(return_value=mock_memories)
        memory_instance.vector_store.keyword_search = Mock(return_value=None)
        memory_instance.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])

        with patch("mem0.memory.main.lemmatize_for_bm25", return_value="test"), \
             patch("mem0.memory.main.extract_entities", return_value=[]):
            result = memory_instance.search("test", filters={"user_id": "test"}, threshold=0)

        assert "results" in result

    def test_search_accepts_threshold_one(self, memory_instance):
        """Search should accept threshold=1.0 (edge case)."""
        mock_memories = []
        memory_instance.vector_store.search = Mock(return_value=mock_memories)
        memory_instance.vector_store.keyword_search = Mock(return_value=None)
        memory_instance.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])

        with patch("mem0.memory.main.lemmatize_for_bm25", return_value="test"), \
             patch("mem0.memory.main.extract_entities", return_value=[]):
            result = memory_instance.search("test", filters={"user_id": "test"}, threshold=1.0)

        assert "results" in result

    def test_search_accepts_top_k_zero(self, memory_instance):
        """Search should accept top_k=0."""
        mock_memories = []
        memory_instance.vector_store.search = Mock(return_value=mock_memories)
        memory_instance.vector_store.keyword_search = Mock(return_value=None)
        memory_instance.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])

        with patch("mem0.memory.main.lemmatize_for_bm25", return_value="test"), \
             patch("mem0.memory.main.extract_entities", return_value=[]):
            result = memory_instance.search("test", filters={"user_id": "test"}, top_k=0)

        assert "results" in result
