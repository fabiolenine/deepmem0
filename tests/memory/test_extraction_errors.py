"""DeepMem0: extraction failures raise ExtractionError instead of returning [].

Measured in production (01-06/10/2026): 23 of 145 queued adds ended "done, no
new facts" because the model's reply was cut at the context limit. The input had
facts; upstream's parser read the truncated JSON as "nothing to store". These
tests pin the opposite: an unusable reply is an ERROR the queue can see, and
only a well-formed `{"memory": [...]}` is an answer.
"""

from unittest.mock import MagicMock

import pytest

from mem0.exceptions import ExtractionError, LLMError
from mem0.memory.main import AsyncMemory, Memory
from mem0.memory.utils import extraction_errors_swallowed, parse_extraction_response


def _setup_mocks(mocker):
    emb = mocker.MagicMock()
    emb.return_value.embed.return_value = [0.1, 0.2, 0.3]
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", emb)
    vs = mocker.MagicMock()
    vs.return_value.search.return_value = []
    mocker.patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[vs.return_value, mocker.MagicMock()])
    mocker.patch("mem0.utils.factory.LlmFactory.create", mocker.MagicMock())
    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())
    mocker.patch("mem0.memory.main.capture_event")


def _prepare(memory, mocker):
    memory.config = mocker.MagicMock()
    memory.config.custom_instructions = None
    memory.custom_instructions = None
    memory.api_version = "v1.1"
    memory.db.get_last_messages = MagicMock(return_value=[])
    memory.db.save_messages = MagicMock()
    return memory


@pytest.fixture
def memory(mocker, monkeypatch):
    monkeypatch.delenv("MEM0_EXTRACTION_ERRORS", raising=False)
    _setup_mocks(mocker)
    return _prepare(Memory(), mocker)


@pytest.fixture
def async_memory(mocker, monkeypatch):
    monkeypatch.delenv("MEM0_EXTRACTION_ERRORS", raising=False)
    _setup_mocks(mocker)
    return _prepare(AsyncMemory(), mocker)


MSG = [{"role": "user", "content": "Eu uso Qdrant 1.18 no projeto."}]


# ---- parser -------------------------------------------------------------

@pytest.mark.parametrize("reply", ['{"memory": []}', '```json\n{"memory": []}\n```', "{}", '{"memory": null}'])
def test_empty_list_is_a_legitimate_answer(reply):
    assert parse_extraction_response(reply) == []


def test_facts_come_back():
    assert parse_extraction_response('{"memory": [{"text": "a"}]}') == [{"text": "a"}]


@pytest.mark.parametrize("reply,cause", [
    ("", "empty"),
    ("   \n", "empty"),
    (None, "empty"),
    ('{"memory": [{"text": "cortado no meio', "unparseable"),   # the production failure
    ("not json at all", "unparseable"),
    ('{"facts": [{"text": "a"}]}', "schema"),                  # wrong key: facts would vanish
    ('{"memory": {"text": "a"}}', "schema"),
    ('[{"text": "a"}]', "schema"),
])
def test_unusable_replies_raise_with_their_cause(reply, cause):
    with pytest.raises(ExtractionError) as exc:
        parse_extraction_response(reply)
    assert exc.value.cause == cause
    assert exc.value.retryable is False
    assert isinstance(exc.value, LLMError)


def test_error_never_carries_the_reply_text():
    secret = '{"facts": [{"text": "senha do cofre é XPTO-123"}]}'
    with pytest.raises(ExtractionError) as exc:
        parse_extraction_response(secret)
    blob = f"{exc.value} {exc.value.details} {exc.value.suggestion}"
    assert "XPTO" not in blob and "senha" not in blob and "facts" not in blob


def test_unknown_cause_is_rejected():
    with pytest.raises(ValueError):
        ExtractionError("network", retryable=True)


# ---- kill switch ---------------------------------------------------------

@pytest.mark.parametrize("value,swallowed", [
    (None, False), ("", False), ("raise", False), ("swallow", True), (" Swallow ", True),
    ("swalow", False),   # a typo must not re-open the loss
    ("off", False), ("0", False),
])
def test_kill_switch_values(monkeypatch, value, swallowed):
    if value is None:
        monkeypatch.delenv("MEM0_EXTRACTION_ERRORS", raising=False)
    else:
        monkeypatch.setenv("MEM0_EXTRACTION_ERRORS", value)
    assert extraction_errors_swallowed() is swallowed


# ---- wired into add (sync) ----------------------------------------------

def test_truncated_reply_raises_and_writes_nothing(memory):
    memory.llm.generate_response.return_value = '{"memory": [{"text": "Eu uso Qd'
    with pytest.raises(ExtractionError) as exc:
        memory._add_to_vector_store(messages=MSG, metadata={}, filters={"user_id": "u"}, infer=True)
    assert exc.value.cause == "unparseable"
    memory.db.save_messages.assert_not_called()
    memory.vector_store.insert.assert_not_called()


def test_llm_exception_becomes_extraction_error_keeping_retryability(memory):
    class Throttled(RuntimeError):
        retryable = True

    class Blocked(RuntimeError):
        retryable = False

    memory.llm.generate_response.side_effect = Throttled("429")
    with pytest.raises(ExtractionError) as exc:
        memory._add_to_vector_store(messages=MSG, metadata={}, filters={"user_id": "u"}, infer=True)
    assert (exc.value.cause, exc.value.retryable) == ("llm_call", True)
    assert isinstance(exc.value.__cause__, Throttled)

    memory.llm.generate_response.side_effect = Blocked("safety")
    with pytest.raises(ExtractionError) as exc:
        memory._add_to_vector_store(messages=MSG, metadata={}, filters={"user_id": "u"}, infer=True)
    assert (exc.value.cause, exc.value.retryable) == ("llm_call", False)


def test_plain_exception_defaults_to_retryable(memory):
    memory.llm.generate_response.side_effect = ConnectionError("refused")
    with pytest.raises(ExtractionError) as exc:
        memory._add_to_vector_store(messages=MSG, metadata={}, filters={"user_id": "u"}, infer=True)
    assert exc.value.retryable is True


def test_legit_empty_answer_still_returns_empty_and_saves_messages(memory):
    memory.llm.generate_response.return_value = '{"memory": []}'
    assert memory._add_to_vector_store(messages=MSG, metadata={}, filters={"user_id": "u"}, infer=True) == []
    memory.db.save_messages.assert_called_once()


def test_kill_switch_restores_upstream_empty_result(memory, monkeypatch):
    monkeypatch.setenv("MEM0_EXTRACTION_ERRORS", "swallow")
    memory.llm.generate_response.return_value = '{"memory": [{"text": "Eu uso Qd'
    assert memory._add_to_vector_store(messages=MSG, metadata={}, filters={"user_id": "u"}, infer=True) == []
    memory.llm.generate_response.side_effect = ConnectionError("refused")
    assert memory._add_to_vector_store(messages=MSG, metadata={}, filters={"user_id": "u"}, infer=True) == []


# ---- wired into add (async) ---------------------------------------------

@pytest.mark.asyncio
async def test_async_truncated_reply_raises(async_memory, mocker):
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", return_value=MagicMock())
    async_memory.llm.generate_response.return_value = '{"memory": [{"text": "Eu uso Qd'
    with pytest.raises(ExtractionError) as exc:
        await async_memory._add_to_vector_store(messages=MSG, metadata={}, effective_filters={"user_id": "u"}, infer=True)
    assert exc.value.cause == "unparseable"
    async_memory.db.save_messages.assert_not_called()


@pytest.mark.asyncio
async def test_async_llm_exception_raises(async_memory, mocker):
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", return_value=MagicMock())
    async_memory.llm.generate_response.side_effect = ConnectionError("refused")
    with pytest.raises(ExtractionError) as exc:
        await async_memory._add_to_vector_store(messages=MSG, metadata={}, effective_filters={"user_id": "u"}, infer=True)
    assert (exc.value.cause, exc.value.retryable) == ("llm_call", True)
