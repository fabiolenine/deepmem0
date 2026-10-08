"""DeepMem0 Gemini contract: explicit controls, usage capture, no silent "".

Built on the SDK's REAL response types, not Mock: a Mock answers any attribute,
so a check on `finish_reason` or `block_reason` would pass against it whether
or not the code reads the field the SDK actually sets.
"""

from unittest.mock import Mock, patch

import pytest
from google.genai import errors, types

import mem0.llms.gemini as gem
from mem0.configs.llms.gemini import GeminiConfig
from mem0.exceptions import LLMResponseError
from mem0.llms.gemini import GeminiLLM

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "oi"}]


@pytest.fixture
def client():
    with patch("mem0.llms.gemini.genai.Client") as cls:
        inst = Mock()
        cls.return_value = inst
        inst._cls = cls
        yield inst


def _resp(text="ok", finish=types.FinishReason.STOP, usage=None, block=None, candidates=True):
    cands = [types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]),
                             finish_reason=finish)] if candidates else []
    return types.GenerateContentResponse(
        candidates=cands,
        usage_metadata=usage,
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block) if block else None,
    )


def _sent_config(client):
    return client.models.generate_content.call_args.kwargs["config"]


# ---- explicit controls ---------------------------------------------------

def test_defaults_send_no_thinking_or_safety(client):
    client.models.generate_content.return_value = _resp()
    GeminiLLM(GeminiConfig(model="m")).generate_response(MSGS)
    cfg = _sent_config(client)
    assert cfg.thinking_config is None and cfg.safety_settings is None


def test_thinking_level_and_budget_are_sent(client):
    client.models.generate_content.return_value = _resp()
    GeminiLLM(GeminiConfig(model="m", thinking_level="MINIMAL")).generate_response(MSGS)
    assert _sent_config(client).thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
    GeminiLLM(GeminiConfig(model="m", thinking_budget=0)).generate_response(MSGS)
    assert _sent_config(client).thinking_config.thinking_budget == 0


def test_safety_threshold_covers_the_four_categories(client):
    client.models.generate_content.return_value = _resp()
    GeminiLLM(GeminiConfig(model="m", safety_threshold="OFF")).generate_response(MSGS)
    settings = _sent_config(client).safety_settings
    assert {s.category for s in settings} == {
        types.HarmCategory.HARM_CATEGORY_HARASSMENT, types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT}
    assert {s.threshold for s in settings} == {types.HarmBlockThreshold.OFF}


def test_timeout_and_retry_reach_the_client_in_milliseconds(client):
    GeminiLLM(GeminiConfig(model="m", vertexai=True, project="p", location="global",
                           timeout_s=12.5, http_retry_attempts=1))
    opts = client._cls.call_args.kwargs["http_options"]
    assert opts.timeout == 12500
    assert opts.retry_options.attempts == 1
    assert client._cls.call_args.kwargs["vertexai"] is True


def test_no_http_options_when_unset(client):
    GeminiLLM(GeminiConfig(model="m", api_key="k"))
    assert "http_options" not in client._cls.call_args.kwargs


# ---- usage ---------------------------------------------------------------

USAGE = types.GenerateContentResponseUsageMetadata(
    prompt_token_count=100, candidates_token_count=20, thoughts_token_count=7, total_token_count=127)


def test_usage_recorded_and_listeners_called(client):
    seen = []
    listener = lambda model, usage: seen.append((model, usage))  # noqa: E731
    gem.add_usage_listener(listener)
    try:
        client.models.generate_content.return_value = _resp(usage=USAGE)
        llm = GeminiLLM(GeminiConfig(model="gemini-x"))
        llm.generate_response(MSGS)
    finally:
        gem.remove_usage_listener(listener)
    expected = {"prompt_token_count": 100, "candidates_token_count": 20,
                "thoughts_token_count": 7, "total_token_count": 127}
    assert llm.last_usage == expected
    assert seen == [("gemini-x", expected)]


def test_usage_counted_even_when_the_response_is_rejected(client):
    seen = []
    listener = lambda model, usage: seen.append(usage)  # noqa: E731
    gem.add_usage_listener(listener)
    try:
        client.models.generate_content.return_value = _resp(finish=types.FinishReason.MAX_TOKENS, usage=USAGE)
        with pytest.raises(LLMResponseError):
            GeminiLLM(GeminiConfig(model="m")).generate_response(MSGS)
    finally:
        gem.remove_usage_listener(listener)
    assert seen and seen[0]["total_token_count"] == 127


def test_broken_listener_does_not_fail_the_call(client):
    def boom(model, usage):
        raise RuntimeError("listener down")
    gem.add_usage_listener(boom)
    try:
        client.models.generate_content.return_value = _resp(text="fine", usage=USAGE)
        assert GeminiLLM(GeminiConfig(model="m")).generate_response(MSGS) == "fine"
    finally:
        gem.remove_usage_listener(boom)


# ---- completeness --------------------------------------------------------

@pytest.mark.parametrize("finish,kind", [
    (types.FinishReason.MAX_TOKENS, "truncated"),
    (types.FinishReason.SAFETY, "blocked"),
    (types.FinishReason.PROHIBITED_CONTENT, "blocked"),
    (types.FinishReason.RECITATION, "blocked"),
    (types.FinishReason.SPII, "blocked"),
])
def test_unfinished_completion_raises(client, finish, kind):
    client.models.generate_content.return_value = _resp(text='{"memory": [{"text": "cor', finish=finish)
    with pytest.raises(LLMResponseError) as exc:
        GeminiLLM(GeminiConfig(model="m")).generate_response(MSGS, response_format={"type": "json_object"})
    assert (exc.value.kind, exc.value.reason, exc.value.retryable) == (kind, finish.name, False)


def test_prompt_block_raises_before_parsing(client):
    client.models.generate_content.return_value = _resp(candidates=False, block=types.BlockedReason.SAFETY)
    with pytest.raises(LLMResponseError) as exc:
        GeminiLLM(GeminiConfig(model="m")).generate_response(MSGS)
    assert (exc.value.kind, exc.value.reason) == ("prompt_blocked", "SAFETY")


def test_no_candidates_without_block_is_retryable(client):
    client.models.generate_content.return_value = _resp(candidates=False)
    with pytest.raises(LLMResponseError) as exc:
        GeminiLLM(GeminiConfig(model="m")).generate_response(MSGS)
    assert (exc.value.kind, exc.value.retryable) == ("no_candidates", True)


def test_stop_returns_text(client):
    client.models.generate_content.return_value = _resp(text='{"memory": []}')
    assert GeminiLLM(GeminiConfig(model="m")).generate_response(MSGS) == '{"memory": []}'


# ---- HTTP errors ---------------------------------------------------------

@pytest.mark.parametrize("code,status,retryable", [
    (429, "RESOURCE_EXHAUSTED", True),
    (503, "UNAVAILABLE", True),
    (500, "INTERNAL", True),
    (400, "INVALID_ARGUMENT", False),
    (403, "PERMISSION_DENIED", False),
])
def test_api_errors_are_tagged_and_propagate(client, code, status, retryable):
    cls = errors.ServerError if code >= 500 else errors.ClientError
    client.models.generate_content.side_effect = cls(code, {"error": {"code": code, "status": status, "message": "x"}})
    with pytest.raises(errors.APIError) as exc:
        GeminiLLM(GeminiConfig(model="m")).generate_response(MSGS)
    assert exc.value.retryable is retryable
