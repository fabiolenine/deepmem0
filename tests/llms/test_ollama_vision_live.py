"""Live smoke: prove that an image actually reaches a real Ollama VLM through
mem0's native vision path (parse_vision_messages -> get_image_description ->
OllamaLLM.generate_response with the image_url->images translation).

A mocked Client can only prove we emit the structure we *expect*; it cannot
prove Ollama consumes it. This runs the whole path against a real local Ollama
VLM and asserts the model describes the image content. If the translation were
wrong (image never reaches the model), the description would not mention the
color.

GATED: skipped unless a local Ollama is reachable AND the vision model is
installed AND Pillow is importable. Set MEM0_LIVE_OLLAMA=0 to force-skip.
Point at a different model with MEM0_LIVE_VLM (default qwen3-vl:4b-instruct).
"""
import base64
import io
import os

import pytest

VLM = os.environ.get("MEM0_LIVE_VLM", "qwen3-vl:4b-instruct")
OLLAMA_URL = os.environ.get("MEM0_LLM_URL", "http://localhost:11434")


def _skip_reason():
    if os.environ.get("MEM0_LIVE_OLLAMA") == "0":
        return "MEM0_LIVE_OLLAMA=0 (force-skip)"
    try:
        from ollama import Client
    except ImportError:
        return "ollama client not installed"
    try:
        import PIL  # noqa: F401
    except ImportError:
        return "Pillow not installed (needed to synthesize the test image)"
    try:
        names = [m.get("model") or m.get("name") for m in Client(host=OLLAMA_URL).list().get("models", [])]
    except Exception as e:
        return f"ollama unreachable at {OLLAMA_URL}: {e}"
    if not any(VLM in (n or "") for n in names):
        return f"vision model {VLM} not installed"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


def _red_square_png() -> bytes:
    from PIL import Image

    img = Image.new("RGB", (128, 128), (220, 20, 20))  # unambiguous red
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _red_square_data_uri() -> str:
    return "data:image/png;base64," + base64.b64encode(_red_square_png()).decode()


def _spy_chat(llm):
    """Record every `client.chat` call, still calling through.

    Asserting on a description alone is circumstantial: a VLM can name a color
    it was never shown. Asserting on what was actually PUT ON THE WIRE is not.
    """
    calls = []
    real = llm.client.chat

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    llm.client.chat = spy
    return calls


def _live_llm():
    from mem0.configs.llms.ollama import OllamaConfig
    from mem0.llms.ollama import OllamaLLM

    # `model` is the text model and is unused here (the image call routes to
    # vision_model); any name works. vision_model carries the actual VLM.
    return OllamaLLM(OllamaConfig(model="llama3.1", vision_model=VLM,
                                  ollama_base_url=OLLAMA_URL, max_tokens=256))


def _mentions_red(description):
    assert description and description.strip(), "VLM returned an empty description"
    low = description.lower()
    # if the image bytes reached the model, it names the color; a broken
    # image_url->images translation would yield an error or a color-less/generic
    # answer. Accept EN or PT.
    assert any(w in low for w in ("red", "vermelh")), (
        f"description did not mention the red color -> image likely did not reach "
        f"the model. Got: {description!r}"
    )


def test_real_vlm_describes_image_through_native_path():
    from mem0.memory.utils import get_image_description

    # get_image_description is exactly what parse_vision_messages calls per image.
    _mentions_red(get_image_description(_red_square_data_uri(), _live_llm(), "auto"))


def test_real_vlm_reached_through_parse_vision_messages_list_branch():
    """End-to-end over the STRICT branch, with a real VLM.

    The test above enters at `get_image_description`, so it never exercises
    `parse_vision_messages` itself -- the per-part validation added for the LIST
    shape (the canonical OpenAI multimodal shape) had no live coverage at all.
    This drives the whole path: strict validation of every image part, then the
    provider call, with real image bytes reaching a real model.
    """
    from mem0.memory.utils import parse_vision_messages

    png = _red_square_png()
    llm = _live_llm()
    calls = _spy_chat(llm)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "What color is this?"},
        {"type": "image_url", "image_url": {"url": _red_square_data_uri()}},
    ]}]
    result = parse_vision_messages(messages, llm=llm, vision_details="auto")

    assert len(result) == 1 and result[0]["role"] == "user"
    # What went on the wire, not just what came back: the model could name a
    # color it was never shown.
    assert len(calls) == 1, f"expected exactly one chat call, got {len(calls)}"
    sent = calls[0]["messages"][0]
    assert sent.get("images"), "no images reached the provider -> image_url->images broke"
    assert base64.b64decode(sent["images"][0]) == png, (
        "the bytes submitted to Ollama are not the PNG this test generated"
    )
    assert calls[0]["model"] == VLM, "image call did not route to the vision model"
    _mentions_red(result[0]["content"])


def test_malformed_part_is_refused_before_any_provider_call():
    """The guard must fire on the real path, not only against mocks.

    Ordering is asserted DIRECTLY (`client.chat` never called), not inferred from
    the error message. Note what the message alone would NOT prove: without the
    guard this would not reach Ollama either -- `_ollama_messages` rejects the
    same part locally, with its own ValueError. The two layers are distinguishable
    by message, but only the spy proves nothing was dispatched.
    """
    from mem0.memory.utils import parse_vision_messages

    llm = _live_llm()
    calls = _spy_chat(llm)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "What color is this?"},
        {"type": "image_url", "image_url": {"url": _red_square_data_uri()}},
        {"type": "image_url", "image_url": {}},  # malformed, and NOT the first
    ]}]
    with pytest.raises(ValueError, match=r"missing image_url\.url"):
        parse_vision_messages(messages, llm=llm, vision_details="auto")
    assert calls == [], "guard did not fire before the provider was called"
