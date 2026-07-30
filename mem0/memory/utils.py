import ast
import hashlib
import logging
import re
import uuid
from typing import Any, Dict, List

from mem0.configs.prompts import (
    AGENT_MEMORY_EXTRACTION_PROMPT,
    FACT_RETRIEVAL_PROMPT,
    USER_MEMORY_EXTRACTION_PROMPT,
)

logger = logging.getLogger(__name__)


def get_fact_retrieval_messages(message, is_agent_memory=False):
    """Get fact retrieval messages based on the memory type.
    
    Args:
        message: The message content to extract facts from
        is_agent_memory: If True, use agent memory extraction prompt, else use user memory extraction prompt
        
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    if is_agent_memory:
        return AGENT_MEMORY_EXTRACTION_PROMPT, f"Input:\n{message}"
    else:
        return USER_MEMORY_EXTRACTION_PROMPT, f"Input:\n{message}"


def get_fact_retrieval_messages_legacy(message):
    """Legacy function for backward compatibility."""
    return FACT_RETRIEVAL_PROMPT, f"Input:\n{message}"


def ensure_json_instruction(system_prompt, user_prompt):
    """Ensure the word 'json' appears in the prompts when using json_object response format.

    OpenAI's API requires the word 'json' to appear in the messages when
    response_format is set to {"type": "json_object"}. When users provide a
    custom_instructions that doesn't include 'json', this causes a
    400 error. This function appends a JSON format instruction to the system
    prompt if 'json' is not already present in either prompt.

    Args:
        system_prompt: The system prompt string
        user_prompt: The user prompt string

    Returns:
        tuple: (system_prompt, user_prompt) with JSON instruction added if needed
    """
    combined = (system_prompt + user_prompt).lower()
    if "json" not in combined:
        system_prompt += (
            "\n\nYou must return your response in valid JSON format "
            "with a 'facts' key containing an array of strings."
        )
    return system_prompt, user_prompt


def parse_messages(messages):
    response = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        # Skip messages without textual content (e.g. assistant tool-call
        # messages that carry `tool_calls` but no `content` key).
        if content is None:
            continue
        if role == "system":
            response += f"system: {content}\n"
        elif role == "user":
            response += f"user: {content}\n"
        elif role == "assistant":
            response += f"assistant: {content}\n"
    return response


def format_entities(entities):
    if not entities:
        return ""

    formatted_lines = []
    for entity in entities:
        simplified = f"{entity['source']} -- {entity['relationship']} -- {entity['destination']}"
        formatted_lines.append(simplified)

    return "\n".join(formatted_lines)

def normalize_facts(raw_facts):
    """Normalize LLM-extracted facts to a list of strings.

    Smaller LLMs (e.g. llama3.1:8b) sometimes return facts as objects
    like {"fact": "..."} or {"text": "..."} instead of plain strings.
    This mirrors the TypeScript FactRetrievalSchema validation.
    """
    if not raw_facts:
        return []
    normalized = []
    for item in raw_facts:
        if isinstance(item, str):
            fact = item
        elif isinstance(item, dict):
            fact = item.get("fact") or item.get("text")
            if fact is None:
                logger.warning("Unexpected fact shape from LLM, skipping: %s", item)
                continue
        else:
            fact = str(item)
        if fact:
            normalized.append(fact)
    return normalized


def normalize_linked_memory_ids(value: Any) -> List[str]:
    """Coerce a persisted ``linked_memory_ids`` payload value into a list of ids.

    Entity payloads are written by many hands — this library, backfill scripts,
    ad-hoc repair scripts — and a single writer that stored ``str(list)`` instead
    of the list is enough to poison every downstream writer, because
    ``set("['a', 'b']")`` silently explodes into CHARACTERS instead of raising.
    One such write produced entity rows holding 22 single-character "ids".

    Shapes tolerated:
      * ``list``        -> element-wise ``str()``, empty/None dropped, order kept
      * ``str``         -> ``ast.literal_eval`` when it parses to a list/tuple
                           (recovers a repr written by a buggy writer); otherwise
                           the string is treated as ONE id -> ``[value]``
      * ``tuple``/``set`` -> list (``set`` sorted, so callers get a stable order)
      * anything else / ``None`` -> ``[]``

    Deliberately does NOT filter by id shape: memory ids are opaque at this layer
    and callers/tests legitimately use non-UUID ids. Deployments that guarantee a
    specific id format enforce that in their own audit tooling, not here.

    Never raises — a malformed payload must not break an add or a delete.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("("):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                logger.warning(
                    "Recovered a serialized linked_memory_ids value (%d ids); "
                    "some writer stored str(list) instead of the list",
                    len(parsed),
                )
                return normalize_linked_memory_ids(list(parsed))
        # Not a serialized container: a bare string is a single id.
        return [text] if text else []
    if isinstance(value, set):
        value = sorted(value, key=str)
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if item is None:
                continue
            text = item if isinstance(item, str) else str(item)
            if text:
                out.append(text)
        return out
    return []


def remove_code_blocks(content: str) -> str:
    """
    Removes enclosing code block markers ```[language] and ``` from a given string.

    Remarks:
    - The function uses a regex pattern to match code blocks that may start with ``` followed by an optional language tag (letters or numbers) and end with ```.
    - If a code block is detected, it returns only the inner content, stripping out the markers.
    - If no code block markers are found, the original content is returned as-is.
    """
    pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, content.strip())
    match_res=match.group(1).strip() if match else content.strip()
    return re.sub(r"<think>.*?</think>", "", match_res, flags=re.DOTALL).strip()



def extract_json(text):
    """
    Extracts JSON content from a string, removing enclosing triple backticks and optional 'json' tag if present.
    If no code block is found, attempts to locate JSON by finding the first '{' and last '}'.
    If that also fails, returns the text as-is.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx : end_idx + 1]
        else:
            json_str = text
    return json_str


def get_image_description(image_obj, llm, vision_details):
    """
    Get the description of the image
    """

    if isinstance(image_obj, str):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "A user is providing an image. Provide a high level description of the image and do not include any additional text.",
                    },
                    {"type": "image_url", "image_url": {"url": image_obj, "detail": vision_details}},
                ],
            },
        ]
    else:
        messages = [image_obj]

    response = llm.generate_response(messages=messages)
    return response


def _image_part_url(part):
    """Extract and validate the URL of one OpenAI-style ``image_url`` content part.

    Single source of truth for BOTH content shapes ``parse_vision_messages``
    accepts: the canonical OpenAI list part
    ``{"type": "image_url", "image_url": {"url": ...}}`` and mem0's legacy
    bare-dict message content.

    Raises ``ValueError`` -- never ``KeyError``/``TypeError`` -- because callers
    classify failures by exception class: a structurally broken payload must be
    poison (fail once, permanently), not an infrastructure blip worth retrying.

    This validates the *message structure* only. Transport encoding (data URI vs
    raw base64 vs local path, size caps) belongs to the provider -- see
    ``mem0/llms/ollama.py:_extract_ollama_image``.
    """
    image_url_obj = part.get("image_url") if isinstance(part, dict) else None
    image_url = image_url_obj.get("url") if isinstance(image_url_obj, dict) else None
    if not isinstance(image_url, str) or not image_url:
        raise ValueError("image_url content part is missing image_url.url")
    return image_url


def parse_vision_messages(messages, llm=None, vision_details="auto"):
    """
    Parse the vision messages from the messages

    Two regimes, deliberately asymmetric:

    * ``llm is None`` (vision disabled) -- TOTAL for a well-formed message: it
      does not raise. Image parts are dropped, but never silently: a dropped
      image is a fact the caller sent and we did not store, so it is logged at
      WARNING. A malformed *container* (a non-dict message) still raises
      ``AttributeError``, which is correct -- consumers classify it as poison.
    * ``llm`` provided (vision enabled) -- STRICT: every image part is validated
      before an LLM call is spent, and any exception raised by the provider
      propagates with its original type, message and traceback.
    """
    returned_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            returned_messages.append(msg)
            continue

        # Skip messages without content (e.g. assistant tool-call messages
        # that carry `tool_calls` but no `content` key).
        if content is None:
            continue

        # Handle message content
        if isinstance(content, list):
            if llm is None:
                text_parts = []
                dropped_images = 0
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type == "text":
                        text = part.get("text")
                        # A non-string `text` would blow up the join below. An
                        # EMPTY string is kept: upstream preserved it, and
                        # dropping it here would silently discard a message that
                        # used to survive -- out of scope for this change.
                        if isinstance(text, str):
                            text_parts.append(text)
                    elif part_type == "image_url":
                        dropped_images += 1
                if dropped_images and text_parts:
                    logger.warning(
                        "parse_vision_messages: vision is disabled (enable_vision=False); "
                        "dropped %d image part(s) from a %s message, keeping its text",
                        dropped_images, role,
                    )
                elif dropped_images:
                    logger.warning(
                        "parse_vision_messages: vision is disabled (enable_vision=False); "
                        "dropped %d image part(s) and discarded the whole %s message "
                        "(it carried no text)",
                        dropped_images, role,
                    )
                if not text_parts:
                    continue
                returned_messages.append({"role": role, "content": " ".join(text_parts)})
            else:
                # Validate EVERY image part before spending an LLM call. This is
                # the canonical OpenAI multimodal shape -- and the very shape
                # `get_image_description` builds -- so it deserves at least the
                # guard the legacy dict shape gets. Validation only: the payload
                # sent to the provider stays the whole `msg`.
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        _image_part_url(part)
                description = get_image_description(msg, llm, vision_details)
                returned_messages.append({"role": role, "content": description})
        elif isinstance(content, dict) and content.get("type") == "image_url":
            if llm is None:
                logger.warning(
                    "parse_vision_messages: vision is disabled (enable_vision=False); "
                    "dropped 1 image part(s) and discarded the whole %s message "
                    "(it carried no text)",
                    role,
                )
                continue
            image_url = _image_part_url(content)
            # NO try/except here, on purpose. The provider's own exception is the
            # diagnosis: mem0/llms/ollama.py raises actionable ValueErrors ("only
            # base64 data URIs are supported", "malformed base64 in data URI",
            # "http(s) image URLs are not supported"). The removed wrapper raised
            # a bare `Exception` claiming the download had failed, which did two
            # harms: it lied (nothing is fetched for a data URI or a local path)
            # and it erased the exception class that consumers use to tell a
            # poison payload from sick infrastructure -- turning a permanent
            # failure into N retried full re-adds.
            description = get_image_description(image_url, llm, vision_details)
            returned_messages.append({"role": role, "content": description})
        else:
            # Regular text content
            returned_messages.append(msg)

    return returned_messages


def process_telemetry_filters(filters):
    """
    Process the telemetry filters
    """
    if filters is None:
        return {}

    encoded_ids = {}
    if "user_id" in filters:
        encoded_ids["user_id"] = hashlib.md5(filters["user_id"].encode()).hexdigest()
    if "agent_id" in filters:
        encoded_ids["agent_id"] = hashlib.md5(filters["agent_id"].encode()).hexdigest()
    if "run_id" in filters:
        encoded_ids["run_id"] = hashlib.md5(filters["run_id"].encode()).hexdigest()

    return list(filters.keys()), encoded_ids


def sanitize_relationship_for_cypher(relationship) -> str:
    """Sanitize relationship text for Cypher queries by replacing problematic characters."""
    char_map = {
        "...": "_ellipsis_",
        "…": "_ellipsis_",
        "。": "_period_",
        "，": "_comma_",
        "；": "_semicolon_",
        "：": "_colon_",
        "！": "_exclamation_",
        "？": "_question_",
        "（": "_lparen_",
        "）": "_rparen_",
        "【": "_lbracket_",
        "】": "_rbracket_",
        "《": "_langle_",
        "》": "_rangle_",
        "'": "_apostrophe_",
        '"': "_quote_",
        "\\": "_backslash_",
        "/": "_slash_",
        "|": "_pipe_",
        "&": "_ampersand_",
        "=": "_equals_",
        "+": "_plus_",
        "*": "_asterisk_",
        "^": "_caret_",
        "%": "_percent_",
        "$": "_dollar_",
        "#": "_hash_",
        "@": "_at_",
        "!": "_bang_",
        "?": "_question_",
        "(": "_lparen_",
        ")": "_rparen_",
        "[": "_lbracket_",
        "]": "_rbracket_",
        "{": "_lbrace_",
        "}": "_rbrace_",
        "<": "_langle_",
        ">": "_rangle_",
        "-": "_",
    }

    # Apply replacements and clean up
    sanitized = relationship
    for old, new in char_map.items():
        sanitized = sanitized.replace(old, new)

    return re.sub(r"_+", "_", sanitized).strip("_")


def remove_spaces_from_entities(
    entity_list: List[Any],
    *,
    sanitize_relationship: bool = True,
) -> List[Dict[str, Any]]:
    """
    Normalize entity relation dicts from LLM/tool output: lowercase, spaces to underscores.

    Skips entries that are not non-empty dicts or that lack any of
    ``source``, ``relationship``, or ``destination`` (avoids KeyError on ``[{}]``
    or partial dicts).
    """
    required = ("source", "relationship", "destination")
    cleaned: List[Dict[str, Any]] = []
    for item in entity_list:
        if not isinstance(item, dict) or not item:
            continue
        if not all(key in item for key in required):
            continue
        item["source"] = item["source"].lower().replace(" ", "_")
        rel = item["relationship"].lower().replace(" ", "_")
        item["relationship"] = sanitize_relationship_for_cypher(rel) if sanitize_relationship else rel
        item["destination"] = item["destination"].lower().replace(" ", "_")
        cleaned.append(item)
    return cleaned

# Namespace fixo para ids determinísticos de entidade. Não muda: mudá-lo faria a
# mesma entidade nascer com id diferente e recriaria a duplicata que ele evita.
ENTITY_ID_NAMESPACE = uuid.UUID("6f9d3a1e-0c4b-5d8a-9e7f-2b1c3d4e5f60")


def normalize_entity_key(text) -> str:
    """Chave de IDENTIDADE de uma entidade — o que decide se duas linhas são a
    mesma.

    POR QUE EXISTE: a identidade era a SIMILARIDADE do vetor (>= 0.95). Isso é
    probabilístico, e o corpus mostrou o preço — `FASE` e `Fase`,
    `docker compose` e `Docker Compose`, `Hilbert transform` e
    `Hilbert Transform` viraram linhas SEPARADAS, cada uma com sua fatia de
    vínculos. Qual delas ganha o boost passa a depender da grafia buscada.

    Casefold + NFKC + espaço colapsado. Separador NÃO é colapsado de propósito:
    `num_ctx` e `num ctx` são coisas diferentes num corpus técnico, e fundi-los
    trocaria uma duplicata por uma colisão.
    """
    import unicodedata

    bruto = unicodedata.normalize("NFKC", str(text or "")).strip()
    return " ".join(bruto.split()).casefold()


def entity_point_id(scope: dict, normalized_key: str) -> str:
    """Id DETERMINÍSTICO da linha de entidade: f(escopo, chave normalizada).

    O escritor fazia sonda-então-UUID-aleatório. O worker HTTP é serial, mas
    hooks e ingestão instanciam `Memory` próprios, então existe corrida
    check-then-insert: dois escritores não acham nada, cada um gera um UUID e
    nascem duas linhas para a mesma entidade. Com o id derivado, o segundo
    escreve NO MESMO ponto — a corrida deixa de criar duplicata e passa a ser um
    lost-update, que o chamador reconcilia relendo.
    """
    partes = "|".join(
        f"{k}={scope.get(k) or ''}" for k in ("user_id", "agent_id", "run_id"))
    return str(uuid.uuid5(ENTITY_ID_NAMESPACE, f"{partes}|{normalized_key}"))


LINK_KEY_PREFIX = "lnk_"


def link_key(memory_id: str) -> str:
    """Chave de payload que representa UM vínculo.

    POR QUE UMA CHAVE POR VÍNCULO: `set_payload` do Qdrant faz MERGE de CHAVES,
    mas SUBSTITUI o valor de uma chave de lista. Guardar os vínculos só numa
    lista transforma toda escrita concorrente em lost-update — medido: 8
    escritores simultâneos na mesma entidade deixam 1 vínculo de 8. Com uma
    chave por vínculo, dois escritores tocando chaves diferentes MERGEIAM, e a
    união vira atômica sem precisar de CAS, que o Qdrant não oferece.

    `linked_memory_ids` continua sendo a lista canônica que todo leitor usa; as
    chaves são a fonte de verdade para RECONSTRUÍ-la.
    """
    return f"{LINK_KEY_PREFIX}{memory_id}"


def links_do_payload(payload: dict) -> list:
    """União de `linked_memory_ids` com as chaves `lnk_*`, ordenada."""
    ids = set(normalize_linked_memory_ids((payload or {}).get("linked_memory_ids")))
    for k, v in (payload or {}).items():
        if k.startswith(LINK_KEY_PREFIX) and v:
            ids.add(k[len(LINK_KEY_PREFIX):])
    return sorted(ids)
