import ast
import hashlib
import logging
import re
import unicodedata
import uuid
from typing import Any, Dict, List

from mem0.configs.prompts import (
    AGENT_MEMORY_EXTRACTION_PROMPT,
    FACT_RETRIEVAL_PROMPT,
    USER_MEMORY_EXTRACTION_PROMPT,
)

logger = logging.getLogger(__name__)

#: Comprimento máximo de um rótulo de locutor. Rótulo maior é REJEITADO, nunca
#: truncado: truncar fundiria "Maria Silva Sant..." e "Maria Silva Sanc..." num
#: mesmo locutor, em silêncio. Rejeitar deixa a mensagem anônima, que é o estado
#: seguro.
MAX_SPEAKER_LABEL = 64

#: Papéis que `parse_messages` sabe renderizar. Papel fora daqui é DESCARTADO —
#: comportamento pré-existente, e o motivo pelo qual o conjunto fechado de
#: locutores tem que ser derivado daqui e não das mensagens cruas: um nome numa
#: mensagem que o modelo nunca vê seria um valor que o validador ACEITA e o
#: modelo só poderia ter inventado.
_PAPEIS_RENDERIZAVEIS = ("system", "user", "assistant")


def normalize_speaker_label(value):
    """Rótulo canônico de locutor, ou ``None`` se não for utilizável.

    FONTE ÚNICA: usada nos três pontos onde um rótulo aparece — saneamento da
    entrada (`name` da mensagem), validação da saída do LLM, e filtro na
    leitura. Escrita e consulta canonizando diferente é filtro exato que erra em
    silêncio, e o Qdrant casa por igualdade.

    ⚠️ NÃO faz casefold, de propósito. Casefoldar fundiria `Maria` e `maria` em
    um locutor só — a mesma razão pela qual `user_id` não é casefoldado
    (fundiria usuários distintos). O custo declarado é o inverso: `Maria` e
    `maria` são locutores DIFERENTES. Limitação conhecida, não descuido.

    ⚠️ Rejeita quebra de linha e caractere de controle porque o rótulo entra num
    prompt cuja gramática é uma linha por turno: um `name` com ``\\n`` forjaria
    turnos inteiros na conversa que o extrator lê. Rejeição é TOTAL — o rótulo
    inteiro cai e a mensagem vira anônima. Sanitizar pela metade guardaria um
    rótulo que o chamador não escreveu e que ninguém consegue consultar depois.
    """
    if not isinstance(value, str):
        # bool/int/dict/lista: um rótulo de locutor é texto. Coagir
        # `42` -> `"42"` dividiria escopo em silêncio se o chamador às vezes
        # mandasse int e às vezes str.
        return None
    rotulo = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(c) == "Cc" for c in rotulo):
        return None
    rotulo = re.sub(r"\s+", " ", rotulo).strip()
    if not rotulo or len(rotulo) > MAX_SPEAKER_LABEL:
        return None
    return rotulo


def mensagens_renderizaveis(messages):
    """``(papel, conteúdo, locutor)`` das mensagens que CHEGAM ao prompt.

    Fonte única do filtro de papel e da resolução de locutor. `parse_messages`
    renderiza a partir daqui e `locutores_das_mensagens` coleta daqui, então as
    duas não podem divergir sobre o que o modelo enxerga.

    ⚠️ Mensagem que não é dict LEVANTA (`AttributeError`), como o `parse_messages`
    original — um container malformado é payload envenenado, e os consumidores
    classificam a exceção como permanente. Engolir aqui transformaria perda total
    em resultado parcial mudo.
    """
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content")
        # Sem conteúdo textual (ex.: mensagem de tool-call que só traz
        # `tool_calls`) não vira turno.
        if content is None or role not in _PAPEIS_RENDERIZAVEIS:
            continue
        yield role, content, normalize_speaker_label(msg.get("name"))


def locutores_das_mensagens(messages):
    """``(conjunto de locutores, uniforme)`` sobre as mensagens EXTRAÍVEIS.

    Extraível = renderizável e não-`system`: `system` é instrução, não
    participante, e um fato nunca lhe é atribuído.

    ``uniforme`` é ``True`` só quando existe pelo menos uma mensagem extraível e
    **TODAS** carregam o MESMO rótulo válido. Não é "há um nome distinto": uma
    conversa ``[user name=Maria, assistant sem nome]`` tem exatamente um nome
    distinto e atribuir tudo à Maria daria a ela os fatos que o assistente
    produziu. Uniformidade é a condição que de fato autoriza a atribuição
    determinística sem perguntar ao modelo.
    """
    rotulos, anonima, extraiveis = set(), False, 0
    for role, _content, locutor in mensagens_renderizaveis(messages):
        if role == "system":
            continue
        extraiveis += 1
        if locutor is None:
            anonima = True
        else:
            rotulos.add(locutor)
    uniforme = bool(extraiveis) and not anonima and len(rotulos) == 1
    return rotulos, uniforme


def speaker_attribution_enabled() -> bool:
    """Kill switch da atribuição a locutor. `MEM0_SPEAKER_ATTRIBUTION`, default ON.

    Desligado: nenhum sufixo entra no prompt, nenhum `actor_id` é gravado, e o
    UPDATE volta a ignorar locutor — o comportamento exato de antes desta versão.

    Lido do ambiente A CADA CHAMADA, e não uma vez no import, pelo mesmo motivo
    de `MEM0_EMBED_MAX_BATCH`: num incidente o operador precisa de um lever que
    funcione com um restart do serviço, sem editar config nem tocar no cliente
    MCP. É também o que torna o contrafactual do gate executável — sem uma forma
    de desligar, "com a funcionalidade off o campo não aparece" não é
    verificável, e uma guarda que não pode ser desligada não pode ser comparada
    contra nada.

    Valor não reconhecido cai no DEFAULT (ligado) em vez de desligar: um typo não
    pode virar desativação silenciosa de uma funcionalidade que o operador
    acredita estar no ar.
    """
    import os
    return (os.environ.get("MEM0_SPEAKER_ATTRIBUTION", "").strip().lower()
            not in ("0", "false", "no", "off"))


def precisa_de_atribuicao_por_llm(rotulos, uniforme) -> bool:
    """Há locutor a decidir que o código sozinho não decide?

    Só então o sufixo entra no prompt. Sem locutor nenhum (100% do tráfego de
    hoje) e com locutor uniforme, o custo em token é ZERO — e isso é orçamento de
    `num_ctx`, não economia: o piso do prompt de extração já é ~42% da janela, e
    este sistema já MEDIU perda total e silenciosa de fatos ao encostar no teto.
    """
    return bool(rotulos) and not uniforme


def resolver_locutor_do_fato(bruto, rotulos, uniforme):
    """Rótulo de locutor a GRAVAR para um fato extraído, ou ``None``.

    O contrato inteiro em uma frase: **o LLM propõe, este código decide** —
    mesma forma de `parse_supersedes_ids`, que resolve índices contra o
    `uuid_mapping` em vez de confiar na saída crua.

    Três caminhos:

    * **uniforme** — todo turno extraível tem o mesmo locutor, então o fato só
      pode ser dele. Determinístico, o modelo nem é consultado.
    * **sem locutor** — nada a atribuir.
    * **decidido pelo modelo** — o valor precisa ser `str`, canonizar, e
      PERTENCER ao conjunto fechado que foi enumerado no prompt.

    ⚠️ A checagem de TIPO vem antes da pertinência, e não é zelo: a saída do LLM
    é lida com `json.loads` cru, então `actor_id` pode chegar como dict ou lista,
    e `{"a": 1} in conjunto` levanta `TypeError: unhashable type` — no meio do
    `add`, derrubando fatos que não tinham nada a ver com atribuição.
    `normalize_speaker_label` recusa qualquer não-`str` antes disso.

    ⚠️ A direção do fracasso é sempre OMITIR, nunca chutar. Campo ausente é
    exatamente o comportamento de hoje e é seguro; rótulo errado é corrupção que
    ninguém detecta olhando o resultado.
    """
    if uniforme:
        return next(iter(rotulos))
    if not rotulos:
        return None
    rotulo = normalize_speaker_label(bruto)
    if rotulo is None or rotulo not in rotulos:
        return None
    return rotulo


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
    """Renderiza a conversa para o extrator, um turno por linha.

    Gramática: ``papel: conteúdo``, ou ``papel (Locutor): conteúdo`` quando a
    mensagem traz um `name` utilizável. Sem `name` a linha é BYTE-IDÊNTICA à de
    antes — o corpus inteiro foi extraído com o formato antigo, e mudá-lo para
    quem não usa locutor seria mexer no que 100% do tráfego de hoje já produz.

    O locutor entra por um marcador próprio, e não colado ao conteúdo, porque o
    prompt já ensina o padrão `Nome: texto` DENTRO do conteúdo (Example 12) —
    duas convenções na mesma posição seriam ambíguas justamente onde a
    atribuição precisa ser exata.
    """
    response = ""
    for role, content, locutor in mensagens_renderizaveis(messages):
        prefixo = f"{role} ({locutor})" if locutor else role
        response += f"{prefixo}: {content}\n"
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


def _sem_perder_locutor(msg, role, content):
    """Mensagem reconstruída que PRESERVA o `name` do original.

    Os ramos multimodais de `parse_vision_messages` remontam a mensagem do zero
    (`{"role", "content"}`) porque o conteúdo muda — texto extraído de partes, ou
    a transcrição do VLM. O `name` evaporava junto, e o resultado era atribuição
    que funcionava em mensagem de texto puro e sumia em mensagem com imagem: uma
    assimetria que ninguém pediu e que só apareceria em produção.

    Repassa o valor CRU, sem canonizar — canonização é do render (`parse_messages`)
    e da validação. Aqui o contrato é não perder o que o chamador mandou.
    """
    nova = {"role": role, "content": content}
    if msg.get("name") is not None:
        nova["name"] = msg["name"]
    return nova


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
                returned_messages.append(
                    _sem_perder_locutor(msg, role, " ".join(text_parts)))
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
                returned_messages.append(_sem_perder_locutor(msg, role, description))
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
            returned_messages.append(_sem_perder_locutor(msg, role, description))
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


def normalize_scope_id(value, name: str):
    """Normaliza um identificador de ESCOPO (``user_id``/``agent_id``/``run_id``).

    É a regra única de identidade de escopo: quem escreve e quem lê têm que
    concordar byte a byte, porque o filtro do vector store é casamento EXATO de
    valor. `" alice"` e `"alice"` são escopos diferentes para o Qdrant, e a
    diferença não aparece como erro — aparece como resultado vazio.

    Contrato:
      * ``None`` passa (a ausência de escopo é decidida pelo chamador);
      * ``int`` vira ``str``; `42` e `"42"` passam a ser o MESMO escopo;
      * ``bool``, ``float`` e qualquer outro tipo são RECUSADOS;
      * espaço nas pontas é aparado; vazio e espaço interno são recusados.

    POR QUE COAGIR ``int`` MAS RECUSAR O RESTO. O contrato de metadata deste
    projeto diz "rejeitar em vez de coagir", e a razão é um incidente real: 17
    memórias gravadas com ``importance='high'`` sumiram de um filtro de alta
    importância, porque a coerção teria que INVENTAR um float. ``int`` é a
    exceção delimitada — `42` → `"42"` não inventa nada, é bijetivo, e UNIFICA o
    escopo em vez de dividi-lo. Já ``str(42.0)`` é `"42.0"`, que não casa com
    `"42"`, e ``str(True)`` é `"True"`, um escopo fantasma: os dois seriam
    divisões SILENCIOSAS de escopo, exatamente o defeito que a doutrina evita.

    ⚠️ ``bool`` precisa de guarda própria porque ``isinstance(True, int)`` é
    ``True`` em Python — sem ela, `True` viraria o escopo `"True"`.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"Invalid {name}: got bool, which is not an identifier. "
            f"Provide a string (or an integer id)."
        )
    if isinstance(value, int):
        value = str(value)
    elif not isinstance(value, str):
        raise ValueError(
            f"Invalid {name}: expected str (or int), got "
            f"{type(value).__name__}. Provide a valid identifier."
        )
    trimmed = value.strip()
    if trimmed == "":
        raise ValueError(
            f"Invalid {name}: cannot be empty or whitespace-only. Provide a valid identifier."
        )
    if any(c.isspace() for c in trimmed):
        raise ValueError(
            f"Invalid {name}: cannot contain whitespace. Provide a valid identifier without spaces."
        )
    return trimmed


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
