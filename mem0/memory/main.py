import asyncio
import concurrent.futures
import gc
import functools
import hashlib
import json
import logging
import math
import os
import re
import random as _random
import threading
import time
import uuid
import warnings
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from pydantic import ValidationError

from mem0.configs.base import RERANK_TIE_BAND, MemoryConfig, MemoryItem
from mem0.configs.enums import MemoryType
from mem0.configs.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    DOCUMENT_TEMPORAL_OVERRIDE,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    build_speaker_attribution_suffix,
    build_temporality_suffix,
    generate_additive_extraction_prompt,
)
from mem0.exceptions import ValidationError as Mem0ValidationError
from mem0.memory.base import MemoryBase
from mem0.memory.setup import mem0_dir, setup_config
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import MEM0_TELEMETRY, capture_event
from mem0.memory.notices import (
    PERFORMANCE_SLOW_QUERY_THRESHOLD_SECONDS,
    detect_scale_threshold_from_add_result,
    detect_scale_threshold_from_top_k,
    detect_decay_usage_from_delete,
    detect_decay_usage_from_delete_all,
    detect_temporal_usage_from_metadata,
    detect_temporal_usage_from_search,
    display_decay_usage_notice,
    display_decay_usage_notice_async,
    display_first_run_notice,
    display_first_run_notice_async,
    display_performance_slow_query_notice,
    display_performance_slow_query_notice_async,
    display_scale_threshold_notice,
    display_scale_threshold_notice_async,
    display_temporal_usage_notice,
    display_temporal_usage_notice_async,
    get_decay_feature_error_message,
    get_decay_feature_error_message_async,
    get_temporal_feature_error_message,
    get_temporal_feature_error_message_async,
)
from mem0.memory.utils import (
    MAX_SPEAKER_LABEL,
    entity_point_id,
    link_key,
    links_do_payload,
    locutores_das_mensagens,
    normalize_entity_key,
    normalize_scope_id,
    normalize_speaker_label,
    extract_json,
    normalize_linked_memory_ids,
    parse_messages,
    parse_vision_messages,
    precisa_de_atribuicao_por_llm,
    process_telemetry_filters,
    remove_code_blocks,
    resolver_locutor_do_fato,
    speaker_attribution_enabled,
)
from mem0.utils.entity_extraction import extract_entities, extract_entities_batch
from mem0.utils.factory import (
    EmbedderFactory,
    LlmFactory,
    RerankerFactory,
    VectorStoreFactory,
)
from mem0.utils.dynamics import (
    DYNAMICS_FIELDS,
    FIELD_FIRST_SEEN,
    OUTCOME_APPLIED,
    OUTCOME_DROPPED,
    OUTCOME_FAILED,
    OUTCOME_MISSING,
    OUTCOME_SUPPRESSED,
    TRIGGER_DEDUP,
    TRIGGER_SEARCH,
    TRIGGER_SIMILAR,
    TRIGGER_UPDATE,
    _anchor_ts,
    _parse_ts as _dynamics_parse_ts,
    boost_from_payload,
    reinforcement_fields,
    should_reinforce,
    utcnow as _dynamics_utcnow,
)
from mem0.utils.lemmatization import lemmatize_for_bm25
from mem0.utils.scoring import (
    ENTITY_BOOST_WEIGHT,
    get_bm25_params,
    normalize_bm25,
    score_and_rank,
)
from mem0.utils.temporality import (
    FIELD_EVENT_DATE,
    FIELD_LINEAGE_SCHEMA,
    FIELD_SUPERSEDED_AT,
    FIELD_SUPERSEDED_BY,
    FIELD_SUPERSEDES,
    FIELD_VERSION_NEXT,
    FIELD_VERSION_PREV,
    LINEAGE_SCHEMA_VERSION,
    RESERVED_LINEAGE_FIELDS,
    event_proximity,
    expand_event_window,
    infer_event_anchor_from_query,
    infer_event_date_from_text,
    parse_as_of,
    parse_event_date,
    parse_supersedes_ids,
    superseded_penalty_applies,
    supersession_inverted,
)
from mem0.vector_stores.base import VectorStoreBase

# Suppress SWIG deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
logger = logging.getLogger(__name__)


# Fields that hold runtime auth/connection objects and must be preserved.
# These are non-serializable objects (e.g. AWSV4SignerAuth, RequestsHttpConnection)
# needed by clients like OpenSearch — not sensitive strings to redact.
_RUNTIME_FIELDS = frozenset({
    "http_auth",
    "auth",
    "connection_class",
    "ssl_context",
})

# Fields that are known to contain sensitive secrets and must be redacted.
_SENSITIVE_FIELDS_EXACT = frozenset({
    "api_key",
    "secret_key",
    "private_key",
    "access_key",
    "password",
    "credentials",
    "credential",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "auth_token",
    "session_token",
    "client_secret",
    "auth_client_secret",
    "azure_client_secret",
    "service_account_json",
    "aws_session_token",
})

# Suffixes that indicate a field likely holds a secret value.
_SENSITIVE_SUFFIXES = (
    "_password",
    "_secret",
    "_token",
    "_credential",
    "_credentials",
)

# Entity parameters that must be passed via filters, not top-level kwargs
ENTITY_PARAMS = frozenset({"user_id", "agent_id", "run_id"})


def _extract_top_level_entity_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """DeepMem0: accept user_id/agent_id/run_id as top-level keyword sugar.

    Upstream 2.0.x raised ValueError demanding filters={...}, breaking every
    pre-2.0 caller for no gain. The scalars are folded into filters instead
    (an explicit filters dict wins on conflict).
    """
    return {k: kwargs.pop(k) for k in ENTITY_PARAMS if k in kwargs}


def _apply_metadata_post_filters(
    memories,
    *,
    min_importance: Optional[float] = None,
    domain: Optional[str] = None,
    memory_type: Optional[str] = None,
    sort_by_importance: bool = False,
):
    """DeepMem0: post-hoc filtering/ordering over classified metadata.

    Operates on the metadata dict of each result (keys such as importance,
    domain and memory_type, typically written by an application-level
    classifier). Memories without the key are excluded by that filter.
    """
    if not memories:
        return memories
    if min_importance is None and not domain and not memory_type and not sort_by_importance:
        return memories

    def _meta(m):
        return (m.get("metadata") or {}) if isinstance(m, dict) else {}

    def _rankable(value):
        """Number that can be compared AND ordered without surprising the caller.

        `isinstance(x, (int, float))` alone is not enough: `bool` is a subclass of
        `int` (so `True` would rank as 1.0) and NaN breaks ordering invariants
        (every comparison is False, so the sort result depends on input order).
        Payloads are written by external callers, so this stays defensive even
        when a write-side contract is in place.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(float(value))
        except (OverflowError, ValueError):
            return False

    out = memories
    if min_importance is not None:
        out = [
            m for m in out
            if _rankable(_meta(m).get("importance"))
            and _meta(m)["importance"] >= min_importance
        ]
    if domain:
        out = [m for m in out if _meta(m).get("domain") == domain]
    if memory_type:
        out = [m for m in out if _meta(m).get("memory_type") == memory_type]
    if sort_by_importance:
        # A chave TEM que espelhar a guarda de isinstance do min_importance acima:
        # importance vinda do caller (metadata= no add) NÃO passa pelo coerce do
        # classificador, então o corpus real tem strings ("high"). Sem a guarda,
        # sorted() compara str com float e derruba a busca inteira com
        # "'<' not supported between instances of 'str' and 'float'" — foi o que
        # aconteceu no Open WebUI em 26/07/2026 (17 memórias 'high' no corpus).
        # Não-numérico ordena como 0.0 em vez de crashar: o filtro já trata o
        # mesmo caso excluindo; ordenar nunca pode ser mais frágil que filtrar.
        out = sorted(
            out,
            key=lambda m: (
                float(_meta(m)["importance"]) if _rankable(_meta(m).get("importance")) else 0.0
            ),
            reverse=True,
        )
    return out


def _validate_and_trim_entity_id(value: Optional[Any], name: str) -> Optional[str]:
    """
    Validates and normalizes an entity ID.
    - Coerces integer ids to str (a database primary key is a legitimate id)
    - Rejects bool, float and any other non-string type
    - Trims leading/trailing whitespace
    - Rejects empty or whitespace-only strings
    - Rejects strings containing internal whitespace

    Thin alias over :func:`mem0.memory.utils.normalize_scope_id`, which is the
    PUBLIC form of this rule. It is public because the same normalization has to
    happen at every boundary that builds a scope filter — including callers
    outside this package — and depending on a leading-underscore name to do it
    would be depending on an implementation detail.

    Args:
        value: The entity ID value to validate
        name: The parameter name (for error messages)

    Returns:
        The trimmed entity ID, or None if input is None

    Raises:
        ValueError: If entity ID is invalid
    """
    return normalize_scope_id(value, name)


def _validate_search_params(threshold: Optional[float] = None, top_k: Optional[int] = None) -> None:
    """
    Validates search parameters.

    Args:
        threshold: Similarity threshold (must be between 0 and 1)
        top_k: Number of results to return (must be non-negative integer)

    Raises:
        ValueError: If threshold or top_k are invalid
    """
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a valid number")
        if threshold < 0 or threshold > 1:
            raise ValueError(
                f"Invalid threshold: {threshold}. Must be between 0 and 1 (inclusive)."
            )
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a valid integer")
        if top_k < 0:
            raise ValueError(
                f"Invalid top_k: {top_k}. Must be a non-negative integer."
            )


def _validate_and_trim_search_query(query: str) -> str:
    """
    Validates and normalizes a search query before embedding/vector search.

    Raises:
        ValueError: If query is not a string or is empty/whitespace-only.
    """
    if not isinstance(query, str):
        raise ValueError("Invalid query: must be a non-empty string.")
    trimmed = query.strip()
    if not trimmed:
        raise ValueError("Invalid query: cannot be empty or whitespace-only.")
    return trimmed


def _is_sensitive_field(field_name: str) -> bool:
    """Check if a field should be redacted for telemetry safety.

    Uses a layered approach:
    1. Runtime fields (allowlist) — always preserved, highest priority.
    2. Exact deny list — known secret field names.
    3. Suffix deny list — catches patterns like db_password, auth_secret, etc.
    """
    name = field_name.lower().strip()
    if name in _RUNTIME_FIELDS:
        return False
    if name in _SENSITIVE_FIELDS_EXACT:
        return True
    return any(name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


def _safe_deepcopy_config(config):
    """Safely deepcopy config, falling back to dict-based cloning for non-serializable objects."""
    try:
        return deepcopy(config)
    except Exception as e:
        logger.debug(f"Deepcopy failed, using dict-based cloning: {e}")

        config_class = type(config)

        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump()
            except Exception:
                clone_dict = dict(config.__dict__)
        else:
            clone_dict = dict(config.__dict__)

        # Restore runtime fields, redact sensitive ones
        for field_name in list(clone_dict.keys()):
            if field_name in _RUNTIME_FIELDS and hasattr(config, field_name):
                clone_dict[field_name] = getattr(config, field_name)
            elif _is_sensitive_field(field_name):
                clone_dict[field_name] = None

        try:
            return config_class(**clone_dict)
        except Exception:
            logger.debug("Config reconstruction failed, returning shallow dict clone")
            return type("Config", (), clone_dict)()


def _normalize_iso_timestamp_to_utc(timestamp: Optional[str]) -> Optional[str]:
    """Normalize timezone-aware ISO timestamps to UTC without rewriting naive values."""
    if not timestamp:
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    # DeepMem0 v0.7.1: strip RESERVED lineage fields from caller metadata so a client
    # cannot inject a version edge (`_mem0_version_*`) via add — only the versioned
    # update transition may write them (critic #3).
    base_metadata_template = {
        k: deepcopy(v) for k, v in input_metadata.items() if k not in RESERVED_LINEAGE_FIELDS
    } if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- validate and add all provided session ids ----------
    session_ids_provided = []

    # Validate and trim entity IDs
    user_id = _validate_and_trim_entity_id(user_id, "user_id")
    agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
    run_id = _validate_and_trim_entity_id(run_id, "run_id")

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


def _build_session_scope(filters):
    """Build deterministic session scope string from entity IDs."""
    parts = []
    for key in sorted(["user_id", "agent_id", "run_id"]):
        val = filters.get(key)
        if val:
            parts.append(f"{key}={val}")
    return "&".join(parts)


def _entity_collection_name(provider: str, collection_name: str) -> str:
    separator = "-" if provider == "s3_vectors" else "_"
    return f"{collection_name}{separator}entities"


# delete_all pagination. Module-level so tests can shrink them.
_DELETE_ALL_PAGE_SIZE = 1000
_DELETE_ALL_MAX_PAGES = 1000


# Cap for the entity-store scans done during cleanup. The number matters less
# than the alarm: a silently truncated scan turns cleanup into a PARTIAL no-op,
# leaving dangling links that nobody will ever go looking for.
ENTITY_SCAN_TOP_K = int(os.environ.get("MEM0_ENTITY_SCAN_TOP_K", "100000"))


def _entity_cleanup_enabled() -> bool:
    """``auto`` (default) always cleans; ``lazy`` restores the old behaviour of
    skipping cleanup in a process that never touched the entity store.

    The old guard (``if self._entity_store is None: return``) was a silent
    correctness hole: a short-lived process that only deletes — a test harness,
    a smoke script, a cron job — never initializes the entity store, so every
    link it should have removed survived forever. Those are exactly the orphan
    rows found in the production corpus.
    """
    return os.environ.get("MEM0_ENTITY_CLEANUP", "auto").strip().lower() != "lazy"


def _scan_entity_rows(store, search_filters):
    """(rows, truncated) for the scope. Truncation is RETURNED, not just logged:
    a silently truncated scan turns cleanup into a partial no-op, and the caller
    has to know that before it commits a delete intent."""
    listed = store.list(filters=search_filters, top_k=ENTITY_SCAN_TOP_K)
    rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
    rows = rows or []
    truncated = len(rows) >= ENTITY_SCAN_TOP_K
    if truncated:
        logger.warning(
            "Entity scan hit the %d-row cap for scope %s — cleanup is INCOMPLETE. "
            "Raise MEM0_ENTITY_SCAN_TOP_K.", ENTITY_SCAN_TOP_K, search_filters)
    return rows, truncated


# ============================================================
# Helpers de IDENTIDADE de entidade — compartilhados pelos DOIS gêmeos
# ============================================================
# ⚠️ Nasceram como métodos só de `Memory`, e `_upsert_entity_async` (que vive em
# `AsyncMemory`) passou a chamá-los: `AttributeError` a cada escrita, engolido
# pelo `except` largo do upsert e transformado num warning. Ou seja, TODA escrita
# de entidade do caminho assíncrono falharia em silêncio. Como função de módulo o
# gêmeo não tem como divergir de novo.

SCOPE_KEYS = ("user_id", "agent_id", "run_id")


def escopo_exato(payload, search_filters) -> bool:
    """A linha pertence EXATAMENTE a este escopo — não a um mais estreito.

    ⚠️ Filtro do Qdrant casa por SUBCONJUNTO: `{user_id: X}` casa qualquer linha
    com esse `user_id`, INCLUSIVE as que também carregam `run_id`. Conferir só as
    chaves BUSCADAS (o que este código fazia) aceita a linha estreita; o que
    decide é a AUSÊNCIA das outras.

    MEDIDO em produção (31/07/2026): a linha `DeepMem0` do escopo de teste
    `{user_id: U, run_id: R}` acumulou 12 vínculos de
    memórias do escopo largo `{user_id: U}` gravadas pelo worker, enquanto
    a linha certa — escopo largo, 108 vínculos — ficava de fora. `check_corpus`
    acusou `entity_cross_scope_rows=1`.

    Isto realinha o LOOKUP com o ID: `entity_point_id` já deriva do escopo
    inteiro, então uma linha de escopo diferente nunca tem o id que este
    escritor usaria — casá-la era garantir divergência entre achar e escrever.
    """
    if not isinstance(payload, dict):
        return False
    for k in SCOPE_KEYS:
        if (payload.get(k) or None) != (search_filters.get(k) or None):
            return False
    return True


def _linhas_de_list(achados):
    """Desembrulha o retorno de `entity_store.list`, VALIDANDO A FORMA.

    O Qdrant devolve o tuple cru do `scroll` (`(pontos, next_page)`); outros
    stores e duplos de teste devolvem outras coisas — e qualquer uma delas é
    truthy. Aceitar truthy como acerto fazia o caminho exato "encontrar" um
    registro inexistente e pular a sonda vetorial.
    """
    if isinstance(achados, tuple):
        achados = achados[0] if achados else []
    if not isinstance(achados, list) or not achados:
        return []
    if isinstance(achados[0], list):
        achados = achados[0]
        if not isinstance(achados, list):
            return []
    return [l for l in achados
            if getattr(l, "id", None) is not None
            and isinstance(getattr(l, "payload", None), dict)]


def entidades_por_chaves(entity_store, chaves, search_filters):
    """Lookup exato EM LOTE: `({chave: linha}, {chaves ambíguas})`.

    Uma chamada de `list()` para todas as chaves, via `MatchAny` em
    `data_normalized` — que é indexado. Filtro de PAYLOAD, não `retrieve` por id
    determinístico: `data_normalized` é independente do id, e depender do id
    seria depender de um invariante que nada enforça.

    ⚠️ LEVANTA em erro do store. Quem chama decide, e a decisão nunca pode ser
    "trata tudo como novo" — isso converte falha de infraestrutura em escrita
    destrutiva, porque o insert no id determinístico SUBSTITUI o payload alheio.

    `ambiguas` = chaves com mais de uma linha no mesmo escopo exato. É o
    invariante de duplicata disparando; escolher uma em silêncio é o que o
    `top_k=1` sobre filtro que casa 2 linhas fazia.
    """
    chaves = [c for c in dict.fromkeys(chaves or []) if c]
    if not chaves:
        return {}, set()

    def _busca(filtros, limite):
        """Devolve (linhas válidas, saturou).

        ⚠️ `top_k` é consumido pelo store ANTES da validação de escopo em
        Python — e o filtro do Qdrant traz também as linhas de escopo
        SUBCONJUNTO. Com o corte cheio, a linha EXATA pode ter ficado de fora,
        e lê-la como ausente leva a inserir no id determinístico, ou seja, a
        apagar payload alheio: a mesma corrupção por outra porta. Saturou =
        não sei responder, e não saber tem de FECHAR.
        """
        brutas = entity_store.list(filters=filtros, top_k=limite)
        vistas = brutas[0] if isinstance(brutas, tuple) and brutas else brutas
        n = len(vistas) if isinstance(vistas, list) else 0
        return _linhas_de_list(brutas), n >= limite

    def _um(chave):
        """Igualdade simples: a forma que TODO backend suporta."""
        return _busca({**search_filters, "data_normalized": chave}, 8)

    truncado = False
    if len(chaves) == 1:
        linhas, truncado = _um(chaves[0])
    else:
        limite = max(2 * len(chaves), len(chaves) + 8)
        try:
            linhas, truncado = _busca(
                {**search_filters, "data_normalized": {"in": chaves}}, limite)
        except Exception as exc:
            # `{"in": ...}` é sintaxe estendida; um backend sem ela não pode
            # degradar para "nada existe" — isso faria o insert no id
            # determinístico substituir payload alheio. Cai para N lookups por
            # igualdade, que é lento e correto, em vez de rápido e destrutivo.
            logger.debug("MatchAny indisponível (%s); lookup por chave", exc)
            linhas = []
            for c in chaves:
                ls, tr = _um(c)
                linhas.extend(ls)
                truncado = truncado or tr

    if truncado:
        raise RuntimeError(
            f"lookup de entidade saturou o limite para {len(chaves)} chave(s): "
            "não é possível afirmar ausência, e afirmar ausência aqui insere "
            "sobre payload alheio")

    alvo = set(chaves)
    encontradas, ambiguas = {}, set()
    for linha in linhas:
        pl = linha.payload
        ch = pl.get("data_normalized")
        if ch not in alvo or not escopo_exato(pl, search_filters):
            continue
        if ch in encontradas:
            ambiguas.add(ch)
            continue
        encontradas[ch] = linha
    for ch in ambiguas:
        encontradas.pop(ch, None)
    if ambiguas:
        logger.warning(
            "entity store: %d chave(s) com linha duplicada no mesmo escopo — "
            "não escolho em silêncio: %s", len(ambiguas), sorted(ambiguas))
    return encontradas, ambiguas


def _mensagens_validas_para_add(messages):
    """Mensagens que o caminho `infer=False` de fato grava, NA ORDEM.

    Separado do laço de escrita para que o embed possa sair dali sem que os
    dois critérios de descarte — formato inválido e `role == "system"` —
    mudem de lugar ou de ordem junto.
    """
    validas = []
    for message_dict in messages:
        if (
            not isinstance(message_dict, dict)
            or message_dict.get("role") is None
            or message_dict.get("content") is None
        ):
            logger.warning(f"Skipping invalid message format: {message_dict}")
            continue
        if message_dict["role"] == "system":
            continue
        validas.append(message_dict)
    return validas


def _embed_map_de(embedding_model, conteudos):
    """`{texto: vetor}` para os textos que embedaram com sucesso.

    ⚠️ Chave por CONTEÚDO é o contrato que `_create_memory` já espera. Textos
    repetidos colapsam numa entrada — o que é correto, porque o vetor de um
    texto é o mesmo —, mas a CONTAGEM devolvida pelo lote tem de bater com a
    dos textos ANTES do zip: `zip` trunca em silêncio, e truncar aqui colaria o
    vetor de um fato em outro.

    Um embed que falha derruba UMA mensagem, não o lote — que era o
    comportamento do laço item a item e não pode piorar por causa do hoist.
    """
    unicos = list(dict.fromkeys(c for c in conteudos if c is not None))
    if not unicos:
        return {}
    try:
        vetores = embedding_model.embed_batch(unicos, "add")
        if len(vetores) != len(unicos):
            raise ValueError(
                f"embed_batch devolveu {len(vetores)} vetores para "
                f"{len(unicos)} textos")
        return dict(zip(unicos, vetores))
    except Exception as exc:
        logger.warning("embed_batch falhou (%s); caindo para item a item", exc)

    mapa = {}
    for texto in unicos:
        try:
            mapa[texto] = embedding_model.embed(texto, "add")
        except Exception as exc:
            logger.warning("Failed to embed message content: %s", exc)
    return mapa


def vincular_entidades_em_lote(entity_store, embedding_model, memory_id,
                               entidades, search_filters):
    """Vincula N entidades a UMA memória com um embed e um lookup, não N de cada.

    MEDIDO no ramo de UPDATE (bge-m3, Qdrant real, collection isolada, 7
    repetições por ponto): o embed é **95-96% do wall** e as idas ao Qdrant
    somam ~18 ms por entidade — `ms/entidade` fica plano em ~450-520 ms de N=1 a
    N=16, ou seja o custo é do OVERHEAD POR CHAMADA. Um lote troca N×~455 ms por
    1×(~500 + 12N) ms.

    ⚠️ Não é o mesmo que o ramo de INSERT, que paga `wait=True` mais
    reconciliação e é muito mais caro — medir aquele e concluir sobre este foi
    erro de uma medição anterior.

    Devolve `True` se tratou o lote; `False` quando o chamador deve cair no
    caminho serial (`_upsert_entity`), que é o comportamento seguro de sempre.

    As REGRAS de identidade não vivem aqui: escopo exato, multiplicidade,
    truncamento e forma de resposta vêm de `entidades_por_chaves` e
    `escopo_exato`, os mesmos que a Fase 7 e o `_upsert_entity` usam. Só a
    orquestração é nova — é o que impede este caminho de virar uma terceira
    regra de identidade.
    """
    por_chave = {}
    for entity_type, entity_text in entidades:
        chave = normalize_entity_key(entity_text)
        if not chave or chave in por_chave:
            continue
        por_chave[chave] = (entity_type, entity_text)
    if not por_chave:
        return True

    chaves = list(por_chave)
    textos = [por_chave[k][1] for k in chaves]

    try:
        vetores = embedding_model.embed_batch(textos, "add")
        if len(vetores) != len(textos):
            raise ValueError(
                f"embed_batch devolveu {len(vetores)} vetores para "
                f"{len(textos)} entidades")
    except Exception as exc:
        # Embed é o ÚNICO passo cujo fracasso pode cair no serial sem risco: o
        # serial refaz o lookup por conta própria e mantém a mesma regra.
        logger.warning("embed_batch de entidade falhou (%s); caminho serial", exc)
        return False

    try:
        achadas, ambiguas = entidades_por_chaves(entity_store, chaves, search_filters)
    except Exception as exc:
        # FAIL-CLOSED: cair no serial aqui repetiria o mesmo lookup quebrado e
        # acabaria inserindo no id determinístico sobre payload alheio.
        logger.warning(
            "lookup exato de entidade falhou (%s) — vínculo de %s pulado",
            exc, memory_id)
        return True
    if ambiguas:
        logger.warning("chave(s) ambígua(s) puladas em %s — %s",
                       memory_id, sorted(ambiguas))

    processar = [i for i, k in enumerate(chaves) if k not in ambiguas]
    faltantes = [i for i in processar if chaves[i] not in achadas]
    sondados_por_indice = {}
    if faltantes:
        sondados = entity_store.search_batch(
            queries=[textos[i] for i in faltantes],
            vectors_list=[vetores[i] for i in faltantes],
            top_k=1,
            filters=search_filters,
        )
        if (not isinstance(sondados, list)
                or len(sondados) != len(faltantes)
                or not all(isinstance(x, list) for x in sondados)):
            logger.warning(
                "search_batch de entidade devolveu forma inesperada — vínculo "
                "de %s pulado", memory_id)
            return True
        for pos, i in enumerate(faltantes):
            sondados_por_indice[i] = sondados[pos]

    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
    for i in processar:
        chave = chaves[i]
        entity_type, entity_text = por_chave[chave]
        alvo = achadas.get(chave)
        if alvo is None:
            candidatas = [m for m in sondados_por_indice.get(i, [])
                          if escopo_exato(getattr(m, "payload", None) or {},
                                          search_filters)]
            if candidatas and candidatas[0].score >= 0.95:
                alvo = candidatas[0]

        if alvo is not None:
            payload = alvo.payload or {}
            raw_linked = payload.get("linked_memory_ids")
            linked_ids = normalize_linked_memory_ids(raw_linked)
            if memory_id not in linked_ids:
                linked_ids.append(memory_id)
            precisa_chave = payload.get("data_normalized") != chave
            if linked_ids != raw_linked or precisa_chave:
                payload["linked_memory_ids"] = linked_ids
                payload["data_normalized"] = chave
                payload[link_key(memory_id)] = 1
                try:
                    entity_store.update(vector_id=alvo.id, vector=None,
                                        payload=payload)
                except Exception as exc:
                    logger.debug("Entity update failed for '%s': %s",
                                 entity_text, exc)
        else:
            to_insert_vectors.append(vetores[i])
            to_insert_ids.append(entity_point_id(search_filters, chave))
            to_insert_payloads.append({
                "data": entity_text,
                "data_normalized": chave,
                "entity_type": entity_type,
                "linked_memory_ids": [memory_id],
                link_key(memory_id): 1,
                **search_filters,
            })

    if to_insert_vectors:
        try:
            entity_store.insert(vectors=to_insert_vectors, ids=to_insert_ids,
                                payloads=to_insert_payloads, wait=True)
        except Exception as exc:
            logger.warning("Batch entity insert failed: %s", exc)
            return True
        for eid in to_insert_ids:
            reconcilia_vinculo(entity_store, eid, memory_id,
                               ENTITY_RECONCILE_ATTEMPTS)
    return True


def resolver_linha_de_entidade(entity_store, entity_text, chave,
                               entity_embedding, search_filters):
    """Decide o destino desta entidade, sem escrever nada.

    Devolve `("update", linha)`, `("insert", None)` ou `("skip", motivo)`.

    Função de MÓDULO de propósito. Os gêmeos sync/async já divergiram nesta
    decisão exata — o assíncrono ficou sem a identidade normalizada quando o
    síncrono ganhou —, e gêmeo que diverge é pior que nenhum: o mesmo corpus
    passa a ter duas regras de identidade conforme o caminho que escreveu.
    Concentrar a decisão aqui é o que impede a terceira ocorrência.

    Toda saída duvidosa é `skip`, nunca `insert`: o insert usa o id
    determinístico e SUBSTITUI o payload alheio, então "não sei" que vira
    escrita é perda de dado.
    """
    try:
        achadas, ambiguas = entidades_por_chaves(
            entity_store, [chave], search_filters)
    except Exception as exc:
        return "skip", f"lookup exato falhou ({exc})"
    if chave in ambiguas:
        return "skip", ("chave duplicada neste escopo "
                        "(resolver com scripts/repair_entity_links.py)")
    exata = achadas.get(chave)
    if exata is not None:
        return "update", exata

    # Sonda vetorial: só alcança a linha LEGADA, sem `data_normalized`
    # (medido: 10 dessas no corpus — a sonda é carga real, não vestígio).
    sondadas = entity_store.search(
        query=entity_text, vectors=entity_embedding, top_k=1,
        filters=search_filters)
    if not isinstance(sondadas, list):
        # Iterar cegamente converteria resposta malformada em "nenhum
        # resultado", e nenhum resultado leva a inserir sobre estado
        # desconhecido.
        return "skip", f"sonda devolveu {type(sondadas).__name__}, não lista"
    # A sonda herda o filtro por SUBCONJUNTO do store: com escopo largo ela
    # devolve a linha estreita, e com texto idêntico o score é 1.0.
    candidatas = [l for l in sondadas
                  if escopo_exato(getattr(l, "payload", None) or {},
                                  search_filters)]
    if candidatas and candidatas[0].score >= 0.95:
        return "update", candidatas[0]
    return "insert", None


def entidade_por_chave(entity_store, chave, search_filters, checks_ref=None):
    """Linha de entidade com esta chave normalizada NESTE escopo, ou None.

    Identidade EXATA, e é a diferença que importa: a identidade era a
    SIMILARIDADE do vetor (>= 0.95), que é probabilística. O corpus mostrou o
    preço — `FASE`/`Fase`, `docker compose`/`Docker Compose`,
    `Hilbert transform`/`Hilbert Transform` viraram linhas SEPARADAS, cada uma
    com sua fatia de vínculos, e qual delas recebe o boost passou a depender
    da grafia que o usuário digitou.

    Fino de `entidades_por_chaves`: mesma regra de escopo e de multiplicidade.
    """
    try:
        encontradas, _amb = entidades_por_chaves(
            entity_store, [chave], search_filters)
    except Exception as exc:      # store sem suporte ao filtro: cai na sonda
        logger.debug("lookup por chave normalizada indisponível: %s", exc)
        return None
    return encontradas.get(chave)


ENTITY_RECONCILE_ATTEMPTS = int(os.environ.get("MEM0_ENTITY_RECONCILE_ATTEMPTS", "6"))
ENTITY_RECONCILE_BACKOFF_S = float(
    os.environ.get("MEM0_ENTITY_RECONCILE_BACKOFF_S", "0.08"))


def reconcilia_vinculo(entity_store, entity_id, memory_id,
                       tentativas: int = ENTITY_RECONCILE_ATTEMPTS):
    """Relê e reanexa o vínculo. ESTREITA a janela de lost-update; NÃO fecha.

    ⚠️ AFIRMAÇÃO CORRIGIDA. Eu escrevi que isto "fecha" a corrida
    check-then-insert. Não fecha, e a análise honesta é esta:

      * ordem que ISTO RESOLVE — A insere · B sobrescreve · A relê: A vê o
        valor de B, reanexa o próprio id, os dois sobrevivem;
      * ordem que NINGUÉM resolve do lado do cliente — A insere · A relê ·
        A termina · B sobrescreve: quando B escreve, A já foi embora. Nenhum
        número de tentativas de A alcança uma escrita posterior a ela.

    Qdrant faz MERGE de CHAVES no `set_payload`, mas o valor de uma chave de
    LISTA é substituído: não existe união atômica para usar, e sem CAS não há
    fecho client-side. O residual é DETECTADO, não prevenido — pela cobertura
    de vínculo do `check_corpus` e porque o próximo add da mesma entidade
    reanexa o id que faltava.

    As tentativas existem para o caso em que a sobrescrita chega durante a
    própria reconciliação; o teto fica declarado e o fracasso vira WARNING,
    porque um vínculo perdido em silêncio foi a origem de todo este trabalho.
    """
    estaveis = 0
    for tentativa in range(1, tentativas + 1):
        try:
            atual = entity_store.get(vector_id=entity_id)
            # CÓPIA: mutar o dict que veio do store faz a leitura seguinte
            # enxergar a própria escrita se o store reaproveitar o objeto —
            # o loop passa a convergir contra si mesmo em vez de contra o
            # estado real. Um teste encontrou isso ao devolver o mesmo dict
            # duas vezes.
            payload = dict((getattr(atual, "payload", None) or {}) if atual else {})
            # UNIÃO com as chaves `lnk_*`: elas sobrevivem ao merge de
            # `set_payload`, a lista não.
            linked = links_do_payload(payload)
            if memory_id in linked:
                # ⚠️ NÃO sair na PRIMEIRA vista. O escritor enxerga o PRÓPRIO
                # insert e ia embora satisfeito; um insert de OUTRO escritor
                # chegando depois apagava tudo e ninguém restava para notar.
                # Exigir observação ESTÁVEL (presente em duas leituras espaçadas)
                # é o que cobre o insert tardio.
                estaveis += 1
                if estaveis >= 2:
                    return True
                if ENTITY_RECONCILE_BACKOFF_S > 0:
                    time.sleep(ENTITY_RECONCILE_BACKOFF_S * tentativa
                               * (0.5 + _random.random()))
                continue
            estaveis = 0
            linked = sorted(set(linked) | {memory_id})
            payload["linked_memory_ids"] = linked
            payload[link_key(memory_id)] = 1
            entity_store.update(vector_id=entity_id, vector=None,
                                     payload=payload)
            logger.info("entity %s: vínculo %s reanexado (tentativa %d)",
                        entity_id, memory_id, tentativa)
            # ESPERA CRESCENTE. Sem ela as tentativas se atropelam e caem todas
            # ANTES de a rajada de inserts concorrentes terminar. MEDIDO com 8
            # escritores simultâneos: 87,5% dos vínculos perdidos, porque
            # `insert` SUBSTITUI o ponto inteiro e apaga o vínculo de quem já
            # havia escrito. Espaçar faz a verificação cair DEPOIS dos inserts,
            # que é quando ela recupera.
            if tentativa < tentativas and ENTITY_RECONCILE_BACKOFF_S > 0:
                # JITTER, não espera fixa. Com backoff uniforme os N escritores
                # continuam SINCRONIZADOS e voltam a colidir nas mesmas janelas —
                # medido: 87,5% de perda com 8 escritores, inalterado pelo
                # backoff sem aleatoriedade. O jitter é o que desempata leituras
                # concorrentes de read-modify-write sem CAS.
                time.sleep(ENTITY_RECONCILE_BACKOFF_S * tentativa
                           * (0.5 + _random.random()))
        except Exception as exc:
            logger.debug("reconciliação de vínculo falhou para %s: %s",
                         entity_id, exc)
            return False
    if ENTITY_RECONCILE_BACKOFF_S > 0:
        time.sleep(ENTITY_RECONCILE_BACKOFF_S * tentativas
                   * (0.5 + _random.random()))
    try:
        atual = entity_store.get(vector_id=entity_id)
        ok = memory_id in links_do_payload(getattr(atual, "payload", None) or {})
    except Exception:
        ok = False
    if not ok:
        logger.warning(
            "entity %s: vínculo %s NÃO grudou em %d tentativas — escrita "
            "concorrente sustentada. O vínculo está perdido e isto é o "
            "aviso, não um detalhe de log.", entity_id, memory_id, tentativas)
    return ok


#: Hook for observability of DELETE, called as
#: ``(memory_id, phase, metrics)`` where `metrics` carries what only the core can
#: see. Same shape as `reinforcement_observer`, and for the same reason: a
#: companion outside the core should not have to patch internals to observe.
#:
#: Instrumentar isto de fora era impossível sem reimplementar a função: `rows` e
#: `truncated` são LOCAIS de `_scan_entity_rows`, e envolver `Memory.delete`
#: público não os expõe. `truncated` é o dado que mais importa — é o único sinal
#: de que a limpeza pode ter deixado vínculo para trás.
#: Must never raise; exceptions are swallowed.
delete_observer = None


def _notify_delete(memory_id, phase, **metrics) -> None:
    observer = delete_observer
    if observer is None:
        return
    try:
        observer(memory_id, phase, metrics)
    except Exception:  # observabilidade nunca derruba o caminho de delete
        pass


def unlink_memory_from_entity_rows(store, memory_id, filters) -> bool:
    """Remove `memory_id` from every entity row in scope. Synchronous.

    Returns True when the cleanup is known to be COMPLETE, False when anything
    was swallowed (scan failure, a row that would not update, a truncated scan).
    The caller needs that answer: committing the delete intent after a failed
    cleanup is what turns a transient error into a permanent dangling link,
    because reconciliation then has nothing left to retry.

    Shared by the sync delete path and by BOTH reconcilers — the async
    reconciler runs at construction time, where there is no event loop to await
    the async variant, and duplicating this logic is how the two paths would
    drift apart.
    """
    search_filters = {k: v for k, v in (filters or {}).items()
                      if k in ("user_id", "agent_id", "run_id") and v}
    ok = True
    t0 = time.time()
    n_linhas = n_scan = 0
    truncated = False
    try:
        rows, truncated = _scan_entity_rows(store, search_filters)
        n_scan = len(rows)
        if truncated:
            ok = False
        for row in rows:
            try:
                payload = getattr(row, "payload", None) or {}
                # Normalizing (instead of `isinstance(...) -> continue`) is what
                # lets cleanup REACH a corrupted row; the old guard made such a
                # row keep its dangling refs forever.
                linked = normalize_linked_memory_ids(payload.get("linked_memory_ids"))
                if memory_id not in linked:
                    continue
                remaining = [mid for mid in linked if mid != memory_id]
                if not remaining:
                    try:
                        store.delete(vector_id=row.id)
                        n_linhas += 1
                    except Exception as e:
                        ok = False
                        logger.debug(f"Entity delete failed for id={row.id}: {e}")
                else:
                    # Payload-only. Unlinking does not change the entity TEXT, so
                    # re-embedding was pure cost: it re-encoded BM25 and rewrote
                    # the vector, perturbing the entity HNSW on every delete.
                    # ORDEM IMPORTA, e me custou duas tentativas: apagar a
                    # chave ANTES do update não adianta, porque o update grava o
                    # payload lido — que ainda a contém — e o MERGE do
                    # `set_payload` a traz de volta.
                    limpo = {k: v for k, v in payload.items()
                             if k != link_key(memory_id)}
                    limpo["linked_memory_ids"] = remaining
                    try:
                        store.update(vector_id=row.id, vector=None, payload=limpo)
                        n_linhas += 1
                    except Exception as e:
                        ok = False
                        logger.debug(f"Entity update failed for id={row.id}: {e}")
                    # `set_payload` faz MERGE: gravar sem a chave NÃO a apaga, e
                    # `links_do_payload` (união de lista + chaves) RESSUSCITARIA
                    # o vínculo recém-deletado — o mesmo defeito de vínculo
                    # pendente que este trabalho existe para eliminar. Store sem
                    # esse suporte torna a limpeza INCOMPLETA, e o veredito é o
                    # que impede comitar o delete sobre limpeza que não houve.
                    try:
                        store.delete_payload_keys(row.id, [link_key(memory_id)])
                    except AttributeError:
                        ok = False
                        logger.warning("store sem delete_payload_keys: %s fica e "
                                       "o vínculo pode ressuscitar",
                                       link_key(memory_id))
                    except Exception as e:
                        ok = False
                        logger.debug(f"delete_payload_keys failed for {row.id}: {e}")
            except Exception as e:
                ok = False
                logger.debug(f"Entity cleanup error: {e}")
    except Exception as e:
        ok = False
        logger.warning(f"Entity store cleanup failed for memory_id={memory_id}: {e}")
    _notify_delete(memory_id, "entity_cleanup",
                   elapsed_ms=round((time.time() - t0) * 1000, 2),
                   rows_scanned=n_scan, rows_touched=n_linhas,
                   truncated=bool(truncated), complete=ok,
                   scope={k: v for k, v in search_filters.items()})
    return ok


def _dynamics_config(config) -> Optional[Any]:
    """The MemoryDynamicsConfig when dynamics is enabled, else None."""
    dyn = getattr(config, "dynamics", None)
    return dyn if dyn is not None and getattr(dyn, "enabled", False) else None


def _temporality_config(config) -> Optional[Any]:
    """The MemoryTemporalityConfig when temporality is enabled, else None."""
    temp = getattr(config, "temporality", None)
    return temp if temp is not None and getattr(temp, "enabled", False) else None


def _mark_superseded(vector_store, db, new_id, new_text, old_ids, new_created_at=None) -> List[Tuple[str, str]]:
    """DeepMem0 v0.3/v0.4: mark supersessions between a new fact and the
    memories the LLM says it replaces.

    Non-destructive: a superseded memory keeps living in the store (search
    demotes it, an as_of anchor can restore it) and gains ``superseded_by`` +
    ``superseded_at``. The first marking wins — an already-superseded memory
    is never re-marked, so ``superseded_at`` stays immutable and chains
    A -> B -> C emerge naturally. Every marking is recorded in the history DB
    as a SUPERSEDED event. Full-payload merge; never raises — supersession is
    bookkeeping and must not break the add.

    v0.4 (async ingestion): the marking direction honors record time. When
    ``new_created_at`` (canonically the fact's submission time) predates an
    existing memory's ``created_at``, the NEW memory is born superseded by
    the existing one instead — a queued fact that lost the race to a direct
    write must never demote fresher truth (see ``supersession_inverted``).

    Returns the ``(superseded_id, superseding_id)`` pairs actually marked.
    """
    now_iso = _dynamics_utcnow().isoformat()
    marked: List[Tuple[str, str]] = []
    new_settled = False  # first marking wins for the born-superseded new memory too
    for old_id in old_ids or []:
        try:
            if old_id == new_id:
                continue
            mem = vector_store.get(vector_id=old_id)
            payload = getattr(mem, "payload", None) if mem is not None else None
            if payload is None:
                continue
            if supersession_inverted(new_created_at, payload.get("created_at")):
                if new_settled:
                    continue
                new_settled = True
                new_mem = vector_store.get(vector_id=new_id)
                new_payload = getattr(new_mem, "payload", None) if new_mem is not None else None
                if new_payload is None or new_payload.get(FIELD_SUPERSEDED_BY):
                    continue
                vector_store.update(
                    vector_id=new_id,
                    payload={**new_payload, FIELD_SUPERSEDED_BY: old_id, FIELD_SUPERSEDED_AT: now_iso},
                )
                try:
                    db.add_history(
                        new_id,
                        new_payload.get("data"),
                        payload.get("data"),
                        "SUPERSEDED",
                        created_at=new_payload.get("created_at"),
                        updated_at=now_iso,
                        actor_id=new_payload.get("actor_id"),
                        role=new_payload.get("role"),
                    )
                except Exception as e:
                    logger.warning(f"Supersession history record failed for {new_id}: {e}")
                marked.append((new_id, old_id))
                continue
            if payload.get(FIELD_SUPERSEDED_BY):
                continue
            vector_store.update(
                vector_id=old_id,
                payload={**payload, FIELD_SUPERSEDED_BY: new_id, FIELD_SUPERSEDED_AT: now_iso},
            )
            try:
                db.add_history(
                    old_id,
                    payload.get("data"),
                    new_text,
                    "SUPERSEDED",
                    created_at=payload.get("created_at"),
                    updated_at=now_iso,
                    actor_id=payload.get("actor_id"),
                    role=payload.get("role"),
                )
            except Exception as e:
                logger.warning(f"Supersession history record failed for {old_id}: {e}")
            marked.append((old_id, new_id))
        except Exception as e:
            logger.warning(f"Supersession marking failed for {old_id}: {e}")
    return marked


# --- DeepMem0 v0.7 (update versioning, roadmap item #7) ---------------------
# Fields on the superseded head (v1) that are NOT inherited by its successor
# (v2): derived (recomputed by _create_memory), version bookkeeping, temporal
# state, and per-source provenance that does not apply to a fresh
# conversational version.
#
# v0.9: the dynamics fields are NO LONGER in this blanket blacklist — whether
# the usage timeline carries forward became an explicit DECISION
# (``version_inherits_dynamics``, handled by _plan_version_dynamics), not a
# side effect of a list. They remain unconditionally blacklisted for CALLER
# metadata (see _VERSION_CALLER_ONLY_BLOCKED): a client must never forge a
# timeline through update(metadata=...).
_VERSION_NON_INHERITED = frozenset({
    "data", "hash", "text_lemmatized", "created_at", "updated_at",
    FIELD_SUPERSEDED_BY, FIELD_SUPERSEDED_AT, FIELD_SUPERSEDES, FIELD_EVENT_DATE,
    FIELD_VERSION_PREV, FIELD_VERSION_NEXT, FIELD_LINEAGE_SCHEMA,
    "source_doc", "page_start", "page_end", "chunk_index", "content_type",
    "task_id",
})

# Dynamics are decided by _plan_version_dynamics for the HEAD side, but a
# CALLER can never write them: derived from the single defining tuple so a new
# dynamics field is forgery-blocked by default (the hand-kept list already let
# two fields slip once).
_VERSION_CALLER_ONLY_BLOCKED = frozenset(DYNAMICS_FIELDS)

# Ownership/scope keys that are IMMUTABLE across versions: a caller can neither
# change nor ADD a scope the head does not have (issue #4490 for actor_id; the
# rest closes cross-scope escalation via a versioned update — critic #9).
_IMMUTABLE_SCOPE = ("user_id", "agent_id", "run_id", "actor_id")
_LINEAGE_SCOPE_KEYS = ("user_id", "agent_id", "run_id")


def _canonizar_filtro_de_locutor(effective_filters):
    """Canoniza `actor_id` no filtro de LEITURA, com a mesma função da escrita.

    O Qdrant casa por igualdade exata. Se a escrita canoniza (`"  Maria  "` vira
    `"Maria"`) e a consulta não, o filtro erra em SILÊNCIO: zero resultado, sem
    erro, sem log — e o chamador conclui que a memória não existe. Um filtro que
    devolve vazio por diferença de espaço é pior que um que recusa.

    Rótulo inválido LEVANTA em vez de ser removido do filtro: remover alargaria a
    consulta para o escopo inteiro e devolveria memórias de TODOS os locutores
    como se fossem de um só — resposta errada apresentada como resposta. A
    exceção é o único desfecho que não mente.
    """
    if "actor_id" not in effective_filters:
        return effective_filters
    bruto = effective_filters["actor_id"]
    rotulo = normalize_speaker_label(bruto)
    if rotulo is None:
        raise ValueError(
            f"actor_id inválido para filtro: {bruto!r}. Um rótulo de locutor é "
            "texto não-vazio, sem quebra de linha nem caractere de controle, com "
            f"até {MAX_SPEAKER_LABEL} caracteres."
        )
    effective_filters["actor_id"] = rotulo
    return effective_filters


def aplicar_escopo_imutavel(meta, head_payload):
    """Impõe o escopo de posse EXATO do head — **inclusive a ausência dele**.

    FONTE ÚNICA da regra, usada pelos dois caminhos de `update()`: o versionado
    (v0.7) e o legado in-place. Eles DIVERGIAM, e a divergência era o buraco: o
    versionado já removia a chave e só a restaurava `if k in head_payload`,
    enquanto o legado só sabia PRESERVAR um valor existente
    (`if "actor_id" in existing_memory.payload`) e deixava passar um valor NOVO
    vindo do `metadata` do chamador.

    A assimetria é o defeito: uma guarda que protege o valor JÁ GRAVADO mas
    aceita gravar um do zero não impede forjar autoria — só impede reescrevê-la.
    E o comentário de `_IMMUTABLE_SCOPE` já declarava a regra certa desde a v0.7:
    *"a caller can neither change nor ADD a scope the head does not have"*. O
    legado contradizia o contrato do próprio módulo.

    Materializa quando `actor_id` começa a ser escrito (v0.15): as memórias
    legadas não têm o campo, então era exatamente nelas — o corpus inteiro — que
    um cliente poderia carimbar um locutor que nunca falou.
    """
    for k in _IMMUTABLE_SCOPE:
        meta.pop(k, None)
        if k in head_payload:
            meta[k] = head_payload[k]
    return meta


def _lineage_scope(payload) -> Tuple:
    """The owner-scope tuple used to fence version-lineage traversal to one owner."""
    p = payload or {}
    return tuple(p.get(k) for k in _LINEAGE_SCOPE_KEYS)


def _build_version_metadata(head_payload, data, caller_metadata, operation_ts,
                            head_id, extract_event_date,
                            inherit_dynamics: bool = False) -> Dict[str, Any]:
    """Field-by-field metadata policy for a new version (v2) minted when ``update()``
    supersedes ``head_payload`` (v1). Pure — no I/O.

    Inherit-by-blacklist: v2 carries forward every head field EXCEPT
    ``_VERSION_NON_INHERITED`` (derived/version-bookkeeping/prior usage/provenance).
    Then:
    - RESERVED lineage fields (``_mem0_version_*``) and IMMUTABLE scope keys are
      STRIPPED from the caller first — a client can neither forge a lineage edge nor
      change ownership of a version (critic #3/#5/#9);
    - remaining caller metadata overrides inherited values;
    - the exact head scope tuple ``(user_id, agent_id, run_id, actor_id)`` is copied
      INCLUDING ABSENCE (immutable);
    - ``created_at = operation_ts``; ``_mem0_version_prev = [head_id]`` +
      ``_mem0_lineage_schema`` stamp the linear update lineage (the shared semantic
      ``supersedes`` is NOT written by an update — it stays semantic-only);
    - ``task_id`` from the caller (provenance); ``event_date`` re-inferred from the
      NEW text.
    """
    caller = dict(caller_metadata or {})
    for k in RESERVED_LINEAGE_FIELDS:
        caller.pop(k, None)          # anti-injection: only _version_update writes these
    for k in _IMMUTABLE_SCOPE:
        caller.pop(k, None)          # ownership is immutable across versions
    # v0.9: dynamics COPY from the head is a decision (inherit_dynamics), never
    # an accident of the blacklist; the caller can never write them regardless.
    head_blocked = _VERSION_NON_INHERITED if inherit_dynamics else (
        _VERSION_NON_INHERITED | _VERSION_CALLER_ONLY_BLOCKED
    )
    caller_blocked = _VERSION_NON_INHERITED | _VERSION_CALLER_ONLY_BLOCKED
    meta: Dict[str, Any] = {
        k: v for k, v in head_payload.items() if k not in head_blocked
    }
    for k, v in caller.items():
        if k not in caller_blocked:
            meta[k] = v
    # exact immutable scope from the head (including absence)
    aplicar_escopo_imutavel(meta, head_payload)
    meta["created_at"] = operation_ts
    meta[FIELD_VERSION_PREV] = [head_id]
    meta[FIELD_LINEAGE_SCHEMA] = LINEAGE_SCHEMA_VERSION
    if caller.get("task_id"):
        meta["task_id"] = caller["task_id"]
    if extract_event_date:
        ev = infer_event_date_from_text(data)
        if ev:
            meta[FIELD_EVENT_DATE] = ev
    return meta


def _plan_version_dynamics(head_payload, dyn, operation_ts, *, inherit):
    """Dynamics side of a versioned update, PURE (no I/O) and shared by the
    sync and async twins so their semantics cannot drift.

    Returns ``(extra_fields, t2_outcome)`` to merge into the new version's
    metadata BEFORE the single ``_create_memory`` write (mirroring the legacy
    in-place T2, which folds the reinforcement into the same atomic write):

    - ``first_seen_at``: the family's first-encounter anchor (head's own
      ``first_seen_at`` when valid, else head's ``created_at`` — propagates
      v1→v2→v3). Stamped whenever inheriting, even with dynamics disabled:
      the anchor is cheap and correct the day dynamics turns on.
    - T2 planned over the HEAD's payload — never the new version's. This is
      what kills the newborn step: with a neutral head, seeding from the new
      version's ``created_at`` (= operation time) plus the T2 event at the
      same instant would mint boost 0.667 out of thin air — WORSE than the
      measured option-A regression (0.5, reverted). Over the head's payload
      the seed adopts the fact's real first encounter.
    - Window suppression suppresses only the T2 EVENT; the copy (done by
      ``_build_version_metadata``) stands. A queued update whose
      ``operation_ts`` predates the head's last event is likewise suppressed —
      the exposed_at discipline: a late job neither back-dates nor re-opens.

    ``t2_outcome`` is None when there was no decision to report (not
    inheriting, or dynamics off); the caller notifies AFTER the verify pass.
    """
    if not inherit:
        return {}, None
    extra: Dict[str, Any] = {}
    anchor = _anchor_ts(head_payload)
    if anchor:
        extra[FIELD_FIRST_SEEN] = anchor
    if dyn is None:
        return extra, None
    now = _dynamics_parse_ts(operation_ts)
    fields, outcome = plan_reinforcement(head_payload, dyn, TRIGGER_UPDATE, now=now)
    if fields:
        extra.update(fields)
    return extra, outcome


def _resolve_chain_head(get_fn, memory_id, max_hops: int = 64) -> Tuple[str, Any]:
    """Follow the DEDICATED update-version link ``_mem0_version_next`` to the current
    head of an update-version chain (v0.7.1). NOT ``superseded_by`` — that is shared
    with semantic supersedence and would branch across mechanisms.

    Used by ``update`` and ``delete``. FAIL-CLOSED (raises, so the operation aborts
    BEFORE any mutation — critic #2) on a cross-scope lineage edge, a cycle, or hop
    exhaustion (corruption). A dangling ``version_next`` (successor missing, e.g.
    purged mid-retry) is treated as the head, preserving retry robustness. A missing
    start record returns ``(memory_id, None)``. Returns ``(head_id, head_record)``.
    """
    first = get_fn(memory_id)
    if first is None:
        return memory_id, None
    anchor = _lineage_scope(getattr(first, "payload", None) or {})
    seen = {memory_id}
    cur_id, cur_mem = memory_id, first
    for _ in range(max_hops):
        nxt = (getattr(cur_mem, "payload", None) or {}).get(FIELD_VERSION_NEXT)
        if not isinstance(nxt, str) or not nxt or nxt == cur_id:
            return cur_id, cur_mem  # head (no successor / malformed / self)
        if nxt in seen:
            raise ValueError(f"Corrupt version lineage: cycle at {nxt}")
        nxt_mem = get_fn(nxt)
        if nxt_mem is None:
            return cur_id, cur_mem  # dangling successor -> current is head
        if _lineage_scope(getattr(nxt_mem, "payload", None) or {}) != anchor:
            raise ValueError(f"Cross-scope version lineage edge {cur_id}->{nxt}; aborting")
        seen.add(nxt)
        cur_id, cur_mem = nxt, nxt_mem
    raise ValueError(f"Version lineage exceeded {max_hops} hops from {memory_id}; aborting")


def _collect_chain(get_fn, memory_id, max_hops: int = 256) -> List[str]:
    """Ids of the WHOLE update-version chain reachable from ``memory_id`` (for delete).

    Resolves the head via ``_mem0_version_next``, then walks ``_mem0_version_prev``
    backward. Uses ONLY the dedicated lineage fields, so semantic supersedence
    siblings (``supersedes=[A,B]``) are NEVER over-collected (critic #1). FAIL-CLOSED:
    raises on a cross-scope member, cycle, or size overflow — so ``delete`` validates
    the whole chain before removing anything. Ordered head-first.
    """
    head_id, head_mem = _resolve_chain_head(get_fn, memory_id, max_hops=max_hops)
    if head_mem is None:
        return []
    anchor = _lineage_scope(getattr(head_mem, "payload", None) or {})
    ids: List[str] = []
    seen: set = set()
    frontier = [(head_id, head_mem)]
    while frontier:
        if len(seen) > max_hops:
            raise ValueError(f"Version lineage too large (>{max_hops}) from {memory_id}; aborting")
        cid, cmem = frontier.pop()
        if cid in seen:
            continue
        seen.add(cid)
        if cmem is None:
            cmem = get_fn(cid)
        if cmem is None:
            continue
        if _lineage_scope(getattr(cmem, "payload", None) or {}) != anchor:
            raise ValueError(f"Cross-scope version lineage member {cid}; aborting")
        ids.append(cid)
        prev = (getattr(cmem, "payload", None) or {}).get(FIELD_VERSION_PREV) or []
        if isinstance(prev, list):
            for anc in prev:
                if isinstance(anc, str) and anc and anc not in seen:
                    frontier.append((anc, None))
    return ids


#: Hook for observability: called as
#: ``(memory_id, trigger, outcome, elapsed_ms, context)`` after EVERY
#: reinforcement decision, whichever trigger made it. Kept as a plain module
#: attribute so the companion (telemetry lives outside the core) can attach
#: without patching three different call sites — the previous shape made T2
#: invisible, since it wrote the timeline inline and never called the shared
#: helper. ``context`` carries the search correlation for T3 (search_id, rank)
#: and is None for write triggers. Must never raise; exceptions are swallowed.
reinforcement_observer = None


def _notify_reinforcement(memory_id, trigger, outcome, elapsed_ms=0.0, context=None) -> None:
    observer = reinforcement_observer
    if observer is None:
        return
    try:
        observer(memory_id, trigger, outcome, elapsed_ms, context)
    except Exception:  # observability must never break bookkeeping
        pass


class ReinforcementTarget(NamedTuple):
    """One memory a search exposed, with everything the write-back needs.

    Correlation travels EXPLICITLY, by value, from the search thread into the
    executor. A thread-local would not survive the hop (the executor runs on its
    own thread) and would be worse than useless on the async path, where many
    tasks share the event-loop thread and would overwrite each other's context.

    ``exposed_at`` is the moment the caller SAW the memory, not the moment the
    worker got around to writing it: under backlog those differ, and the
    difference lands straight in the ACT-R timeline and in both windows.

    ``snapshot`` is the metadata the search already carried — used only as a
    NEGATIVE pre-filter (skip a fetch that the window will surely suppress);
    the fresh payload remains the authority.
    """

    memory_id: str
    rank: int  # 1-based, position in the FINAL page the caller received
    search_id: str
    exposed_at: Any
    snapshot: dict


def plan_reinforcement(payload, dyn, trigger, *, now=None):
    """THE single decision point for every trigger: should this memory be
    reinforced now, and with which fields?

    Returns ``(fields_or_None, outcome)``. T1/T3 hand the fields to the vector
    store themselves; T2 merges them into the update it is already writing, so
    the content update and its reinforcement stay one atomic write. Centralizing
    only the DECISION (not the write) keeps that atomicity while making all three
    triggers observable through one place.
    """
    payload = payload or {}
    if not should_reinforce(
        payload,
        now=now,
        window_seconds=dyn.reinforcement_window,
        trigger=trigger,
        search_window_seconds=getattr(dyn, "reinforce_on_search_window", 0) or 0,
    ):
        return None, OUTCOME_SUPPRESSED
    fields = reinforcement_fields(
        payload, now=now, max_timestamps=dyn.max_timestamps, trigger=trigger
    )
    return fields, OUTCOME_APPLIED


def _reinforce_memory(vector_store, dyn, memory_id, payload, *, trigger=TRIGGER_DEDUP,
                      now=None, context=None) -> str:
    """One reinforcement event on a memory. Returns the structured outcome.

    Writes ONLY the dynamics fields when the store merges payloads (Qdrant's
    set_payload does), instead of re-writing the full payload read moments
    earlier: that read-modify-write silently reverted any field a concurrent
    writer had changed in between. Stores that REPLACE the payload still get the
    full merge, or they would lose everything else.

    This narrows the blast radius; it does NOT make the timeline concurrency-safe.
    The read-modify-write on the dynamics fields themselves remains: two
    reinforcements racing can both read the same history and one overwrites the
    other, and an UPDATE that upserts a full payload built from an older snapshot
    can erase a reinforcement that landed in between. Reinforcement is
    best-effort bookkeeping — a lost event costs one unrecorded exposure, and the
    telemetry counts what was applied so the loss is measurable rather than
    invisible.

    Never raises — reinforcement is bookkeeping and must not break add/update/search.
    """
    started = time.time()
    outcome = OUTCOME_FAILED
    try:
        payload = payload or {}
        # `now` = when the memory was ENCOUNTERED. For T3 that is the exposure
        # instant, so a late job neither back-dates nor re-opens a window: if a
        # write landed after the exposure, the window check sees a negative
        # delta and suppresses, which is the correct ordering.
        fields, outcome = plan_reinforcement(payload, dyn, trigger, now=now)
        if fields is None:
            return outcome
        if getattr(vector_store, "PAYLOAD_UPDATE_MERGES", False):
            vector_store.update(vector_id=memory_id, payload=fields)
        else:
            vector_store.update(vector_id=memory_id, payload={**payload, **fields})
        return outcome
    except Exception as e:
        logger.warning(f"Reinforcement failed for memory {memory_id}: {e}")
        outcome = OUTCOME_FAILED
        return outcome
    finally:
        _notify_reinforcement(memory_id, trigger, outcome,
                              (time.time() - started) * 1000.0, context)


#: Digit-sequence tokens ("8188", "0.7.1", "1,5"). The labeled probe (2026-07-27,
#: 41 real boundary pairs) showed the false-positive class for T1S is "same
#: template, one identifier swapped": IDs 8188 vs 8189 sit at cosine
#: 0.9754 — NO threshold kills it without also killing genuine restatements.
#: All 6 labeled false positives differed in digits; zero true positives in the
#: >=0.95 band did. Same principle as the extraction oracle's "alien number".
_DIGIT_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _digit_tokens(text):
    """Multiset of numeric tokens, decimal separator normalized."""
    return Counter(t.replace(",", ".") for t in _DIGIT_TOKEN_RE.findall(text or ""))


def _digits_compatible(text_a, text_b) -> bool:
    """True when one text's numeric tokens are a sub-multiset of the other's.

    Subset (not equality) so a restatement that ADDS a clause with a new number
    still counts, while a swapped identifier (8188 vs 8189) never does: neither
    side contains the other.
    """
    a, b = _digit_tokens(text_a), _digit_tokens(text_b)
    return not (a - b) or not (b - a)


def _similar_reinforcement_target(vector_store, text, embedding, search_filters, dyn):
    """Nearest eligible near-paraphrase of an extracted fact, or None (T1S).

    Why a dedicated per-fact search: the add path's existing_results carry a
    score against the WHOLE conversation embedding — useless as a per-fact
    signal — and no vectors, so cosine cannot be computed locally. One dense
    query per fact (~10-30ms) against an add dominated by 26-37s of extraction.

    FAIL-OPEN by contract: this is enrichment on the hot add path. Any failure
    returns None and the add keeps inserting — a Qdrant hiccup must never cost
    a memory. Callers must not invoke this for facts carrying a `supersedes`
    mark (a correction reinforcing what it corrects is the worst case).
    """
    # Config reads coerced defensively: a mocked/duck-typed config must render
    # the trigger INERT, never crash the add path (this guard sits before the
    # fail-open try on purpose — it is itself part of the fail-open contract).
    try:
        threshold = float(getattr(dyn, "reinforce_similarity_threshold", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if not getattr(dyn, "reinforce_on_similar", False) or threshold <= 0:
        return None
    try:
        hits = vector_store.search(query=text, vectors=embedding, top_k=3,
                                   filters=search_filters)
        for hit in hits or []:
            score = getattr(hit, "score", None)
            if score is None or score < threshold:
                return None  # ordered by similarity: below threshold, stop
            pl = getattr(hit, "payload", None) or {}
            if pl.get(FIELD_SUPERSEDED_BY):
                continue  # demoted fact: a boost would fight the penalty
            if not _digits_compatible(text, pl.get("data")):
                continue  # swapped identifier — the measured FP class
            return str(hit.id), float(score)
        return None
    except Exception as e:
        logger.warning(f"T1S similarity lookup failed (fail-open, add continues): {e}")
        _notify_reinforcement(None, TRIGGER_SIMILAR, OUTCOME_FAILED, 0.0,
                              {"stage": "similarity_lookup"})
        return None


def _apply_similar_reinforcements(vector_store, dyn, pending):
    """Apply deferred T1S reinforcements. ``pending``: (target_id, score, new_id).

    Runs AFTER persist and _mark_superseded, and re-reads each target FRESH —
    no payload snapshot. The fresh read both narrows the lost-update window
    from seconds (the persist span) to milliseconds — same move as the T2
    reorder — and re-checks eligibility for free: a target superseded moments
    ago by another fact of this very batch shows its superseded_by here.
    Best-effort contract as measured for T2×T3; each attempt notifies with a
    structured outcome either way.
    """
    for target_id, score, new_id, *rest in pending:
        new_hash = rest[0] if rest else None
        try:
            fresh = vector_store.get(vector_id=target_id)
        except Exception as e:
            logger.warning(f"T1S fresh read failed for {target_id}: {e}")
            _notify_reinforcement(target_id, TRIGGER_SIMILAR, OUTCOME_FAILED, 0.0,
                                  {"stage": "fresh_read", "from_add": new_id})
            continue
        payload = getattr(fresh, "payload", None) if fresh is not None else None
        if payload is None:
            _notify_reinforcement(target_id, TRIGGER_SIMILAR, OUTCOME_MISSING, 0.0,
                                  {"from_add": new_id})
            continue
        if payload.get(FIELD_SUPERSEDED_BY):
            _notify_reinforcement(target_id, TRIGGER_SIMILAR, OUTCOME_SUPPRESSED, 0.0,
                                  {"reason": "superseded", "from_add": new_id})
            continue
        _reinforce_memory(vector_store, dyn, target_id, payload,
                          trigger=TRIGGER_SIMILAR,
                          context={"similarity": round(score, 6), "from_add": new_id,
                                   # v0.10.1: hashes de CONTEÚDO no evento — sem eles o
                                   # júri não distingue "texto mutado depois" de "texto
                                   # da época" (limitação declarada, agora fechada p/
                                   # eventos novos; antigos seguem indetectáveis)
                                   "target_hash": payload.get("hash"),
                                   "from_add_hash": new_hash})


def _validate_historical(historical, as_of, temp) -> bool:
    """v0.10: valida o modo RECORDAÇÃO HISTÓRICA. Pure — sem I/O.

    O modo é um caminho EXPLÍCITO (decisão de produto, 28/07/2026): `as_of`
    sozinho preserva o comportamento de sempre (filtro record-time +
    elegibilidade v0.9); `historical=True` é quem muda a semântica — recordar
    "o que eu sabia na época" não reforça nada e não usa peso de uso no
    ranking. Fail-fast: âncora ausente ou feature desligada viram erro claro,
    nunca um modo silenciosamente diferente do pedido.
    """
    if not historical:
        return False
    if temp is None or not getattr(temp, "historical_recall", False):
        raise ValueError(
            "historical recall is disabled (temporality.historical_recall=false "
            "or temporality disabled)"
        )
    if as_of is None:
        raise ValueError("historical=True requires as_of (a recollection needs its anchor)")
    return True


def _annotate_known_successors(memories) -> int:
    """v0.10: marca resultados que têm um sucessor EXPLÍCITO conhecido.

    HONESTO por contrato: detecta `superseded_by` (correção semântica v0.3 ou
    versão de update v0.7) — NÃO detecta "qualquer fato mais novo sobre o
    assunto" (detecção por assunto é outra feature). Só o booleano: resolver o
    id da versão atual exige travessia de cadeia com riscos próprios
    (ciclo/deletado/escopo) e o aviso não precisa dele.
    """
    n = 0
    for doc in memories or []:
        meta = doc.get("metadata") or {}
        if meta.get(FIELD_SUPERSEDED_BY):
            doc["has_newer_version"] = True
            n += 1
    return n


def _t3_enabled(dyn, reinforce) -> bool:
    """Whether this particular search reinforces what it returns.

    ``reinforce`` is the per-CALL override: ``False`` opts a search out even with
    T3 globally on. Measurement harnesses MUST pass it — a golden set that runs
    its own queries against the live corpus would otherwise reinforce its own
    expected targets on every run, inflating the very metric it exists to
    protect. ``None`` defers to the configured default.
    """
    if dyn is None or not dyn.reinforce_on_search:
        return False
    return True if reinforce is None else bool(reinforce)


def _t3_targets(dyn, memories, *, search_id, exposed_at) -> List[ReinforcementTarget]:
    """What a search reinforces: the top ``reinforce_top_n`` returned (0 = all).

    Being returned at rank 9 is list filler, not recall; reinforcing a whole page
    would put the timeline at the mercy of ``limit`` rather than of use.
    """
    top_n = getattr(dyn, "reinforce_top_n", 0) or 0
    selected = memories[:top_n] if top_n > 0 else memories
    return [
        ReinforcementTarget(
            memory_id=doc["id"],
            rank=i,
            search_id=search_id,
            exposed_at=exposed_at,
            # cópia: o dict de metadata pertence ao resultado que volta ao
            # caller, e "viaja por valor" só é verdade se ninguém puder mutá-lo
            # depois que o alvo já foi montado.
            snapshot=dict(doc.get("metadata") or {}),
        )
        for i, doc in enumerate(selected, start=1)
        if doc.get("id")
        # v0.9: um registro supersedido não é reforçado pela exposição — com a
        # timeline copiada ao sucessor, reforçar o velho recriaria o double-dip
        # que a máscara removeu (o t1s já pulava; T3 ficava inconsistente).
        and not (doc.get("metadata") or {}).get(FIELD_SUPERSEDED_BY)
    ]


#: Serial, bounded write-back. One thread per search created an unbounded number
#: of threads under load and gave no way to see a backlog; a single worker with a
#: capped queue makes the pressure visible (dropped events are counted, not lost
#: silently) and keeps reinforcement strictly off the hot path.
#: Backlog cap in TARGETS (memories), the same unit the counters use: mixing
#: "jobs pending" with "memories dropped" made the gauge unreadable.
_REINFORCE_MAX_PENDING = 256
_reinforce_executor = None
_reinforce_pending = 0
_reinforce_dropped = 0
_reinforce_lock = threading.Lock()


def _get_reinforce_executor():
    """Lazy singleton. Creation happens UNDER the lock: check-then-act outside it
    let two threads build two executors, leaking one silently."""
    global _reinforce_executor
    with _reinforce_lock:
        if _reinforce_executor is None:
            _reinforce_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="deepmem0-reinforce"
            )
        return _reinforce_executor


def reinforcement_backlog() -> Dict[str, int]:
    """Backpressure signal for telemetry, both counters in MEMORIES.

    ``dropped`` is cumulative since process start — it resets on restart, so read
    it as a rate, never as a lifetime total.
    """
    with _reinforce_lock:
        return {"pending": _reinforce_pending, "dropped": _reinforce_dropped}


def _reinforce_hits_in_background(vector_store, dyn, targets) -> None:
    """T3: reinforce searched-and-returned memories off the hot path.

    Each memory is re-fetched so the window check and the merge run against fresh
    payload, not the search snapshot — the snapshot only skips work that would
    surely be suppressed.
    """
    global _reinforce_pending, _reinforce_dropped

    if not targets:
        return
    window = getattr(dyn, "reinforcement_window", 0) or 0
    search_window = getattr(dyn, "reinforce_on_search_window", 0) or 0

    def _run():
        global _reinforce_pending
        try:
            for target in targets:
                try:
                    mem = vector_store.get(vector_id=target.memory_id)
                    context = {"search_id": target.search_id, "rank": target.rank}
                    if mem is None:
                        _notify_reinforcement(target.memory_id, TRIGGER_SEARCH,
                                              OUTCOME_MISSING, 0.0, context)
                        continue
                    _reinforce_memory(
                        vector_store, dyn, target.memory_id,
                        getattr(mem, "payload", None), trigger=TRIGGER_SEARCH,
                        now=target.exposed_at, context=context,
                    )
                except Exception as e:
                    logger.debug(f"Access reinforcement skipped for {target.memory_id}: {e}")
                    _notify_reinforcement(target.memory_id, TRIGGER_SEARCH, OUTCOME_FAILED,
                                          0.0, {"search_id": target.search_id,
                                                "rank": target.rank})
        finally:
            with _reinforce_lock:
                _reinforce_pending -= len(targets)

    # NEGATIVE pre-filter only: the snapshot the search already carried can prove
    # a target is inside its window, and then there is no reason to pay a fetch.
    # It can never prove the opposite — "looks eligible" still goes to the fresh
    # payload, which decides.
    eligible = [
        t for t in targets
        if should_reinforce(t.snapshot, now=t.exposed_at, window_seconds=window,
                            trigger=TRIGGER_SEARCH, search_window_seconds=search_window)
    ]
    for skipped in (t for t in targets if t not in eligible):
        _notify_reinforcement(skipped.memory_id, TRIGGER_SEARCH, OUTCOME_SUPPRESSED, 0.0,
                              {"search_id": skipped.search_id, "rank": skipped.rank,
                               "prefiltered": True})
    if not eligible:
        return
    targets = eligible

    with _reinforce_lock:
        if _reinforce_pending + len(targets) > _REINFORCE_MAX_PENDING:
            _reinforce_dropped += len(targets)
            dropped = list(targets)
            logger.warning(
                "Reinforcement backlog full (%d pending); dropped %d search hits",
                _reinforce_pending, len(targets),
            )
        else:
            _reinforce_pending += len(targets)
            dropped = None
    if dropped is not None:
        # A drop is an EVENT, not just a counter: without this the stream showed
        # nothing at all for exposures that were thrown away.
        for t in dropped:
            _notify_reinforcement(t.memory_id, TRIGGER_SEARCH, OUTCOME_DROPPED, 0.0,
                                  {"search_id": t.search_id, "rank": t.rank})
        return
    try:
        _get_reinforce_executor().submit(_run)
    except Exception as exc:  # submit itself can fail (interpreter shutdown)
        with _reinforce_lock:
            _reinforce_pending -= len(targets)  # or the gauge leaks forever
        logger.warning("Reinforcement submit failed: %s", exc)
        for t in targets:
            _notify_reinforcement(t.memory_id, TRIGGER_SEARCH, OUTCOME_FAILED, 0.0,
                                  {"search_id": t.search_id, "rank": t.rank})


_rerank_contract_warned = False


def _relevance_from_rerank_score(rerank_score) -> float:
    """Read a reranker score as RELEVANCE, per the ``BaseReranker`` contract.

    ``rerank_score`` is already an absolute relevance in [0, 1] — every provider
    emits one (sentence-transformers applies ``nn.Sigmoid`` for ``num_labels=1``
    cross-encoders; Cohere and ZeroEntropy return ``relevance_score``; the LLM
    reranker scores 0-1; HuggingFace sigmoids its logits). This used to apply a
    sigmoid here, which was a SECOND sigmoid: it squeezed [0, 1] into
    [0.5, 0.731], so the superseded penalty (0.2, documented as a [0, 1]-scale
    constant) consumed 86% of the available range instead of 20%, and the tie
    bands measured a compressed axis.

    It went unnoticed because the UNADJUSTED order is invariant: a sigmoid is
    monotonic, so with no penalty and no tie-break the primary sort is exactly
    the reranker's own order either way. That invariance does NOT extend to the
    adjusted order — subtracting a constant and grouping by a fixed band are not
    scale-free, which is the whole reason this mattered. Measured over 407
    recorded production pools, 11 group differently under the two spaces.

    A provider that breaks the contract (e.g. ``HuggingFaceReranker`` with
    ``normalize=False``, which emits raw logits) is CLAMPED, not sigmoided:
    guessing that an out-of-range value must be a logit would silently re-enter
    the bug for anything that is merely miscalibrated. Clamping is monotonic and
    the sort below is stable over the reranker's own ordering, so ranking ORDER
    survives; the penalty and the tie bands do not. Warned once per process —
    this fires per document, and a per-call warning would bury the log.
    """
    global _rerank_contract_warned
    value = float(rerank_score)
    if 0.0 <= value <= 1.0:
        return value
    if not _rerank_contract_warned:
        _rerank_contract_warned = True
        logger.warning(
            "rerank_score %r is outside the [0, 1] contract (see BaseReranker); "
            "clamping. Ranking order is preserved, but the superseded penalty and "
            "the post-rerank tie bands are degraded. A reranker configured to emit "
            "raw logits (e.g. normalize=False) is the usual cause.",
            value,
        )
    # NaN fails every comparison, so max() returns 0.0 — fail low, not through.
    return min(1.0, max(0.0, value))


def _apply_post_rerank_adjustments(memories, dyn=None, temp=None, as_of=None, event_anchor=None, historical=False) -> List[Dict[str, Any]]:
    """Blend ACT-R activation (v0.2), the superseded penalty (v0.3) and event-time
    proximity (v0.6) into the reranked order.

    RELEVANCE is the reranker's score, an absolute [0, 1] relevance (see
    ``_relevance_from_rerank_score``); the superseded penalty (v0.3) is
    subtracted from it — deliberately strong enough that a superseded fact loses
    to its current replacement even when slightly more similar, waived for
    memories superseded only after an ``as_of`` anchor.

    ACT-R activation (v0.2) and event proximity (v0.6) are BOUNDED TIE-BREAKERS,
    never additive terms. Measured 2026-07-21: the additive form
    (``base + weight*activation``) overturned DECISIVE reranker gaps, because the
    reranker's real operating range on this corpus is narrow — a 0.15 relevance
    preference was flipped by a reinforced 0.08 boost. The factorial ablation
    over the golden showed the additive form was net-negative (hit@1 0.914 vs
    0.943 without it). So these signals only reorder
    candidates that are within the shared reranker tie band of each other — a
    genuine reranker tie — and never touch a decision the reranker made with
    margin. Within a tie, the secondary key is ``(event_proximity, activation)``:
    an explicit date named in the query is a stronger intent signal than usage
    recency, so proximity precedes activation; with no anchor, every proximity is
    0.0 and ordering falls through to activation exactly as before. Memories
    without dynamics/temporality/event fields keep their reranked order.
    """
    # v0.2 tie-break runs whenever dynamics is on and has a band — INDEPENDENT of
    # dynamics.weight, which gates ONLY the fusion term (mirroring how v0.6 gates
    # its event tie-break). Conflating the two made "tie-break only" unreachable:
    # zeroing the fusion weight to keep exposure bias out of pool composition also
    # silently killed the bounded post-rerank tie-break, the one form of activation
    # the 2026-07-21 ablation actually vindicated.
    # v0.10: no modo recordação a ativação é inerte também no tie-break.
    dyn_active = (dyn is not None and (getattr(dyn, "tie_band", 0.0) or 0.0) > 0
                  and not historical)
    temp_active = temp is not None and temp.superseded_penalty > 0
    # v0.6 tie-break runs whenever event_ranking is on and the query has an anchor
    # — INDEPENDENT of event_ranking_weight (weight only gates the fusion term, so
    # weight=0 is a pure tie-break mode with zero divisor interaction).
    event_active = (
        temp is not None
        and getattr(temp, "event_ranking", False)
        and event_anchor is not None
    )
    if not memories or (not dyn_active and not temp_active and not event_active):
        return memories
    now = _dynamics_utcnow()
    event_window_days = getattr(temp, "event_window_days", 30) if temp is not None else 30
    enriched = []
    for doc in memories:
        meta = doc.get("metadata") or {}
        rerank_score = doc.get("rerank_score")
        if rerank_score is None:
            base = doc.get("score") or 0.0
        else:
            base = _relevance_from_rerank_score(rerank_score)
        # v0.9: the superseded decision comes FIRST — it both applies the
        # penalty (as before) and MASKS activation: with the timeline copied to
        # the successor, boosting the old version too would double-dip. Same
        # predicate for both, so as_of views keep historical activation.
        sup_applies = superseded_penalty_applies(
            {
                FIELD_SUPERSEDED_BY: meta.get(FIELD_SUPERSEDED_BY),
                FIELD_SUPERSEDED_AT: meta.get(FIELD_SUPERSEDED_AT),
            },
            as_of=as_of,
        )
        boost = 0.0
        if dyn_active and not sup_applies:
            boost = boost_from_payload(
                {
                    "reinforced_at": meta.get("reinforced_at"),
                    "access_count": meta.get("access_count"),
                    "created_at": doc.get("created_at"),
                    # anchor decoupled from the version's created_at (v0.9)
                    "first_seen_at": meta.get("first_seen_at"),
                },
                now=now,
                decay=dyn.decay,
            )
            if boost > 0:
                doc["activation"] = round(boost, 4)
        eprox = 0.0
        if event_active:
            eprox = event_proximity(event_anchor, meta.get(FIELD_EVENT_DATE), event_window_days)
            if eprox > 0:
                doc["event_proximity"] = round(eprox, 4)
        if temp_active and sup_applies:
            doc["superseded_penalty"] = temp.superseded_penalty
            base -= temp.superseded_penalty
        enriched.append({"doc": doc, "base": base, "boost": boost, "eprox": eprox})

    # Primary order: relevance (reranker score minus superseded penalty).
    enriched.sort(key=lambda e: e["base"], reverse=True)
    if not dyn_active and not event_active:
        return [e["doc"] for e in enriched]

    # Tie-break: two DECOUPLED stable passes, each reordering only within runs of
    # candidates whose relevance is within its OWN band of the run leader (a
    # genuine reranker tie); outside the band the reranker's decision stands. The
    # passes use independent bands so widening one never widens the other — the
    # activation window (ACT-R, dyn.tie_band) stays tight even on a dated query
    # where the event band may be wider. Activation runs first, then event, so an
    # explicit date in the query (a deliberate intent signal) takes precedence
    # over usage recency within its band while activation still breaks the tighter
    # ties it owns. Neither can overturn a decisive reranker margin (>> its band).
    def _tie_pass(items, band, key):
        band = band or 0.0
        if band <= 0:
            return items
        out, i, n = [], 0, len(items)
        while i < n:
            leader = items[i]["base"]
            j = i + 1
            while j < n and leader - items[j]["base"] < band:
                j += 1
            group = items[i:j]
            group.sort(key=key, reverse=True)  # stable: equal keys keep prior order
            out.extend(group)
            i = j
        return out

    if dyn_active:
        act_band = dyn.tie_band if dyn is not None else RERANK_TIE_BAND
        enriched = _tie_pass(enriched, act_band, lambda e: e["boost"])
    if event_active:
        ev_band = getattr(temp, "event_tie_band", RERANK_TIE_BAND)
        enriched = _tie_pass(enriched, ev_band, lambda e: e["eprox"])
    return [e["doc"] for e in enriched]


def _apply_activation_post_rerank(memories, dyn) -> List[Dict[str, Any]]:
    """v0.2 entry point, kept as a thin wrapper over the combined adjuster."""
    return _apply_post_rerank_adjustments(memories, dyn=dyn)


setup_config()
logger = logging.getLogger(__name__)

_PROJECT_UPDATE_UNSUPPORTED_ERROR = "Project updates are not supported by the OSS Memory SDK."


class _OSSProject:
    def update(
        self,
        custom_instructions: Optional[str] = None,
        custom_categories: Optional[list] = None,
        retrieval_criteria: Optional[list] = None,
        multilingual: Optional[bool] = None,
        decay: Optional[bool] = None,
    ):
        if decay is True:
            raise ValueError(get_decay_feature_error_message("sync", "project.update", "decay"))
        raise ValueError(_PROJECT_UPDATE_UNSUPPORTED_ERROR)


class _AsyncOSSProject:
    async def update(
        self,
        custom_instructions: Optional[str] = None,
        custom_categories: Optional[list] = None,
        retrieval_criteria: Optional[list] = None,
        multilingual: Optional[bool] = None,
        decay: Optional[bool] = None,
    ):
        if decay is True:
            raise ValueError(await get_decay_feature_error_message_async("async", "project.update", "decay"))
        raise ValueError(_PROJECT_UPDATE_UNSUPPORTED_ERROR)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        # DeepMem0: propagate the corpus language into the vector store's BM25
        # encoder unless the user pinned vector_store.config.language explicitly.
        if (
            getattr(self.config, "language", "en") != "en"
            and self.config.vector_store.provider == "qdrant"
            and getattr(self.config.vector_store.config, "language", None) is None
        ):
            self.config.vector_store.config.language = self.config.language

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self.custom_instructions = self.config.custom_instructions

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        # DeepMem0 v0.7: per-process lock serializing versioned update transitions
        # so concurrent updates to the same chain resolve-head -> mint -> mark ->
        # verify atomically and form a LINEAR chain, not two competing heads
        # (roadmap item #7). Cross-process concurrency needs an external single
        # writer (the MCP worker is serial) or a distributed lock.
        self._version_lock = threading.Lock()
        # Entity store is initialized lazily on first use
        self._entity_store = None

        if MEM0_TELEMETRY:
            # Create telemetry config manually to avoid deepcopy issues with thread locks
            telemetry_config_dict = {}
            if hasattr(self.config.vector_store.config, 'model_dump'):
                # For pydantic models
                telemetry_config_dict = self.config.vector_store.config.model_dump()
            else:
                # For other objects, manually copy common attributes
                for attr in ['host', 'port', 'path', 'api_key', 'index_name', 'dimension', 'metric']:
                    if hasattr(self.config.vector_store.config, attr):
                        telemetry_config_dict[attr] = getattr(self.config.vector_store.config, attr)

            # Override collection name for telemetry
            telemetry_config_dict['collection_name'] = "mem0migrations"

            # Set path for file-based vector stores
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config_dict['path'] = os.path.join(mem0_dir, provider_path)
                os.makedirs(telemetry_config_dict['path'], exist_ok=True)

            # Create the config object using the same class as the original
            telemetry_config = self.config.vector_store.config.__class__(**telemetry_config_dict)
            self._telemetry_vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, telemetry_config
            )
        if getattr(type(self.vector_store), "keyword_search", None) is VectorStoreBase.keyword_search:
            logger.warning(
                "The '%s' vector store does not support keyword search. "
                "Hybrid (BM25) scoring will be disabled and search will use "
                "semantic similarity only. To enable hybrid search, switch to a "
                "store with keyword_search support (e.g. qdrant, elasticsearch, pgvector).",
                self.config.vector_store.provider,
            )

        # DeepMem0 v0.7.2: finish any delete interrupted by a crash (no-op if none).
        try:
            self.reconcile_pending_deletes()
        except Exception as e:
            logger.warning(f"Delete-intent reconciliation skipped: {e}")

        capture_event("mem0.init", self, {"sync_type": "sync"})

    @property
    def project(self):
        return _OSSProject()

    @property
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        if self._entity_store is None:
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            entity_collection = _entity_collection_name(self.config.vector_store.provider, self.collection_name)
            # Set collection name on the cloned config
            if hasattr(entity_config, 'collection_name'):
                entity_config.collection_name = entity_collection
            elif isinstance(entity_config, dict):
                entity_config['collection_name'] = entity_collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                if hasattr(entity_config, "client"):
                    entity_config.client = self.vector_store.client
                elif isinstance(entity_config, dict):
                    entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        return self._entity_store

    def _entidade_por_chave(self, chave, search_filters):
        return entidade_por_chave(self.entity_store, chave, search_filters)

    def _upsert_entity(self, entity_text, entity_type, memory_id, filters):
        """Upsert an entity into the entity store, linking it to a memory."""
        try:
            entity_embedding = self.embedding_model.embed(entity_text, "add")
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
            chave = normalize_entity_key(entity_text)

            acao, alvo = resolver_linha_de_entidade(
                self.entity_store, entity_text, chave, entity_embedding,
                search_filters)
            if acao == "skip":
                logger.warning("entidade %r: upsert pulado — %s",
                               entity_text, alvo)
                return

            if acao == "update":
                # Update existing entity's linked_memory_ids
                match = alvo
                payload = match.payload or {}
                raw_linked = payload.get("linked_memory_ids")
                linked_ids = normalize_linked_memory_ids(raw_linked)
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                # Write when the id is new OR when normalizing changed the value:
                # that second case is how a row corrupted by some other writer
                # heals on its next touch instead of staying broken forever.
                # `data_normalized` também cura na primeira vez que a linha é
                # tocada: sem isso a linha legada nunca entra no lookup exato e a
                # duplicata por caixa continua nascendo ao lado dela.
                precisa_chave = payload.get("data_normalized") != chave
                if linked_ids != raw_linked or precisa_chave:
                    payload["linked_memory_ids"] = linked_ids
                    payload["data_normalized"] = chave
                    payload[link_key(memory_id)] = 1
                    self.entity_store.update(
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                # Id DETERMINÍSTICO = f(escopo, chave). Era sonda-então-UUID-
                # aleatório: o worker HTTP é serial, mas hooks e ingestão
                # instanciam `Memory` próprios, então dois escritores que não
                # acham nada geravam dois UUIDs e nasciam DUAS linhas para a
                # mesma entidade. Agora o segundo escreve no MESMO ponto, e a
                # corrida vira lost-update — reconciliado logo abaixo.
                entity_id = entity_point_id(search_filters, chave)
                entity_payload = {
                    "data": entity_text,
                    "data_normalized": chave,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    # chave por vínculo: sobrevive a `set_payload` concorrente
                    link_key(memory_id): 1,
                    **{k: v for k, v in search_filters.items()},
                }
                self.entity_store.insert(
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                    wait=True,   # leitura-após-escrita: ver o comentário no store
                )
                self._reconcilia_vinculo(entity_id, memory_id)
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}': {e}")

    def _reconcilia_vinculo(self, entity_id, memory_id, tentativas=None):
        # `tentativas=None` -> usa o default do módulo (e portanto o env).
        # Declarar `= 4` aqui ANULAVA `MEM0_ENTITY_RECONCILE_ATTEMPTS` e fez
        # duas medições com janela maior não mudarem nada — o knob existia e não
        # chegava a lugar nenhum.
        return reconcilia_vinculo(
            self.entity_store, entity_id, memory_id,
            ENTITY_RECONCILE_ATTEMPTS if tentativas is None else tentativas)

    def _remove_memory_from_entity_store(self, memory_id, filters):
        """Strip `memory_id` from every entity record scoped to `filters`.

        For each entity whose `linked_memory_ids` contains `memory_id`:
          - remove the id; if the list becomes empty, delete the entity record.
          - otherwise update the payload only (no re-embed: the entity text is
            unchanged, and a full upsert would churn the entity HNSW).

        Errors on individual entities are swallowed at debug level; outer
        failures at warning level, so the primary delete path is never broken by
        entity cleanup.
        """
        if not _entity_cleanup_enabled():
            return True
        return unlink_memory_from_entity_rows(self.entity_store, memory_id, filters)

    def _link_entities_for_memory(self, memory_id, text, filters):
        """Extract entities from `text` and link them to `memory_id` in the
        entity store, scoped to `filters`. Simpler single-memory variant of
        Phase 7 in add(): per-entity search-then-update-or-insert via the
        existing `_upsert_entity` helper. Non-fatal on any failure.
        """
        try:
            entities = extract_entities(text, language=self.config.language)
            if not entities:
                return
            search_filters = {k: v for k, v in filters.items()
                              if k in SCOPE_KEYS and v}
            if vincular_entidades_em_lote(
                    self.entity_store, self.embedding_model, memory_id,
                    entities, search_filters):
                return
            # Lote recusou (falha de embed): serial, que refaz o lookup por
            # conta própria sob a MESMA regra de identidade.
            seen = set()
            for entity_type, entity_text in entities:
                key = normalize_entity_key(entity_text)
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    self._upsert_entity(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}': {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id}: {e}")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        temporal_context: str = "conversation",
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            timestamp (Any, optional): Platform-only temporal parameter. Not supported in OSS.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            temporal_context (str, optional): "conversation" (default) resolves relative dates
                ("yesterday") against the observation/ingestion time. "document" disables that
                resolution: document dates are historical facts, taken only as written — a date
                without a year is never completed with the current year. Use for add_document.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            Mem0ValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """
        if timestamp is not None:
            raise ValueError(get_temporal_feature_error_message("sync", "add", "timestamp"))
        if temporal_context not in ("conversation", "document"):
            # fail-closed: um typo ("Document", "doc") viraria silenciosamente o modo
            # conversacional — e o override de datas de documento sumiria sem sinal.
            raise ValueError(
                f"temporal_context inválido: {temporal_context!r} (use 'conversation' ou 'document')"
            )

        temporal_usage_notice = detect_temporal_usage_from_metadata(metadata)
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            scale_threshold_notice = detect_scale_threshold_from_add_result(self, results)
            if temporal_usage_notice:
                display_temporal_usage_notice(self, "sync", "add", *temporal_usage_notice)
            elif scale_threshold_notice:
                display_scale_threshold_notice(self, "sync", "add", *scale_threshold_notice)
            else:
                display_first_run_notice(self, "sync", "add")
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        vector_store_result = self._add_to_vector_store(
            messages, processed_metadata, effective_filters, infer,
            prompt=prompt, temporal_context=temporal_context,
        )
        scale_threshold_notice = detect_scale_threshold_from_add_result(self, vector_store_result)
        if temporal_usage_notice:
            display_temporal_usage_notice(self, "sync", "add", *temporal_usage_notice)
        elif scale_threshold_notice:
            display_scale_threshold_notice(self, "sync", "add", *scale_threshold_notice)
        else:
            display_first_run_notice(self, "sync", "add")
        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer, prompt=None, temporal_context="conversation"):
        if not infer:
            # Um embed POR MENSAGEM custava uma ida ao embedder cada — e o custo
            # é dominado pela CHAMADA, não pelo item (medido contra bge-m3:
            # ~500 ms de overhead por chamada, ~12 ms por item curto). Aqui o
            # embed sai do laço; `_create_memory` já aceita vetor pré-computado.
            validas = _mensagens_validas_para_add(messages)
            embed_map = _embed_map_de(
                self.embedding_model, [m["content"] for m in validas])

            returned_memories = []
            for message_dict in validas:
                msg_content = message_dict["content"]
                if msg_content not in embed_map:
                    # Fallback por item já falhou para esta mensagem: pular UMA
                    # é o comportamento de antes; derrubar o lote inteiro seria
                    # regressão introduzida pelo próprio hoist.
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                mem_id = self._create_memory(msg_content, embed_map, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        # === V3 PHASED BATCH PIPELINE ===

        # Phase 0: Context gathering
        session_scope = _build_session_scope(filters)
        # DeepMem0: a document must NOT read from nor write to the conversational
        # message history — otherwise its chunks bleed into later adds via last_k
        # (proven with a reservation-number canary; leaks PII). Each doc chunk is
        # extracted standalone.
        skip_doc_history = temporal_context == "document"
        last_messages = [] if skip_doc_history else self.db.get_last_messages(session_scope, limit=10)
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        query_embedding = self.embedding_model.embed(parsed_messages, "search")
        existing_results = self.vector_store.search(
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        existing_memories = []
        uuid_mapping = {}
        for idx, mem in enumerate(existing_results):
            uuid_mapping[str(idx)] = mem.id
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction (single call)
        is_agent_scoped = bool(filters.get("agent_id")) and not filters.get("user_id")
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if is_agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX
        temp = _temporality_config(self.config)
        if temp is not None:
            # DeepMem0 v0.3: same call also detects supersession (+ event_date).
            system_prompt += build_temporality_suffix(include_event_date=temp.extract_event_date)
        if temporal_context == "document":
            # DeepMem0: a document keeps its OWN dates; disable Observation-Date
            # resolution so a year-less date is never filled with the current year.
            system_prompt += DOCUMENT_TEMPORAL_OVERRIDE

        # DeepMem0 v0.15: per-fact speaker. `rotulos` comes from the messages that
        # actually REACH the prompt (parse_vision_messages already ran, so `name`
        # survived the multimodal branches) — a label the model never saw rendered
        # must never be a value the validator accepts.
        rotulos_locutor, locutor_uniforme = (
            locutores_das_mensagens(messages) if speaker_attribution_enabled()
            else (set(), False)
        )
        if precisa_de_atribuicao_por_llm(rotulos_locutor, locutor_uniforme):
            system_prompt += build_speaker_attribution_suffix(rotulos_locutor)

        custom_instr = prompt or self.custom_instructions

        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
            # DeepMem0: extract facts in the input's language for non-English
            # corpora (upstream ships this flag but never sets it).
            use_input_language=(getattr(self.config, "language", "en") != "en"),
        )

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

        # Parse response
        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                extracted_memories = []
            else:
                try:
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")
            extracted_memories = []

        if not extracted_memories:
            # Save messages even if nothing extracted
            if not skip_doc_history:
                self.db.save_messages(messages, session_scope)
            return []

        # Phase 3: Batch embed all extracted memory texts
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        try:
            mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        except Exception:
            # Fallback: embed individually
            embed_map = {}
            for text in mem_texts:
                try:
                    embed_map[text] = self.embedding_model.embed(text, "add")
                except Exception as e:
                    logger.warning(f"Failed to embed memory text: {e}")

        # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
        # Build map of existing hashes for dedup (and DeepMem0 v0.2 reinforcement)
        existing_by_hash = {}
        for mem in existing_results:
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            if h:
                existing_by_hash[h] = mem

        dyn = _dynamics_config(self.config)
        records = []  # (memory_id, text, embedding, payload)
        pending_supersessions = []  # (new_memory_id, new_text, [old_ids], new_created_at) — applied after persist
        pending_similarity = []  # DeepMem0 v0.8 (T1S): (target_id, score, new_memory_id) — applied after persist
        pending_similarity_targets = set()  # one reinforcement per target per add
        seen_hashes = set()  # dedup within the current batch
        for mem in extracted_memories:
            text = mem.get("text")
            if not text or text not in embed_map:
                continue

            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_by_hash or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match): {text[:50]}")
                # DeepMem0 v0.2 (T1): a re-encountered fact is the strongest
                # reinforcement signal — the upstream silent no-op becomes the hook.
                # (An identical fact replaces nothing — its supersedes mark, if
                # any, is ignored by design.)
                # v0.9: a SUPERSEDED copy is deduped but NOT reinforced — its
                # timeline lives on the successor now; boosting the old record
                # would rebuild the double-dip the mask removed (t1s already
                # skipped superseded targets; T1 was the inconsistency).
                existing = existing_by_hash.get(mem_hash)
                if (dyn is not None and existing is not None
                        and not (existing.payload or {}).get(FIELD_SUPERSEDED_BY)):
                    _reinforce_memory(
                        self.vector_store, dyn, existing.id, existing.payload,
                        trigger=TRIGGER_DEDUP,
                    )
                continue
            seen_hashes.add(mem_hash)

            text_lemmatized = lemmatize_for_bm25(text, language=self.config.language)

            memory_id = str(uuid.uuid4())
            mem_metadata = deepcopy(metadata)
            mem_metadata["data"] = text
            mem_metadata["text_lemmatized"] = text_lemmatized
            mem_metadata["hash"] = mem_hash
            if "created_at" not in mem_metadata:
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            if mem.get("attributed_to"):
                mem_metadata["attributed_to"] = mem["attributed_to"]
            # DeepMem0 v0.15: WHO SPOKE. Uniform conversation resolves in code;
            # otherwise the model's proposal only survives if it is a `str` that
            # canonicalizes into the closed set enumerated in the prompt.
            # Anything else omits the field — which is exactly today's behaviour,
            # and the only failure direction that cannot corrupt attribution.
            locutor = resolver_locutor_do_fato(
                mem.get("actor_id"), rotulos_locutor, locutor_uniforme)
            if locutor:
                mem_metadata["actor_id"] = locutor
            # DeepMem0 v0.2 (option B): creation does NOT put the memory on the
            # timeline — it stays neutral until its first reinforcement (T1/T2/T3).
            if temp is not None:
                # DeepMem0 v0.3: the LLM references existing memories by their
                # presented index; resolve through uuid_mapping (hallucinated
                # ids are discarded) and defer marking until after persist.
                supersedes_ids = parse_supersedes_ids(mem.get("supersedes"), uuid_mapping)
                if supersedes_ids:
                    mem_metadata[FIELD_SUPERSEDES] = supersedes_ids
                    pending_supersessions.append((memory_id, text, supersedes_ids, mem_metadata["created_at"]))
                if temp.extract_event_date:
                    event_date = parse_event_date(mem.get("event_date"))
                    if temporal_context == "document":
                        # medido: o extrator pequeno escreve a data no TEXTO do
                        # fato mas omite o campo (0/185); e pode emitir uma data
                        # VÁLIDA-mas-ERRADA (ex.: ano corrente). Em modo documento
                        # a data ESCRITA vence: se o texto tem exatamente UMA data
                        # completa, ela é a verdade (cross-validação do parecer).
                        text_date = infer_event_date_from_text(text)
                        if text_date and event_date and event_date != text_date:
                            logger.warning(
                                f"event_date do LLM ({event_date}) contradiz a data do texto "
                                f"({text_date}) em modo documento — usando a do texto"
                            )
                            event_date = text_date
                        elif not event_date:
                            event_date = text_date
                    if event_date:
                        mem_metadata["event_date"] = event_date

            # DeepMem0 v0.8 (T1S): a NEW fact whose nearest corpus neighbor is a
            # near-paraphrase reinforces that neighbor — and is still inserted;
            # nothing is suppressed. A fact carrying ANY supersedes mark never
            # reinforces (a correction is a near-paraphrase with one changed
            # value; boosting what it corrects is the worst case). Decision now,
            # application deferred until after persist.
            if dyn is not None and not mem.get("supersedes"):
                _sim = _similar_reinforcement_target(
                    self.vector_store, text, embed_map[text], search_filters, dyn
                )
                if _sim is not None and _sim[0] not in pending_similarity_targets:
                    pending_similarity_targets.add(_sim[0])
                    pending_similarity.append((_sim[0], _sim[1], memory_id, mem_hash))

            records.append((memory_id, text, embed_map[text], mem_metadata))

        if not records:
            if not skip_doc_history:
                self.db.save_messages(messages, session_scope)
            return []

        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        all_ids = [r[0] for r in records]
        all_payloads = [r[3] for r in records]

        failed_persist = set()
        try:
            self.vector_store.insert(
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
            )
        except Exception:
            # Fallback: insert one by one
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                try:
                    self.vector_store.insert(vectors=[vec], ids=[mid], payloads=[pay])
                except Exception as e:
                    logger.error(f"Failed to insert memory {mid}: {e}")
                    failed_persist.add(mid)

        # DeepMem0 v0.3: mark superseded memories only AFTER the new facts are
        # persisted (never point a memory at a replacement that failed to land).
        superseded_events = []
        if pending_supersessions:
            try:
                for new_id, new_text, old_ids, new_created in pending_supersessions:
                    superseded_events.extend(
                        _mark_superseded(
                            self.vector_store, self.db, new_id, new_text, old_ids, new_created_at=new_created
                        )
                    )
            except Exception as e:
                logger.warning(f"Supersession marking pass failed: {e}")

        # DeepMem0 v0.8 (T1S): apply deferred semantic reinforcements only after
        # the new facts landed AND supersession marks were written — the fresh
        # read inside re-checks eligibility, so a target superseded by another
        # fact of this very batch is skipped. A fact that failed to persist
        # reinforces nobody.
        if pending_similarity:
            _apply_similar_reinforcements(
                self.vector_store, dyn,
                [p for p in pending_similarity if p[2] not in failed_persist],
            )

        # Batch history
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            for r in records
        ]
        try:
            self.db.batch_add_history(history_records)
        except Exception:
            # Fallback: add one by one
            for hr in history_records:
                try:
                    self.db.add_history(hr["memory_id"], None, hr["new_memory"], "ADD", created_at=hr.get("created_at"))
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']}: {e}")

        # Phase 7: Batch entity linking
        try:
            all_texts = [r[1] for r in records]
            all_entities = extract_entities_batch(
                all_texts, language=self.config.language)

            # 7a: Global dedup — collect unique entities across all memories
            global_entities = {}  # normalized_key -> (entity_type, entity_text, set of memory_ids)
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text in entities:
                    key = normalize_entity_key(entity_text)
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            if global_entities:
                ordered_keys = list(global_entities.keys())
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Single batch embed for all unique entities
                try:
                    entity_embeddings = self.embedding_model.embed_batch(entity_texts, "add")
                except Exception:
                    # Fallback: embed individually, use None for failures
                    entity_embeddings = []
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(self.embedding_model.embed(t, "add"))
                        except Exception:
                            entity_embeddings.append(None)


                if len(entity_embeddings) != len(ordered_keys):
                    logger.warning(
                        "embed_batch returned %d vectors for %d entity texts — "
                        "padding/truncating to avoid dropping entity links",
                        len(entity_embeddings),
                        len(ordered_keys),
                    )
                    entity_embeddings = list(entity_embeddings[: len(ordered_keys)])
                    entity_embeddings += [None] * (len(ordered_keys) - len(entity_embeddings))

                # Filter out entities with failed embeddings
                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                if valid:
                    valid_indices, valid_keys = zip(*valid)
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]

                    # 7c: lookup EXATO por chave, em lote, e SÓ ENTÃO a sonda.
                    # A Fase 7 rodava só com a sonda vetorial — a regra
                    # probabilística que o cutover de 30/07 substituiu em
                    # `_upsert_entity` e que aqui ficou. Duas regras de
                    # identidade no mesmo corpus, e a mais fraca no caminho
                    # mais quente (todo `add` com infer=True).
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    try:
                        por_chave, _amb = entidades_por_chaves(
                            self.entity_store, valid_keys, search_filters)
                    except Exception as exc:
                        # FAIL-CLOSED. "trata tudo como novo" converteria falha
                        # de infraestrutura em escrita destrutiva: o insert usa
                        # o id determinístico e SUBSTITUI o payload alheio.
                        logger.warning(
                            "lookup exato de entidade falhou (%s) — Fase 7 "
                            "abortada; o caminho serial reconcilia depois", exc)
                        raise

                    # Chave AMBÍGUA sai do processamento inteiro. Sem isto ela
                    # cai em `faltantes`, vai para a sonda, e acaba atualizando
                    # uma das duplicatas ou INSERINDO no id determinístico —
                    # apagando payload alheio. A guarda de multiplicidade
                    # existia só no caminho serial; aqui, o mais quente, ela
                    # estava pela metade, o que é o mesmo que não existir.
                    if _amb:
                        logger.warning(
                            "Fase 7: %d chave(s) ambígua(s) puladas — %s",
                            len(_amb), sorted(_amb))
                    processar = [j for j, k in enumerate(valid_keys)
                                 if k not in _amb]

                    faltantes = [j for j in processar
                                 if valid_keys[j] not in por_chave]
                    existing_matches = [[] for _ in valid_keys]
                    if faltantes:
                        sondados = self.entity_store.search_batch(
                            queries=[valid_texts[j] for j in faltantes],
                            vectors_list=[valid_vectors[j] for j in faltantes],
                            top_k=1,
                            filters=search_filters,
                        )
                        # Forma validada, e FECHA se não bater. Resposta curta
                        # ou de outro tipo era lida como "não achou" — e não
                        # achar leva a inserir no id determinístico.
                        if (not isinstance(sondados, list)
                                or len(sondados) != len(faltantes)
                                # CADA entrada tem de ser a lista de matches
                                # daquela consulta. Validar só a externa deixa
                                # passar `[<objeto>]`, que é iterável, rende
                                # zero matches e portanto INSERE — silencioso,
                                # ao contrário da lista curta, que ao menos
                                # estoura IndexError.
                                or not all(isinstance(x, list) for x in sondados)):
                            raise RuntimeError(
                                f"search_batch devolveu {type(sondados).__name__} "
                                f"com {len(sondados) if isinstance(sondados, list) else '?'} "
                                f"entradas para {len(faltantes)} consultas")
                        for pos, j in enumerate(faltantes):
                            existing_matches[j] = sondados[pos]

                    # 7d: Separate into inserts vs updates
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    for j in processar:
                        key = valid_keys[j]
                        entity_type, entity_text, memory_ids = global_entities[key]
                        matches = existing_matches[j] if j < len(existing_matches) else []
                        # a sonda herda o filtro por SUBCONJUNTO: recusa o que
                        # não for deste escopo exato (ver `escopo_exato`).
                        matches = [m for m in matches
                                   if escopo_exato(getattr(m, "payload", None) or {},
                                                   search_filters)]
                        exata = por_chave.get(key)
                        if exata is not None:
                            matches = [exata]

                        if matches and (exata is not None or matches[0].score >= 0.95):
                            # Update existing entity
                            match = matches[0]
                            payload = match.payload or {}
                            # normalize_* is load-bearing here: a str payload fed
                            # straight to set() iterates CHARACTER BY CHARACTER,
                            # which is how real entity rows lost their links.
                            linked = set(normalize_linked_memory_ids(payload.get("linked_memory_ids")))
                            linked |= memory_ids
                            payload["linked_memory_ids"] = sorted(linked)
                            payload["data_normalized"] = key
                            payload.update({link_key(m): 1 for m in memory_ids})
                            try:
                                self.entity_store.update(
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}': {e}")
                        else:
                            # New entity — collect for batch insert
                            to_insert_vectors.append(valid_vectors[j])
                            # ⚠️ ESTE é o escritor que roda em todo `add` com
                            # infer=True — a Fase 7 em lote. Ele ficou com
                            # `uuid4()` e sem `data_normalized` enquanto
                            # `_upsert_entity` ganhava identidade determinística,
                            # e o resultado seria pior que não ter consertado
                            # nada: o corpus passa a ter DUAS regras de
                            # identidade conforme o caminho que escreveu.
                            to_insert_ids.append(
                                entity_point_id(search_filters, key))
                            to_insert_payloads.append({
                                "data": entity_text,
                                "data_normalized": key,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **{link_key(m): 1 for m in memory_ids},
                                **search_filters,
                            })

                    # 7e: Single batch insert for all new entities
                    if to_insert_vectors:
                        try:
                            self.entity_store.insert(
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                                # leitura-após-escrita: o default do Qdrant NÃO
                                # espera, e escrita confirmada mas ainda
                                # invisível é o que fez o escritor seguinte
                                # recriar a linha e apagar vínculos alheios
                                # (30/07). `_upsert_entity` já esperava; este
                                # caminho, o mais quente, não.
                                wait=True,
                            )
                        except Exception as e:
                            logger.warning(f"Batch entity insert failed: {e}")
        except Exception as e:
            logger.warning(f"Batch entity linking failed: {e}")

        # Phase 8: Save messages + return
        if not skip_doc_history:
            self.db.save_messages(messages, session_scope)

        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            for r in records
        ]
        # DeepMem0 v0.3: surface supersessions to the caller (additive entries).
        # v0.4: pairs may point either way — a queued fact that arrived late is
        # born superseded by the fresher existing one (superseded_id == new id).
        returned_memories.extend(
            {"id": superseded_id, "event": "SUPERSEDED", "superseded_by": superseding_id}
            for superseded_id, superseding_id in superseded_events
        )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        return returned_memories

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            display_first_run_notice(self, "sync", "get")
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "attributed_to",
            "role",
            "memory_scope",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        display_first_run_notice(self, "sync", "get")
        return result_item

    def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if top_k is invalid.
        """
        # Reject top-level entity params - must use filters instead
        _scope_kwargs = _extract_top_level_entity_params(kwargs)
        if _scope_kwargs:
            filters = {**_scope_kwargs, **(filters or {})}

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )
        _canonizar_filtro_de_locutor(effective_filters)

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )

        all_memories_result = self._get_all_from_vector_store(effective_filters, limit)

        if scale_threshold_notice:
            display_scale_threshold_notice(self, "sync", "get_all", *scale_threshold_notice)
        else:
            display_first_run_notice(self, "sync", "get_all")
        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "attributed_to",
            "role",
            "memory_scope",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: Optional[bool] = None,
        explain: bool = False,
        reference_date: Optional[Any] = None,
        min_importance: Optional[float] = None,
        domain: Optional[str] = None,
        memory_type: Optional[str] = None,
        sort_by_importance: bool = False,
        as_of: Optional[str] = None,
        event_from: Optional[str] = None,
        event_to: Optional[str] = None,
        reinforce: Optional[bool] = None,
        search_id: Optional[str] = None,
        historical: bool = False,
        **kwargs,
    ):
        """
        Searches for memories based on a query.

        Args:
            query (str): Query to search for.
            top_k (int, optional): Maximum number of results to return. Defaults to 20.
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}

                Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            threshold (float, optional): Minimum score for a memory to be included. Defaults to 0.1.
            rerank (bool, optional): Whether to rerank results. Defaults to False.
            explain (bool, optional): Whether to include score_details for each result. Defaults to False.
            reference_date (Any, optional): Platform-only temporal parameter. Not supported in OSS.
            as_of (str, optional): DeepMem0 v0.3 RECORD-time anchor (ISO date/datetime) — restrict
                results to memories that already existed then (filters on created_at) and restore
                the world as it was. Answers "what did I know on X". DeepMem0 runtime only.
            event_from (str, optional): DeepMem0 v0.6 EVENT-time window start (inclusive). Full or
                partial ISO date — "2023" = whole year, "2023-10" = whole month, "2023-10-17" = day.
                Filters on event_date (WHEN the fact happened, distinct from as_of's record-time).
                Memories without an event_date are EXCLUDED while the window is active. One side
                alone = open interval. DeepMem0 runtime only.
            event_to (str, optional): DeepMem0 v0.6 EVENT-time window end (inclusive), same partial
                expansion. When neither event_from/event_to is given, a single date named in the
                query auto-anchors ranking (event_ranking) without filtering anything out.

        Returns:
            dict: A dictionary containing the search results under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
                  DeepMem0 also echoes "as_of" (record-time anchor), "event_anchor" ({"from","to"}
                  auto-detected from the query) OR "event_filter" ({"from","to"} explicit window;
                  mutually exclusive with event_anchor) when those apply.

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if threshold/top_k values are invalid.
        """
        if reference_date is not None:
            raise ValueError(get_temporal_feature_error_message("sync", "search", "reference_date"))

        # DeepMem0 v0.3: as-of anchor — "what did I know / what held on that date".
        as_of_iso, as_of_dt = (None, None)
        if as_of is not None and _temporality_config(self.config) is not None:
            as_of_iso, as_of_dt = parse_as_of(as_of)

        # DeepMem0 v0.10: recordação histórica — decisão derivada UMA vez e
        # lida por todos (gate de reforço, fusão, adjuster); recordar nunca
        # reforça, mesmo com reinforce=True explícito. Config só é tocada
        # quando o modo foi PEDIDO (instâncias nuas em testes upstream não
        # têm config; mesmo padrão do as_of/v0.6).
        if historical:
            _validate_historical(historical, as_of, _temporality_config(self.config))
            reinforce = False

        # DeepMem0 v0.6: event-time window — validate caller bounds fail-fast
        # (mirrors as_of) EVEN when temporality is off, so a malformed date is
        # never a config-dependent silent no-op. Application is gated below.
        event_from_iso, event_to_iso = (None, None)
        if event_from is not None or event_to is not None:
            event_from_iso, event_to_iso = expand_event_window(event_from, event_to)
        event_anchor = None

        # Reject top-level entity params - must use filters instead
        _scope_kwargs = _extract_top_level_entity_params(kwargs)
        if _scope_kwargs:
            filters = {**_scope_kwargs, **(filters or {})}

        # Validate search parameters (before applying defaults)
        _validate_search_params(threshold=threshold, top_k=top_k)
        query = _validate_and_trim_search_query(query)
        temporal_usage_notice = detect_temporal_usage_from_search(query, filters)

        # Validate and trim entity IDs in filters
        effective_filters = filters.copy() if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )
        _canonizar_filtro_de_locutor(effective_filters)
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        # Apply enhanced metadata filtering if advanced operators are detected
        if self._has_advanced_operators(effective_filters):
            processed_filters = self._process_metadata_filters(effective_filters)
            # Remove logical/operator keys that have been reprocessed
            for logical_key in ("AND", "OR", "NOT"):
                effective_filters.pop(logical_key, None)
            for fk in list(effective_filters.keys()):
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    effective_filters.pop(fk, None)
            effective_filters.update(processed_filters)

        # DeepMem0 v0.3: record-time anchor — only memories that already existed
        # at the as_of instant participate (applies to the dense AND keyword
        # legs, before the over-fetch; Qdrant auto-detects a DatetimeRange for
        # ISO values). A caller-provided created_at bound is tightened, never
        # loosened.
        if as_of_iso is not None:
            existing_created = effective_filters.get("created_at")
            if isinstance(existing_created, dict):
                current_lte = existing_created.get("lte")
                existing_created["lte"] = (
                    min(current_lte, as_of_iso) if isinstance(current_lte, str) else as_of_iso
                )
            else:
                effective_filters["created_at"] = {"lte": as_of_iso}

        # DeepMem0 v0.6: auto-detect a single event-time expression in the query
        # for ranking — suppressed when the caller passed an explicit window (they
        # already stated intent). Gated by event_ranking; the fusion term is
        # separately gated by event_ranking_weight > 0 downstream. Placed after
        # filter validation so self.config is only touched once the request is
        # well-formed (mirrors as_of's post-validation config access).
        _search_config = getattr(self, "config", None)
        if event_from_iso is None and event_to_iso is None and _search_config is not None:
            _ev_cfg = _temporality_config(_search_config)
            if _ev_cfg is not None and getattr(_ev_cfg, "event_ranking", False):
                event_anchor = infer_event_anchor_from_query(query)

        # DeepMem0 v0.6: explicit event-time window filter (event_date range).
        # Record-time as_of and event-time window compose (AND'ed in the store).
        # Applied only when temporality is enabled (mirror as_of). A FRESH nested
        # dict is written so the caller's filter object is never mutated; an
        # existing event_date bound is tightened, never loosened. Undated memories
        # never match a range on a missing field, so they drop out of the window.
        if (event_from_iso is not None or event_to_iso is not None) and _temporality_config(self.config) is not None:
            bound = {}
            if event_from_iso is not None:
                bound["gte"] = event_from_iso
            if event_to_iso is not None:
                bound["lte"] = event_to_iso
            existing_event = effective_filters.get(FIELD_EVENT_DATE)
            if isinstance(existing_event, dict):
                merged = dict(existing_event)
                if "gte" in bound:
                    cur = merged.get("gte")
                    merged["gte"] = max(cur, bound["gte"]) if isinstance(cur, str) else bound["gte"]
                if "lte" in bound:
                    cur = merged.get("lte")
                    merged["lte"] = min(cur, bound["lte"]) if isinstance(cur, str) else bound["lte"]
                effective_filters[FIELD_EVENT_DATE] = merged
            else:
                effective_filters[FIELD_EVENT_DATE] = bound

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
                "explain": explain,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        # DeepMem0: a configured reranker is ON by default (upstream defaulted
        # rerank=False, so a configured reranker silently never ran unless every
        # caller opted in), and it sees an OVER-FETCHED candidate pool — reranking
        # only the fused top-k cannot recover targets that the additive fusion
        # buried under keyword-boosted competitors (measured on a PT corpus:
        # hit@1 0.857 -> 0.886, one extra recall, with pool=20).
        if rerank is None:
            rerank = self.reranker is not None
        fetch_limit = limit
        if rerank and self.reranker:
            fetch_limit = max(2 * limit, getattr(self.config, "rerank_pool", 20))

        search_start = time.perf_counter()
        original_memories = self._search_vector_store(
            query, effective_filters, fetch_limit, threshold, explain=explain, as_of_dt=as_of_dt,
            dense_anchors=(getattr(self.config, "rerank_dense_anchors", 5)
                           if (rerank and self.reranker) else 0),
            event_anchor=event_anchor,
            historical=historical,
        )
        search_elapsed_seconds = time.perf_counter() - search_start

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            try:
                reranked_memories = self.reranker.rerank(query, original_memories, fetch_limit)
                original_memories = reranked_memories
                # DeepMem0 v0.2/v0.3: blend ACT-R activation and the superseded
                # penalty into the reranked order (the fusion-stage signals only
                # shape the pool; the cross-encoder re-sorts it, so both must
                # also speak after the reranker — in a single sort).
                dyn = _dynamics_config(self.config)
                temp = _temporality_config(self.config)
                if dyn is not None or temp is not None:
                    original_memories = _apply_post_rerank_adjustments(
                        original_memories, dyn=dyn, temp=temp, as_of=as_of_dt,
                        event_anchor=event_anchor, historical=historical,
                    )
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")
        # DeepMem0: cut the over-fetched pool back to the requested top_k.
        original_memories = original_memories[:limit]
        original_memories = _apply_metadata_post_filters(
            original_memories,
            min_importance=min_importance,
            domain=domain,
            memory_type=memory_type,
            sort_by_importance=sort_by_importance,
        )

        # DeepMem0 v0.2 (T3, opt-in): being retrieved is itself a re-encounter.
        # Only the memories actually returned to the caller are reinforced,
        # asynchronously, so the hot path never pays for the write-back.
        dyn = _dynamics_config(self.config)
        if _t3_enabled(dyn, reinforce) and original_memories:
            # exposed_at é AGORA — o instante em que o caller viu estas memórias.
            # O worker pode rodar muito depois; usar o relógio dele deslocaria a
            # linha do tempo e as duas janelas.
            _reinforce_hits_in_background(
                self.vector_store, dyn,
                _t3_targets(dyn, original_memories,
                            search_id=search_id or uuid.uuid4().hex[:16],
                            exposed_at=_dynamics_utcnow()),
            )

        if temporal_usage_notice:
            display_temporal_usage_notice(self, "sync", "search", *temporal_usage_notice)
        elif scale_threshold_notice:
            display_scale_threshold_notice(self, "sync", "search", *scale_threshold_notice)
        elif search_elapsed_seconds > PERFORMANCE_SLOW_QUERY_THRESHOLD_SECONDS:
            display_performance_slow_query_notice(
                self,
                "sync",
                "search",
                search_elapsed_seconds,
                top_k,
                len(original_memories),
            )
        else:
            display_first_run_notice(self, "sync", "search")
        response = {"results": original_memories}
        if as_of_iso is not None:
            response["as_of"] = as_of_iso
        # DeepMem0 v0.10: no modo recordação, avisa quais resultados têm
        # sucessor EXPLÍCITO conhecido ("há fatos mais atuais") + echo do modo.
        if historical:
            _n_newer = _annotate_known_successors(original_memories)
            response["historical_recall"] = {
                "as_of": as_of_iso, "results_with_newer_version": _n_newer,
            }
        # DeepMem0 v0.6: echo the auto-detected ranking anchor OR the explicit
        # filter window (mutually exclusive — an explicit window suppresses
        # auto-detection). event_anchor is echoed whenever an anchor was found,
        # independent of whether any candidate matched it.
        if event_anchor is not None:
            response["event_anchor"] = {"from": event_anchor[0], "to": event_anchor[1]}
        elif event_from_iso is not None or event_to_iso is not None:
            response["event_filter"] = {"from": event_from_iso, "to": event_to_iso}
        return response

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        def merge_filters(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Merge source into target, deep-merging nested operator dicts for the same key."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    target[key].update(value)
                else:
                    target[key] = value

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                merge_filters(processed_filters, process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.
        
        Args:
            filters: Dictionary of filters to check
            
        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False
            
        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    def _search_vector_store(self, query, filters, limit, threshold=0.1, explain=False, as_of_dt=None, dense_anchors=0, event_anchor=None, historical=False):
        # Guard against None threshold (backward compat)
        if threshold is None:
            threshold = 0.1

        # Step 1: Preprocess query
        query_lemmatized = lemmatize_for_bm25(query, language=self.config.language)
        query_entities = extract_entities(query, language=self.config.language)

        # Step 2: Embed query
        embeddings = self.embedding_model.embed(query, "search")

        # Step 3: Semantic search (over-fetch for scoring pool)
        internal_limit = max(limit * 4, 60)
        semantic_results = self.vector_store.search(
            query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4: Keyword search (if store supports it)
        keyword_results = self.vector_store.keyword_search(
            query=query_lemmatized, top_k=internal_limit, filters=filters
        )

        # Step 5: Compute BM25 scores from keyword results
        bm25_scores = {}
        if keyword_results is not None:
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                if raw_score and raw_score > 0:
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6: Compute entity boosts
        entity_boosts = {}
        if query_entities:
            entity_boosts = self._compute_entity_boosts(query_entities, filters)

        # Step 7: Build candidate set from semantic results
        candidates = []
        for mem in semantic_results:
            mem_id = str(mem.id)
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": mem.payload if hasattr(mem, 'payload') else {},
            })

        # Step 7b (DeepMem0 v0.2): lazy ACT-R activation over the candidate pool.
        # Derived from each candidate's reinforcement timeline at query time —
        # memories without a history stay neutral (no key in the dict).
        activation_boosts = {}
        dyn = _dynamics_config(self.config)
        # v0.10: recordação histórica não usa peso de uso — ativação inerte.
        if dyn is not None and dyn.weight > 0 and not historical:
            now = _dynamics_utcnow()
            for cand in candidates:
                # v0.9 MASK: a superseded record gets NO activation — with the
                # timeline COPIED to its successor, boosting both would let the
                # family double-dip (penalty − boost partially cancel on the old
                # one). Same predicate as the penalty, so as_of time travel keeps
                # historical activation for the then-current version.
                if superseded_penalty_applies(cand["payload"], as_of=as_of_dt):
                    continue
                boost = boost_from_payload(cand["payload"], now=now, decay=dyn.decay)
                if boost > 0:
                    activation_boosts[cand["id"]] = boost

        # Step 7c (DeepMem0 v0.3): superseded facts are demoted, never excluded.
        # Anchor-aware: with an as_of, a memory superseded only AFTER the anchor
        # was still the current fact then, so its penalty is waived.
        superseded_penalties = {}
        temp = _temporality_config(self.config)
        if temp is not None and temp.superseded_penalty > 0:
            for cand in candidates:
                if superseded_penalty_applies(cand["payload"], as_of=as_of_dt):
                    superseded_penalties[cand["id"]] = temp.superseded_penalty

        # Step 7d (DeepMem0 v0.6): event-time proximity boosts over the candidate
        # pool when the query named a date. FUSION-stage only, gated by
        # event_ranking_weight > 0 (weight=0 => tie-break-only, no divisor growth).
        # Memories without an event_date stay neutral (no key in the dict).
        event_boosts = {}
        if (temp is not None and getattr(temp, "event_ranking", False)
                and temp.event_ranking_weight > 0 and event_anchor):
            event_window_days = getattr(temp, "event_window_days", 30)
            for cand in candidates:
                prox = event_proximity(event_anchor, (cand["payload"] or {}).get(FIELD_EVENT_DATE), event_window_days)
                if prox > 0:
                    event_boosts[cand["id"]] = prox

        # Step 8: Score and rank
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=limit,
            explain=explain,
            activation_boosts=activation_boosts,
            activation_weight=dyn.weight if dyn is not None else 0.0,
            penalties=superseded_penalties or None,
            event_boosts=event_boosts or None,
            event_weight=temp.event_ranking_weight if temp is not None else 0.0,
        )

        # DeepMem0: DENSE ANCHORS — a fusão corta o pool por score FUNDIDO, então
        # um alvo denso-forte enterrado por boosts ruidosos (entity/activation de
        # competidores) sai do pool ANTES do reranker e o resgate-por-rerank da F1
        # nunca acontece (medido: alvo denso rank 1-2, fundido rank 21-40, sumia
        # do top-10 quando o corpus cresceu 620->984). Garantia: o denso-top-N
        # sempre entra no pool do reranker — só ADICIONA candidatos; o
        # cross-encoder decide. Ativo apenas no caminho com rerank.
        if dense_anchors > 0:
            seen_ids = {r["id"] for r in scored_results}
            for cand in candidates[:dense_anchors]:
                if cand["id"] not in seen_ids:
                    scored_results.append(cand)
                    seen_ids.add(cand["id"])

        # Step 9: Format results
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "attributed_to",
            "role",
            "memory_scope",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        original_memories = []
        for scored in scored_results:
            payload = scored.get("payload") or {}

            if not payload.get("data"):
                continue  # Skip candidates with no payload data

            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]

            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                if not memory_item_dict.get("metadata"):
                    memory_item_dict["metadata"] = {}
                memory_item_dict["metadata"].update(additional_metadata)
            if explain and "score_details" in scored:
                memory_item_dict["score_details"] = scored["score_details"]

            original_memories.append(memory_item_dict)

        return original_memories

    def _compute_entity_boosts(self, query_entities, filters):
        """Compute per-memory entity boosts from entity store search.

        For each extracted entity from the query:
        1. Embed the entity text
        2. Search the entity store (threshold >= 0.5)
        3. For each matched entity, boost its linked memories

        Returns:
            Dict mapping memory_id (str) -> max entity boost [0, 0.5].
        """
        # Deduplicate entities (max 8)
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = normalize_entity_key(entity_text)
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        try:
            entity_texts = [text for _, text in deduped]
            embeddings = self.embedding_model.embed_batch(entity_texts, "search")

            if len(embeddings) != len(entity_texts):
                logger.warning(
                    "embed_batch returned %d vectors for %d texts — skipping entity boost",
                    len(embeddings),
                    len(entity_texts),
                )
                return memory_boosts

            entity_store = self.entity_store

            def _search_entity(entity_text, embedding):
                return entity_store.search(
                    query=entity_text, vectors=embedding, top_k=500, filters=search_filters
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    pool.submit(_search_entity, text, emb): text
                    for text, emb in zip(entity_texts, embeddings)
                }

                for future in concurrent.futures.as_completed(futures):
                    try:
                        matches = future.result()
                    except Exception as e:
                        logger.warning("Entity boost search failed for one entity: %s", e)
                        continue

                    for match in matches:
                        similarity = match.score if hasattr(match, 'score') else 0.0
                        if similarity < 0.5:
                            continue

                        payload = match.payload if hasattr(match, 'payload') else {}
                        linked_memory_ids = payload.get("linked_memory_ids", [])
                        if not isinstance(linked_memory_ids, list):
                            # Deliberately fail CLOSED here rather than normalize:
                            # a malformed value must not become retrieval-active,
                            # because a non-empty entity_boosts also raises the
                            # score divisor for every candidate in the query.
                            # But it must not be SILENT either — silence is how a
                            # dead entity boost went unnoticed for weeks. Writers
                            # normalize and self-heal; the reader only reports.
                            logger.warning(
                                "Entity %s has a malformed linked_memory_ids (%s); "
                                "its boost is being skipped. Run "
                                "check_corpus.py / repair_entity_links.py.",
                                getattr(match, "id", "?"), type(linked_memory_ids).__name__)
                            continue

                        num_linked = max(len(linked_memory_ids), 1)
                        memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                        boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                        for memory_id in linked_memory_ids:
                            if memory_id:
                                memory_key = str(memory_id)
                                memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    def update(self, memory_id, data, metadata: Optional[Dict[str, Any]] = None):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        if metadata:  # strip reserved lineage fields (anti-injection, incl. legacy path)
            metadata = {k: v for k, v in metadata.items() if k not in RESERVED_LINEAGE_FIELDS}
        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        # DeepMem0 v0.7.1: the versioned path returns (current_id, superseded_head_id)
        # resolved INSIDE the transition lock, so old_id matches the version actually
        # superseded even under concurrency (critic #4). Legacy in-place returns a
        # single id (current == old).
        returned = self._update_memory(memory_id, data, existing_embeddings, metadata)
        if isinstance(returned, tuple):
            current_id, old_id = returned
        else:
            current_id = old_id = returned
        display_first_run_notice(self, "sync", "update")
        return {"message": "Memory updated successfully!", "id": current_id, "old_id": old_id}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})

        # DeepMem0 v0.7.1: delete removes the WHOLE UPDATE-VERSION chain (only the
        # dedicated lineage, NEVER semantic supersedence siblings — critic #1), under
        # the lock. Transactional shape (critic #6): _collect_chain PREFLIGHTS the
        # whole chain (raises fail-closed on cross-scope/corruption before any
        # mutation); then delete HISTORICAL versions first and the current HEAD LAST,
        # so a mid-failure leaves the current fact alive (not only demoted history);
        # partial failure returns the exact remaining ids for a deterministic retry.
        temp = _temporality_config(self.config)
        if temp is not None and getattr(temp, "version_on_update", False):
            with self._version_lock:
                chain = _collect_chain(
                    lambda vid: self.vector_store.get(vector_id=vid), memory_id
                )
                if not chain:
                    raise ValueError(f"Memory with id {memory_id} not found")
                order = list(reversed(chain))  # chain is head-first -> head deleted LAST
                deleted: List[str] = []
                for cid in order:
                    existing = self.vector_store.get(vector_id=cid)
                    if existing is None:
                        continue
                    try:
                        self._delete_memory(cid, existing)
                        deleted.append(cid)
                    except Exception as e:
                        remaining = [x for x in order if x not in deleted]
                        logger.error(f"Partial version-chain delete for {memory_id}: {e}; remaining={remaining}")
                        return {"message": "Memory partially deleted; retry with a remaining id",
                                "deleted": deleted, "remaining": remaining}
        else:
            existing_memory = self.vector_store.get(vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found")
            self._delete_memory(memory_id, existing_memory)
        decay_usage_notice = detect_decay_usage_from_delete()
        if decay_usage_notice:
            display_decay_usage_notice(self, "sync", "delete", *decay_usage_notice)
        else:
            display_first_run_notice(self, "sync", "delete")
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        # ⚠️ Normalizar ANTES de montar `filters`, e antes do teste de verdade.
        # Duas razões, nessa ordem:
        #   1. o filtro do store é casamento EXATO — `delete_all(user_id=" alice")`
        #      não casava nada e a chamada voltava SEM ERRO, tendo apagado zero.
        #      Escopo errado em delete é silencioso por natureza: não há resultado
        #      vazio para ninguém estranhar;
        #   2. `if user_id:` antes da normalização descarta `0`, e a chamada morre
        #      em "At least one filter is required" — mensagem errada para o
        #      defeito certo. Depois da coerção, `0` é `"0"`, que é verdadeiro.
        # Este trecho é uma REGRESSÃO NOSSA: o upstream valida aqui, e a
        # reescrita de paginação deste método deixou a validação para trás.
        user_id = _validate_and_trim_entity_id(user_id, "user_id")
        agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
        run_id = _validate_and_trim_entity_id(run_id, "run_id")

        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        # Paginate. `list()` defaults to top_k=100 in several stores (Qdrant
        # among them), so the old single untruncated call silently deleted at
        # most one page and reported that as the whole job.
        deleted = 0
        attempted = set()
        for _ in range(_DELETE_ALL_MAX_PAGES):
            page = self.vector_store.list(filters=filters, top_k=_DELETE_ALL_PAGE_SIZE)[0]
            # Stop on the first page that offers nothing NEW. Termination cannot
            # rest on "the page came back empty": a row we failed to delete, or a
            # store that does not reflect deletes immediately, would be re-listed
            # forever. Each iteration either claims at least one unseen id or ends.
            fresh = [m for m in page if m.id not in attempted]
            if not fresh:
                break
            for memory in fresh:
                attempted.add(memory.id)
                try:
                    self._delete_memory(memory.id)
                    deleted += 1
                except Exception as e:
                    logger.warning(f"delete_all: memory {memory.id} could not be deleted: {e}")
        else:
            logger.warning("delete_all: page cap (%d) reached — scope may not be drained",
                           _DELETE_ALL_MAX_PAGES)
        if len(attempted) != deleted:
            logger.warning("delete_all: %d of %d memories could not be deleted",
                           len(attempted) - deleted, len(attempted))

        logger.info(f"Deleted {deleted} memories")

        decay_usage_notice = detect_decay_usage_from_delete_all(deleted)
        if decay_usage_notice:
            display_decay_usage_notice(self, "sync", "delete_all", *decay_usage_notice)
        else:
            display_first_run_notice(self, "sync", "delete_all")
        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        history = self.db.get_history(memory_id)
        display_first_run_notice(self, "sync", "history")
        return history

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data, language=self.config.language)
        # DeepMem0 v0.2: creation stays neutral until the first reinforcement.

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _version_update(self, memory_id, data, existing_embeddings, metadata, temp):
        """DeepMem0 v0.7.1 versioned update. Under the per-instance version lock
        (serializes multi-thread on the singleton Memory — the production write path;
        cross-instance/process needs an external single writer, e.g. the serial MCP
        worker). Resolves the head via the DEDICATED ``_mem0_version_next`` lineage
        (fail-closed on cross-scope/cycle — critic #2), mints a new version, and links
        the lineage in BOTH directions (``version_next`` on the old, ``version_prev``
        on the new). Keeps writing ``superseded_by``/``superseded_at`` for ranking/
        as_of but NOT the shared semantic ``supersedes`` (critic #5). Direction honors
        record time: an update older than the head is born superseded by it (the head
        stays current and simply gains the older record as a version predecessor).
        Strict + compensating: on any failure the head is restored to its EXACT
        original and the new version deleted. Returns ``(current_id, superseded_head_id)``
        resolved INSIDE the lock so the caller reports lineage matching reality
        (critic #4).
        """
        _HEAD_MUT = (FIELD_SUPERSEDED_BY, FIELD_SUPERSEDED_AT, FIELD_VERSION_NEXT, FIELD_VERSION_PREV)

        def _hist(mem_id, old_txt, new_txt, cre, upd, src):
            try:
                self.db.add_history(mem_id, old_txt, new_txt, "SUPERSEDED", created_at=cre,
                                    updated_at=upd, actor_id=src.get("actor_id"), role=src.get("role"))
            except Exception as e:
                logger.warning(f"Supersession history record failed for {mem_id}: {e}")

        with self._version_lock:
            head_id, head_mem = _resolve_chain_head(
                lambda vid: self.vector_store.get(vector_id=vid), memory_id
            )
            if head_mem is None:
                raise ValueError(
                    f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'"
                )
            head_payload = dict(getattr(head_mem, "payload", None) or {})
            caller = dict(metadata or {})
            operation_ts = caller.get("created_at") or _dynamics_utcnow().isoformat()
            born_superseded = supersession_inverted(operation_ts, head_payload.get("created_at"))
            # DeepMem0 v0.9: an updated fact is the SAME fact, evolved — the new
            # version COPIES the usage timeline (head untouched: crash-safe with
            # queued-update retries, as_of keeps historical activation) and gets
            # a T2 event. Decided BEFORE metadata is built; never for a
            # born-superseded record (the head stays current there).
            inherit_dyn = (
                bool(getattr(temp, "version_inherits_dynamics", False))
                and not born_superseded
            )
            v2_meta = _build_version_metadata(
                head_payload, data, caller, operation_ts, head_id,
                getattr(temp, "extract_event_date", True),
                inherit_dynamics=inherit_dyn,
            )
            dyn_extra, t2_outcome = _plan_version_dynamics(
                head_payload, _dynamics_config(self.config), operation_ts,
                inherit=inherit_dyn,
            )
            v2_meta.update(dyn_extra)
            if born_superseded:
                # v2 is OLDER than the head: born superseded BY it. In the version
                # lineage v2 -> head (head gains v2 as a predecessor below). No
                # predecessor of its own. superseded_at = head's record-time so an
                # as_of between operation_ts and it still restores v2.
                v2_meta[FIELD_VERSION_PREV] = []
                v2_meta[FIELD_VERSION_NEXT] = head_id
                v2_meta[FIELD_SUPERSEDED_BY] = head_id
                v2_meta[FIELD_SUPERSEDED_AT] = head_payload.get("created_at") or operation_ts

            head_restore = {**head_payload, **{k: head_payload.get(k) for k in _HEAD_MUT}}
            head_modified = False
            new_id = None
            try:
                new_id = self._create_memory(data, existing_embeddings, metadata=v2_meta)
                session_filters = {k: v2_meta[k] for k in ("user_id", "agent_id", "run_id") if v2_meta.get(k)}
                self._link_entities_for_memory(new_id, data, session_filters)
                if born_superseded:
                    head_prev = list(head_payload.get(FIELD_VERSION_PREV) or [])
                    head_prev.append(new_id)
                    self.vector_store.update(vector_id=head_id, payload={**head_payload, FIELD_VERSION_PREV: head_prev})
                    head_modified = True
                    _hist(new_id, data, head_payload.get("data"), operation_ts, operation_ts, head_payload)
                    cp = (getattr(self.vector_store.get(vector_id=new_id), "payload", None) or {})
                    if cp.get(FIELD_VERSION_NEXT) != head_id:
                        raise RuntimeError(f"Version transition verify failed: {new_id} not born superseded by {head_id}")
                    current_id = head_id
                else:
                    self.vector_store.update(
                        vector_id=head_id,
                        payload={**head_payload, FIELD_SUPERSEDED_BY: new_id,
                                 FIELD_SUPERSEDED_AT: operation_ts, FIELD_VERSION_NEXT: new_id},
                    )
                    head_modified = True
                    _hist(head_id, head_payload.get("data"), data, head_payload.get("created_at"), operation_ts, head_payload)
                    check = self.vector_store.get(vector_id=head_id)
                    cp = (getattr(check, "payload", None) or {}) if check is not None else None
                    if cp is None or cp.get(FIELD_VERSION_NEXT) != new_id or cp.get(FIELD_SUPERSEDED_BY) != new_id:
                        raise RuntimeError(f"Version transition verify failed: head {head_id} not linked to {new_id}")
                    if self.vector_store.get(vector_id=new_id) is None:
                        raise RuntimeError(f"Version transition verify failed: new version {new_id} missing")
                    current_id = new_id
            except Exception:
                if head_modified:
                    try:
                        self.vector_store.update(vector_id=head_id, payload=head_restore)
                    except Exception as re_:
                        logger.error(f"Compensation restore of head {head_id} failed: {re_}")
                if new_id is not None:
                    try:
                        self._delete_memory(new_id)
                    except Exception as ce:
                        logger.error(f"Compensation delete of {new_id} failed: {ce}")
                raise
            # T2 notify only after the transition verified — same discipline as
            # the legacy path: never report a reinforcement a failed write
            # would have left unpersisted (compensation deletes the carrier).
            if t2_outcome is not None:
                _notify_reinforcement(new_id, TRIGGER_UPDATE, t2_outcome)
            logger.info(f"Versioned update: head={head_id} new={new_id} current={current_id} born={born_superseded}")
            return current_id, head_id

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        temp = _temporality_config(self.config)
        if temp is not None and getattr(temp, "version_on_update", False):
            # v0.9: o T2 EXISTE neste modo — a versão nova herda a timeline do
            # head (cópia) e ganha o evento T2, via _plan_version_dynamics
            # (gated por version_inherits_dynamics; pinado em teste).
            return self._version_update(memory_id, data, existing_embeddings, metadata, temp)
        logger.info(f"Updating memory with {data=}")

        # Embedding ANTES da leitura autoritativa: é o trecho lento (centenas de
        # ms), e enquanto ele rodava o payload lido antes ia ficando velho — um
        # T3 que chegasse nesse intervalo era apagado pelo upsert de payload
        # completo. Ler depois estreita a janela de segundos para milissegundos.
        # (Não a elimina: sobra `leitura → T3 escreve → upsert`. Garantia real
        # exigiria lock por memory_id entre T2 e T3, ou CAS no store.)
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(existing_memory.payload)
        if metadata is not None:
            new_metadata.update(metadata)

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data, language=self.config.language)
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Ownership scope is immutable after creation (issue #4490 for actor_id).
        # MESMA regra do caminho versionado, e a ausência conta: o guard antigo
        # só preservava um valor JÁ existente, então um `metadata={"actor_id": ...}`
        # do chamador carimbava autoria em memória que não tinha nenhuma — e
        # nenhuma é o estado de todo o corpus legado.
        aplicar_escopo_imutavel(new_metadata, existing_memory.payload)

        # DeepMem0 v0.2 (T2): an updated fact is alive — reinforce its timeline.
        # Inside the reinforcement window the content update still applies; only
        # the reinforcement bookkeeping is suppressed (fields carry over as-is).
        dyn = _dynamics_config(self.config)
        if dyn is not None:
            _fields, _outcome = plan_reinforcement(
                existing_memory.payload, dyn, TRIGGER_UPDATE
            )
            if _fields:
                new_metadata.update(_fields)
            # NÃO notifica aqui: a decisão foi tomada, mas o reforço só existe
            # depois que a escrita do update pega. Emitir "applied" antes do
            # write faria a telemetria afirmar um reforço que uma falha de
            # vector_store deixaria sem persistir.

        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        if dyn is not None:
            _notify_reinforcement(memory_id, TRIGGER_UPDATE, _outcome)
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        self._remove_memory_from_entity_store(memory_id, session_filters)
        self._link_entities_for_memory(memory_id, data, session_filters)

        return memory_id

    def _delete_memory(self, memory_id, existing_memory=None, *, op_id=None):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = self.vector_store.get(vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data", "")
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = existing_memory.payload or {}
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}
        # DeepMem0 v0.7.2: durable delete intent (crash-consistency). own_intent -> this
        # call opens a fresh PENDING intent; a caller-supplied op_id (reconciliation)
        # resumes an existing one. The intent state (not the tombstone) is the
        # authority on completion, so a crash at any point below is reconcilable.
        own_intent = op_id is None
        if own_intent:
            op_id = str(uuid.uuid4())
            if self.db is not None:
                before_image = json.dumps({
                    "data": prev_value, "created_at": created_at,
                    "actor_id": existing_memory.payload.get("actor_id"),
                    "role": existing_memory.payload.get("role"),
                }, sort_keys=True)
                self.db.begin_delete(op_id, memory_id,
                                     scope=json.dumps(session_filters, sort_keys=True),
                                     before_image=before_image)
        self.vector_store.delete(vector_id=memory_id)
        # Tombstone is IDEMPOTENT: never duplicate the DELETE row on retry/reconcile.
        if self.db is not None and not self.db.has_delete_tombstone(memory_id):
            self.db.add_history(
                memory_id,
                prev_value,
                None,
                "DELETE",
                created_at=created_at,
                updated_at=updated_at,
                actor_id=existing_memory.payload.get("actor_id"),
                role=existing_memory.payload.get("role"),
                is_deleted=1,
            )
        # Entity-store cleanup BEFORE committing the intent, and the commit is
        # CONDITIONAL on it. Ordering alone was not enough: the helper swallows
        # its errors, so commit-anyway turned a transient failure into a
        # permanent dangling link — reconciliation had nothing left to retry.
        # Cleanup is idempotent, so replaying it on reconcile is safe.
        cleaned = self._remove_memory_from_entity_store(memory_id, session_filters)

        if self.db is not None:
            if cleaned:
                self.db.commit_delete(op_id)
            else:
                logger.warning(
                    "Delete of %s: entity cleanup incomplete — leaving the intent "
                    "PENDING so reconciliation retries it.", memory_id)

        return memory_id

    def reconcile_pending_deletes(self) -> int:
        """DeepMem0 v0.7.2 — finish any delete interrupted by a crash (roadmap #7).

        For each ``pending`` delete intent: delete the vector if it is still present,
        ensure the DELETE tombstone exists (idempotent), and commit the intent. The
        intent STATE (not the tombstone) is the authority on completion, so this
        converges from a crash at ANY point — no lost tombstone, no false 'completed'
        signal for a live vector. Idempotent; a single cheap query when nothing is
        pending. Entity cleanup RUNS here too (it used to be skipped as "benign
        residue" — it is not: those stale links are exactly the dangling refs the
        corpus audit now reports, and they outlive the memory forever). Called
        once at init.
        """
        if getattr(self, "db", None) is None:
            return 0
        try:
            pending = self.db.list_pending_deletes()
        except Exception as e:
            logger.warning(f"Could not read pending delete intents: {e}")
            return 0
        reconciled = 0
        for intent in pending:
            mid, op = intent["memory_id"], intent["op_id"]
            try:
                before = {}
                if intent.get("before_image"):
                    try:
                        before = json.loads(intent["before_image"])
                    except Exception:
                        before = {}
                existing = self.vector_store.get(vector_id=mid)
                spared = False
                cleaned = True   # a SPARED id has nothing to clean
                if existing is not None:
                    # ABA guard: only delete if the CURRENT vector is the SAME memory the
                    # intent targeted (created_at identity). A REUSED id (import/restore/manual)
                    # is SPARED — we never delete a different memory that took the id.
                    cur_created = (existing.payload or {}).get("created_at")
                    orig_created = before.get("created_at")
                    if orig_created and cur_created and cur_created != orig_created:
                        logger.warning(f"Reconcile: id {mid} appears REUSED (created_at differs) — sparing current vector")
                        spared = True
                    else:
                        self.vector_store.delete(vector_id=mid)
                if not spared:
                    # The ABA guard extends to the entity store: stripping links for
                    # a REUSED id would silently unlink the *new* memory that took it.
                    try:
                        scope = json.loads(intent.get("scope") or "{}")
                    except Exception:
                        scope = {}
                    if not isinstance(scope, dict):
                        scope = {}
                    cleaned = self._remove_memory_from_entity_store(mid, scope)
                if not self.db.has_delete_tombstone(mid):
                    # faithful tombstone from the before-image (survives a crash that hit
                    # before the tombstone was written)
                    self.db.add_history(
                        mid, before.get("data"), None, "DELETE",
                        created_at=before.get("created_at"),
                        updated_at=datetime.now(timezone.utc).isoformat(),
                        actor_id=before.get("actor_id"), role=before.get("role"),
                        is_deleted=1,
                    )
                if cleaned:
                    self.db.commit_delete(op)
                    reconciled += 1
                else:
                    # Intent stays PENDING on purpose: an incomplete entity
                    # cleanup that we commit anyway is a dangling link nobody
                    # will ever come back for.
                    logger.warning(
                        "Reconcile of %s: entity cleanup incomplete — intent stays PENDING.", mid)
            except Exception as e:
                logger.warning(f"Reconcile of pending delete {op} ({mid}) failed: {e}")
        if reconciled:
            logger.info(f"Reconciled {reconciled} pending delete(s) on startup.")
        return reconciled

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")

        if hasattr(self.db, "connection") and self.db.connection:
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            self.db.connection.close()

        self.db = SQLiteManager(self.config.history_db_path)

        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        # Reset entity store if initialized
        if self._entity_store is not None:
            try:
                self._entity_store.reset()
            except Exception as e:
                logger.warning(f"Failed to reset entity store: {e}")
            self._entity_store = None

        capture_event("mem0.reset", self, {"sync_type": "sync"})
        display_first_run_notice(self, "sync", "reset")

    def close(self):
        """Release resources held by this Memory instance (SQLite connections, etc.)."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")


class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        # DeepMem0: propagate the corpus language into the vector store's BM25
        # encoder unless the user pinned vector_store.config.language explicitly.
        if (
            getattr(self.config, "language", "en") != "en"
            and self.config.vector_store.provider == "qdrant"
            and getattr(self.config.vector_store.config, "language", None) is None
        ):
            self.config.vector_store.config.language = self.config.language

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self.custom_instructions = self.config.custom_instructions
        # DeepMem0 v0.7: async lock serializing versioned update transitions into a
        # linear chain within the event loop (roadmap item #7).
        self._version_lock = asyncio.Lock()
        self._entity_store = None

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        if MEM0_TELEMETRY:
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            telemetry_config.collection_name = "mem0migrations"
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config.path = os.path.join(mem0_dir, provider_path)
                os.makedirs(telemetry_config.path, exist_ok=True)
            self._telemetry_vector_store = VectorStoreFactory.create(self.config.vector_store.provider, telemetry_config)

        if getattr(type(self.vector_store), "keyword_search", None) is VectorStoreBase.keyword_search:
            logger.warning(
                "The '%s' vector store does not support keyword search. "
                "Hybrid (BM25) scoring will be disabled and search will use "
                "semantic similarity only. To enable hybrid search, switch to a "
                "store with keyword_search support (e.g. qdrant, elasticsearch, pgvector).",
                self.config.vector_store.provider,
            )

        # DeepMem0 v0.7.2: finish any delete interrupted by a crash (no-op if none).
        try:
            self.reconcile_pending_deletes()
        except Exception as e:
            logger.warning(f"Delete-intent reconciliation skipped: {e}")

        capture_event("mem0.init", self, {"sync_type": "async"})

    @property
    def project(self):
        return _AsyncOSSProject()

    @property
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        if self._entity_store is None:
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            entity_collection = _entity_collection_name(self.config.vector_store.provider, self.collection_name)
            if hasattr(entity_config, 'collection_name'):
                entity_config.collection_name = entity_collection
            elif isinstance(entity_config, dict):
                entity_config['collection_name'] = entity_collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                if hasattr(entity_config, "client"):
                    entity_config.client = self.vector_store.client
                elif isinstance(entity_config, dict):
                    entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        return self._entity_store

    def _entidade_por_chave(self, chave, search_filters):
        return entidade_por_chave(self.entity_store, chave, search_filters)

    def _reconcilia_vinculo(self, entity_id, memory_id, tentativas=None):
        # `tentativas=None` -> usa o default do módulo (e portanto o env).
        # Declarar `= 4` aqui ANULAVA `MEM0_ENTITY_RECONCILE_ATTEMPTS` e fez
        # duas medições com janela maior não mudarem nada — o knob existia e não
        # chegava a lugar nenhum.
        return reconcilia_vinculo(
            self.entity_store, entity_id, memory_id,
            ENTITY_RECONCILE_ATTEMPTS if tentativas is None else tentativas)

    async def _upsert_entity_async(self, entity_text, entity_type, memory_id, filters):
        """Async variant of `_upsert_entity` — per-entity search-then-update-or-insert."""
        try:
            entity_embedding = await asyncio.to_thread(self.embedding_model.embed, entity_text, "add")
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}

            chave = normalize_entity_key(entity_text)

            # ⚠️ O gêmeo assíncrono ficou para trás quando o síncrono ganhou
            # identidade normalizada, e um gêmeo que diverge é pior que nenhum:
            # o mesmo corpus passa a ter duas regras de identidade dependendo de
            # qual caminho escreveu. A decisão agora é UMA — a função de módulo
            # `resolver_linha_de_entidade` —, então não há o que divergir:
            # escopo exato, multiplicidade e fail-closed valem nos dois.
            acao, alvo = await asyncio.to_thread(
                resolver_linha_de_entidade, self.entity_store, entity_text,
                chave, entity_embedding, search_filters)
            if acao == "skip":
                logger.warning("entidade %r: upsert pulado — %s",
                               entity_text, alvo)
                return

            if acao == "update":
                match = alvo
                payload = match.payload or {}
                raw_linked = payload.get("linked_memory_ids")
                linked_ids = normalize_linked_memory_ids(raw_linked)
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                # Same self-heal as the sync twin: normalizing changing the value
                # is itself a reason to write — and so is a row that still lacks
                # the identity key, because it can never be found by the exact
                # lookup and keeps having its case-variant duplicate born beside it.
                precisa_chave = payload.get("data_normalized") != chave
                if linked_ids != raw_linked or precisa_chave:
                    payload["linked_memory_ids"] = linked_ids
                    payload["data_normalized"] = chave
                    payload[link_key(memory_id)] = 1
                    await asyncio.to_thread(
                        self.entity_store.update,
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                entity_id = entity_point_id(search_filters, chave)
                entity_payload = {
                    "data": entity_text,
                    "data_normalized": chave,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    # chave por vínculo: sobrevive a `set_payload` concorrente
                    link_key(memory_id): 1,
                    **{k: v for k, v in search_filters.items()},
                }
                await asyncio.to_thread(
                    functools.partial(self.entity_store.insert, wait=True),
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
                await asyncio.to_thread(self._reconcilia_vinculo, entity_id, memory_id)
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}' (async): {e}")

    async def _bulk_clear_entity_store(self, filters):
        """Delete all entity records matching the given scope filters.

        Used by delete_all to avoid the race condition that occurs when
        concurrent _delete_memory coroutines each try to read-modify-write
        the same entity rows' linked_memory_ids lists.
        """
        if not _entity_cleanup_enabled():
            return True
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        ok = True
        try:
            rows, truncated = await asyncio.to_thread(
                _scan_entity_rows, self.entity_store, search_filters)
            if truncated:
                ok = False
            for row in rows:
                try:
                    await asyncio.to_thread(self.entity_store.delete, vector_id=row.id)
                except Exception as e:
                    ok = False
                    logger.debug(f"Bulk entity delete failed for id={row.id}: {e}")
        except Exception as e:
            ok = False
            logger.warning(f"Bulk entity store cleanup failed: {e}")
        return ok

    async def _remove_memory_from_entity_store(self, memory_id, filters):
        """Async variant of `Memory._remove_memory_from_entity_store`.

        Delegates to the same helper off-thread rather than mirroring the logic:
        the two copies had already drifted once (the sync one alone carried the
        `isinstance` guard fix), and entity cleanup is not hot enough to justify
        per-call awaits.
        """
        if not _entity_cleanup_enabled():
            return True
        return await asyncio.to_thread(
            unlink_memory_from_entity_rows, self.entity_store, memory_id, filters)

    async def _link_entities_for_memory(self, memory_id, text, filters):
        """Async variant of `Memory._link_entities_for_memory`."""
        try:
            entities = await asyncio.to_thread(
                extract_entities, text, self.config.language)
            if not entities:
                return
            search_filters = {k: v for k, v in filters.items()
                              if k in SCOPE_KEYS and v}
            # Mesma função de módulo do gêmeo síncrono: a orquestração do lote
            # não pode divergir mais do que as regras de identidade divergiram.
            if await asyncio.to_thread(
                    vincular_entidades_em_lote, self.entity_store,
                    self.embedding_model, memory_id, entities, search_filters):
                return
            seen = set()
            for entity_type, entity_text in entities:
                key = normalize_entity_key(entity_text)
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    await self._upsert_entity_async(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}' (async): {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id} (async): {e}")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        temporal_context: str = "conversation",
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            timestamp (Any, optional): Platform-only temporal parameter. Not supported in OSS.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None.
                                         Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """
        if timestamp is not None:
            raise ValueError(await get_temporal_feature_error_message_async("async", "add", "timestamp"))
        if temporal_context not in ("conversation", "document"):
            # fail-closed (espelha o sync): typo não pode virar modo conversacional mudo
            raise ValueError(
                f"temporal_context inválido: {temporal_context!r} (use 'conversation' ou 'document')"
            )

        temporal_usage_notice = detect_temporal_usage_from_metadata(metadata)
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            scale_threshold_notice = await asyncio.to_thread(detect_scale_threshold_from_add_result, self, results)
            if temporal_usage_notice:
                await display_temporal_usage_notice_async(self, "async", "add", *temporal_usage_notice)
            elif scale_threshold_notice:
                await display_scale_threshold_notice_async(self, "async", "add", *scale_threshold_notice)
            else:
                await display_first_run_notice_async(self, "async", "add")
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        vector_store_result = await self._add_to_vector_store(
            messages, processed_metadata, effective_filters, infer,
            prompt=prompt, temporal_context=temporal_context,
        )
        scale_threshold_notice = await asyncio.to_thread(detect_scale_threshold_from_add_result, self, vector_store_result)
        if temporal_usage_notice:
            await display_temporal_usage_notice_async(self, "async", "add", *temporal_usage_notice)
        elif scale_threshold_notice:
            await display_scale_threshold_notice_async(self, "async", "add", *scale_threshold_notice)
        else:
            await display_first_run_notice_async(self, "async", "add")
        return {"results": vector_store_result}

    async def _add_to_vector_store(
        self,
        messages: list,
        metadata: dict,
        effective_filters: dict,
        infer: bool,
        prompt: Optional[str] = None,
        temporal_context: str = "conversation",
    ):
        if not infer:
            returned_memories = []
            # Espelho do gêmeo síncrono: o embed sai do laço. Mesma função de
            # módulo nos dois, para a regra de descarte e a semântica de falha
            # não divergirem — que é como os gêmeos de entidade divergiram.
            validas = _mensagens_validas_para_add(messages)
            embed_map = await asyncio.to_thread(
                _embed_map_de, self.embedding_model,
                [m["content"] for m in validas])

            for message_dict in validas:
                msg_content = message_dict["content"]
                if msg_content not in embed_map:
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                mem_id = await self._create_memory(msg_content, embed_map, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        # === V3 PHASED BATCH PIPELINE (async) ===

        # Phase 0: Context gathering
        session_scope = _build_session_scope(effective_filters)
        # DeepMem0: documents don't touch the conversational message history (read
        # or write) — else chunks bleed into later adds via last_k (proven). See sync.
        skip_doc_history = temporal_context == "document"
        last_messages = [] if skip_doc_history else await asyncio.to_thread(self.db.get_last_messages, session_scope, 10)
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        search_filters = {k: v for k, v in effective_filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        query_embedding = await asyncio.to_thread(self.embedding_model.embed, parsed_messages, "search")
        existing_results = await asyncio.to_thread(
            self.vector_store.search,
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        existing_memories = []
        uuid_mapping = {}
        for idx, mem in enumerate(existing_results):
            uuid_mapping[str(idx)] = mem.id
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction (single call)
        is_agent_scoped = bool(effective_filters.get("agent_id")) and not effective_filters.get("user_id")
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if is_agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX
        temp = _temporality_config(self.config)
        if temp is not None:
            # DeepMem0 v0.3: same call also detects supersession (+ event_date).
            system_prompt += build_temporality_suffix(include_event_date=temp.extract_event_date)
        if temporal_context == "document":
            # DeepMem0: a document keeps its OWN dates; disable Observation-Date
            # resolution so a year-less date is never filled with the current year.
            system_prompt += DOCUMENT_TEMPORAL_OVERRIDE

        # DeepMem0 v0.15: per-fact speaker. `rotulos` comes from the messages that
        # actually REACH the prompt (parse_vision_messages already ran, so `name`
        # survived the multimodal branches) — a label the model never saw rendered
        # must never be a value the validator accepts.
        rotulos_locutor, locutor_uniforme = (
            locutores_das_mensagens(messages) if speaker_attribution_enabled()
            else (set(), False)
        )
        if precisa_de_atribuicao_por_llm(rotulos_locutor, locutor_uniforme):
            system_prompt += build_speaker_attribution_suffix(rotulos_locutor)

        custom_instr = prompt or self.custom_instructions

        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
            # DeepMem0: extract facts in the input's language for non-English
            # corpora (upstream ships this flag but never sets it).
            use_input_language=(getattr(self.config, "language", "en") != "en"),
        )

        try:
            response = await asyncio.to_thread(
                self.llm.generate_response,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"LLM extraction failed (async): {e}")
            return []

        # Parse response
        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                extracted_memories = []
            else:
                try:
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        except Exception as e:
            logger.error(f"Error parsing extraction response (async): {e}")
            extracted_memories = []

        if not extracted_memories:
            if not skip_doc_history:
                await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            return []

        # Phase 3: Batch embed all extracted memory texts
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        try:
            mem_embeddings_list = await asyncio.to_thread(self.embedding_model.embed_batch, mem_texts, "add")
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        except Exception:
            embed_map = {}
            for text in mem_texts:
                try:
                    embed_map[text] = await asyncio.to_thread(self.embedding_model.embed, text, "add")
                except Exception as e:
                    logger.warning(f"Failed to embed memory text (async): {e}")

        # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
        existing_by_hash = {}
        for mem in existing_results:
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            if h:
                existing_by_hash[h] = mem

        dyn = _dynamics_config(self.config)
        records = []
        pending_supersessions = []  # (new_memory_id, new_text, [old_ids], new_created_at) — applied after persist
        pending_similarity = []  # DeepMem0 v0.8 (T1S): (target_id, score, new_memory_id) — applied after persist
        pending_similarity_targets = set()  # one reinforcement per target per add
        seen_hashes = set()
        for mem in extracted_memories:
            text = mem.get("text")
            if not text or text not in embed_map:
                continue

            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_by_hash or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match, async): {text[:50]}")
                # DeepMem0 v0.2 (T1): re-encounter reinforces the existing memory.
                # (An identical fact replaces nothing — supersedes mark ignored.)
                # v0.9: supersedido é deduplicado mas NÃO reforçado (ver sync).
                existing = existing_by_hash.get(mem_hash)
                if (dyn is not None and existing is not None
                        and not (existing.payload or {}).get(FIELD_SUPERSEDED_BY)):
                    await asyncio.to_thread(
                        _reinforce_memory,
                        self.vector_store, dyn, existing.id, existing.payload,
                        trigger=TRIGGER_DEDUP,
                    )
                continue
            seen_hashes.add(mem_hash)

            text_lemmatized = lemmatize_for_bm25(text, language=self.config.language)

            memory_id = str(uuid.uuid4())
            mem_metadata = deepcopy(metadata)
            mem_metadata["data"] = text
            mem_metadata["text_lemmatized"] = text_lemmatized
            mem_metadata["hash"] = mem_hash
            if "created_at" not in mem_metadata:
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            if mem.get("attributed_to"):
                mem_metadata["attributed_to"] = mem["attributed_to"]
            # DeepMem0 v0.15: WHO SPOKE. Uniform conversation resolves in code;
            # otherwise the model's proposal only survives if it is a `str` that
            # canonicalizes into the closed set enumerated in the prompt.
            # Anything else omits the field — which is exactly today's behaviour,
            # and the only failure direction that cannot corrupt attribution.
            locutor = resolver_locutor_do_fato(
                mem.get("actor_id"), rotulos_locutor, locutor_uniforme)
            if locutor:
                mem_metadata["actor_id"] = locutor
            # DeepMem0 v0.2 (option B): creation stays neutral until the first reinforcement.
            if temp is not None:
                # DeepMem0 v0.3: resolve LLM-referenced indices via uuid_mapping.
                supersedes_ids = parse_supersedes_ids(mem.get("supersedes"), uuid_mapping)
                if supersedes_ids:
                    mem_metadata[FIELD_SUPERSEDES] = supersedes_ids
                    pending_supersessions.append((memory_id, text, supersedes_ids, mem_metadata["created_at"]))
                if temp.extract_event_date:
                    event_date = parse_event_date(mem.get("event_date"))
                    if temporal_context == "document":
                        # medido: o extrator pequeno escreve a data no TEXTO do
                        # fato mas omite o campo (0/185); e pode emitir uma data
                        # VÁLIDA-mas-ERRADA (ex.: ano corrente). Em modo documento
                        # a data ESCRITA vence: se o texto tem exatamente UMA data
                        # completa, ela é a verdade (cross-validação do parecer).
                        text_date = infer_event_date_from_text(text)
                        if text_date and event_date and event_date != text_date:
                            logger.warning(
                                f"event_date do LLM ({event_date}) contradiz a data do texto "
                                f"({text_date}) em modo documento — usando a do texto"
                            )
                            event_date = text_date
                        elif not event_date:
                            event_date = text_date
                    if event_date:
                        mem_metadata["event_date"] = event_date

            # DeepMem0 v0.8 (T1S): semantic re-encounter — see the sync twin.
            if dyn is not None and not mem.get("supersedes"):
                _sim = await asyncio.to_thread(
                    _similar_reinforcement_target,
                    self.vector_store, text, embed_map[text], search_filters, dyn,
                )
                if _sim is not None and _sim[0] not in pending_similarity_targets:
                    pending_similarity_targets.add(_sim[0])
                    pending_similarity.append((_sim[0], _sim[1], memory_id, mem_hash))

            records.append((memory_id, text, embed_map[text], mem_metadata))

        if not records:
            if not skip_doc_history:
                await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            return []

        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        all_ids = [r[0] for r in records]
        all_payloads = [r[3] for r in records]

        failed_persist = set()
        try:
            await asyncio.to_thread(
                self.vector_store.insert,
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
            )
        except Exception:
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                try:
                    await asyncio.to_thread(self.vector_store.insert, vectors=[vec], ids=[mid], payloads=[pay])
                except Exception as e:
                    logger.error(f"Failed to insert memory {mid} (async): {e}")
                    failed_persist.add(mid)

        # DeepMem0 v0.3: mark superseded memories only AFTER the new facts landed.
        superseded_events = []
        if pending_supersessions:
            try:
                for new_id, new_text, old_ids, new_created in pending_supersessions:
                    marked = await asyncio.to_thread(
                        _mark_superseded, self.vector_store, self.db, new_id, new_text, old_ids,
                        new_created_at=new_created,
                    )
                    superseded_events.extend(marked)
            except Exception as e:
                logger.warning(f"Supersession marking pass failed (async): {e}")

        # DeepMem0 v0.8 (T1S): deferred application — see the sync twin.
        if pending_similarity:
            await asyncio.to_thread(
                _apply_similar_reinforcements,
                self.vector_store, dyn,
                [p for p in pending_similarity if p[2] not in failed_persist],
            )

        # Batch history
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            for r in records
        ]
        try:
            await asyncio.to_thread(self.db.batch_add_history, history_records)
        except Exception:
            for hr in history_records:
                try:
                    await asyncio.to_thread(
                        self.db.add_history, hr["memory_id"], None, hr["new_memory"], "ADD",
                        created_at=hr.get("created_at")
                    )
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']} (async): {e}")

        # Phase 7: Batch entity linking
        try:
            all_texts = [r[1] for r in records]
            all_entities = await asyncio.to_thread(
                extract_entities_batch, all_texts, 32, self.config.language)

            # 7a: Global dedup
            global_entities = {}
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text in entities:
                    key = normalize_entity_key(entity_text)
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            if global_entities:
                ordered_keys = list(global_entities.keys())
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Batch embed entities
                try:
                    entity_embeddings = await asyncio.to_thread(self.embedding_model.embed_batch, entity_texts, "add")
                except Exception:
                    entity_embeddings = []
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(await asyncio.to_thread(self.embedding_model.embed, t, "add"))
                        except Exception:
                            entity_embeddings.append(None)

                if len(entity_embeddings) != len(ordered_keys):
                    logger.warning(
                        "embed_batch returned %d vectors for %d entity texts — "
                        "padding/truncating to avoid dropping entity links",
                        len(entity_embeddings),
                        len(ordered_keys),
                    )
                    entity_embeddings = list(entity_embeddings[: len(ordered_keys)])
                    entity_embeddings += [None] * (len(ordered_keys) - len(entity_embeddings))

                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                if valid:
                    valid_indices, valid_keys = zip(*valid)
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]

                    # 7c: lookup EXATO por chave, em lote, e SÓ ENTÃO a sonda —
                    # espelho do gêmeo síncrono. Rodava só com a sonda vetorial,
                    # a regra probabilística que o cutover de 30/07 substituiu.
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    try:
                        por_chave, _amb = await asyncio.to_thread(
                            entidades_por_chaves, self.entity_store,
                            valid_keys, search_filters)
                    except Exception as exc:
                        # FAIL-CLOSED: "trata tudo como novo" converteria falha
                        # de infraestrutura em escrita destrutiva no id
                        # determinístico.
                        logger.warning(
                            "lookup exato de entidade falhou (%s) — Fase 7 "
                            "async abortada", exc)
                        raise

                    if _amb:
                        logger.warning(
                            "Fase 7 async: %d chave(s) ambígua(s) puladas — %s",
                            len(_amb), sorted(_amb))
                    processar = [j for j, k in enumerate(valid_keys)
                                 if k not in _amb]

                    faltantes = [j for j in processar
                                 if valid_keys[j] not in por_chave]
                    existing_matches = [[] for _ in valid_keys]
                    if faltantes:
                        sondados = await asyncio.to_thread(
                            self.entity_store.search_batch,
                            queries=[valid_texts[j] for j in faltantes],
                            vectors_list=[valid_vectors[j] for j in faltantes],
                            top_k=1,
                            filters=search_filters,
                        )
                        if (not isinstance(sondados, list)
                                or len(sondados) != len(faltantes)
                                or not all(isinstance(x, list) for x in sondados)):
                            raise RuntimeError(
                                f"search_batch devolveu {type(sondados).__name__} "
                                f"com {len(sondados) if isinstance(sondados, list) else '?'} "
                                f"entradas para {len(faltantes)} consultas")
                        for pos, j in enumerate(faltantes):
                            existing_matches[j] = sondados[pos]

                    # 7d: Separate into inserts vs updates
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    for j in processar:
                        key = valid_keys[j]
                        entity_type, entity_text, memory_ids = global_entities[key]
                        matches = existing_matches[j] if j < len(existing_matches) else []
                        matches = [m for m in matches
                                   if escopo_exato(getattr(m, "payload", None) or {},
                                                   search_filters)]
                        exata = por_chave.get(key)
                        if exata is not None:
                            matches = [exata]

                        if matches and (exata is not None or matches[0].score >= 0.95):
                            match = matches[0]
                            payload = match.payload or {}
                            # normalize_* is load-bearing here: a str payload fed
                            # straight to set() iterates CHARACTER BY CHARACTER,
                            # which is how real entity rows lost their links.
                            linked = set(normalize_linked_memory_ids(payload.get("linked_memory_ids")))
                            linked |= memory_ids
                            payload["linked_memory_ids"] = sorted(linked)
                            payload["data_normalized"] = key
                            payload.update({link_key(m): 1 for m in memory_ids})
                            try:
                                await asyncio.to_thread(
                                    self.entity_store.update,
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}' (async): {e}")
                        else:
                            to_insert_vectors.append(valid_vectors[j])
                            # ⚠️ ESTE é o escritor que roda em todo `add` com
                            # infer=True — a Fase 7 em lote. Ele ficou com
                            # `uuid4()` e sem `data_normalized` enquanto
                            # `_upsert_entity` ganhava identidade determinística,
                            # e o resultado seria pior que não ter consertado
                            # nada: o corpus passa a ter DUAS regras de
                            # identidade conforme o caminho que escreveu.
                            to_insert_ids.append(
                                entity_point_id(search_filters, key))
                            to_insert_payloads.append({
                                "data": entity_text,
                                "data_normalized": key,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **{link_key(m): 1 for m in memory_ids},
                                **search_filters,
                            })

                    # 7e: Batch insert new entities
                    if to_insert_vectors:
                        try:
                            await asyncio.to_thread(
                                # leitura-após-escrita, igual ao gêmeo síncrono:
                                # o default do Qdrant NÃO espera, e escrita
                                # confirmada mas invisível é o que fez o
                                # escritor seguinte recriar a linha e apagar
                                # vínculos alheios (30/07).
                                functools.partial(self.entity_store.insert,
                                                  wait=True),
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        except Exception as e:
                            logger.warning(f"Batch entity insert failed (async): {e}")
        except Exception as e:
            logger.warning(f"Batch entity linking failed (async): {e}")

        # Phase 8: Save messages + return
        if not skip_doc_history:
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)

        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            for r in records
        ]
        # DeepMem0 v0.3: surface supersessions to the caller (additive entries).
        # v0.4: pairs may point either way — a queued fact that arrived late is
        # born superseded by the fresher existing one (superseded_id == new id).
        returned_memories.extend(
            {"id": superseded_id, "event": "SUPERSEDED", "superseded_by": superseding_id}
            for superseded_id, superseding_id in superseded_events
        )

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        return returned_memories

    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "async"})
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if not memory:
            await display_first_run_notice_async(self, "async", "get")
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "attributed_to",
            "role",
            "memory_scope",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        await display_first_run_notice_async(self, "async", "get")
        return result_item

    async def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if top_k is invalid.
        """
        # Reject top-level entity params - must use filters instead
        _scope_kwargs = _extract_top_level_entity_params(kwargs)
        if _scope_kwargs:
            filters = {**_scope_kwargs, **(filters or {})}

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )
        _canonizar_filtro_de_locutor(effective_filters)

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"}
        )

        all_memories_result = await self._get_all_from_vector_store(effective_filters, limit)

        if scale_threshold_notice:
            await display_scale_threshold_notice_async(self, "async", "get_all", *scale_threshold_notice)
        else:
            await display_first_run_notice_async(self, "async", "get_all")
        return {"results": all_memories_result}

    async def _get_all_from_vector_store(self, filters, limit):
        memories_result = await asyncio.to_thread(self.vector_store.list, filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "attributed_to",
            "role",
            "memory_scope",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    async def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: Optional[bool] = None,
        explain: bool = False,
        reference_date: Optional[Any] = None,
        min_importance: Optional[float] = None,
        domain: Optional[str] = None,
        memory_type: Optional[str] = None,
        sort_by_importance: bool = False,
        as_of: Optional[str] = None,
        event_from: Optional[str] = None,
        event_to: Optional[str] = None,
        reinforce: Optional[bool] = None,
        search_id: Optional[str] = None,
        historical: bool = False,
        **kwargs,
    ):
        """
        Searches for memories based on a query.

        Args:
            query (str): Query to search for.
            top_k (int, optional): Maximum number of results to return. Defaults to 20.
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}

                Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            threshold (float, optional): Minimum score for a memory to be included. Defaults to 0.1.
            rerank (bool, optional): Whether to rerank results. Defaults to False.
            explain (bool, optional): Whether to include score_details for each result. Defaults to False.
            reference_date (Any, optional): Platform-only temporal parameter. Not supported in OSS.
            as_of (str, optional): DeepMem0 v0.3 RECORD-time anchor (ISO date/datetime) — restrict
                results to memories that already existed then (filters on created_at) and restore
                the world as it was. Answers "what did I know on X". DeepMem0 runtime only.
            event_from (str, optional): DeepMem0 v0.6 EVENT-time window start (inclusive). Full or
                partial ISO date — "2023" = whole year, "2023-10" = whole month, "2023-10-17" = day.
                Filters on event_date (WHEN the fact happened, distinct from as_of's record-time).
                Memories without an event_date are EXCLUDED while the window is active. One side
                alone = open interval. DeepMem0 runtime only.
            event_to (str, optional): DeepMem0 v0.6 EVENT-time window end (inclusive), same partial
                expansion. When neither event_from/event_to is given, a single date named in the
                query auto-anchors ranking (event_ranking) without filtering anything out.

        Returns:
            dict: A dictionary containing the search results under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
                  DeepMem0 also echoes "as_of" (record-time anchor), "event_anchor" ({"from","to"}
                  auto-detected from the query) OR "event_filter" ({"from","to"} explicit window;
                  mutually exclusive with event_anchor) when those apply.

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if threshold/top_k values are invalid.
        """
        if reference_date is not None:
            raise ValueError(
                await get_temporal_feature_error_message_async("async", "search", "reference_date")
            )

        # DeepMem0 v0.3: as-of anchor — "what did I know / what held on that date".
        as_of_iso, as_of_dt = (None, None)
        if as_of is not None and _temporality_config(self.config) is not None:
            as_of_iso, as_of_dt = parse_as_of(as_of)

        # DeepMem0 v0.10: recordação histórica — decisão derivada UMA vez e
        # lida por todos (gate de reforço, fusão, adjuster); recordar nunca
        # reforça, mesmo com reinforce=True explícito. Config só é tocada
        # quando o modo foi PEDIDO (instâncias nuas em testes upstream não
        # têm config; mesmo padrão do as_of/v0.6).
        if historical:
            _validate_historical(historical, as_of, _temporality_config(self.config))
            reinforce = False

        # DeepMem0 v0.6: event-time window — validate caller bounds fail-fast
        # (mirrors as_of) EVEN when temporality is off, so a malformed date is
        # never a config-dependent silent no-op. Application is gated below.
        event_from_iso, event_to_iso = (None, None)
        if event_from is not None or event_to is not None:
            event_from_iso, event_to_iso = expand_event_window(event_from, event_to)
        event_anchor = None

        # Reject top-level entity params - must use filters instead
        _scope_kwargs = _extract_top_level_entity_params(kwargs)
        if _scope_kwargs:
            filters = {**_scope_kwargs, **(filters or {})}

        # Validate search parameters (before applying defaults)
        _validate_search_params(threshold=threshold, top_k=top_k)
        query = _validate_and_trim_search_query(query)
        temporal_usage_notice = detect_temporal_usage_from_search(query, filters)

        # Validate and trim entity IDs in filters
        effective_filters = filters.copy() if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )
        _canonizar_filtro_de_locutor(effective_filters)

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        # Apply enhanced metadata filtering if advanced operators are detected
        if self._has_advanced_operators(effective_filters):
            processed_filters = self._process_metadata_filters(effective_filters)
            # Remove logical/operator keys that have been reprocessed
            for logical_key in ("AND", "OR", "NOT"):
                effective_filters.pop(logical_key, None)
            for fk in list(effective_filters.keys()):
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    effective_filters.pop(fk, None)
            effective_filters.update(processed_filters)

        # DeepMem0 v0.3: record-time anchor — only memories that already existed
        # at the as_of instant participate (applies to the dense AND keyword
        # legs, before the over-fetch; Qdrant auto-detects a DatetimeRange for
        # ISO values). A caller-provided created_at bound is tightened, never
        # loosened.
        if as_of_iso is not None:
            existing_created = effective_filters.get("created_at")
            if isinstance(existing_created, dict):
                current_lte = existing_created.get("lte")
                existing_created["lte"] = (
                    min(current_lte, as_of_iso) if isinstance(current_lte, str) else as_of_iso
                )
            else:
                effective_filters["created_at"] = {"lte": as_of_iso}

        # DeepMem0 v0.6: auto-detect a single event-time expression in the query
        # for ranking — suppressed when the caller passed an explicit window (they
        # already stated intent). Gated by event_ranking; the fusion term is
        # separately gated by event_ranking_weight > 0 downstream. Placed after
        # filter validation so self.config is only touched once the request is
        # well-formed (mirrors as_of's post-validation config access).
        _search_config = getattr(self, "config", None)
        if event_from_iso is None and event_to_iso is None and _search_config is not None:
            _ev_cfg = _temporality_config(_search_config)
            if _ev_cfg is not None and getattr(_ev_cfg, "event_ranking", False):
                event_anchor = infer_event_anchor_from_query(query)

        # DeepMem0 v0.6: explicit event-time window filter (event_date range).
        # Record-time as_of and event-time window compose (AND'ed in the store).
        # Applied only when temporality is enabled (mirror as_of). A FRESH nested
        # dict is written so the caller's filter object is never mutated; an
        # existing event_date bound is tightened, never loosened. Undated memories
        # never match a range on a missing field, so they drop out of the window.
        if (event_from_iso is not None or event_to_iso is not None) and _temporality_config(self.config) is not None:
            bound = {}
            if event_from_iso is not None:
                bound["gte"] = event_from_iso
            if event_to_iso is not None:
                bound["lte"] = event_to_iso
            existing_event = effective_filters.get(FIELD_EVENT_DATE)
            if isinstance(existing_event, dict):
                merged = dict(existing_event)
                if "gte" in bound:
                    cur = merged.get("gte")
                    merged["gte"] = max(cur, bound["gte"]) if isinstance(cur, str) else bound["gte"]
                if "lte" in bound:
                    cur = merged.get("lte")
                    merged["lte"] = min(cur, bound["lte"]) if isinstance(cur, str) else bound["lte"]
                effective_filters[FIELD_EVENT_DATE] = merged
            else:
                effective_filters[FIELD_EVENT_DATE] = bound

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "async",
                "threshold": threshold,
                "explain": explain,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        # DeepMem0: a configured reranker is ON by default (upstream defaulted
        # rerank=False, so a configured reranker silently never ran unless every
        # caller opted in), and it sees an OVER-FETCHED candidate pool — reranking
        # only the fused top-k cannot recover targets that the additive fusion
        # buried under keyword-boosted competitors (measured on a PT corpus:
        # hit@1 0.857 -> 0.886, one extra recall, with pool=20).
        if rerank is None:
            rerank = self.reranker is not None
        fetch_limit = limit
        if rerank and self.reranker:
            fetch_limit = max(2 * limit, getattr(self.config, "rerank_pool", 20))

        search_start = time.perf_counter()
        original_memories = await self._search_vector_store(
            query, effective_filters, fetch_limit, threshold, explain=explain, as_of_dt=as_of_dt,
            dense_anchors=(getattr(self.config, "rerank_dense_anchors", 5)
                           if (rerank and self.reranker) else 0),
            event_anchor=event_anchor,
            historical=historical,
        )
        search_elapsed_seconds = time.perf_counter() - search_start

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            try:
                # Run reranking in thread pool to avoid blocking async loop
                reranked_memories = await asyncio.to_thread(
                    self.reranker.rerank, query, original_memories, fetch_limit
                )
                original_memories = reranked_memories
                # DeepMem0 v0.2/v0.3: activation + superseded penalty, single sort.
                dyn = _dynamics_config(self.config)
                temp = _temporality_config(self.config)
                if dyn is not None or temp is not None:
                    original_memories = _apply_post_rerank_adjustments(
                        original_memories, dyn=dyn, temp=temp, as_of=as_of_dt,
                        event_anchor=event_anchor, historical=historical,
                    )
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")
        # DeepMem0: cut the over-fetched pool back to the requested top_k.
        original_memories = original_memories[:limit]
        original_memories = _apply_metadata_post_filters(
            original_memories,
            min_importance=min_importance,
            domain=domain,
            memory_type=memory_type,
            sort_by_importance=sort_by_importance,
        )

        # DeepMem0 v0.2 (T3, opt-in): reinforce returned memories off the hot path.
        dyn = _dynamics_config(self.config)
        if _t3_enabled(dyn, reinforce) and original_memories:
            # exposed_at é AGORA — o instante em que o caller viu estas memórias.
            # O worker pode rodar muito depois; usar o relógio dele deslocaria a
            # linha do tempo e as duas janelas.
            _reinforce_hits_in_background(
                self.vector_store, dyn,
                _t3_targets(dyn, original_memories,
                            search_id=search_id or uuid.uuid4().hex[:16],
                            exposed_at=_dynamics_utcnow()),
            )

        if temporal_usage_notice:
            await display_temporal_usage_notice_async(self, "async", "search", *temporal_usage_notice)
        elif scale_threshold_notice:
            await display_scale_threshold_notice_async(self, "async", "search", *scale_threshold_notice)
        elif search_elapsed_seconds > PERFORMANCE_SLOW_QUERY_THRESHOLD_SECONDS:
            await display_performance_slow_query_notice_async(
                self,
                "async",
                "search",
                search_elapsed_seconds,
                top_k,
                len(original_memories),
            )
        else:
            await display_first_run_notice_async(self, "async", "search")
        response = {"results": original_memories}
        if as_of_iso is not None:
            response["as_of"] = as_of_iso
        # DeepMem0 v0.10: no modo recordação, avisa quais resultados têm
        # sucessor EXPLÍCITO conhecido ("há fatos mais atuais") + echo do modo.
        if historical:
            _n_newer = _annotate_known_successors(original_memories)
            response["historical_recall"] = {
                "as_of": as_of_iso, "results_with_newer_version": _n_newer,
            }
        # DeepMem0 v0.6: echo the auto-detected ranking anchor OR the explicit
        # filter window (mutually exclusive — an explicit window suppresses
        # auto-detection). event_anchor is echoed whenever an anchor was found,
        # independent of whether any candidate matched it.
        if event_anchor is not None:
            response["event_anchor"] = {"from": event_anchor[0], "to": event_anchor[1]}
        elif event_from_iso is not None or event_to_iso is not None:
            response["event_filter"] = {"from": event_from_iso, "to": event_to_iso}
        return response

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        def merge_filters(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Merge source into target, deep-merging nested operator dicts for the same key."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    target[key].update(value)
                else:
                    target[key] = value

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                merge_filters(processed_filters, process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False

        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    async def _search_vector_store(self, query, filters, limit, threshold=0.1, explain=False, as_of_dt=None, dense_anchors=0, event_anchor=None, historical=False):
        if threshold is None:
            threshold = 0.1

        # Step 1: Preprocess query (CPU-bound)
        query_lemmatized = await asyncio.to_thread(lemmatize_for_bm25, query, self.config.language)
        query_entities = await asyncio.to_thread(
            extract_entities, query, self.config.language)

        # Step 2: Embed query
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")

        # Step 3: Semantic search (over-fetch)
        internal_limit = max(limit * 4, 60)
        semantic_results = await asyncio.to_thread(
            self.vector_store.search, query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4: Keyword search (if store supports it)
        keyword_results = await asyncio.to_thread(
            self.vector_store.keyword_search, query=query_lemmatized, top_k=internal_limit, filters=filters
        )

        # Step 5: Compute BM25 scores
        bm25_scores = {}
        if keyword_results is not None:
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                if raw_score and raw_score > 0:
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6: Compute entity boosts
        entity_boosts = {}
        if query_entities:
            entity_boosts = await self._compute_entity_boosts_async(query_entities, filters)

        # Step 7: Build candidate set from semantic results
        candidates = []
        for mem in semantic_results:
            mem_id = str(mem.id)
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": mem.payload if hasattr(mem, 'payload') else {},
            })

        # Step 7b (DeepMem0 v0.2): lazy ACT-R activation over the candidate pool.
        activation_boosts = {}
        dyn = _dynamics_config(self.config)
        # v0.10: recordação histórica não usa peso de uso — ativação inerte.
        if dyn is not None and dyn.weight > 0 and not historical:
            now = _dynamics_utcnow()
            for cand in candidates:
                # v0.9 MASK: a superseded record gets NO activation — with the
                # timeline COPIED to its successor, boosting both would let the
                # family double-dip (penalty − boost partially cancel on the old
                # one). Same predicate as the penalty, so as_of time travel keeps
                # historical activation for the then-current version.
                if superseded_penalty_applies(cand["payload"], as_of=as_of_dt):
                    continue
                boost = boost_from_payload(cand["payload"], now=now, decay=dyn.decay)
                if boost > 0:
                    activation_boosts[cand["id"]] = boost

        # Step 7c (DeepMem0 v0.3): superseded facts are demoted, never excluded.
        # Anchor-aware: with an as_of, a memory superseded only AFTER the anchor
        # was still the current fact then, so its penalty is waived.
        superseded_penalties = {}
        temp = _temporality_config(self.config)
        if temp is not None and temp.superseded_penalty > 0:
            for cand in candidates:
                if superseded_penalty_applies(cand["payload"], as_of=as_of_dt):
                    superseded_penalties[cand["id"]] = temp.superseded_penalty

        # Step 7d (DeepMem0 v0.6): event-time proximity boosts over the candidate
        # pool when the query named a date. FUSION-stage only, gated by
        # event_ranking_weight > 0 (weight=0 => tie-break-only, no divisor growth).
        # Memories without an event_date stay neutral (no key in the dict).
        event_boosts = {}
        if (temp is not None and getattr(temp, "event_ranking", False)
                and temp.event_ranking_weight > 0 and event_anchor):
            event_window_days = getattr(temp, "event_window_days", 30)
            for cand in candidates:
                prox = event_proximity(event_anchor, (cand["payload"] or {}).get(FIELD_EVENT_DATE), event_window_days)
                if prox > 0:
                    event_boosts[cand["id"]] = prox

        # Step 8: Score and rank
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=limit,
            explain=explain,
            activation_boosts=activation_boosts,
            activation_weight=dyn.weight if dyn is not None else 0.0,
            penalties=superseded_penalties or None,
            event_boosts=event_boosts or None,
            event_weight=temp.event_ranking_weight if temp is not None else 0.0,
        )

        # DeepMem0: DENSE ANCHORS — a fusão corta o pool por score FUNDIDO, então
        # um alvo denso-forte enterrado por boosts ruidosos (entity/activation de
        # competidores) sai do pool ANTES do reranker e o resgate-por-rerank da F1
        # nunca acontece (medido: alvo denso rank 1-2, fundido rank 21-40, sumia
        # do top-10 quando o corpus cresceu 620->984). Garantia: o denso-top-N
        # sempre entra no pool do reranker — só ADICIONA candidatos; o
        # cross-encoder decide. Ativo apenas no caminho com rerank.
        if dense_anchors > 0:
            seen_ids = {r["id"] for r in scored_results}
            for cand in candidates[:dense_anchors]:
                if cand["id"] not in seen_ids:
                    scored_results.append(cand)
                    seen_ids.add(cand["id"])

        # Step 9: Format results
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "attributed_to",
            "role",
            "memory_scope",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        original_memories = []
        for scored in scored_results:
            payload = scored.get("payload") or {}
            if not payload.get("data"):
                continue

            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]

            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                if not memory_item_dict.get("metadata"):
                    memory_item_dict["metadata"] = {}
                memory_item_dict["metadata"].update(additional_metadata)
            if explain and "score_details" in scored:
                memory_item_dict["score_details"] = scored["score_details"]

            original_memories.append(memory_item_dict)

        return original_memories

    async def _compute_entity_boosts_async(self, query_entities, filters):
        """Async version of entity boost computation."""
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = normalize_entity_key(entity_text)
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        try:
            entity_texts = [text for _, text in deduped]
            embeddings = await asyncio.to_thread(self.embedding_model.embed_batch, entity_texts, "search")

            if len(embeddings) != len(entity_texts):
                logger.warning(
                    "embed_batch returned %d vectors for %d texts — skipping entity boost",
                    len(embeddings),
                    len(entity_texts),
                )
                return memory_boosts

            sem = asyncio.Semaphore(4)

            async def _search_entity(entity_text, embedding):
                async with sem:
                    return await asyncio.to_thread(
                        self.entity_store.search,
                        query=entity_text,
                        vectors=embedding,
                        top_k=500,
                        filters=search_filters,
                    )

            results = await asyncio.gather(
                *(_search_entity(text, emb) for text, emb in zip(entity_texts, embeddings)),
                return_exceptions=True,
            )

            for matches in results:
                if isinstance(matches, BaseException):
                    logger.warning("Entity boost search failed for one entity: %s", matches)
                    continue

                for match in matches:
                    similarity = match.score if hasattr(match, 'score') else 0.0
                    if similarity < 0.5:
                        continue

                    payload = match.payload if hasattr(match, 'payload') else {}
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    if not isinstance(linked_memory_ids, list):
                        # Fail closed, but audibly — see the sync twin.
                        logger.warning(
                            "Entity %s has a malformed linked_memory_ids (%s); "
                            "its boost is being skipped. Run "
                            "check_corpus.py / repair_entity_links.py.",
                            getattr(match, "id", "?"), type(linked_memory_ids).__name__)
                        continue

                    num_linked = max(len(linked_memory_ids), 1)
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    for memory_id in linked_memory_ids:
                        if memory_id:
                            memory_key = str(memory_id)
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    async def update(self, memory_id, data, metadata: Optional[Dict[str, Any]] = None):
        """
        Update a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> await m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "async"})

        if metadata:  # strip reserved lineage fields (anti-injection, incl. legacy path)
            metadata = {k: v for k, v in metadata.items() if k not in RESERVED_LINEAGE_FIELDS}
        embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")
        existing_embeddings = {data: embeddings}

        # DeepMem0 v0.7.1: old_id comes from inside the transition lock (critic #4).
        returned = await self._update_memory(memory_id, data, existing_embeddings, metadata)
        if isinstance(returned, tuple):
            current_id, old_id = returned
        else:
            current_id = old_id = returned
        await display_first_run_notice_async(self, "async", "update")
        return {"message": "Memory updated successfully!", "id": current_id, "old_id": old_id}

    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "async"})

        # DeepMem0 v0.7.1: whole UPDATE-version chain, transactional (preflight via
        # _collect_chain which raises fail-closed; historical-first, head-last; partial
        # failure returns remaining ids). Semantic supersedence siblings are untouched.
        temp = _temporality_config(self.config)
        if temp is not None and getattr(temp, "version_on_update", False):
            async with self._version_lock:
                chain = await asyncio.to_thread(
                    _collect_chain, lambda vid: self.vector_store.get(vector_id=vid), memory_id
                )
                if not chain:
                    raise ValueError(f"Memory with id {memory_id} not found")
                order = list(reversed(chain))
                deleted: List[str] = []
                for cid in order:
                    existing = await asyncio.to_thread(self.vector_store.get, vector_id=cid)
                    if existing is None:
                        continue
                    try:
                        await self._delete_memory(cid, existing)
                        deleted.append(cid)
                    except Exception as e:
                        remaining = [x for x in order if x not in deleted]
                        logger.error(f"Partial version-chain delete for {memory_id}: {e}; remaining={remaining}")
                        return {"message": "Memory partially deleted; retry with a remaining id",
                                "deleted": deleted, "remaining": remaining}
        else:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found")
            await self._delete_memory(memory_id, existing_memory)
        decay_usage_notice = detect_decay_usage_from_delete()
        if decay_usage_notice:
            await display_decay_usage_notice_async(self, "async", "delete", *decay_usage_notice)
        else:
            await display_first_run_notice_async(self, "async", "delete")
        return {"message": "Memory deleted successfully!"}

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        # Mesma normalização do gêmeo sync — ver o comentário longo lá. O gêmeo
        # async precisa dela por conta própria: não delega para o sync, e já
        # houve defeito que existia só neste lado por essa razão.
        user_id = _validate_and_trim_entity_id(user_id, "user_id")
        agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
        run_id = _validate_and_trim_entity_id(run_id, "run_id")

        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"})
        # Paginate — see the sync twin: `list()` defaults to top_k=100 in several
        # stores, so one untruncated call deleted at most a page and called it done.
        deleted = 0
        errors = []
        attempted = set()
        succeeded = []
        # ⚠️ Inicializado ANTES do laço. Estava atribuído só no FIM do corpo, e o
        # `break` da primeira página vazia pulava a atribuição sem executar o
        # `else` — então `delete_all` num escopo SEM memórias morria com
        # `UnboundLocalError` no `vector_scope_empty` lá embaixo. Escopo vazio é
        # justamente o caso comum (id errado, escopo já drenado), e este gêmeo
        # async nunca foi exercitado porque nada em produção chama `delete_all`:
        # o MCP usa `safe_bulk_delete`. Reproduzido em a473615 antes da correção.
        hit_page_cap = False
        for _ in range(_DELETE_ALL_MAX_PAGES):
            page = (await asyncio.to_thread(
                self.vector_store.list, filters=filters, top_k=_DELETE_ALL_PAGE_SIZE))[0]
            # Same "nothing new" termination as the sync twin — see there.
            fresh = [m for m in page if m.id not in attempted]
            if not fresh:
                break
            attempted.update(m.id for m in fresh)
            results = await asyncio.gather(
                *[self._delete_memory(m.id, skip_entity_cleanup=True) for m in fresh],
                return_exceptions=True)
            page_errors = [r for r in results if isinstance(r, BaseException)]
            errors.extend(page_errors)
            deleted += len(results) - len(page_errors)
            succeeded.extend(m.id for m, r in zip(fresh, results)
                             if not isinstance(r, BaseException))
        else:
            hit_page_cap = True
            logger.warning("delete_all: page cap (%d) reached — scope may not be drained",
                           _DELETE_ALL_MAX_PAGES)

        # "Zero delete errors" is NOT enough to justify wiping every entity row in
        # the scope: the page cap, or a memory written concurrently, can leave live
        # memories behind. Verify the scope is actually empty first.
        try:
            leftover = (await asyncio.to_thread(
                self.vector_store.list, filters=filters, top_k=1))[0]
        except Exception:
            leftover = [None]          # unknown -> assume not empty (fail safe)
        vector_scope_empty = not leftover and not hit_page_cap

        # Entity cleanup, ONCE, after every page — not per page. No
        # `_entity_store is not None` guard: a process that only ever deletes
        # never initializes the store, and skipping cleanup there is precisely
        # what leaves orphan rows behind. The helper decides.
        #
        # Bulk clear wipes EVERY entity row in the scope, which is only sound
        # when the scope really emptied. With a partial failure, memories are
        # still alive and would silently lose their entity rows — so fall back
        # to per-memory unlinking for the ones that did get deleted.
        # `skip_entity_cleanup=True` above means each _delete_memory committed its
        # intent WITHOUT cleaning — so the durability guarantee has to be
        # re-established here, per memory, using the intent ids we deferred.
        cleanup_ok = True
        if errors or not vector_scope_empty:
            if errors:
                logger.warning(
                    "delete_all: %d deletions failed — unlinking per memory instead of "
                    "bulk-clearing, so surviving memories keep their entity rows", len(errors))
            else:
                logger.warning(
                    "delete_all: scope not verified empty — unlinking per memory "
                    "instead of bulk-clearing")
            for mid in succeeded:
                if not await self._remove_memory_from_entity_store(mid, filters):
                    cleanup_ok = False
        else:
            cleanup_ok = await self._bulk_clear_entity_store(filters)
        if not cleanup_ok:
            logger.warning(
                "delete_all: entity cleanup INCOMPLETE for %s — dangling links may "
                "remain; re-run or repair with scripts/repair_entity_links.py", filters)

        if errors:
            logger.warning("Failed to delete %d memories", len(errors))
            for err in errors:
                logger.warning("Delete error: %s", err)

        logger.info(f"Deleted {deleted} memories")

        decay_usage_notice = detect_decay_usage_from_delete_all(deleted)
        if decay_usage_notice:
            await display_decay_usage_notice_async(self, "async", "delete_all", *decay_usage_notice)
        else:
            await display_first_run_notice_async(self, "async", "delete_all")
        return {"message": "Memories deleted successfully!"}

    async def history(self, memory_id):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "async"})
        history = await asyncio.to_thread(self.db.get_history, memory_id)
        await display_first_run_notice_async(self, "async", "history")
        return history

    async def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, memory_action="add")

        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data, language=self.config.language)
        # DeepMem0 v0.2: creation stays neutral until the first reinforcement.

        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        return memory_id

    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (llm, optional): LLM to use for the procedural memory creation. Defaults to None.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        try:
            from langchain_core.messages.utils import (
                convert_to_messages,  # type: ignore
            )
        except Exception:
            logger.error(
                "Import error while loading langchain-core. Please install 'langchain-core' to use procedural memory."
            )
            raise

        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        try:
            if llm is not None:
                parsed_messages = convert_to_messages(parsed_messages)
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                procedural_memory = response.content
            else:
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
                procedural_memory = remove_code_blocks(procedural_memory)
        
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    async def _version_update(self, memory_id, data, existing_embeddings, metadata, temp):
        """Async mirror of ``_version_update`` (DeepMem0 v0.7.1). Dedicated
        ``_mem0_version_next/prev`` lineage, fail-closed scope guard, born-superseded
        reverse link, strict verify + compensation restoring the head's exact original.
        Returns ``(current_id, superseded_head_id)`` from inside the lock."""
        _HEAD_MUT = (FIELD_SUPERSEDED_BY, FIELD_SUPERSEDED_AT, FIELD_VERSION_NEXT, FIELD_VERSION_PREV)

        async def _hist(mem_id, old_txt, new_txt, cre, upd, src):
            try:
                await asyncio.to_thread(self.db.add_history, mem_id, old_txt, new_txt, "SUPERSEDED",
                                        created_at=cre, updated_at=upd, actor_id=src.get("actor_id"), role=src.get("role"))
            except Exception as e:
                logger.warning(f"Supersession history record failed for {mem_id}: {e}")

        async with self._version_lock:
            head_id, head_mem = await asyncio.to_thread(
                _resolve_chain_head, lambda vid: self.vector_store.get(vector_id=vid), memory_id
            )
            if head_mem is None:
                raise ValueError(
                    f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'"
                )
            head_payload = dict(getattr(head_mem, "payload", None) or {})
            caller = dict(metadata or {})
            operation_ts = caller.get("created_at") or _dynamics_utcnow().isoformat()
            born_superseded = supersession_inverted(operation_ts, head_payload.get("created_at"))
            # DeepMem0 v0.9: cópia da timeline + T2 — ver o twin sync.
            inherit_dyn = (
                bool(getattr(temp, "version_inherits_dynamics", False))
                and not born_superseded
            )
            v2_meta = _build_version_metadata(
                head_payload, data, caller, operation_ts, head_id,
                getattr(temp, "extract_event_date", True),
                inherit_dynamics=inherit_dyn,
            )
            dyn_extra, t2_outcome = _plan_version_dynamics(
                head_payload, _dynamics_config(self.config), operation_ts,
                inherit=inherit_dyn,
            )
            v2_meta.update(dyn_extra)
            if born_superseded:
                v2_meta[FIELD_VERSION_PREV] = []
                v2_meta[FIELD_VERSION_NEXT] = head_id
                v2_meta[FIELD_SUPERSEDED_BY] = head_id
                v2_meta[FIELD_SUPERSEDED_AT] = head_payload.get("created_at") or operation_ts

            head_restore = {**head_payload, **{k: head_payload.get(k) for k in _HEAD_MUT}}
            head_modified = False
            new_id = None
            try:
                new_id = await self._create_memory(data, existing_embeddings, metadata=v2_meta)
                session_filters = {k: v2_meta[k] for k in ("user_id", "agent_id", "run_id") if v2_meta.get(k)}
                await self._link_entities_for_memory(new_id, data, session_filters)
                if born_superseded:
                    head_prev = list(head_payload.get(FIELD_VERSION_PREV) or [])
                    head_prev.append(new_id)
                    await asyncio.to_thread(self.vector_store.update, vector_id=head_id,
                                            payload={**head_payload, FIELD_VERSION_PREV: head_prev})
                    head_modified = True
                    await _hist(new_id, data, head_payload.get("data"), operation_ts, operation_ts, head_payload)
                    _m = await asyncio.to_thread(self.vector_store.get, vector_id=new_id)
                    if (getattr(_m, "payload", None) or {}).get(FIELD_VERSION_NEXT) != head_id:
                        raise RuntimeError(f"Version transition verify failed: {new_id} not born superseded by {head_id}")
                    current_id = head_id
                else:
                    await asyncio.to_thread(
                        self.vector_store.update, vector_id=head_id,
                        payload={**head_payload, FIELD_SUPERSEDED_BY: new_id,
                                 FIELD_SUPERSEDED_AT: operation_ts, FIELD_VERSION_NEXT: new_id},
                    )
                    head_modified = True
                    await _hist(head_id, head_payload.get("data"), data, head_payload.get("created_at"), operation_ts, head_payload)
                    check = await asyncio.to_thread(self.vector_store.get, vector_id=head_id)
                    cp = (getattr(check, "payload", None) or {}) if check is not None else None
                    if cp is None or cp.get(FIELD_VERSION_NEXT) != new_id or cp.get(FIELD_SUPERSEDED_BY) != new_id:
                        raise RuntimeError(f"Version transition verify failed: head {head_id} not linked to {new_id}")
                    if await asyncio.to_thread(self.vector_store.get, vector_id=new_id) is None:
                        raise RuntimeError(f"Version transition verify failed: new version {new_id} missing")
                    current_id = new_id
            except Exception:
                if head_modified:
                    try:
                        await asyncio.to_thread(self.vector_store.update, vector_id=head_id, payload=head_restore)
                    except Exception as re_:
                        logger.error(f"Compensation restore of head {head_id} failed: {re_}")
                if new_id is not None:
                    try:
                        await self._delete_memory(new_id)
                    except Exception as ce:
                        logger.error(f"Compensation delete of {new_id} failed: {ce}")
                raise
            # T2 notify pós-verify — mesma disciplina do twin sync.
            if t2_outcome is not None:
                _notify_reinforcement(new_id, TRIGGER_UPDATE, t2_outcome)
            logger.info(f"Versioned update (async): head={head_id} new={new_id} current={current_id} born={born_superseded}")
            return current_id, head_id

    async def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        temp = _temporality_config(self.config)
        if temp is not None and getattr(temp, "version_on_update", False):
            # v0.9: T2 existe neste modo via _plan_version_dynamics (ver sync).
            return await self._version_update(memory_id, data, existing_embeddings, metadata, temp)
        logger.info(f"Updating memory with {data=}")

        # Embedding ANTES da leitura autoritativa — mesma razão do caminho sync:
        # o trecho lento envelhecia o payload e o upsert de payload completo
        # apagava um T3 que chegasse no intervalo.
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")

        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(existing_memory.payload)
        if metadata is not None:
            new_metadata.update(metadata)

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data, language=self.config.language)
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Ownership scope is immutable after creation (issue #4490 for actor_id).
        # MESMA regra do caminho versionado, e a ausência conta: o guard antigo
        # só preservava um valor JÁ existente, então um `metadata={"actor_id": ...}`
        # do chamador carimbava autoria em memória que não tinha nenhuma — e
        # nenhuma é o estado de todo o corpus legado.
        aplicar_escopo_imutavel(new_metadata, existing_memory.payload)

        # DeepMem0 v0.2 (T2): an updated fact is alive — reinforce its timeline.
        dyn = _dynamics_config(self.config)
        if dyn is not None:
            _fields, _outcome = plan_reinforcement(
                existing_memory.payload, dyn, TRIGGER_UPDATE
            )
            if _fields:
                new_metadata.update(_fields)
            # NÃO notifica aqui: a decisão foi tomada, mas o reforço só existe
            # depois que a escrita do update pega. Emitir "applied" antes do
            # write faria a telemetria afirmar um reforço que uma falha de embed
            # ou de vector_store deixaria sem persistir.

        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        if dyn is not None:
            _notify_reinforcement(memory_id, TRIGGER_UPDATE, _outcome)
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        await self._remove_memory_from_entity_store(memory_id, session_filters)
        await self._link_entities_for_memory(memory_id, data, session_filters)

        return memory_id

    async def _delete_memory(self, memory_id, existing_memory=None, skip_entity_cleanup=False, *, op_id=None):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data", "")
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = existing_memory.payload or {}
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}

        # DeepMem0 v0.7.2: durable delete intent (crash-consistency) — mirrors sync.
        own_intent = op_id is None
        if own_intent:
            op_id = str(uuid.uuid4())
            if self.db is not None:
                before_image = json.dumps({
                    "data": prev_value, "created_at": created_at,
                    "actor_id": existing_memory.payload.get("actor_id"),
                    "role": existing_memory.payload.get("role"),
                }, sort_keys=True)
                await asyncio.to_thread(
                    self.db.begin_delete, op_id, memory_id,
                    json.dumps(session_filters, sort_keys=True), before_image,
                )
        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        # Tombstone is IDEMPOTENT: never duplicate the DELETE row on retry/reconcile.
        if self.db is not None and not await asyncio.to_thread(self.db.has_delete_tombstone, memory_id):
            await asyncio.to_thread(
                self.db.add_history,
                memory_id,
                prev_value,
                None,
                "DELETE",
                created_at=created_at,
                updated_at=updated_at,
                actor_id=existing_memory.payload.get("actor_id"),
                role=existing_memory.payload.get("role"),
                is_deleted=1,
            )
        # Cleanup BEFORE the commit, and the commit is CONDITIONAL on it —
        # see the sync twin. Cleanup is idempotent.
        cleaned = True
        if not skip_entity_cleanup:
            cleaned = await self._remove_memory_from_entity_store(memory_id, session_filters)

        if self.db is not None:
            if cleaned:
                await asyncio.to_thread(self.db.commit_delete, op_id)
            else:
                logger.warning(
                    "Delete of %s: entity cleanup incomplete — leaving the intent "
                    "PENDING so reconciliation retries it.", memory_id)

        return memory_id

    def reconcile_pending_deletes(self) -> int:
        """DeepMem0 v0.7.2 — finish crash-interrupted deletes (sync, one-shot at init).

        Same contract as ``Memory.reconcile_pending_deletes``; runs synchronously
        (no event loop yet at construction) using the sync vector-store/db calls.
        """
        if getattr(self, "db", None) is None:
            return 0
        try:
            pending = self.db.list_pending_deletes()
        except Exception as e:
            logger.warning(f"Could not read pending delete intents: {e}")
            return 0
        reconciled = 0
        for intent in pending:
            mid, op = intent["memory_id"], intent["op_id"]
            try:
                before = {}
                if intent.get("before_image"):
                    try:
                        before = json.loads(intent["before_image"])
                    except Exception:
                        before = {}
                existing = self.vector_store.get(vector_id=mid)
                spared = False
                cleaned = True   # a SPARED id has nothing to clean
                if existing is not None:
                    # ABA guard: only delete if the CURRENT vector is the SAME memory the
                    # intent targeted (created_at identity). A REUSED id (import/restore/manual)
                    # is SPARED — we never delete a different memory that took the id.
                    cur_created = (existing.payload or {}).get("created_at")
                    orig_created = before.get("created_at")
                    if orig_created and cur_created and cur_created != orig_created:
                        logger.warning(f"Reconcile: id {mid} appears REUSED (created_at differs) — sparing current vector")
                        spared = True
                    else:
                        self.vector_store.delete(vector_id=mid)
                if not spared and _entity_cleanup_enabled():
                    # The ABA guard extends to the entity store: stripping links for
                    # a REUSED id would silently unlink the *new* memory that took it.
                    try:
                        scope = json.loads(intent.get("scope") or "{}")
                    except Exception:
                        scope = {}
                    cleaned = unlink_memory_from_entity_rows(
                        self.entity_store, mid, scope if isinstance(scope, dict) else {})
                if not self.db.has_delete_tombstone(mid):
                    # faithful tombstone from the before-image (survives a crash that hit
                    # before the tombstone was written)
                    self.db.add_history(
                        mid, before.get("data"), None, "DELETE",
                        created_at=before.get("created_at"),
                        updated_at=datetime.now(timezone.utc).isoformat(),
                        actor_id=before.get("actor_id"), role=before.get("role"),
                        is_deleted=1,
                    )
                if cleaned:
                    self.db.commit_delete(op)
                    reconciled += 1
                else:
                    # Intent stays PENDING on purpose: an incomplete entity
                    # cleanup that we commit anyway is a dangling link nobody
                    # will ever come back for.
                    logger.warning(
                        "Reconcile of %s: entity cleanup incomplete — intent stays PENDING.", mid)
            except Exception as e:
                logger.warning(f"Reconcile of pending delete {op} ({mid}) failed: {e}")
        if reconciled:
            logger.info(f"Reconciled {reconciled} pending delete(s) on startup.")
        return reconciled

    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")
        await asyncio.to_thread(self.vector_store.delete_col)

        gc.collect()

        if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
            await asyncio.to_thread(self.vector_store.client.close)

        if hasattr(self.db, "connection") and self.db.connection:
            await asyncio.to_thread(lambda: self.db.connection.execute("DROP TABLE IF EXISTS history"))
            await asyncio.to_thread(self.db.connection.close)

        self.db = SQLiteManager(self.config.history_db_path)

        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )

        if self._entity_store is not None:
            try:
                await asyncio.to_thread(self._entity_store.reset)
            except Exception as e:
                logger.warning(f"Failed to reset entity store: {e}")
            self._entity_store = None

        capture_event("mem0.reset", self, {"sync_type": "async"})
        await display_first_run_notice_async(self, "async", "reset")

    def close(self):
        """Release resources held by this AsyncMemory instance."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    async def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
