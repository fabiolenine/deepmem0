"""
Shared spaCy model loader.

Consolidates spaCy model loading into a single module so that
entity_extraction and lemmatization share one instance instead of
each loading their own copy from disk.

DeepMem0 (N3): the loader is a cache PER LANGUAGE. It used to hard-code
`en_core_web_sm` and every caller got the English pipeline whatever the
configured language was — which on Portuguese text tags all five tokens of
"Ontem eu viajei para Recife" as PROPN, pronoun and verb included, and the
auxiliary `tem` as NOUN. Every POS-keyed rule downstream was therefore inert or
harmful. With `pt_core_news_sm` the same sentence tags as
`ADV PRON VERB ADP PROPN` and `doc.ents` returns real entities.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

# language code -> spaCy package name
MODEL_BY_LANGUAGE = {
    "en": "en_core_web_sm",
    "pt": "pt_core_news_sm",
}
DEFAULT_LANGUAGE = "en"

_nlp_full: dict = {}
_nlp_lemma: dict = {}
_load_failed_full: set = set()
_load_failed_lemma: set = set()
# ⚠️ REENTRANTE. `_fallback` chama `_load` de dentro do `with _lock`, e com um
# `Lock` simples isso é DEADLOCK — o processo trava. O caminho que dispara é
# justamente o de degradar com elegância: língua configurada sem o modelo dela e
# `MEM0_SPACY_STRICT=0`. Um teste do carregador pegou; nenhuma execução normal
# pegaria, porque em produção o modelo está presente.
_lock = threading.RLock()


def _code(language) -> str:
    return (language or DEFAULT_LANGUAGE).split("-")[0].lower()


def model_name(language=None) -> str:
    """spaCy package for a language, falling back to English."""
    return MODEL_BY_LANGUAGE.get(_code(language), MODEL_BY_LANGUAGE[DEFAULT_LANGUAGE])


def model_available(language=None) -> bool:
    """Whether the language's model is INSTALLED — sem tocar a rede.

    `_ensure_model_available` tenta `spacy.cli.download`, e uma sonda de
    readiness que dispara download pendura no primeiro ambiente sem rede (medido:
    10 minutos até o timeout). Readiness pergunta "está instalado?", não
    "consegue instalar?".
    """
    try:
        import spacy

        return bool(spacy.util.is_package(model_name(language)))
    except Exception:
        return False


def entity_pipeline_status(language=None) -> dict:
    """Diagnóstico para a sonda: qual modelo, instalado, e se degradaria.

    `degraded` é o que a readiness tem que reprovar: língua configurada sem o
    modelo dela significa POS inutilizável (medido: verbo português volta PROPN,
    auxiliar volta NOUN), não apenas "um pouco pior".
    """
    code = _code(language)
    suportado = code in MODEL_BY_LANGUAGE
    nome = model_name(code)
    instalado = model_available(code)
    # Idioma NÃO SUPORTADO conta como degradado. `model_name` cai em inglês para
    # código desconhecido, então perguntar só "o modelo está instalado?" devolvia
    # `True` para uma língua sem pipeline nenhum — o mesmo silêncio que este
    # trabalho existe para eliminar, só deslocado do português para as outras.
    return {
        "language": code,
        "supported": suportado,
        "model": nome if suportado else None,
        "installed": instalado if suportado else False,
        "is_default_language": code == DEFAULT_LANGUAGE,
        "degraded": (not suportado or not instalado) and code != DEFAULT_LANGUAGE,
        "strict": _strict(code),
        "load_failed": code in _load_failed_full,
    }


def _strict(code: str) -> bool:
    """Whether a missing model must RAISE instead of degrading silently.

    ⚠️ Isto era opt-in (`MEM0_SPACY_STRICT`, default off) e o critério de aceite
    pede o contrário: modelo ausente num deployment configurado para outra língua
    tem que REPROVAR, não desligar o sistema de entidades em silêncio. Com o
    default off, o silêncio continuava sendo o comportamento padrão — que é
    exatamente como um pipeline inglês acabou pontuando um corpus português.

    A regra agora distingue os dois casos, o que satisfaz o critério sem quebrar
    quem atualiza:
      * língua configurada NÃO é o default (ex.: `pt`) e o modelo dela falta ->
        LEVANTA. É a incompatibilidade que o critério proíbe silenciar, e cair no
        inglês ali produz POS inutilizável (medido: `viajei/PROPN`, `tem/NOUN`).
      * língua é o default (inglês) -> mantém o comportamento do upstream
        (avisa e devolve None), porque não há para onde cair e romper aí
        quebraria qualquer instalação sem o modelo baixado.

    `MEM0_SPACY_STRICT=0` continua disponível como escape hatch explícito para
    quem aceita a degradação — o padrão passou a ser seguro, e afrouxar exige um
    ato deliberado.
    """
    bruto = os.environ.get("MEM0_SPACY_STRICT", "").strip().lower()
    if bruto in ("0", "false", "no", "off"):
        return False
    if bruto in ("1", "true", "yes", "on"):
        return True
    return code != DEFAULT_LANGUAGE


def _ensure_model_available(name: str):
    """Download the model if spaCy is installed but the package is missing."""
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy is not installed. Install it with: pip install mem0ai[nlp]"
        )

    if not spacy.util.is_package(name):
        logger.info("Downloading spaCy model %s...", name)
        try:
            from spacy.cli import download

            download(name)
            logger.info("spaCy model %s downloaded successfully", name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download spaCy model {name}: {e}. "
                f"Please install manually: python -m spacy download {name}"
            ) from e


def _load(cache: dict, failed: set, language, disable, rotulo: str):
    code = _code(language)
    if code in failed:
        return _fallback(cache, failed, code, disable, rotulo)
    cached = cache.get(code)
    if cached is not None:
        return cached
    name = model_name(code)
    with _lock:
        cached = cache.get(code)
        if cached is not None:
            return cached
        if code in failed:
            return _fallback(cache, failed, code, disable, rotulo)
        try:
            _ensure_model_available(name)
            import spacy

            cache[code] = spacy.load(name, disable=disable) if disable else spacy.load(name)
            logger.info("spaCy %s model loaded for %r (%s)", rotulo, code, name)
            return cache[code]
        except BaseException as e:  # DeepMem0: spacy.cli.download may sys.exit(1)
            failed.add(code)
            if _strict(code):
                raise RuntimeError(
                    f"spaCy {rotulo} model {name!r} for language {code!r} is "
                    f"unavailable ({e}). Falling back to the "
                    f"{MODEL_BY_LANGUAGE[DEFAULT_LANGUAGE]} pipeline would give "
                    f"unusable POS tags for {code!r} text — measured: a "
                    "Portuguese verb comes back PROPN and an auxiliary comes "
                    "back NOUN. Install it with "
                    f"`python -m spacy download {name}`, or set "
                    "MEM0_SPACY_STRICT=0 to accept the degradation explicitly."
                ) from e
            logger.warning("Failed to load spaCy %s model %s for %r: %s",
                           rotulo, name, code, e)
            return _fallback(cache, failed, code, disable, rotulo)


def _fallback(cache: dict, failed: set, code: str, disable, rotulo: str):
    """English pipeline as a last resort — LOUD, never silent."""
    if code == DEFAULT_LANGUAGE:
        return None
    logger.warning(
        "spaCy has no usable model for %r; falling back to the %s pipeline. "
        "POS tags and entities for %r text will be unreliable.",
        code, MODEL_BY_LANGUAGE[DEFAULT_LANGUAGE], code)
    return _load(cache, failed, DEFAULT_LANGUAGE, disable, rotulo)


def get_nlp_full(language=None):
    """spaCy model with all pipelines (NER, tagger, parser) for entity extraction."""
    return _load(_nlp_full, _load_failed_full, language, None, "full")


def get_nlp_lemma(language=None):
    """spaCy model with only the lemmatizer, for BM25 text processing."""
    return _load(_nlp_lemma, _load_failed_lemma, language, ["ner", "parser"], "lemma")
