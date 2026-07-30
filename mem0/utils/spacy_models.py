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
_lock = threading.Lock()


def _code(language) -> str:
    return (language or DEFAULT_LANGUAGE).split("-")[0].lower()


def model_name(language=None) -> str:
    """spaCy package for a language, falling back to English."""
    return MODEL_BY_LANGUAGE.get(_code(language), MODEL_BY_LANGUAGE[DEFAULT_LANGUAGE])


def _strict() -> bool:
    """Whether a missing model must raise instead of degrading silently.

    A deployment configured for Portuguese whose Portuguese model is absent used
    to fall back to English and say nothing — which is exactly how an English
    pipeline ended up scoring a Portuguese corpus for months. `MEM0_SPACY_STRICT`
    turns that into a startup failure. Default off, so the public behaviour is
    unchanged for anyone upgrading.
    """
    return os.environ.get("MEM0_SPACY_STRICT", "").strip().lower() in (
        "1", "true", "yes", "on")


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
            if _strict():
                raise RuntimeError(
                    f"spaCy {rotulo} model {name!r} for language {code!r} is "
                    f"unavailable ({e}) and MEM0_SPACY_STRICT is on. Entity "
                    "extraction would silently fall back to another language's "
                    "pipeline, which is the defect this flag exists to prevent."
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
