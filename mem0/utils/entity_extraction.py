"""
Entity extraction from text using spaCy NLP.

Extracts four types of entities from text:
- **Proper nouns**: Capitalized multi-word sequences (person names, places, brands)
- **Quoted text**: Text in single or double quotes (titles, specific terms)
- **Noun compounds**: Multi-word noun phrases with specific modifiers (e.g., "machine learning")
- **Noun fallback**: Single nouns from circumstantial compound patterns

Public API:
    extract_entities(text: str) -> List[Tuple[str, str]]

Internal:
    _extract_entities_from_doc(doc) -> List[Tuple[str, str]]
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Words that are too generic to be useful as entity heads
_GENERIC_HEADS = {
    "thing", "stuff", "way", "time", "experience", "situation", "case",
    "fact", "matter", "issue", "idea", "thought", "feeling", "place",
    "area", "part", "kind", "type", "sort", "lot", "bit", "day", "year",
    "week", "month", "moment", "instance", "example", "technique",
    "method", "approach", "process", "step", "tool", "result", "outcome",
    "goal", "task", "item", "topic", "scale", "size", "level", "degree",
    "amount", "number", "style", "look", "color", "colour", "shape",
    "form", "piece", "section", "side", "end", "edge", "surface", "point",
}

# Modifiers that describe circumstance, not content
_CIRCUMSTANTIAL_MODS = {
    "solo", "individual", "team", "group", "joint", "collaborative",
    "first", "last", "next", "previous", "final", "initial", "main", "side",
}

# Adjectives too vague to make a compound entity specific
_NON_SPECIFIC_ADJ = {
    "many", "few", "several", "some", "any", "all", "most", "more",
    "less", "much", "little", "enough", "various", "numerous", "multiple",
    "countless", "great", "good", "bad", "nice", "terrible", "awful",
    "awesome", "amazing", "wonderful", "horrible", "excellent", "poor",
    "best", "worst", "fine", "okay", "new", "old", "recent", "past",
    "future", "current", "previous", "next", "last", "first", "latest",
    "early", "late", "former", "modern", "ancient", "big", "small",
    "large", "tiny", "huge", "enormous", "long", "short", "tall", "high",
    "low", "wide", "narrow", "thick", "thin", "deep", "shallow",
    "similar", "different", "same", "other", "another", "such", "certain",
    "important", "main", "major", "minor", "key", "primary", "real",
    "actual", "true", "whole", "entire", "full", "complete", "total",
    "basic", "simple", "interesting", "boring", "exciting", "special",
    "particular", "general", "common", "unique", "rare", "typical",
    "usual", "normal", "regular", "possible", "likely", "potential",
    "available", "necessary", "only", "solo", "individual", "team",
    "group", "joint", "collaborative", "final", "initial", "side",
}

# Generic tail words to strip from compound entities
_GENERIC_ENDINGS = {
    "work", "works", "job", "jobs", "task", "tasks", "stuff", "things",
    "thing", "info", "information", "details", "data", "content",
    "material", "materials", "activities", "activity", "efforts", "effort",
    "options", "option", "choices", "choice", "results", "result",
    "output", "outputs", "products", "product", "items", "item",
}

# Capitalized single words that are too generic to be proper nouns
_GENERIC_CAPS = {
    "works", "items", "things", "stuff", "resources", "options", "tips",
    "ideas", "steps", "ways", "methods", "tools", "features", "benefits",
    "examples", "details", "notes", "instructions", "guidelines",
    "recommendations", "suggestions", "overview", "summary", "conclusion",
    "introduction", "pros", "cons", "advantages", "disadvantages",
}


# ============================================================
# LÉXICO POR IDIOMA (N2)
# ============================================================
# As listas acima são 100% inglesas e rodavam sobre texto português — o mesmo
# erro que o fork já corrigiu para o BM25 (`lemmatization.py`: "o lematizador
# inglês é ruído, ou pior, em texto não-inglês"), nunca corrigido para entidades.
#
# Medido no corpus PT: `Fase` e `Item` viravam PROPER porque `_GENERIC_CAPS` só
# conhece `works`/`items`/`steps`; e `CONCLUÍDO`, `DECISÃO`, `REMOVIDO`,
# `CRÍTICO` viravam PROPER porque uma palavra em CAIXA ALTA no meio da frase é
# indistinguível de sigla para um pipeline sem léxico da língua.
#
# A LÓGICA DE POS NÃO MUDA. Só o vocabulário: com um parser PT (N3) o ramo C
# passa a ter POS e noun_chunks confiáveis, e desligá-lo agora desperdiçaria
# justamente o que o N3 traz.
_PT_GENERIC_HEADS = {
    "coisa", "jeito", "forma", "tempo", "experiência", "situação", "caso",
    "fato", "questão", "ideia", "pensamento", "sentimento", "lugar", "área",
    "parte", "tipo", "monte", "dia", "ano", "semana", "mês", "momento",
    "instância", "exemplo", "técnica", "método", "abordagem", "processo",
    "passo", "etapa", "fase", "ferramenta", "resultado", "objetivo", "meta",
    "tarefa", "tópico", "escala", "tamanho", "nível", "grau", "quantidade",
    "número", "estilo", "cor", "formato", "peça", "seção", "lado", "fim",
    "borda", "superfície", "ponto", "item", "coisas", "detalhe",
}

_PT_CIRCUMSTANTIAL_MODS = {
    "individual", "equipe", "grupo", "conjunto", "primeiro", "último",
    "próximo", "anterior", "final", "inicial", "principal", "lateral",
}

_PT_NON_SPECIFIC_ADJ = {
    "muitos", "muitas", "poucos", "poucas", "vários", "várias", "alguns",
    "algumas", "qualquer", "todos", "todas", "maioria", "mais", "menos",
    "bastante", "diversos", "diversas", "múltiplos", "grande", "bom", "boa",
    "ruim", "ótimo", "péssimo", "excelente", "melhor", "pior", "novo", "nova",
    "velho", "antigo", "recente", "passado", "futuro", "atual", "anterior",
    "próximo", "último", "primeiro", "cedo", "tarde", "moderno", "pequeno",
    "enorme", "longo", "curto", "alto", "baixo", "largo", "estreito",
    "grosso", "fino", "profundo", "raso", "similar", "parecido", "diferente",
    "mesmo", "mesma", "outro", "outra", "tal", "certo", "importante",
    "principal", "maior", "menor", "chave", "primário", "real", "verdadeiro",
    "inteiro", "completo", "total", "básico", "simples", "interessante",
    "especial", "particular", "geral", "comum", "único", "raro", "típico",
    "usual", "normal", "regular", "possível", "provável", "potencial",
    "disponível", "necessário", "apenas", "só",
}

_PT_GENERIC_ENDINGS = {
    "trabalho", "trabalhos", "tarefa", "tarefas", "coisas", "coisa", "info",
    "informação", "informações", "detalhes", "dados", "conteúdo", "material",
    "materiais", "atividades", "atividade", "esforços", "esforço", "opções",
    "opção", "escolhas", "escolha", "resultados", "resultado", "saída",
    "saídas", "produtos", "produto", "itens", "item",
}

_PT_GENERIC_CAPS = {
    "fase", "fases", "item", "itens", "etapa", "etapas", "resumo", "visão",
    "conclusão", "conclusões", "introdução", "observação", "observações",
    "nota", "notas", "exemplo", "exemplos", "detalhes", "recomendações",
    "sugestões", "vantagens", "desvantagens", "prós", "contras", "passos",
    "métodos", "ferramentas", "opções", "ideias", "dicas", "recursos",
    "resultado", "resultados", "decisão", "decisões", "objetivo", "objetivos",
    "contexto", "motivo", "causa", "problema", "solução", "conteúdo",
}

_LEXICONS = {
    "en": {
        "generic_heads": _GENERIC_HEADS,
        "circumstantial": _CIRCUMSTANTIAL_MODS,
        "non_specific_adj": _NON_SPECIFIC_ADJ,
        "generic_endings": _GENERIC_ENDINGS,
        "generic_caps": _GENERIC_CAPS,
        # Sufixos de ênfase ficam FORA do inglês: a coluna EN do golden é
        # congelada e não há defeito medido lá.
        "upper_emphasis_suffixes": (),
    },
    "pt": {
        "generic_heads": _GENERIC_HEADS | _PT_GENERIC_HEADS,
        "circumstantial": _CIRCUMSTANTIAL_MODS | _PT_CIRCUMSTANTIAL_MODS,
        "non_specific_adj": _NON_SPECIFIC_ADJ | _PT_NON_SPECIFIC_ADJ,
        "generic_endings": _GENERIC_ENDINGS | _PT_GENERIC_ENDINGS,
        "generic_caps": _GENERIC_CAPS | _PT_GENERIC_CAPS,
        # Sufixos verbais/nominais do português. Comprimento NÃO serve como
        # critério — a 1ª versão usava `len >= 6` e matou `PYTHONPATH`, que é
        # identificador legítimo. O que separa ênfase de sigla é MORFOLOGIA:
        # `REMOVIDO`/`CONCLUÍDO` (-IDO), `APARECEM` (-EM), `DECISÃO` (-ÃO) são
        # palavras flexionadas; `PYTHONPATH`, `FFT`, `HNSW` não têm flexão.
        "upper_emphasis_suffixes": (
            "ADO", "ADOS", "ADA", "ADAS", "IDO", "IDOS", "IDA", "IDAS",
            "ÇÃO", "ÇÕES", "SÃO", "SÕES", "MENTE", "AGEM", "ANDO", "ENDO",
            "INDO", "AVAM", "IAM", "AREM", "EREM", "IREM", "ECEM", "ARAM",
        ),
    },
}


def _lexicon(language):
    """Léxico do idioma, com fallback EXPLÍCITO para inglês.

    Idioma desconhecido cai em inglês e AVISA uma vez: silêncio aqui
    reproduziria o defeito original — um pipeline inglês rodando sobre outra
    língua sem ninguém saber.
    """
    code = (language or "en").split("-")[0].lower()
    lex = _LEXICONS.get(code)
    if lex is None:
        if code not in _LEXICONS_AVISADOS:
            _LEXICONS_AVISADOS.add(code)
            logger.warning(
                "entity extraction has no lexicon for language %r; falling back "
                "to English. Spans from this text will be filtered by English "
                "word lists.", code)
        lex = _LEXICONS["en"]
    return lex


_LEXICONS_AVISADOS = set()

# Uma palavra INTEIRA em caixa alta é sigla ou ênfase. Acento decide sozinho
# (sigla não leva acento em nenhuma das duas línguas); comprimento decide o
# resto, e o teto vem do léxico porque é escolha de idioma.
_ACENTOS = "ÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇÑ"


def _e_enfase_em_caixa_alta(txt: str, lex) -> bool:
    """CAIXA ALTA é sigla ou ênfase? Decidem acento e morfologia, não tamanho.

    A 1ª versão usava `len >= 6` e o golden pegou o preço: `PYTHONPATH` sumiu.
    Sigla não flexiona; palavra escrita em caixa alta para dar ênfase, sim.
    """
    if " " in txt or len(txt) < 4 or not txt.isupper():
        return False
    sufixos = lex.get("upper_emphasis_suffixes") or ()
    if not sufixos:
        return False                     # idioma sem regra declarada: não mexe
    if any(c in _ACENTOS for c in txt):
        return True                      # CONCLUÍDO, DECISÃO, CRÍTICO
    return txt.endswith(sufixos)         # REMOVIDO, APARECEM, DESAPARECEM


# Markdown/formatting markers to skip during extraction
# Connectors allowed INSIDE a proper-noun sequence (branch A).
#
# `in`, `at`, `for` and `is` were in this set and glued distinct entities into one
# span: `Northwind in São Paulo` became a single PROPER, and the substring
# cleanup below then DELETED `Northwind` for being contained in it. Measured on
# the production corpus: the two memories a human would call the `Northwind`
# answer had no entity link at all because of this.
#
# `is` is a verb — it was never defensible. `of`, `the`, `and` and `'s` stay:
# they are internal to real names (`Bank of America`, `The Rolling Stones`).
_CONNECTORS = {"'s", "of", "the", "and"}

# Sanity caps. An entity is a NAME, not a clause: a span with six words is a
# badly cut sentence. Measured (29/07/2026): 16 of the 81 spans the extractor
# produced from the 37 golden queries were over 5 tokens, e.g.
# 'definição de harness segundo o artigo de Martin Fowler sobre engenharia de'.
# Garbage spans match garbage entity rows (`'que datas foi'` matched
# `'idade e data de nascimento'` at 0.67), which is where the boost's fan-out —
# a median of 250 memories boosted per query — comes from.
MAX_ENTITY_TOKENS = 5
MAX_ENTITY_CHARS = 60

_FORMATTING_MARKERS = {"*", "-", "+", "\u2022", "\u2013", "\u2014", "#", "##", "###", "**", "__"}


def _is_sentence_start(tokens: list, idx: int) -> bool:
    """Check if a token is at the start of a sentence or after formatting."""
    if idx == 0:
        return True
    tok = tokens[idx]
    if tok.is_sent_start:
        return True
    prev = tokens[idx - 1].text
    return prev in ".!?:" or prev in _FORMATTING_MARKERS or "\n" in prev


def _strip_generic_ending(toks: list, endings=None) -> list:
    """Remove generic trailing words from compound token sequences."""
    if len(toks) <= 1:
        return toks
    endings = _GENERIC_ENDINGS if endings is None else endings
    last = toks[-1].lemma_.lower() if hasattr(toks[-1], "lemma_") else toks[-1].lower()
    return toks[:-1] if last in endings and len(toks) > 2 else toks


def _lemmatize_compound(toks: list) -> str:
    """Join compound tokens, lemmatizing nouns."""
    return " ".join(t.lemma_ if t.pos_ == "NOUN" else t.text for t in toks)


def _has_artifacts(txt: str) -> bool:
    """Check for formatting artifacts that indicate non-entity text."""
    return any(
        [
            "**" in txt or "__" in txt or ":*" in txt,
            re.search(r"\s\*\s|\s\*$|^\*\s", txt),
            "  " in txt or "\n" in txt or "\t" in txt,
            len(txt) > 100,
            txt.startswith(("\u2022", "-", "+", "\u2013", "\u2014")),
        ]
    )


def extract_entities(text: str, language: str = "en") -> List[Tuple[str, str]]:
    """Extract named entities, quoted text, and noun compounds from text.

    This is the public API that accepts a string. It loads the spaCy model
    internally and delegates to _extract_entities_from_doc().

    Args:
        text: Input text to extract entities from.

    Returns:
        Deduplicated list of (entity_type, entity_text) tuples.
        Entity types: PROPER, QUOTED, COMPOUND, NOUN.
        Returns empty list if spaCy is unavailable.
    """
    from mem0.utils.spacy_models import get_nlp_full

    nlp = get_nlp_full()
    if nlp is None:
        return []

    doc = nlp(text)
    return _extract_entities_from_doc(doc, _lexicon(language))


def extract_entities_batch(texts: List[str], batch_size: int = 32,
                           language: str = "en") -> List[List[Tuple[str, str]]]:
    """Extract entities from multiple texts using spaCy's nlp.pipe() for batched NER.

    Uses spaCy's efficient batch processing pipeline instead of calling
    nlp() individually per text. Significantly faster for multiple texts.

    Args:
        texts: List of input texts to extract entities from.
        batch_size: Number of texts to process in each spaCy batch.
        language: Lexicon to filter spans with. Unknown codes fall back to
            English and log a warning — silence here would reproduce the very
            defect this parameter exists to fix.

    Returns:
        List of entity lists, one per input text. Each entity list contains
        (entity_type, entity_text) tuples. Returns list of empty lists if
        spaCy is unavailable.
    """
    if not texts:
        return []

    from mem0.utils.spacy_models import get_nlp_full

    nlp = get_nlp_full()
    if nlp is None:
        return [[] for _ in texts]

    lex = _lexicon(language)
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        results.append(_extract_entities_from_doc(doc, lex))
    return results


def _extract_entities_from_doc(doc, lex=None) -> List[Tuple[str, str]]:
    """Extract entities from a spaCy Doc object.

    Ported from platform's shared.core.utils.entity_extraction.extract_entities().
    """
    lex = lex or _LEXICONS["en"]
    entities: List[Tuple[str, str]] = []
    text = doc.text
    tokens = list(doc)

    # === PROPER NOUN SEQUENCES ===
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.text in _FORMATTING_MARKERS:
            i += 1
            continue
        is_cap = tok.text and tok.text[0].isupper()
        is_label = i + 1 < len(tokens) and tokens[i + 1].text == ":"

        if is_cap and not is_label and tok.pos_ in {"PROPN", "NOUN", "ADJ"}:
            seq = [(tok, i)]
            j = i + 1
            while j < len(tokens):
                t = tokens[j]
                if (t.text and t.text[0].isupper()) or t.text.lower() in _CONNECTORS:
                    seq.append((t, j))
                    j += 1
                else:
                    break
            # Strip trailing function words
            while seq and seq[-1][0].text.lower() in _CONNECTORS:
                seq.pop()
            if seq:
                has_mid_cap = any(
                    not _is_sentence_start(tokens, idx)
                    for (t, idx) in seq
                    if t.text[0].isupper() and t.text.lower() not in _CONNECTORS
                )
                if has_mid_cap:
                    phrase = "".join(t.text_with_ws for (t, idx) in seq).strip()
                    if len(phrase) > 2:
                        entities.append(("PROPER", phrase))
            i = j
        else:
            i += 1

    # === QUOTED TEXT ===
    for m in re.finditer(r'"([^"]+)"', text):
        if len(m.group(1).strip()) > 2:
            entities.append(("QUOTED", m.group(1).strip()))
    for m in re.finditer(r"(?:^|[\s\(\[{,;])'([^']+)'(?=[\s\.,;:!?\)\]]|$)", text):
        if len(m.group(1).strip()) > 2:
            entities.append(("QUOTED", m.group(1).strip()))

    # === NOUN-NOUN COMPOUNDS ===
    for chunk in doc.noun_chunks:
        chunk_tokens = list(chunk)
        split_indices: list = []
        poss_splits: list = []
        for idx, tok in enumerate(chunk_tokens):
            if tok.dep_ == "case" and tok.text in {"'s", "\u2019s", "'"}:
                split_indices.append(idx)
                poss_splits.append(idx)
            elif tok.pos_ == "PUNCT" and tok.text in {"'", '"', "\u2018", "\u2019", "\u201c", "\u201d"}:
                split_indices.append(idx)

        if split_indices:
            groups: list = []
            prev = 0
            for split_idx in split_indices:
                if split_idx > prev:
                    groups.append(chunk_tokens[prev:split_idx])
                if split_idx in poss_splits:
                    next_split = next((s for s in split_indices if s > split_idx), None)
                    owned = chunk_tokens[split_idx + 1: next_split if next_split else len(chunk_tokens)]
                    if owned:
                        first_content = next((t for t in owned if t.pos_ not in {"PUNCT", "PART"}), None)
                        if not (first_content and first_content.text and first_content.text[0].isupper()):
                            prev = next_split if next_split else len(chunk_tokens)
                            continue
                prev = split_idx + 1
            if prev < len(chunk_tokens):
                groups.append(chunk_tokens[prev:])
        else:
            groups = [chunk_tokens]

        for group in groups:
            if not group:
                continue
            head = next((t for t in reversed(group) if t.pos_ in {"NOUN", "PROPN"}), None)
            if not head:
                continue
            head_generic = head.lemma_.lower() in lex["generic_heads"]
            content = [
                t
                for t in group
                if t.pos_ not in {"DET", "PRON", "PUNCT", "PART", "ADP", "SCONJ", "NUM"} and (t.pos_ == "ADJ" or not t.is_stop)
            ]
            if not content:
                continue

            compound_toks = [t for t in content if t.dep_ == "compound"]
            adj_toks = [t for t in content if t.pos_ == "ADJ" or t.dep_ == "amod"]
            has_spec_adj = any(t.lemma_.lower() not in lex["non_specific_adj"]
                               for t in adj_toks)
            if head_generic and not has_spec_adj and not compound_toks:
                continue

            if compound_toks:
                is_circ = any(t.lemma_.lower() in lex["circumstantial"]
                              for t in compound_toks)
                if is_circ:
                    val = head.lemma_ if head.pos_ == "NOUN" else head.text
                    if len(val) > 2:
                        entities.append(("NOUN", val))
                else:
                    filtered = _strip_generic_ending(
                        [t for t in content
                         if not (t.pos_ == "ADJ"
                                 and t.lemma_.lower() in lex["non_specific_adj"])],
                        lex["generic_endings"],
                    )
                    if filtered:
                        phrase = _lemmatize_compound(filtered)
                        if len(phrase) > 3 and " " in phrase:
                            entities.append(("COMPOUND", phrase))
            elif len(content) > 1 and has_spec_adj:
                filtered = _strip_generic_ending(
                    [t for t in content
                     if not ((t.pos_ == "ADJ" or t.dep_ == "amod")
                             and t.lemma_.lower() in lex["non_specific_adj"])],
                    lex["generic_endings"],
                )
                if filtered:
                    phrase = _lemmatize_compound(filtered)
                    if len(phrase) > 3 and " " in phrase:
                        entities.append(("COMPOUND", phrase))

    # === FALLBACK: Mis-tagged VERB heads ===
    processed = {e[1].lower() for e in entities if e[0] == "COMPOUND"}
    generic_verb_heads = lex["generic_heads"] | {"find", "buy", "purchase",
                                                 "sale", "deal", "trip", "visit"}

    def collect_compounds(head):
        return [t for t in doc if t.head == head and t.dep_ == "compound"]

    for tok in doc:
        if tok.pos_ == "VERB" and tok.dep_ in {"pobj", "dobj", "nsubj"}:
            comps = sorted(collect_compounds(tok), key=lambda t: t.i)
            if comps:
                # NEVER append the verb. `comps + [tok]` produced spans like
                # `Meridian faz` and `Mem0 foi concluída` — a name plus a
                # conjugated verb is not an entity under any reading. When
                # `comps` is a single token the `" " in phrase` guard below
                # drops it, which is the right outcome too.
                phrase_toks = comps
                phrase = " ".join(t.text for t in phrase_toks)
                if phrase.lower() not in processed and len(phrase) > 3 and " " in phrase:
                    entities.append(("COMPOUND", phrase))
                    processed.add(phrase.lower())

    # === DEDUPLICATION & CLEANUP ===
    seen: set = set()
    deduped = []
    for t, e in entities:
        k = e.lower().strip()
        if k not in seen and len(k) > 2:
            seen.add(k)
            deduped.append((t, e))

    cleaned: List[Tuple[str, str]] = []
    for etype, etext in deduped:
        txt = re.sub(r"^\*+\s*|\s*\*+$", "", etext.strip())
        txt = re.sub(r"\s*:+$", "", txt)
        txt = re.sub(r"^\d+\s*\.\s*", "", txt)
        if not txt or len(txt) <= 2 or _has_artifacts(txt):
            continue
        if etype == "PROPER" and " " not in txt and txt.lower() in lex["generic_caps"]:
            continue
        # `CONCLUÍDO`, `DECISÃO`, `REMOVIDO`: caixa alta de ÊNFASE virava PROPER
        # porque nada distinguia ênfase de sigla sem léxico da língua.
        if etype == "PROPER" and _e_enfase_em_caixa_alta(txt, lex):
            continue
        if len(txt.split()) > MAX_ENTITY_TOKENS or len(txt) > MAX_ENTITY_CHARS:
            continue
        cleaned.append((etype, txt))

    # Keep best type per entity (PROPER > COMPOUND > QUOTED > NOUN)
    type_pri = {"PROPER": 0, "COMPOUND": 1, "QUOTED": 2, "NOUN": 3, "VERB": 4}
    best: dict = {}
    for t, e in cleaned:
        k = e.lower()
        if k not in best or type_pri.get(t, 99) < type_pri.get(best[k][0], 99):
            best[k] = (t, e)
    deduped = list(best.values())

    # Remove entities that are whole-word substrings of longer entities.
    # Word-boundary anchoring avoids dropping distinct entities that only share a
    # leading substring (e.g. "Sam" must survive alongside "Samsung").
    # A PROPER is never suppressed. `Northwind` was deleted because
    # `Northwind in São Paulo` contained it — the short span is the one that
    # matches how anyone searches, and losing it cost the two memories a human
    # would call the answer their only entity link. Suppression stays for
    # COMPOUND/NOUN/QUOTED, where a longer span really does subsume a fragment.
    all_lower = [e[1].lower() for e in deduped]
    return [
        (t, e)
        for t, e in deduped
        if t == "PROPER"
        or not any(e.lower() != o and re.search(rf"\b{re.escape(e.lower())}\b", o)
                   for o in all_lower)
    ]
