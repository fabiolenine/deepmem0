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
import os
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
    "run", "runs", "rodada", "rodadas", "bloco", "blocos", "seção", "seções",
    "run", "runs", "rodada", "rodadas", "bloco", "blocos", "seção", "seções",
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
        "trailing_function_words": ("of", "the", "in", "and", "for", "at",
                                    "to", "with", "on", "by", "from", "as"),
        # `doc.ents` do modelo inglês é o que o upstream já tinha e nunca
        # consultou; ligá-lo em EN mudaria a coluna congelada sem defeito medido.
        "use_ner": False,
        "trust_pos": False,
        "identifiers": True,
        "connectors": _CONNECTORS,
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
        # Cauda funcional. O manifesto do N2 pegou o efeito colateral de filtrar
        # adjetivo não-específico dentro do compound: `Engenharia de Dados` vira
        # `Engenharia de`, `Head / Diretor de`, `Head de` — span terminado em
        # preposição, que o E-GOLD deixa passar (2 tokens, sem verbo, dentro dos
        # caps). Entidade não termina em preposição em nenhuma das duas línguas.
        "trailing_function_words": ("de", "da", "do", "das", "dos", "em", "no",
                                    "na", "nos", "nas", "para", "por", "com",
                                    "sem", "sob", "sobre", "entre", "e", "ou",
                                    "a", "o", "as", "os", "ao", "à", "/"),
        # Com `pt_core_news_sm` o POS finalmente vale: `viajei` sai VERB (era
        # PROPN), `tem` sai VERB (era NOUN). Só aqui as regras ancoradas em POS
        # deixam de ser inertes — e `doc.ents`, que o extrator NUNCA consultou
        # em 4 ramos artesanais, passa a ser a fonte primária de nome próprio.
        "use_ner": True,
        "trust_pos": True,
        "identifiers": True,
        # `de`/`da`/`do` são INTERNOS a nome próprio em português — `Rio de
        # Janeiro`, `Vitória da Conquista`. O N1 tirou preposição da whitelist
        # porque em inglês `in`/`at` COLAVAM entidades distintas
        # (`Northwind in São Paulo`); em português a preposição de genitivo faz
        # o oposto. `e` fica FORA: colaria `Qdrant e Ollama`.
        "connectors": _CONNECTORS | {"de", "da", "do", "das", "dos"},
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




# Rótulos de NER -> tipo interno. Nome de pessoa, lugar, organização e "misc"
# são todos NOME PRÓPRIO para efeito de entidade; o tipo interno só existe para
# a precedência de deduplicação (PROPER > COMPOUND > QUOTED > NOUN).
_NER_LABEL_TO_TYPE = {
    "PER": "PROPER", "PERSON": "PROPER",
    "LOC": "PROPER", "GPE": "PROPER", "FAC": "PROPER",
    "ORG": "PROPER",
    "MISC": "PROPER", "PRODUCT": "PROPER", "EVENT": "PROPER",
    "WORK_OF_ART": "PROPER", "LANGUAGE": "PROPER", "NORP": "PROPER",
}


# ============================================================
# IDENTIFICADORES TÉCNICOS (N4)
# ============================================================
# Classe descoberta pelo golden depois que o N3 destravou o resto: `num_ctx`,
# `llama-tiny3`, `embed-v2`, `linked_memory_ids`, `legacy_job_v2`, `Graph7` sumiam
# TODOS. O ramo de nome próprio exige inicial maiúscula, e o NER não os conhece.
#
# O POS não serve aqui, e isso é MEDIDO, não suposto — com o modelo português os
# mesmos identificadores saem `num_ctx/VERB`, `linked_memory_ids/ADJ`,
# `legacy_job_v2/ADJ`, `Graph7/NUM`. A forma serve: o tokenizador os mantém
# inteiros, e a marca é separador interno ou dígito.
#
# O DÍGITO OU O UNDERSCORE SÃO OBRIGATÓRIOS. Sem isso, `guarda-chuva` e
# `bem-vindo` entrariam como identificador — palavra hifenizada comum é
# exatamente o falso positivo que uma regra só de separador criaria.
_RE_IDENT = re.compile(
    r"""^(?:
          [A-Za-z][A-Za-z0-9]*(?:[_][A-Za-z0-9_]+)+     # num_ctx, linked_memory_ids
        | [A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*\d[A-Za-z0-9.\-]*  # embed-v2, Graph7, llama-tiny3
        )$""",
    re.X,
)


def _identificadores(doc) -> List[Tuple[str, str]]:
    """Tokens com forma de identificador, incluindo caminhos com barra.

    A junção por `/` varre a RUN inteira e aceita se QUALQUER parte tiver forma
    de identificador: `BAAI/bge-reranker-v2-m3` começa por `BAAI`, que sozinho é
    só uma palavra maiúscula — partir só de partes que já casam a forma deixava
    o prefixo de fora e devolvia as duas metades soltas.
    """
    achados: List[Tuple[str, str]] = []
    toks = list(doc)
    i = 0
    while i < len(toks):
        # extensão máxima de uma run colada por "/"
        j = i
        while (j + 2 < len(toks) and toks[j + 1].text == "/"
               and not toks[j].whitespace_ and not toks[j + 1].whitespace_
               and re.match(r"^[A-Za-z0-9][\w.\-]*$", toks[j + 2].text)):
            j += 2
        # `event_tie_band=0.002` chega como UM token, e o `=` reprovava a forma
        # inteira: o identificador sumia junto com o valor.
        partes = [toks[k].text.split("=", 1)[0] for k in range(i, j + 1, 2)]
        if any(_RE_IDENT.match(x) for x in partes):
            span = "/".join(partes)
            if len(span) > 2:
                achados.append(("PROPER", span))
            i = j + 1
            continue
        i += 1
    return achados

def _apara_cauda_funcional(frase: str, lex) -> str:
    """Remove palavras funcionais do FIM do span.

    Achado pelo manifesto do N2, não pelo golden: filtrar adjetivo não-específico
    dentro do compound deixava a preposição órfã (`Engenharia de`). O golden não
    via — dois tokens, sem verbo, dentro dos caps —, o manifesto viu porque
    compara span a span em texto real.
    """
    caudas = lex.get("trailing_function_words") or ()
    if not caudas:
        return frase
    partes = frase.split()
    while len(partes) > 1 and partes[-1].lower().strip(",;:") in caudas:
        partes.pop()
    return " ".join(partes)


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

    nlp = get_nlp_full(language)
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

    nlp = get_nlp_full(language)
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

    # === TECHNICAL IDENTIFIERS ===
    # Kill switch para ATRIBUIÇÃO: N3 (modelo/NER/POS) e N4 (identificadores)
    # entraram no mesmo commit, e sem isto o efeito de cada um não se separa —
    # exigência explícita do plano. Também serve de escape hatch se a heurística
    # de forma se mostrar ruidosa em algum corpus.
    if lex.get("identifiers") and os.environ.get(
            "MEM0_ENTITY_IDENTIFIERS", "").strip().lower() not in ("0", "off", "false"):
        entities.extend(_identificadores(doc))

    # === NAMED ENTITIES (doc.ents) ===
    # O extrator tinha 4 ramos artesanais e NUNCA consultava `doc.ents` — o
    # reconhecedor treinado do próprio modelo que ele já carregava. União em
    # nível de span, com precedência para o NER na sobreposição (a dedup por
    # tipo abaixo garante isso: PROPER vence COMPOUND para o mesmo texto).
    faixas_ner: List[Tuple[int, int]] = []
    if lex.get("use_ner"):
        confia = lex.get("trust_pos")
        for ent in getattr(doc, "ents", ()):
            tipo = _NER_LABEL_TO_TYPE.get(ent.label_)
            if not tipo or len(ent.text.strip()) <= 2:
                continue
            # O reconhecedor erra: em `o docker compose derruba` devolve
            # `compose derruba` como MISC. Entidade não contém verbo — e só com
            # o modelo da língua dá para saber que `derruba` é verbo.
            if confia and any(t.pos_ in {"VERB", "AUX"} for t in ent):
                continue
            entities.append((tipo, ent.text.strip()))
            faixas_ner.append((ent.start_char, ent.end_char))

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
                if (t.text and t.text[0].isupper()) or t.text.lower() in lex["connectors"]:
                    seq.append((t, j))
                    j += 1
                else:
                    break
            # Strip trailing function words
            while seq and seq[-1][0].text.lower() in lex["connectors"]:
                seq.pop()
            if seq:
                # A guarda `has_mid_cap` existe porque, sem POS confiável, uma
                # maiúscula de início de frase é indistinguível de nome próprio —
                # e em português com modelo inglês TODA palavra saía PROPN
                # (medido: `Ontem/PROPN eu/PROPN viajei/PROPN`). Com o modelo da
                # língua o POS vale, e `Northwind/PROPN encerrou/VERB` é legível:
                # início de frase deixa de ser motivo para descartar a entidade.
                # Era esta guarda que apagava a resposta que um humano daria.
                confia_pos = lex.get("trust_pos")
                has_mid_cap = any(
                    (not _is_sentence_start(tokens, idx)
                     or (confia_pos and t.pos_ == "PROPN"))
                    for (t, idx) in seq
                    if t.text[0].isupper() and t.text.lower() not in lex["connectors"]
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
            # VERB/AUX só entram na exclusão quando o POS é confiável: com o
            # modelo inglês sobre português o verbo sai NOUN/PROPN (`tem/NOUN`),
            # então filtrar por POS era inerte lá — e mexer nisso em inglês
            # moveria a coluna congelada do golden sem defeito medido.
            _fora = {"DET", "PRON", "PUNCT", "PART", "ADP", "SCONJ", "NUM"}
            if lex.get("trust_pos"):
                _fora = _fora | {"VERB", "AUX", "CCONJ", "ADV"}
            content = [
                t
                for t in group
                if t.pos_ not in _fora and (t.pos_ == "ADJ" or not t.is_stop)
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
                        phrase = _apara_cauda_funcional(
                            _lemmatize_compound(filtered), lex)
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
                    phrase = _apara_cauda_funcional(
                        _lemmatize_compound(filtered), lex)
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
        # Aspas e pontuação órfãs na borda. O manifesto do N3 as pegou em texto
        # real (`" Definitivo`, `" mem0`) — aspas desbalanceadas no corpus, que
        # nenhuma frase de teste isolada reproduz. Entidade não começa nem
        # termina em pontuação.
        txt = txt.strip(" \t\n\"'“”‘’«»(){}[]<>,;·—–-").strip()
        if not txt or len(txt) <= 2 or _has_artifacts(txt):
            continue
        if etype == "PROPER" and " " not in txt and txt.lower() in lex["generic_caps"]:
            continue
        # `CONCLUÍDO`, `DECISÃO`, `REMOVIDO`: caixa alta de ÊNFASE virava PROPER
        # porque nada distinguia ênfase de sigla sem léxico da língua.
        if etype == "PROPER" and _e_enfase_em_caixa_alta(txt, lex):
            continue
        # `Fase 7`, `Item 3`: palavra genérica + número é RÓTULO DE SEÇÃO. A
        # checagem de genérico só olhava span de UMA palavra, então bastava o
        # número ao lado para escapar dela.
        _p = txt.split()
        if (etype == "PROPER" and len(_p) == 2 and _p[1].isdigit()
                and _p[0].lower() in lex["generic_caps"]):
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
    sobreviventes = [
        (t, e)
        for t, e in deduped
        if t == "PROPER"
        or not any(e.lower() != o and re.search(rf"\b{re.escape(e.lower())}\b", o)
                   for o in all_lower)
    ]

    # EMBRULHO: span que contém outro span JÁ EMITIDO e acrescenta só
    # cola — preposição, conjunção ou uma palavra solta. `Northwind in São
    # Paulo` sobre `Northwind` + `São Paulo`; `legacy_job_v2 falhava` sobre
    # `legacy_job_v2`. O N1 provou que o span curto é o que as pessoas buscam;
    # aqui o longo deixa de ser emitido junto, em vez de apagar o curto.
    curtos = {e.lower() for t, e in sobreviventes}
    cola = set(lex.get("trailing_function_words") or ()) | {"in", "of", "the",
                                                            "at", "on", "for"}

    def _e_embrulho(texto: str) -> bool:
        baixo = texto.lower()
        palavras = baixo.split()
        if len(palavras) < 2:
            return False
        for outro in curtos:
            if outro == baixo or " " not in texto and outro not in baixo:
                continue
            if not re.search(rf"\b{re.escape(outro)}\b", baixo):
                continue
            resto = re.sub(rf"\b{re.escape(outro)}\b", " ", baixo).split()
            if resto and all(w in cola or w in curtos for w in resto):
                return True
            # `legacy_job_v2 falhava`: cabeça com FORMA de identificador mais uma
            # palavra solta. O gate é a forma, não "qualquer PROPER" — foi a
            # versão larga disso que comeu `Samsung phone`, compound legítimo.
            if (len(resto) == 1 and _RE_IDENT.match(outro)
                    and baixo.startswith(outro)):
                return True
        return False

    # ⚠️ APENAS onde o POS é confiável, e NUNCA sobre QUOTED. A 1ª versão tinha
    # uma cláusula "sobra exatamente 1 palavra" que comeu `Samsung phone` e
    # `The Great Gatsby` — compound legítimo e citação verbatim do usuário. Só
    # COLA (preposição, artigo, conjunção) autoriza descartar o span longo.
    if not lex.get("trust_pos"):
        return sobreviventes
    sobreviventes = [(t, e) for t, e in sobreviventes
                     if t == "QUOTED" or not _e_embrulho(e)]

    # PRECEDÊNCIA DO RECONHECEDOR NA SOBREPOSIÇÃO PARCIAL.
    # A dedup por tipo acima resolve texto IDÊNTICO, e o embrulho resolve
    # contenção — sobreposição parcial ficava sem regra. `Sao Paulo Guarulhos`
    # (ramo artesanal) contra `Sao Paulo` + `Guarulhos` (NER) só desaparecia por
    # acidente, porque a sobra caía em outra regra. O critério pede união em
    # nível de span com precedência do `doc.ents`, e precedência exige OFFSET.
    if faixas_ner:
        textos_ner = {t.lower() for _tp, t in entities[:len(faixas_ner)]}

        def _cruza_parcialmente(txt: str) -> bool:
            if txt.lower() in textos_ner:
                return False                     # é o próprio span do NER
            pos = text.find(txt)
            while pos >= 0:
                fim = pos + len(txt)
                for ini_e, fim_e in faixas_ner:
                    inter = min(fim, fim_e) - max(pos, ini_e)
                    if inter <= 0:
                        continue
                    contido = pos >= ini_e and fim <= fim_e
                    contem = ini_e >= pos and fim_e <= fim
                    if not contido and not contem:
                        return True              # cruza sem conter nem estar contido
                pos = text.find(txt, pos + 1)
            return False

        sobreviventes = [(t, e) for t, e in sobreviventes
                         if t == "QUOTED" or not _cruza_parcialmente(e)]
    return sobreviventes
