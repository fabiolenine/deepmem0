"""Projeção de leitura: `attributed_to` tem de CHEGAR ao chamador.

O defeito consertado aqui é um **valor de escrita morta**. `attributed_to` era
gravado no payload em todo `add` com `infer=True`, mas caía numa fresta:
estava em `core_and_promoted_keys` (o que o EXCLUI do balde `metadata`) e fora de
`promoted_payload_keys` (que é o que de fato COPIA para o resultado). Escrito
sempre, devolvido nunca.

MEDIDO no corpus de produção antes da correção: **1079 de 1218 memórias** (88,6%)
carregavam o campo — `user` 896, `assistant` 12, `document` 171 — e nenhum leitor
o devolvia.

⚠️ `document` NÃO está no vocabulário do contrato do prompt (`user|assistant`);
ele vem do prompt de documento do MCP. O vocabulário real tem 3 valores, e este
teste fixa isso em vez de fingir que são 2 — validar retroativamente invalidaria
171 memórias legítimas.

Há SEIS cópias da mesma lista (get/get_all/search × sync/async). Copiar-e-colar
deriva: por isso o primeiro teste é ESTÁTICO e quebra se qualquer sítio divergir,
e os demais são de COMPORTAMENTO, um por sítio. Um teste estático sozinho
provaria que as listas são iguais mas não que a promoção funciona; um teste de
comportamento sozinho passaria enquanto cinco sítios apodrecessem sem cobertura.
"""
import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mem0.memory.main import AsyncMemory, Memory

#: A ordem importa para o teste estático (é comparação de lista), e é a ordem em
#: que os campos aparecem no fonte.
ESPERADAS = ["user_id", "agent_id", "run_id", "actor_id", "attributed_to",
             "role", "memory_scope"]
SITIOS_ESPERADOS = 6


def _listas_promovidas_no_fonte():
    """Toda atribuição literal a `promoted_payload_keys` em main.py, via AST.

    Ler o FONTE em vez de importar é de propósito: as listas são locais dentro
    dos métodos, não constantes de módulo, então não há objeto para inspecionar
    em tempo de execução. O AST é o único jeito de ver as seis.
    """
    fonte = Path(inspect.getfile(Memory)).read_text()
    arvore = ast.parse(fonte)
    achadas = []
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Assign):
            continue
        alvos = [t.id for t in no.targets if isinstance(t, ast.Name)]
        if "promoted_payload_keys" not in alvos:
            continue
        if not isinstance(no.value, ast.List):
            achadas.append((no.lineno, None))  # forma inesperada: reprova adiante
            continue
        achadas.append((no.lineno, [e.value for e in no.value.elts
                                    if isinstance(e, ast.Constant)]))
    return achadas


class TestSeisSitiosNaoDerivam:
    """Guarda contra deriva entre as cópias.

    Não centralizo as listas numa constante de propósito: isso aumentaria o diff
    contra o upstream no arquivo mais rebaseado do fork. A guarda substitui a
    refatoração — se alguém editar cinco sítios e esquecer o sexto, quebra aqui.
    """

    def test_ha_exatamente_seis_sitios(self):
        achadas = _listas_promovidas_no_fonte()
        assert len(achadas) == SITIOS_ESPERADOS, (
            f"esperava {SITIOS_ESPERADOS} sítios de promoção, achei {len(achadas)} "
            f"nas linhas {[l for l, _ in achadas]}. Um sítio novo (ou removido) "
            f"muda o contrato de leitura e precisa de decisão explícita."
        )

    def test_todas_as_listas_sao_identicas(self):
        achadas = _listas_promovidas_no_fonte()
        distintas = {tuple(chaves) if chaves is not None else None
                     for _, chaves in achadas}
        assert len(distintas) == 1, (
            "os sítios de promoção DIVERGIRAM — get/get_all/search deixariam de "
            f"devolver o mesmo conjunto de campos. Variantes: {distintas}"
        )

    def test_o_conjunto_promovido_e_o_declarado(self):
        achadas = _listas_promovidas_no_fonte()
        for linha, chaves in achadas:
            assert chaves == ESPERADAS, f"linha {linha}: {chaves} != {ESPERADAS}"

    def test_attributed_to_esta_promovido(self):
        """O defeito específico: presente no fonte, ausente da promoção."""
        for linha, chaves in _listas_promovidas_no_fonte():
            assert "attributed_to" in (chaves or []), (
                f"linha {linha}: attributed_to voltou a ser escrita morta"
            )


def _ponto(payload, id_="mem-1"):
    p = MagicMock()
    p.id = id_
    p.payload = dict(payload)
    p.score = 0.9
    return p


BASE = {"data": "Prefere respostas técnicas", "hash": "h1", "user_id": "U",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00"}


def _memoria(mocker, cls):
    from tests.memory.test_main import _build_memory_instance
    return _build_memory_instance(mocker, cls)


#: As SEIS leituras públicas, como dados. Um só teste percorre todas: cobertura
#: por construção, e um sítio novo (ou removido) muda esta tabela, não some.
#: São os métodos PÚBLICOS de propósito — testar `_search_vector_store` provaria
#: a projeção e deixaria de fora a validação de escopo e o over-fetch que o
#: público envolve, que é justamente por onde um cliente real passa.
LEITURAS = ["get", "get_all", "search"]


async def _ler(mem, qual, ponto, assincrono):
    """Executa a leitura pública `qual` devolvendo um resultado do `ponto`."""
    if qual == "get":
        mem.vector_store.get.return_value = ponto
        r = mem.get("mem-1")
        return (await r) if assincrono else r
    if qual == "get_all":
        mem.vector_store.list.return_value = [[ponto]]
        r = mem.get_all(filters={"user_id": "U"}, top_k=10)
        saida = (await r) if assincrono else r
        return (saida["results"] if isinstance(saida, dict) else saida)[0]
    mem.vector_store.search.return_value = [ponto]
    r = mem.search("q", filters={"user_id": "U"}, top_k=10)
    saida = (await r) if assincrono else r
    return (saida["results"] if isinstance(saida, dict) else saida)[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("qual", LEITURAS)
@pytest.mark.parametrize("cls,assincrono", [(Memory, False), (AsyncMemory, True)],
                         ids=["sync", "async"])
@pytest.mark.parametrize("valor", ["user", "assistant", "document"])
async def test_projecao_publica_devolve_atribuicao(mocker, qual, cls, assincrono, valor):
    """Os 3 valores REAIS do vocabulário, nas 6 leituras públicas.

    `document` está aqui porque o corpus tem 171 memórias com ele e o contrato do
    prompt só menciona duas — um teste que cobrisse só `user`/`assistant`
    documentaria uma ficção.
    """
    mem = _memoria(mocker, cls)
    r = await _ler(mem, qual, _ponto({**BASE, "attributed_to": valor,
                                      "actor_id": "Maria"}), assincrono)
    assert r["attributed_to"] == valor
    assert r["actor_id"] == "Maria"
    # promovido ao topo E fora do balde `metadata` — nunca nos dois lugares,
    # senão clientes divergem conforme onde leiam.
    assert "attributed_to" not in (r.get("metadata") or {})
    assert "actor_id" not in (r.get("metadata") or {})


@pytest.mark.asyncio
@pytest.mark.parametrize("qual", LEITURAS)
@pytest.mark.parametrize("cls,assincrono", [(Memory, False), (AsyncMemory, True)],
                         ids=["sync", "async"])
async def test_projecao_publica_sem_atribuicao_nao_quebra(mocker, qual, cls, assincrono):
    """Ausência é estado VÁLIDO e permanente: 1218 memórias legadas sem
    `actor_id`, 139 sem `attributed_to`."""
    mem = _memoria(mocker, cls)
    r = await _ler(mem, qual, _ponto(BASE), assincrono)
    assert "attributed_to" not in r and "actor_id" not in r
    assert r["memory"] == BASE["data"]


class TestMemoriaSemOCampoNaoQuebra:
    """As 1218 memórias legadas não têm `actor_id`, e 139 não têm `attributed_to`.

    A promoção é guardada por `if key in payload`: ausência é estado VÁLIDO e
    permanente, não um caso a consertar depois. Sem esta guarda a leitura do
    corpus existente quebraria — é o contrário do que a mudança quer.
    """

    def test_get_sem_atribuicao(self, mocker):
        mem = _memoria(mocker, Memory)
        mem.vector_store.get.return_value = _ponto(BASE)
        r = mem.get("mem-1")
        assert "attributed_to" not in r
        assert "actor_id" not in r
        assert r["memory"] == BASE["data"]

    def test_search_sem_atribuicao(self, mocker):
        mem = _memoria(mocker, Memory)
        mem.vector_store.search.return_value = [_ponto(BASE)]
        r = mem._search_vector_store("q", filters={"user_id": "U"}, limit=10)
        assert "attributed_to" not in r[0]
        assert "actor_id" not in r[0]

    def test_campo_promovido_nao_vaza_para_metadata(self, mocker):
        """Promovido ao topo E fora do balde `metadata` — não nos dois lugares.

        Duplicar o valor faria clientes divergirem conforme onde lessem.
        """
        mem = _memoria(mocker, Memory)
        mem.vector_store.get.return_value = _ponto(
            {**BASE, "attributed_to": "user", "actor_id": "Maria", "importance": 0.8}
        )
        r = mem.get("mem-1")
        assert r["attributed_to"] == "user" and r["actor_id"] == "Maria"
        assert r["metadata"] == {"importance": 0.8}
