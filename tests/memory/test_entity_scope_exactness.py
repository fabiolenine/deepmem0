"""Escopo de entidade: casar por IGUALDADE, não por subconjunto.

Motivado por corrupção MEDIDA em produção (31/07/2026). O filtro do Qdrant casa
por SUBCONJUNTO: `{user_id: U}` casa qualquer linha com esse `user_id`,
**inclusive** as que também carregam `run_id`. Resultado no corpus real: a linha
`DeepMem0` do escopo de teste `{user_id: U, run_id: R}`
acumulou **12 vínculos de memórias do escopo largo** `{user_id: U}`, todas
gravadas pelo worker de produção — enquanto a linha certa, de escopo largo e 108
vínculos, ficava de fora.

O id determinístico JÁ era exato (`f({user_id})` != `f({user_id, run_id})`); o
lookup é que não era. Estes testes fixam a igualdade nos DOIS escritores:
`_upsert_entity` (caminho de update) e a Fase 7 (todo `add` com infer=True).

`test_scope_mismatch_is_not_a_hit` em `test_entity_identity.py` cobre um caso
DIFERENTE e mais fácil — valor de `user_id` divergente. Subconjunto é o caso que
passava.
"""
from unittest.mock import MagicMock, patch

import pytest

from mem0.memory.main import Memory
from mem0.memory.utils import entity_point_id, normalize_entity_key

LARGO = {"user_id": "U"}
ESTREITO = {"user_id": "U", "run_id": "R"}


def _linha(id_, data, escopo, vinculos):
    linha = MagicMock()
    linha.id = id_
    linha.score = 1.0
    linha.payload = {"data": data, "data_normalized": normalize_entity_key(data),
                     "linked_memory_ids": list(vinculos), **escopo}
    return linha


class TestUpsertEntityExigeEscopoIgual:
    """Caminho de `update()`/versionamento."""

    def _memory(self):
        mem = MagicMock(spec=Memory)
        mem.embedding_model = MagicMock()
        mem.embedding_model.embed.return_value = [0.1, 0.2]
        mem.entity_store = MagicMock()
        mem._entidade_por_chave = Memory._entidade_por_chave.__get__(mem)
        mem._reconcilia_vinculo = MagicMock()
        return mem

    def test_linha_de_escopo_estreito_nao_serve_a_escrita_larga(self):
        """O defeito exato que corrompeu o corpus.

        O store, filtrando por `{user_id}`, DEVOLVE a linha que também tem
        `run_id` — é assim que o Qdrant funciona. Quem tem de recusar é o
        escritor.
        """
        mem = self._memory()
        estreita = _linha("id-estreito", "DeepMem0", ESTREITO, ["m-antiga"])
        mem.entity_store.list.return_value = [estreita]
        mem.entity_store.search.return_value = []

        Memory._upsert_entity(mem, "DeepMem0", "PROPER", "m-nova", dict(LARGO))

        assert not mem.entity_store.update.called, (
            "vínculo do escopo largo foi parar na linha do escopo de teste — "
            "é a corrupção medida em 31/07")
        mem.entity_store.insert.assert_called_once()
        assert mem.entity_store.insert.call_args.kwargs["ids"] == [
            entity_point_id(LARGO, normalize_entity_key("DeepMem0"))]

    def test_linha_larga_nao_serve_a_escrita_estreita(self):
        """A direção inversa: uma escrita escopada não pode escrever na linha
        compartilhada, senão o escopo de teste polui o corpus real."""
        mem = self._memory()
        larga = _linha("id-largo", "DeepMem0", LARGO, ["m1", "m2"])
        mem.entity_store.list.return_value = [larga]
        mem.entity_store.search.return_value = []

        Memory._upsert_entity(mem, "DeepMem0", "PROPER", "m-teste", dict(ESTREITO))

        assert not mem.entity_store.update.called
        assert mem.entity_store.insert.call_args.kwargs["ids"] == [
            entity_point_id(ESTREITO, normalize_entity_key("DeepMem0"))]

    def test_sonda_vetorial_tambem_recusa_escopo_diferente(self):
        """A sonda tem o mesmo filtro por subconjunto — e devolve `score=1.0`
        porque o texto é idêntico. Sem a checagem, ela funde do mesmo jeito."""
        mem = self._memory()
        mem.entity_store.list.return_value = []
        estreita = _linha("id-estreito", "DeepMem0", ESTREITO, ["m-antiga"])
        del estreita.payload["data_normalized"]          # linha legada
        mem.entity_store.search.return_value = [estreita]

        Memory._upsert_entity(mem, "DeepMem0", "PROPER", "m-nova", dict(LARGO))

        assert not mem.entity_store.update.called
        mem.entity_store.insert.assert_called_once()

    def test_falha_do_lookup_nao_pode_virar_escrita_no_caminho_serial(self):
        """FAIL-CLOSED também aqui (BLOCKER 2 da r2).

        Havia um fallback para a sonda "por compatibilidade de backend" — mas a
        compatibilidade já está dentro de `entidades_por_chaves` (`_um`,
        igualdade simples). Este `except` só é alcançado por falha REAL do
        store, e sonda-e-insert converte falha de infra em escrita destrutiva.
        """
        mem = self._memory()
        mem.entity_store.list.side_effect = RuntimeError("Qdrant fora do ar")
        mem.entity_store.search.return_value = []

        Memory._upsert_entity(mem, "DeepMem0", "PROPER", "m1", dict(LARGO))

        assert not mem.entity_store.insert.called
        assert not mem.entity_store.update.called

    def test_escopo_igual_continua_casando(self):
        """Controle positivo: sem ele os testes acima passariam com um escritor
        que simplesmente nunca casa nada."""
        mem = self._memory()
        mem.entity_store.list.return_value = [
            _linha("id-largo", "DeepMem0", LARGO, ["m1"])]

        Memory._upsert_entity(mem, "DeepMem0", "PROPER", "m2", dict(LARGO))

        mem.entity_store.update.assert_called_once()
        assert not mem.entity_store.insert.called

    def test_multiplas_linhas_para_a_mesma_chave_nao_viram_escolha_silenciosa(self):
        """Duas linhas no MESMO escopo exato é o invariante de duplicata
        disparando. `top_k=1` pegava a primeira arbitrariamente."""
        mem = self._memory()
        mem.entity_store.list.return_value = [
            _linha("id-a", "DeepMem0", LARGO, ["m1"]),
            _linha("id-b", "DeepMem0", LARGO, ["m2"]),
        ]
        mem.entity_store.search.return_value = []

        Memory._upsert_entity(mem, "DeepMem0", "PROPER", "m3", dict(LARGO))

        # Inserir cria a TERCEIRA linha; atualizar é a escolha arbitrária que
        # este trabalho remove. As DUAS asserções são necessárias: só a de
        # insert passava mesmo com o comportamento antigo (que atualizava a
        # primeira em silêncio) — guarda sem potência, pega na mutação M5.
        assert not mem.entity_store.insert.called, (
            "com duplicata detectada, inserir cria a TERCEIRA linha")
        assert not mem.entity_store.update.called, (
            "escolheu uma das duplicatas em silêncio — é exatamente o "
            "`top_k=1` sobre filtro que casa 2 linhas")


class TestGemeoAsyncTemAsMesmasGuardas:
    """O assíncrono já divergiu do síncrono nesta decisão exata uma vez.

    A decisão mora em `resolver_linha_de_entidade`, função de módulo que os dois
    chamam — mas "é espelho" é afirmação, não prova. Estes testes exercitam o
    caminho async de verdade.
    """

    def _memory(self):
        from mem0.memory.main import AsyncMemory
        mem = MagicMock(spec=AsyncMemory)
        mem.embedding_model = MagicMock()
        mem.embedding_model.embed.return_value = [0.1, 0.2]
        mem.entity_store = MagicMock()
        mem._reconcilia_vinculo = MagicMock()
        return mem

    def _roda(self, mem, escopo):
        import asyncio as _aio

        from mem0.memory.main import AsyncMemory
        _aio.run(AsyncMemory._upsert_entity_async(
            mem, "DeepMem0", "PROPER", "m-nova", dict(escopo)))

    def test_async_recusa_linha_de_escopo_estreito(self):
        mem = self._memory()
        mem.entity_store.list.return_value = [
            _linha("id-estreito", "DeepMem0", ESTREITO, ["m-antiga"])]
        mem.entity_store.search.return_value = []

        self._roda(mem, LARGO)

        assert not mem.entity_store.update.called
        mem.entity_store.insert.assert_called_once()

    def test_async_fecha_na_falha_de_lookup(self):
        mem = self._memory()
        mem.entity_store.list.side_effect = RuntimeError("Qdrant fora do ar")
        mem.entity_store.search.return_value = []

        self._roda(mem, LARGO)

        assert not mem.entity_store.insert.called
        assert not mem.entity_store.update.called

    def test_async_pula_chave_ambigua(self):
        mem = self._memory()
        mem.entity_store.list.return_value = [
            _linha("id-a", "DeepMem0", LARGO, ["m1"]),
            _linha("id-b", "DeepMem0", LARGO, ["m2"]),
        ]
        mem.entity_store.search.return_value = []

        self._roda(mem, LARGO)

        assert not mem.entity_store.insert.called
        assert not mem.entity_store.update.called

    def test_async_controle_positivo_escopo_igual_casa(self):
        """Sem ele, os três acima passariam com um escritor que nunca escreve."""
        mem = self._memory()
        mem.entity_store.list.return_value = [
            _linha("id-largo", "DeepMem0", LARGO, ["m1"])]

        self._roda(mem, LARGO)

        mem.entity_store.update.assert_called_once()
        assert not mem.entity_store.insert.called


class TestFase7Async:
    """A OUTRA metade do gêmeo assíncrono.

    `TestGemeoAsyncTemAsMesmasGuardas` cobre `_upsert_entity_async`; a Fase 7
    async (`AsyncMemory._add_to_vector_store`) ficou sem teste, e "é espelho" é
    afirmação, não prova — foi exatamente o argumento com que esta classe irmã
    nasceu.
    """

    def _mem(self, retorno_do_probe):
        from mem0.memory.main import AsyncMemory
        embedder = MagicMock()
        embedder.return_value.embed.return_value = [0.1, 0.2, 0.3]
        vstore = MagicMock()
        vstore.return_value.search.return_value = []
        es = MagicMock()
        es.search_batch.return_value = [retorno_do_probe or []]
        es.list.return_value = []
        with patch("mem0.utils.factory.EmbedderFactory.create", embedder), \
             patch("mem0.utils.factory.VectorStoreFactory.create",
                   side_effect=[vstore.return_value, es]), \
             patch("mem0.utils.factory.LlmFactory.create", MagicMock()), \
             patch("mem0.memory.storage.SQLiteManager", MagicMock()):
            mem = AsyncMemory()
            mem.custom_instructions = None
            mem.db.get_last_messages = MagicMock(return_value=[])
            mem.db.save_messages = MagicMock()
            mem.db.add_history = MagicMock()
            mem.embedding_model.embed_batch = MagicMock(
                side_effect=lambda t, *a, **k: [[0.1, 0.2, 0.3] for _ in t])
            mem.llm.generate_response = MagicMock(
                return_value='{"memory": [{"text": "o projeto DeepMem0 usa Qdrant"}]}')
            mem._entity_store = es
            return mem, es

    def _roda(self, mem, store, escopo):
        import asyncio as _aio
        with patch("mem0.memory.main.extract_entities_batch",
                   return_value=[[("PROPER", "DeepMem0")]]) as ee:
            out = _aio.run(mem._add_to_vector_store(
                [{"role": "user", "content": "o projeto DeepMem0 usa Qdrant"}],
                {}, dict(escopo), True))
        assert out, "o add async não criou memória — a Fase 7 async não rodou"
        assert ee.called
        assert store.search_batch.called or store.list.called
        return out

    def test_async_probe_nao_funde_linha_de_outro_escopo(self):
        estreita = _linha("id-estreito", "DeepMem0", ESTREITO, ["m-antiga"])
        mem, store = self._mem([estreita])
        self._roda(mem, store, LARGO)
        assert not store.update.called
        assert store.insert.call_args.kwargs["ids"] == [
            entity_point_id(LARGO, normalize_entity_key("DeepMem0"))]

    def test_async_chave_ambigua_e_pulada(self):
        mem, store = self._mem([])
        store.list.return_value = [
            _linha("id-a", "DeepMem0", LARGO, ["m1"]),
            _linha("id-b", "DeepMem0", LARGO, ["m2"]),
        ]
        self._roda(mem, store, LARGO)
        assert not store.insert.called
        assert not store.update.called

    def test_async_falha_de_lookup_fecha(self):
        mem, store = self._mem([])
        store.list.side_effect = RuntimeError("Qdrant fora do ar")
        self._roda(mem, store, LARGO)
        assert not store.insert.called
        assert not store.update.called

    def test_async_lookup_truncado_fecha(self):
        mem, store = self._mem([])
        store.list.return_value = [
            _linha(f"id-{i}", "DeepMem0", ESTREITO, [f"m{i}"]) for i in range(64)]
        self._roda(mem, store, LARGO)
        assert not store.insert.called
        assert not store.update.called

    def test_async_search_batch_com_entrada_nao_lista_fecha(self):
        mem, store = self._mem([])
        store.search_batch.return_value = [MagicMock()]
        self._roda(mem, store, LARGO)
        assert not store.insert.called
        assert not store.update.called

    def test_async_insert_em_lote_espera_a_escrita(self):
        mem, store = self._mem([])
        self._roda(mem, store, LARGO)
        store.insert.assert_called()
        assert store.insert.call_args.kwargs.get("wait") is True, (
            "insert de entidade async sem wait=True")


class TestEscopoAusenteNaoEAcerto:
    """A direção B: payload SEM a chave que a escrita tem.

    Duas direções, e só uma é a corrupção medida:
      A) payload tem chave que a escrita NÃO tem (`run_id=teste` vs escrita
         sem `run_id`) — é o que pôs 12 vínculos largos na linha estreita;
      B) payload NÃO tem chave que a escrita tem (linha sem `user_id`).

    B só acontece se o store IGNORAR o filtro — um `MatchValue(user_id=u)` não
    devolve linha sem `user_id`, e MEDIDO em 31/07/2026 há 0 dessas em 6155.
    Fica estrito nas duas assim mesmo: o id determinístico deriva do escopo
    INTEIRO, então casar B faria o lookup achar uma linha cujo id não é o que
    este escritor usaria para inserir — achar e escrever divergiriam de novo,
    que é o defeito de fundo deste trabalho.

    Este teste existe porque a alternativa era deixar a decisão implícita nos
    fixtures de quatro testes pré-existentes.
    """

    def _memory(self):
        mem = MagicMock(spec=Memory)
        mem.embedding_model = MagicMock()
        mem.embedding_model.embed.return_value = [0.1, 0.2]
        mem.entity_store = MagicMock()
        mem._entidade_por_chave = Memory._entidade_por_chave.__get__(mem)
        mem._reconcilia_vinculo = MagicMock()
        return mem

    def test_linha_sem_escopo_nao_casa_escrita_escopada(self):
        mem = self._memory()
        sem_escopo = MagicMock()
        sem_escopo.id = "id-sem-escopo"
        sem_escopo.score = 1.0
        sem_escopo.payload = {"data": "DeepMem0", "data_normalized": "deepmem0",
                              "linked_memory_ids": ["m1"]}
        mem.entity_store.list.return_value = [sem_escopo]
        mem.entity_store.search.return_value = []

        Memory._upsert_entity(mem, "DeepMem0", "PROPER", "m2", dict(LARGO))

        assert not mem.entity_store.update.called
        assert mem.entity_store.insert.call_args.kwargs["ids"] == [
            entity_point_id(LARGO, normalize_entity_key("DeepMem0"))], (
            "o insert usa o id derivado do escopo INTEIRO; casar uma linha de "
            "escopo diferente faria achar e escrever divergirem")


class TestFase7 :
    """Caminho de `add()` com infer=True — o mais quente que existe."""

    def _mem_com_fase7(self, retorno_do_probe):
        """Memory mockado que chega até a Fase 7.

        ⚠️ A resposta do LLM tem de vir na chave `memory` — é UMA chamada, não
        duas. Com a chave errada `extracted_memories` fica vazio, o add retorna
        cedo e a Fase 7 NUNCA roda: os testes passariam vácuos. Foi o que
        aconteceu na primeira versão deste arquivo, e é por isso que `_roda`
        devolve o veredito de "a fase rodou".
        """
        embedder = MagicMock()
        embedder.return_value.embed.return_value = [0.1, 0.2, 0.3]
        vstore = MagicMock()
        vstore.return_value.search.return_value = []
        entity_store = MagicMock()
        entity_store.search_batch.return_value = [retorno_do_probe or []]
        entity_store.list.return_value = []
        llm = MagicMock()
        with patch("mem0.utils.factory.EmbedderFactory.create", embedder), \
             patch("mem0.utils.factory.VectorStoreFactory.create",
                   side_effect=[vstore.return_value, entity_store]), \
             patch("mem0.utils.factory.LlmFactory.create", llm), \
             patch("mem0.memory.storage.SQLiteManager", MagicMock()):
            mem = Memory()
            mem.custom_instructions = None
            mem.db.get_last_messages = MagicMock(return_value=[])
            mem.db.save_messages = MagicMock()
            mem.db.add_history = MagicMock()
            mem.embedding_model.embed_batch = MagicMock(
                side_effect=lambda texts, *a, **k: [[0.1, 0.2, 0.3] for _ in texts])
            mem.llm.generate_response = MagicMock(
                return_value='{"memory": [{"text": "o projeto DeepMem0 usa Qdrant"}]}')
            mem._entity_store = entity_store
            return mem, entity_store

    def _roda(self, mem, store, escopo):
        with patch("mem0.memory.main.extract_entities_batch",
                   return_value=[[("PROPER", "DeepMem0")]]) as ee:
            out = mem._add_to_vector_store(
                [{"role": "user", "content": "o projeto DeepMem0 usa Qdrant"}],
                {}, dict(escopo), True)
        # ANTI-VÁCUO: sem isto, toda asserção de "não fez X" passa quando a fase
        # simplesmente não executou.
        assert out, "o add não criou memória nenhuma — a Fase 7 não rodou"
        assert ee.called, "extract_entities_batch não foi chamado"
        assert store.search_batch.called or store.list.called, (
            "a Fase 7 não consultou o entity store")
        return out

    def test_probe_nao_pode_fundir_linha_de_outro_escopo(self):
        """A sonda devolve a linha estreita com `score=1.0` (texto idêntico).
        Quem tem de recusar é o escritor."""
        estreita = _linha("id-estreito", "DeepMem0", ESTREITO, ["m-antiga"])
        mem, store = self._mem_com_fase7([estreita])
        self._roda(mem, store, LARGO)

        assert not store.update.called, (
            "a Fase 7 fundiu vínculo largo numa linha de escopo estreito — "
            "é a corrupção medida em 31/07")
        store.insert.assert_called()
        assert store.insert.call_args.kwargs["ids"] == [
            entity_point_id(LARGO, normalize_entity_key("DeepMem0"))]

    def test_clobber_probe_que_erra_nao_pode_apagar_vinculos_alheios(self):
        """A guarda de clobber (G4).

        A sonda erra (score 0.80) sobre uma linha que EXISTE no id determinístico.
        A Fase 7 então insere nesse MESMO id — e `insert` substitui o payload
        inteiro, apagando os vínculos das outras memórias.
        """
        fraco = MagicMock()
        fraco.score = 0.80
        fraco.id = "outra-coisa"
        fraco.payload = {"data": "Outra"}
        mem, store = self._mem_com_fase7([fraco])
        # a linha real existe no escopo largo, com vínculos de OUTRAS memórias
        existente = _linha(
            entity_point_id(LARGO, normalize_entity_key("DeepMem0")),
            "DeepMem0", LARGO, ["m-alheia-1", "m-alheia-2"])
        store.list.return_value = [existente]

        self._roda(mem, store, LARGO)

        if store.insert.called:
            for p in (store.insert.call_args.kwargs.get("payloads") or []):
                if p.get("data_normalized") == "deepmem0":
                    pytest.fail(
                        "insert no id determinístico SUBSTITUI o payload e apaga "
                        f"m-alheia-1/2; payload novo = {p.get('linked_memory_ids')}")
        assert store.update.called, "a linha existente tinha de ser atualizada"
        vinculos = store.update.call_args.kwargs["payload"]["linked_memory_ids"]
        assert "m-alheia-1" in vinculos and "m-alheia-2" in vinculos

    def test_falha_de_lookup_nao_pode_virar_escrita(self):
        """FAIL-CLOSED (BLOCKER 2).

        Se o lookup exato falha, "trata tudo como novo" insere no id
        determinístico — e `insert` SUBSTITUI o payload, apagando os vínculos de
        quem já estava lá. Falha de infraestrutura não pode virar escrita
        destrutiva: a Fase 7 aborta e o caminho serial reconcilia depois.
        """
        mem, store = self._mem_com_fase7([])
        store.list.side_effect = RuntimeError("Qdrant fora do ar")

        self._roda(mem, store, LARGO)

        assert not store.insert.called, (
            "lookup falhou e a Fase 7 inseriu assim mesmo — é o caminho que "
            "apaga vínculo alheio")
        assert not store.update.called

    def test_chave_ambigua_nao_e_processada_pela_fase7(self):
        """A guarda de multiplicidade valia só no caminho serial.

        Chave ambígua era removida de `encontradas` e caía em `faltantes` —
        ia para a sonda e acabava atualizando uma das duplicatas, ou inserindo
        no id determinístico e apagando payload alheio.
        """
        mem, store = self._mem_com_fase7([])
        store.list.return_value = [
            _linha("id-a", "DeepMem0", LARGO, ["m1"]),
            _linha("id-b", "DeepMem0", LARGO, ["m2"]),
        ]
        self._roda(mem, store, LARGO)

        assert not store.insert.called, "inserir cria a TERCEIRA linha"
        assert not store.update.called, "escolheu uma das duplicatas em silêncio"

    def test_lookup_truncado_nao_pode_virar_escrita(self):
        """`top_k` é consumido ANTES da validação de escopo em Python.

        O filtro traz também linhas de escopo SUBCONJUNTO; com o corte cheio a
        linha EXATA pode ter ficado de fora, e lê-la como ausente insere no id
        determinístico. Saturou = não sei responder.
        """
        mem, store = self._mem_com_fase7([])
        # satura qualquer limite plausível com linhas de OUTRO escopo
        store.list.return_value = [
            _linha(f"id-{i}", "DeepMem0", ESTREITO, [f"m{i}"]) for i in range(64)]
        self._roda(mem, store, LARGO)

        assert not store.insert.called, (
            "afirmou ausência sobre um retorno truncado e inseriu")
        assert not store.update.called

    def test_search_batch_com_entrada_que_nao_e_lista_nao_pode_virar_escrita(self):
        """A forma perigosa é esta, não a lista curta.

        Lista curta estoura `IndexError` e a fase aborta por acidente — a
        mutação que desliga a guarda passava assim mesmo. Já uma entrada que
        não é lista é ITERÁVEL, rende zero matches, e portanto INSERE no id
        determinístico em silêncio.
        """
        mem, store = self._mem_com_fase7([])
        store.search_batch.return_value = [MagicMock()]   # 1 entrada, não-lista
        self._roda(mem, store, LARGO)

        assert not store.insert.called, (
            "entrada malformada virou 'não achou' e inseriu sobre estado "
            "desconhecido")
        assert not store.update.called

    def test_search_batch_curto_tambem_fecha(self):
        """Contagem errada: hoje fecharia por IndexError; a guarda torna o
        motivo explícito em vez de acidental."""
        mem, store = self._mem_com_fase7([])
        store.search_batch.return_value = []      # 0 entradas para 1 consulta
        self._roda(mem, store, LARGO)

        assert not store.insert.called
        assert not store.update.called

    def test_insert_em_lote_espera_a_escrita(self):
        """`wait=True` — o default do Qdrant NÃO espera, e escrita confirmada
        mas invisível foi o que causou a recriação e o clobber em 30/07."""
        mem, store = self._mem_com_fase7([])
        self._roda(mem, store, LARGO)

        store.insert.assert_called()
        assert store.insert.call_args.kwargs.get("wait") is True, (
            "insert de entidade sem wait=True")
