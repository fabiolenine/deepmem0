"""Entity identity: the normalized key and the deterministic point id.

Identity used to be vector SIMILARITY (>= 0.95), which is probabilistic. A real
corpus paid for it: `FASE`/`Fase`, `docker compose`/`Docker Compose` and
`Hilbert transform`/`Hilbert Transform` became SEPARATE rows, each holding a
slice of the links — so which one receives the boost depended on the spelling the
user happened to type.
"""
from unittest.mock import MagicMock

import pytest

from mem0.memory.utils import (
    ENTITY_ID_NAMESPACE,
    entity_point_id,
    normalize_entity_key,
)


class TestNormalizedKey:
    @pytest.mark.parametrize("a,b", [
        ("FASE", "Fase"),
        ("docker compose", "Docker Compose"),
        ("Hilbert transform", "Hilbert Transform"),
        ("  São  Paulo ", "são paulo"),
    ])
    def test_case_and_spacing_collapse(self, a, b):
        assert normalize_entity_key(a) == normalize_entity_key(b)

    @pytest.mark.parametrize("a,b", [
        ("num_ctx", "num ctx"),
        ("bge-m3", "bge m3"),
        ("linked_memory_ids", "linked memory ids"),
    ])
    def test_separators_are_NOT_collapsed(self, a, b):
        """In a technical corpus `num_ctx` and `num ctx` are different things;
        merging them would trade a duplicate for a collision."""
        assert normalize_entity_key(a) != normalize_entity_key(b)

    def test_empty_and_none(self):
        assert normalize_entity_key(None) == ""
        assert normalize_entity_key("   ") == ""


class TestDeterministicId:
    def test_same_key_same_id(self):
        scope = {"user_id": "u"}
        assert (entity_point_id(scope, normalize_entity_key("Fase"))
                == entity_point_id(scope, normalize_entity_key("FASE")))

    def test_scope_changes_the_id(self):
        chave = normalize_entity_key("Fase")
        assert (entity_point_id({"user_id": "a"}, chave)
                != entity_point_id({"user_id": "b"}, chave))

    def test_agent_and_run_participate(self):
        chave = normalize_entity_key("X1")
        base = {"user_id": "u"}
        assert entity_point_id(base, chave) != entity_point_id(
            {**base, "agent_id": "ag"}, chave)
        assert entity_point_id(base, chave) != entity_point_id(
            {**base, "run_id": "r"}, chave)

    def test_missing_scope_keys_are_stable(self):
        """`{}` and `{user_id: None}` must not produce different rows."""
        chave = normalize_entity_key("X1")
        assert entity_point_id({}, chave) == entity_point_id({"user_id": None},
                                                             chave)

    def test_is_a_uuid5_of_the_pinned_namespace(self):
        import uuid
        chave = normalize_entity_key("Fase")
        esperado = uuid.uuid5(
            ENTITY_ID_NAMESPACE, f"user_id=u|agent_id=|run_id=|{chave}")
        assert entity_point_id({"user_id": "u"}, chave) == str(esperado)


class TestWriterUsesTheKeyFirst:
    """The exact lookup must come BEFORE the vector probe, and a legacy row must
    gain `data_normalized` the first time it is touched — otherwise it never
    enters the exact lookup and the case duplicate keeps being born next to it.
    """

    def _memory(self):
        from mem0.memory.main import Memory

        mem = MagicMock(spec=Memory)
        mem.embedding_model = MagicMock()
        mem.embedding_model.embed.return_value = [0.1, 0.2]
        mem.entity_store = MagicMock()
        mem._entidade_por_chave = Memory._entidade_por_chave.__get__(mem)
        mem._reconcilia_vinculo = MagicMock()
        return mem

    def test_exact_hit_skips_the_vector_probe(self):
        from mem0.memory.main import Memory

        mem = self._memory()
        linha = MagicMock()
        linha.id = "id-existente"
        # o payload de produção carrega o escopo (`**search_filters` no insert);
        # omiti-lo aqui fazia a validação de escopo recusar o acerto — e essa
        # validação existe porque um filtro IGNORADO pelo store devolveria a
        # primeira linha qualquer, que seria fundida com esta.
        linha.payload = {"data": "Fase", "data_normalized": "fase",
                         "user_id": "u", "linked_memory_ids": ["m1"]}
        mem.entity_store.list.return_value = [linha]

        Memory._upsert_entity(mem, "FASE", "PROPER", "m2", {"user_id": "u"})

        mem.entity_store.search.assert_not_called()
        mem.entity_store.insert.assert_not_called()
        payload = mem.entity_store.update.call_args.kwargs["payload"]
        assert payload["linked_memory_ids"] == ["m1", "m2"]

    def test_legacy_row_gains_the_key_when_touched(self):
        from mem0.memory.main import Memory

        mem = self._memory()
        mem.entity_store.list.return_value = []
        legado = MagicMock()
        legado.id = "id-legado"
        legado.score = 0.99
        legado.payload = {"data": "Fase", "linked_memory_ids": ["m1"]}
        mem.entity_store.search.return_value = [legado]

        Memory._upsert_entity(mem, "Fase", "PROPER", "m1", {"user_id": "u"})

        payload = mem.entity_store.update.call_args.kwargs["payload"]
        assert payload["data_normalized"] == "fase", (
            "linha legada sem a chave nunca entra no lookup exato")

    def test_new_row_uses_the_deterministic_id_and_reconciles(self):
        from mem0.memory.main import Memory

        mem = self._memory()
        mem.entity_store.list.return_value = []
        mem.entity_store.search.return_value = []

        Memory._upsert_entity(mem, "Fase", "PROPER", "m1", {"user_id": "u"})

        ids = mem.entity_store.insert.call_args.kwargs["ids"]
        assert ids == [entity_point_id({"user_id": "u"},
                                       normalize_entity_key("Fase"))]
        payload = mem.entity_store.insert.call_args.kwargs["payloads"][0]
        assert payload["data_normalized"] == "fase"
        mem._reconcilia_vinculo.assert_called_once()

    def test_probe_below_threshold_still_inserts(self):
        from mem0.memory.main import Memory

        mem = self._memory()
        mem.entity_store.list.return_value = []
        fraco = MagicMock()
        fraco.score = 0.80
        fraco.payload = {"data": "Outra"}
        mem.entity_store.search.return_value = [fraco]

        Memory._upsert_entity(mem, "Fase", "PROPER", "m1", {"user_id": "u"})
        mem.entity_store.insert.assert_called_once()

    def test_scope_mismatch_is_not_a_hit(self):
        """Filtro ignorado pelo store devolveria linha de OUTRO escopo, e fundi-la
        misturaria memória de usuários diferentes."""
        from mem0.memory.main import Memory

        mem = self._memory()
        outro = MagicMock()
        outro.id = "id-outro-escopo"
        outro.payload = {"data": "Fase", "data_normalized": "fase",
                         "user_id": "OUTRO", "linked_memory_ids": ["mX"]}
        mem.entity_store.list.return_value = [outro]
        mem.entity_store.search.return_value = []

        Memory._upsert_entity(mem, "Fase", "PROPER", "m1", {"user_id": "u"})
        mem.entity_store.insert.assert_called_once()

    def test_malformed_store_result_is_not_a_hit(self):
        """`list()` de um store que mude o tipo de retorno é truthy; aceitar
        truthy como acerto faria o caminho exato 'encontrar' registro
        inexistente e pular a sonda vetorial."""
        from mem0.memory.main import Memory

        for lixo in ("nao e lista", 42, [object()], [None]):
            mem = self._memory()
            mem.entity_store.list.return_value = lixo
            mem.entity_store.search.return_value = []
            Memory._upsert_entity(mem, "Fase", "PROPER", "m1", {"user_id": "u"})
            mem.entity_store.insert.assert_called_once()


class TestCleanupVerdict:
    """`unlink_memory_from_entity_rows` devolve o veredito de COMPLETUDE.

    O chamador precisa dessa resposta: comitar a intenção de delete depois de uma
    limpeza que falhou é o que transforma erro transitório em vínculo pendente
    PERMANENTE — a reconciliação não tem mais o que repetir.

    O caso do scan truncado é o que nenhum smoke contra Qdrant real alcança sem
    um corpus de 100k entidades, e é justamente onde o silêncio custaria caro.
    """

    def test_truncated_scan_returns_false(self):
        from unittest.mock import MagicMock

        from mem0.memory.main import ENTITY_SCAN_TOP_K, unlink_memory_from_entity_rows

        store = MagicMock()
        # o store devolve EXATAMENTE o teto -> pode haver mais linhas invisíveis
        linhas = []
        for i in range(ENTITY_SCAN_TOP_K):
            r = MagicMock()
            r.id, r.payload = f"e{i}", {"data": "x", "linked_memory_ids": ["m1"]}
            linhas.append(r)
        store.list.return_value = [linhas]

        ok = unlink_memory_from_entity_rows(store, "m1", {"user_id": "u"})
        assert ok is False, ("scan no teto pode ter deixado linha de fora; dizer "
                            "'completo' aí é mentir para o chamador")

    def test_complete_scan_returns_true(self):
        from unittest.mock import MagicMock

        from mem0.memory.main import unlink_memory_from_entity_rows

        store = MagicMock()
        r = MagicMock()
        r.id, r.payload = "e1", {"data": "x", "linked_memory_ids": ["m1", "m2"]}
        store.list.return_value = [[r]]

        assert unlink_memory_from_entity_rows(store, "m1", {"user_id": "u"}) is True
        payload = store.update.call_args.kwargs["payload"]
        assert payload["linked_memory_ids"] == ["m2"]

    def test_row_that_fails_to_update_returns_false(self):
        from unittest.mock import MagicMock

        from mem0.memory.main import unlink_memory_from_entity_rows

        store = MagicMock()
        r = MagicMock()
        r.id, r.payload = "e1", {"data": "x", "linked_memory_ids": ["m1", "m2"]}
        store.list.return_value = [[r]]
        store.update.side_effect = RuntimeError("qdrant fora do ar")

        assert unlink_memory_from_entity_rows(store, "m1", {"user_id": "u"}) is False


class TestDeleteObserver:
    """`delete_observer` expõe o que SÓ o core enxerga.

    Instrumentar isto de fora era impossível sem reimplementar a função: `rows` e
    `truncated` são LOCAIS de `_scan_entity_rows`, e envolver `Memory.delete`
    público não os expõe. `truncated` é o dado que mais importa — é o único sinal
    de que a limpeza pode ter deixado vínculo para trás.
    """

    def _store(self, n_linhas=1):
        from unittest.mock import MagicMock
        store = MagicMock()
        linhas = []
        for i in range(n_linhas):
            r = MagicMock()
            r.id, r.payload = f"e{i}", {"data": "x", "linked_memory_ids": ["m1", "m2"]}
            linhas.append(r)
        store.list.return_value = [linhas]
        return store

    def test_emits_metrics_the_caller_cannot_see(self, monkeypatch):
        from mem0.memory import main as M

        vistos = []
        monkeypatch.setattr(M, "delete_observer",
                            lambda mid, fase, met: vistos.append((mid, fase, met)))
        M.unlink_memory_from_entity_rows(self._store(3), "m1", {"user_id": "u"})

        assert len(vistos) == 1
        mid, fase, met = vistos[0]
        assert mid == "m1" and fase == "entity_cleanup"
        assert met["rows_scanned"] == 3 and met["rows_touched"] == 3
        assert met["truncated"] is False and met["complete"] is True
        assert met["elapsed_ms"] >= 0 and met["scope"] == {"user_id": "u"}

    def test_truncation_is_visible_to_the_observer(self, monkeypatch):
        from mem0.memory import main as M

        vistos = []
        monkeypatch.setattr(M, "delete_observer",
                            lambda mid, fase, met: vistos.append(met))
        M.unlink_memory_from_entity_rows(
            self._store(M.ENTITY_SCAN_TOP_K), "m1", {"user_id": "u"})

        assert vistos[0]["truncated"] is True
        assert vistos[0]["complete"] is False

    def test_observer_that_raises_never_breaks_the_delete(self, monkeypatch):
        """Observabilidade não pode derrubar o caminho de delete."""
        from mem0.memory import main as M

        def _explode(*a, **k):
            raise RuntimeError("coletor fora do ar")

        monkeypatch.setattr(M, "delete_observer", _explode)
        assert M.unlink_memory_from_entity_rows(
            self._store(1), "m1", {"user_id": "u"}) is True

    def test_no_observer_is_a_noop(self, monkeypatch):
        from mem0.memory import main as M

        monkeypatch.setattr(M, "delete_observer", None)
        assert M.unlink_memory_from_entity_rows(
            self._store(1), "m1", {"user_id": "u"}) is True
