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
