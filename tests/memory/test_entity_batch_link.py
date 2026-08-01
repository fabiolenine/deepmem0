"""Vínculo de entidade EM LOTE no caminho de update (P1b).

`_link_entities_for_memory` chamava `_upsert_entity` por entidade: N embeds, N
lookups. MEDIDO no ramo de UPDATE (bge-m3 + Qdrant reais, collection isolada, 7
repetições por ponto): o embed é **95-96% do wall**, as idas ao Qdrant somam
~18 ms por entidade, e `ms/entidade` fica plano em ~450-520 ms de N=1 a N=16 —
custo de OVERHEAD POR CHAMADA, não de trabalho.

⚠️ Registro de um erro de medição: uma primeira versão do benchmark usava
`user_id` novo por repetição, o que torna toda entidade inédita e mede o ramo de
INSERT (com `wait=True` + reconciliação, muito mais caro). Concluir sobre o
update a partir daquilo levou à afirmação errada de que a escrita dominava.

As REGRAS de identidade não são reimplementadas aqui — vêm de
`entidades_por_chaves`/`escopo_exato`, os mesmos da Fase 7 e do `_upsert_entity`.
Estes testes fixam que o lote (a) não é uma terceira regra e (b) degrada para o
serial só onde é seguro.
"""
from unittest.mock import MagicMock, patch

from mem0.memory.main import Memory, vincular_entidades_em_lote
from mem0.memory.utils import entity_point_id, normalize_entity_key

LARGO = {"user_id": "U"}
ESTREITO = {"user_id": "U", "run_id": "R"}
ENTS = [("PROPER", "DeepMem0"), ("PROPER", "Qdrant"), ("PROPER", "Ollama")]


def _linha(id_, data, escopo, vinculos, score=1.0):
    linha = MagicMock()
    linha.id = id_
    linha.score = score
    linha.payload = {"data": data, "data_normalized": normalize_entity_key(data),
                     "linked_memory_ids": list(vinculos), **escopo}
    return linha


def _store(linhas=None):
    st = MagicMock()
    st.list.return_value = linhas or []
    st.search_batch.return_value = []
    return st


def _ids_atualizados(st):
    """Ids que receberam `update`.

    ⚠️ Asserir `not st.update.called` é grosseiro DEMAIS aqui: a reconciliação
    pós-insert relê e reanexa o vínculo, e faz isso com `update` nas linhas
    RECÉM-CRIADAS. Proibir toda escrita reprovaria o comportamento correto. O
    que importa é QUAL linha foi tocada.
    """
    ids = set()
    for chamada in st.update.call_args_list:
        vid = chamada.kwargs.get("vector_id")
        if vid is None and chamada.args:
            vid = chamada.args[0]
        ids.add(str(vid))
    return ids


def _embedder():
    em = MagicMock()
    em.embed_batch.side_effect = lambda ts, *a, **k: [[float(len(t))] for t in ts]
    return em


class TestUmEmbedEUmLookup:
    def test_tres_entidades_um_embed_batch_e_um_lookup(self):
        st, em = _store(), _embedder()
        st.search_batch.return_value = [[], [], []]
        assert vincular_entidades_em_lote(st, em, "m1", ENTS, LARGO) is True
        assert em.embed_batch.call_count == 1
        assert em.embed.call_count == 0
        assert st.list.call_count == 1
        assert st.search_batch.call_count == 1
        assert st.insert.call_count == 1        # UM insert em lote

    def test_insert_em_lote_espera_a_escrita(self):
        st, em = _store(), _embedder()
        st.search_batch.return_value = [[], [], []]
        vincular_entidades_em_lote(st, em, "m1", ENTS, LARGO)
        assert st.insert.call_args.kwargs["wait"] is True

    def test_chaves_repetidas_colapsam(self):
        st, em = _store(), _embedder()
        st.search_batch.return_value = [[]]
        vincular_entidades_em_lote(
            st, em, "m1", [("PROPER", "FASE"), ("PROPER", "Fase")], LARGO)
        assert em.embed_batch.call_args[0][0] == ["FASE"]

    def test_ids_dos_inserts_sao_os_deterministicos(self):
        st, em = _store(), _embedder()
        st.search_batch.return_value = [[], [], []]
        vincular_entidades_em_lote(st, em, "m1", ENTS, LARGO)
        esperados = [entity_point_id(LARGO, normalize_entity_key(t))
                     for _tp, t in ENTS]
        assert st.insert.call_args.kwargs["ids"] == esperados


class TestMesmasRegrasDeIdentidade:
    def test_linha_de_escopo_estreito_nao_serve(self):
        st = _store([_linha("id-estreito", "DeepMem0", ESTREITO, ["mX"])])
        st.search_batch.return_value = [[], [], []]
        vincular_entidades_em_lote(st, _embedder(), "m1", ENTS, LARGO)
        assert "id-estreito" not in _ids_atualizados(st)
        assert len(st.insert.call_args.kwargs["ids"]) == 3

    def test_linha_do_escopo_certo_e_atualizada_nao_reinserida(self):
        st = _store([_linha("id-largo", "DeepMem0", LARGO, ["mX"])])
        st.search_batch.return_value = [[], []]
        vincular_entidades_em_lote(st, _embedder(), "m1", ENTS, LARGO)
        assert "id-largo" in _ids_atualizados(st)
        payload = next(c.kwargs["payload"] for c in st.update.call_args_list
                       if c.kwargs.get("vector_id") == "id-largo")
        assert payload["linked_memory_ids"] == ["mX", "m1"]
        assert payload["lnk_m1"] == 1
        assert len(st.insert.call_args.kwargs["ids"]) == 2

    def test_chave_ambigua_e_pulada_sem_escrita(self):
        st = _store([_linha("id-a", "DeepMem0", LARGO, ["m1"]),
                     _linha("id-b", "DeepMem0", LARGO, ["m2"])])
        st.search_batch.return_value = [[], []]
        vincular_entidades_em_lote(st, _embedder(), "m1", ENTS, LARGO)
        ids = st.insert.call_args.kwargs["ids"]
        assert entity_point_id(LARGO, "deepmem0") not in ids
        assert not ({"id-a", "id-b"} & _ids_atualizados(st)), (
            "escolheu uma das duplicatas")

    def test_sonda_que_erra_o_escopo_nao_funde(self):
        st = _store()
        st.search_batch.return_value = [
            [_linha("id-estreito", "DeepMem0", ESTREITO, ["mX"])], [], []]
        vincular_entidades_em_lote(st, _embedder(), "m1", ENTS, LARGO)
        assert "id-estreito" not in _ids_atualizados(st)
        assert len(st.insert.call_args.kwargs["ids"]) == 3


class TestDegradacao:
    def test_falha_de_embed_devolve_False_para_o_serial(self):
        st, em = _store(), _embedder()
        em.embed_batch.side_effect = RuntimeError("ollama fora")
        assert vincular_entidades_em_lote(st, em, "m1", ENTS, LARGO) is False
        assert not st.insert.called

    def test_falha_de_lookup_FECHA_e_nao_cai_no_serial(self):
        """Cair no serial repetiria o mesmo lookup quebrado e acabaria
        inserindo no id determinístico sobre payload alheio.

        ⚠️ `search_batch` devolve resposta BEM FORMADA de propósito. Com `[]`
        (o default do duplo) a validação de forma reprovaria antes, e o teste
        passaria pelo motivo errado — mascarando esta guarda com a outra. Foi
        exatamente o que a mutação P1b-M3 revelou ao sobreviver.
        """
        st, em = _store(), _embedder()
        st.search_batch.return_value = [[], [], []]
        st.list.side_effect = RuntimeError("Qdrant fora")
        assert vincular_entidades_em_lote(st, em, "m1", ENTS, LARGO) is True
        assert not st.insert.called
        assert not st.update.called

    def test_search_batch_com_forma_errada_fecha(self):
        st, em = _store(), _embedder()
        st.search_batch.return_value = [MagicMock()]
        assert vincular_entidades_em_lote(st, em, "m1", ENTS, LARGO) is True
        assert not st.insert.called
        assert not st.update.called

    def test_reconciliacao_roda_para_as_linhas_inseridas(self):
        """A reconciliação pós-insert é o que estreita a corrida de
        lost-update; sem ela o lote seria mais rápido e menos seguro."""
        st, em = _store(), _embedder()
        st.search_batch.return_value = [[], [], []]
        vincular_entidades_em_lote(st, em, "m1", ENTS, LARGO)
        inseridos = set(st.insert.call_args.kwargs["ids"])
        assert inseridos & _ids_atualizados(st), (
            "nenhuma linha recém-inserida foi reconciliada")

    def test_contagem_do_embed_batch_errada_cai_no_serial(self):
        st, em = _store(), _embedder()
        em.embed_batch.side_effect = lambda ts, *a, **k: [[1.0]]
        assert vincular_entidades_em_lote(st, em, "m1", ENTS, LARGO) is False
        assert not st.insert.called


class TestIntegracaoComOLinkEntities:
    def _mem(self):
        mem = MagicMock(spec=Memory)
        mem.config = MagicMock()
        mem.config.language = "pt"
        mem.entity_store = _store()
        mem.entity_store.search_batch.return_value = [[], [], []]
        mem.embedding_model = _embedder()
        return mem

    def test_link_entities_usa_o_lote(self):
        mem = self._mem()
        with patch("mem0.memory.main.extract_entities", return_value=ENTS):
            Memory._link_entities_for_memory(mem, "m1", "texto", LARGO)
        assert mem.embedding_model.embed_batch.call_count == 1
        assert not mem._upsert_entity.called

    def test_link_entities_cai_no_serial_quando_o_lote_recusa(self):
        mem = self._mem()
        mem.embedding_model.embed_batch.side_effect = RuntimeError("ollama fora")
        with patch("mem0.memory.main.extract_entities", return_value=ENTS):
            Memory._link_entities_for_memory(mem, "m1", "texto", LARGO)
        assert mem._upsert_entity.call_count == 3
