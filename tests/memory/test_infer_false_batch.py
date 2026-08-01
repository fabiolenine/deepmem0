"""Hoist do embed no caminho `infer=False` (P3).

O laço fazia UM embed por mensagem, e o custo do embedder é dominado pela
CHAMADA, não pelo item (medido contra bge-m3: ~500 ms de overhead por chamada,
~12 ms por item curto). Tirar o embed do laço é barato — e é exatamente o tipo
de mudança que quebra coisa em silêncio, porque o chamador casa vetor com texto.

O que estes testes fixam é o que o hoist pode estragar sem levantar exceção:
ordem, descartes (`role == "system"` e formato inválido), conteúdos repetidos, e
a semântica de falha — um embed ruim derrubava UMA mensagem, e não pode passar a
derrubar o lote.
"""
from unittest.mock import MagicMock, patch

import pytest

from mem0.memory.main import (
    AsyncMemory,
    Memory,
    _embed_map_de,
    _mensagens_validas_para_add,
)


class TestSelecaoDeMensagens:
    def test_descarta_system_e_formato_invalido_preservando_ordem(self):
        msgs = [
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": "primeira"},
            "nao e dict",
            {"role": "assistant"},                       # sem content
            {"content": "sem role"},
            {"role": "assistant", "content": "segunda"},
        ]
        out = _mensagens_validas_para_add(msgs)
        assert [m["content"] for m in out] == ["primeira", "segunda"]

    def test_lista_vazia(self):
        assert _mensagens_validas_para_add([]) == []


class TestEmbedMap:
    def test_um_unico_embed_batch_para_todos(self):
        em = MagicMock()
        em.embed_batch.return_value = [[1.0], [2.0]]
        mapa = _embed_map_de(em, ["a", "b"])
        assert em.embed_batch.call_count == 1
        assert em.embed.call_count == 0
        assert mapa == {"a": [1.0], "b": [2.0]}

    def test_conteudos_repetidos_embedam_uma_vez_so(self):
        """Texto repetido tem o MESMO vetor; embedá-lo duas vezes é só custo."""
        em = MagicMock()
        em.embed_batch.return_value = [[1.0], [2.0]]
        mapa = _embed_map_de(em, ["a", "b", "a", "a"])
        assert em.embed_batch.call_args[0][0] == ["a", "b"]
        assert mapa == {"a": [1.0], "b": [2.0]}

    def test_contagem_errada_do_lote_nao_e_zipada_em_silencio(self):
        """`zip` trunca sem avisar — e truncar aqui cola o vetor de um fato em
        outro. Tem de cair no item a item, não devolver mapa parcial."""
        em = MagicMock()
        em.embed_batch.return_value = [[1.0]]            # 1 para 3
        em.embed.side_effect = lambda t, a: [float(len(t))]
        mapa = _embed_map_de(em, ["aa", "bbb", "cccc"])
        assert mapa == {"aa": [2.0], "bbb": [3.0], "cccc": [4.0]}
        assert em.embed.call_count == 3

    def test_falha_do_lote_cai_para_item_a_item(self):
        em = MagicMock()
        em.embed_batch.side_effect = RuntimeError("ollama fora do ar")
        em.embed.side_effect = lambda t, a: [float(len(t))]
        mapa = _embed_map_de(em, ["aa", "bbb"])
        assert mapa == {"aa": [2.0], "bbb": [3.0]}

    def test_falha_de_UM_item_nao_derruba_os_outros(self):
        """O laço antigo perdia uma mensagem; o hoist não pode perder todas."""
        em = MagicMock()
        em.embed_batch.side_effect = RuntimeError("falhou")

        def _embed(t, a):
            if t == "ruim":
                raise RuntimeError("texto problemático")
            return [float(len(t))]

        em.embed.side_effect = _embed
        mapa = _embed_map_de(em, ["ok", "ruim", "outro"])
        assert "ruim" not in mapa
        assert mapa == {"ok": [2.0], "outro": [5.0]}


class TestCaminhoCompleto:
    def _mem(self, embed_batch_retorno=None, erro=None):
        mem = MagicMock(spec=Memory)
        mem.embedding_model = MagicMock()
        if erro:
            mem.embedding_model.embed_batch.side_effect = erro
        else:
            mem.embedding_model.embed_batch.side_effect = (
                lambda ts, *a, **k: [[float(len(t))] for t in ts])
        mem.embedding_model.embed.side_effect = lambda t, a: [float(len(t))]
        mem._create_memory = MagicMock(side_effect=lambda d, e, m: f"id-{d}")
        return mem

    def test_ordem_e_conteudo_do_retorno(self):
        mem = self._mem()
        out = Memory._add_to_vector_store(
            mem,
            [{"role": "system", "content": "x"},
             {"role": "user", "content": "primeira"},
             {"role": "assistant", "content": "segunda", "name": "bot"}],
            {}, {}, False)
        assert [r["memory"] for r in out] == ["primeira", "segunda"]
        assert [r["role"] for r in out] == ["user", "assistant"]
        assert out[1]["actor_id"] == "bot"
        assert mem.embedding_model.embed_batch.call_count == 1

    def test_cada_memoria_recebe_o_vetor_do_proprio_texto(self):
        """A asserção que pega desalinhamento: o vetor é derivado do texto."""
        mem = self._mem()
        Memory._add_to_vector_store(
            mem,
            [{"role": "user", "content": "aa"},
             {"role": "user", "content": "bbbb"}],
            {}, {}, False)
        for chamada in mem._create_memory.call_args_list:
            texto, embed_map, _meta = chamada[0]
            assert embed_map[texto] == [float(len(texto))]

    def test_conteudos_identicos_geram_duas_memorias(self):
        """Colapsar no mapa de embed não pode colapsar as MEMÓRIAS."""
        mem = self._mem()
        out = Memory._add_to_vector_store(
            mem,
            [{"role": "user", "content": "igual"},
             {"role": "assistant", "content": "igual"}],
            {}, {}, False)
        assert len(out) == 2
        assert mem._create_memory.call_count == 2

    def test_mensagem_sem_vetor_e_pulada_sem_derrubar_as_outras(self):
        mem = self._mem(erro=RuntimeError("lote falhou"))

        def _embed(t, a):
            if t == "ruim":
                raise RuntimeError("nao embeda")
            return [float(len(t))]

        mem.embedding_model.embed.side_effect = _embed
        out = Memory._add_to_vector_store(
            mem,
            [{"role": "user", "content": "ok"},
             {"role": "user", "content": "ruim"},
             {"role": "user", "content": "outro"}],
            {}, {}, False)
        assert [r["memory"] for r in out] == ["ok", "outro"]


class TestParidadeAsync:
    def _mem(self):
        mem = MagicMock(spec=AsyncMemory)
        mem.embedding_model = MagicMock()
        mem.embedding_model.embed_batch.side_effect = (
            lambda ts, *a, **k: [[float(len(t))] for t in ts])

        async def _cria(d, e, m):
            return f"id-{d}"

        mem._create_memory = MagicMock(side_effect=_cria)
        return mem

    def test_async_mesma_ordem_e_mesmos_descartes(self):
        import asyncio
        mem = self._mem()
        out = asyncio.run(AsyncMemory._add_to_vector_store(
            mem,
            [{"role": "system", "content": "x"},
             {"role": "user", "content": "primeira"},
             {"role": "assistant", "content": "segunda"}],
            {}, {}, False))
        assert [r["memory"] for r in out] == ["primeira", "segunda"]
        assert mem.embedding_model.embed_batch.call_count == 1

    def test_async_um_embed_batch_e_nao_um_por_mensagem(self):
        import asyncio
        mem = self._mem()
        asyncio.run(AsyncMemory._add_to_vector_store(
            mem,
            [{"role": "user", "content": f"m{i}"} for i in range(5)],
            {}, {}, False))
        assert mem.embedding_model.embed_batch.call_count == 1
        assert mem.embedding_model.embed.call_count == 0

    # --- paridade DE VERDADE: os casos que quebram, não só o caminho feliz ---
    # A r3 já tinha apontado exatamente esta lacuna na Fase 7 async. Cobrir só
    # ordem/descarte/contagem e chamar de "paridade" é a mesma falha de novo.

    def test_async_cada_memoria_recebe_o_vetor_do_proprio_texto(self):
        import asyncio
        mem = self._mem()
        asyncio.run(AsyncMemory._add_to_vector_store(
            mem,
            [{"role": "user", "content": "aa"},
             {"role": "user", "content": "bbbb"}],
            {}, {}, False))
        for chamada in mem._create_memory.call_args_list:
            texto, embed_map, _meta = chamada[0]
            assert embed_map[texto] == [float(len(texto))]

    def test_async_conteudos_identicos_geram_duas_memorias(self):
        import asyncio
        mem = self._mem()
        out = asyncio.run(AsyncMemory._add_to_vector_store(
            mem,
            [{"role": "user", "content": "igual"},
             {"role": "assistant", "content": "igual"}],
            {}, {}, False))
        assert len(out) == 2
        assert mem._create_memory.call_count == 2
        # colapsa no embed (mesmo texto, mesmo vetor), não nas memórias
        assert mem.embedding_model.embed_batch.call_args[0][0] == ["igual"]

    def test_async_contagem_errada_do_lote_cai_no_item_a_item(self):
        import asyncio
        mem = self._mem()
        mem.embedding_model.embed_batch.side_effect = (
            lambda ts, *a, **k: [[1.0]])          # 1 vetor para 2 textos
        mem.embedding_model.embed.side_effect = lambda t, a: [float(len(t))]
        out = asyncio.run(AsyncMemory._add_to_vector_store(
            mem,
            [{"role": "user", "content": "aa"},
             {"role": "user", "content": "bbbb"}],
            {}, {}, False))
        assert [r["memory"] for r in out] == ["aa", "bbbb"]
        assert mem.embedding_model.embed.call_count == 2

    def test_async_falha_parcial_pula_uma_e_mantem_as_outras(self):
        import asyncio
        mem = self._mem()
        mem.embedding_model.embed_batch.side_effect = RuntimeError("lote falhou")

        def _embed(t, a):
            if t == "ruim":
                raise RuntimeError("nao embeda")
            return [float(len(t))]

        mem.embedding_model.embed.side_effect = _embed
        out = asyncio.run(AsyncMemory._add_to_vector_store(
            mem,
            [{"role": "user", "content": "ok"},
             {"role": "user", "content": "ruim"},
             {"role": "user", "content": "outro"}],
            {}, {}, False))
        assert [r["memory"] for r in out] == ["ok", "outro"]
