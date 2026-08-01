"""Teto de lote do `OllamaEmbedding.embed_batch` (P2).

⚠️ A justificativa NÃO é a que se supõe, e isso é o registro que importa.
MEDIDO contra o `bge-m3` real (GPU de 8 GB, 01/08/2026): **nenhuma falha até
32768 itens numa única requisição** — sem erro, sem timeout — e a VRAM do modelo
não cresce com o lote. ⚠️ A medição parou num limiar de wall-time escolhido
(744 s/chamada), NÃO numa falha; logo o que se afirma é "nenhuma falha até
32768", não "não existe fronteira". O teto não previne falha de payload. O que
ele limita é:

* **latência de uma chamada** — 743,6 s em 32768 itens curtos; 63,8 s em 1024
  chunks de ~1,8k chars contra 16,0 s em 256 (cliente MCP estoura antes disso);
* **raio de uma falha** — uma requisição que morre derruba o lote inteiro e joga
  o chamador no fallback item a item.

256 é onde `ms/item` já estabilizou nos dois perfis (12,0 ms/item em texto curto,
62,7 ms/item em ~1,8k chars), então o teto não custa vazão.

O que estes testes protegem é o que o fatiamento pode quebrar em silêncio:
ORDEM e CONTAGEM. O chamador casa vetor com texto por POSIÇÃO — em
`_add_to_vector_store` o `embed_map` é montado com `zip(mem_texts, ...)` —, então
uma troca de ordem cola o embedding de um fato em outro sem erro nenhum.
"""
from unittest.mock import MagicMock, patch

import pytest

from mem0.embeddings.ollama import DEFAULT_MAX_BATCH, OllamaEmbedding


def _embedder(monkeypatch=None):
    with patch("mem0.embeddings.ollama.Client") as cli:
        cli.return_value.list.return_value = {"models": [{"model": "bge-m3:latest"}]}
        emb = OllamaEmbedding()
        emb.config.model = "bge-m3"
        return emb


class TestFatiamento:
    def _com_cliente(self, n_textos, limite=None):
        emb = _embedder()
        chamadas = []

        def _embed(model=None, input=None):
            chamadas.append(list(input))
            # vetor identificável por texto: expõe troca de ordem
            return {"embeddings": [[float(len(t)), hash(t) % 1000] for t in input]}

        emb.client.embed = MagicMock(side_effect=_embed)
        textos = [f"texto numero {i}" for i in range(n_textos)]
        if limite is not None:
            with patch.dict("os.environ", {"MEM0_EMBED_MAX_BATCH": str(limite)}):
                out = emb.embed_batch(textos, "add")
        else:
            out = emb.embed_batch(textos, "add")
        return textos, out, chamadas

    def test_lote_abaixo_do_teto_vai_numa_chamada(self):
        _t, out, chamadas = self._com_cliente(10, limite=256)
        assert len(chamadas) == 1
        assert len(out) == 10

    def test_lote_acima_do_teto_e_fatiado(self):
        _t, out, chamadas = self._com_cliente(10, limite=4)
        assert [len(c) for c in chamadas] == [4, 4, 2]
        assert len(out) == 10

    def test_ordem_sobrevive_ao_fatiamento(self):
        """O chamador casa vetor com texto por POSIÇÃO. Trocar a ordem cola o
        embedding de um fato em outro — sem exceção, sem log."""
        textos, out, _c = self._com_cliente(10, limite=3)
        esperado = [[float(len(t)), hash(t) % 1000] for t in textos]
        assert out == esperado

    def test_fronteira_exata_nao_gera_pedaco_vazio(self):
        _t, out, chamadas = self._com_cliente(8, limite=4)
        assert [len(c) for c in chamadas] == [4, 4]
        assert len(out) == 8

    def test_lista_vazia_nao_chama_o_cliente(self):
        emb = _embedder()
        emb.client.embed = MagicMock()
        assert emb.embed_batch([], "add") == []
        emb.client.embed.assert_not_called()


class TestContagemPorPedaco:
    """Conferir só o total deixaria um pedaço curto ser compensado por outro
    longo — e o desalinhamento resultante é o defeito silencioso do topo."""

    def test_pedaco_curto_no_meio_levanta(self):
        emb = _embedder()
        respostas = [
            {"embeddings": [[1.0], [2.0], [3.0]]},   # ok
            {"embeddings": [[4.0]]},                 # curto: 1 para 3
            {"embeddings": [[5.0], [6.0], [7.0]]},   # ok
        ]
        emb.client.embed = MagicMock(side_effect=respostas)
        with patch.dict("os.environ", {"MEM0_EMBED_MAX_BATCH": "3"}):
            with pytest.raises(ValueError, match="chunk"):
                emb.embed_batch([f"t{i}" for i in range(9)], "add")

    def test_pedaco_longo_compensando_curto_tambem_levanta(self):
        """O caso que um check só-no-total deixaria passar: 2 + 4 = 6 = total."""
        emb = _embedder()
        respostas = [
            {"embeddings": [[1.0], [2.0]]},                    # 2 para 3
            {"embeddings": [[3.0], [4.0], [5.0], [6.0]]},      # 4 para 3
        ]
        emb.client.embed = MagicMock(side_effect=respostas)
        with patch.dict("os.environ", {"MEM0_EMBED_MAX_BATCH": "3"}):
            with pytest.raises(ValueError, match="chunk"):
                emb.embed_batch([f"t{i}" for i in range(6)], "add")


class TestTeto:
    def test_default_documentado(self):
        assert DEFAULT_MAX_BATCH == 256

    @pytest.mark.parametrize("valor", ["0", "-5", "nao-e-numero", ""])
    def test_valor_invalido_cai_no_default_em_vez_de_desligar_o_teto(self, valor):
        """`0` desligaria o fatiamento; string inválida estouraria no meio de um
        add. Nos dois casos o desfecho seguro é o default."""
        from mem0.embeddings.ollama import _max_batch
        with patch.dict("os.environ", {"MEM0_EMBED_MAX_BATCH": valor}):
            assert _max_batch() == DEFAULT_MAX_BATCH

    def test_env_ajusta_sem_restart(self):
        from mem0.embeddings.ollama import _max_batch
        with patch.dict("os.environ", {"MEM0_EMBED_MAX_BATCH": "32"}):
            assert _max_batch() == 32
        with patch.dict("os.environ", {"MEM0_EMBED_MAX_BATCH": "64"}):
            assert _max_batch() == 64
