"""Contrato do arquivo de configuração do CLI.

O CLI lê a configuração de um arquivo gerado pelo servidor MCP. Este contrato é
o que impede que os dois lados divirjam — e divergência aqui não daria erro,
daria resultado plausível e errado (outra collection, reranker desligado).
"""

import json

import pytest

from mem0.cli import config as cfg


def _escrever(tmp_path, documento, nome="config.json"):
    caminho = tmp_path / nome
    caminho.write_text(json.dumps(documento), encoding="utf-8")
    return caminho


VALIDO = {
    "contract_version": cfg.CONTRACT_VERSION,
    "config": {
        "vector_store": {"provider": "qdrant",
                         "config": {"collection_name": "col_x", "url": "http://q:6333"}},
        "embedder": {"provider": "ollama", "config": {"model": "bge-m3"}},
        "llm": {"provider": "ollama", "config": {"model": "qwen"}},
        "reranker": {"provider": "hf"},
        "language": "pt",
    },
}


class TestResolucaoDeCaminho:
    def test_explicito_vence_tudo(self, tmp_path, monkeypatch):
        monkeypatch.setenv(cfg.ENV_PATH, "/do/env.json")
        assert cfg.resolve_path("/do/flag.json") == cfg.Path("/do/flag.json")

    def test_env_vence_o_padrao(self, monkeypatch):
        monkeypatch.setenv(cfg.ENV_PATH, "/do/env.json")
        assert cfg.resolve_path() == cfg.Path("/do/env.json")

    def test_padrao_quando_nada_declarado(self, monkeypatch):
        monkeypatch.delenv(cfg.ENV_PATH, raising=False)
        assert cfg.resolve_path() == cfg.DEFAULT_PATH

    def test_env_em_branco_nao_conta(self, monkeypatch):
        monkeypatch.setenv(cfg.ENV_PATH, "   ")
        assert cfg.resolve_path() == cfg.DEFAULT_PATH


class TestCarga:
    def test_arquivo_valido(self, tmp_path):
        caminho = _escrever(tmp_path, VALIDO)
        assert cfg.load(str(caminho))["language"] == "pt"

    def test_ausente_diz_como_gerar(self, tmp_path):
        """A mensagem tem de ser acionável: é o erro do primeiro uso."""
        with pytest.raises(cfg.ConfigError) as exc:
            cfg.load(str(tmp_path / "nao_existe.json"))
        assert "config export" in str(exc.value)

    def test_versao_incompativel_e_RECUSADA(self, tmp_path):
        """Ler contrato de outra versão pela metade daria resultado errado
        sem erro — o que é pior do que falhar."""
        caminho = _escrever(tmp_path, {**VALIDO, "contract_version": 999})
        with pytest.raises(cfg.ConfigError, match="999"):
            cfg.load(str(caminho))

    def test_sem_contract_version_e_recusado(self, tmp_path):
        doc = {k: v for k, v in VALIDO.items() if k != "contract_version"}
        with pytest.raises(cfg.ConfigError, match="contract_version"):
            cfg.load(str(_escrever(tmp_path, doc)))

    def test_json_quebrado(self, tmp_path):
        caminho = tmp_path / "quebrado.json"
        caminho.write_text("{ não é json", encoding="utf-8")
        with pytest.raises(cfg.ConfigError):
            cfg.load(str(caminho))

    def test_sem_a_chave_config(self, tmp_path):
        with pytest.raises(cfg.ConfigError, match="config"):
            cfg.load(str(_escrever(tmp_path, {"contract_version": cfg.CONTRACT_VERSION})))

    def test_config_vazio_e_recusado(self, tmp_path):
        doc = {"contract_version": cfg.CONTRACT_VERSION, "config": {}}
        with pytest.raises(cfg.ConfigError):
            cfg.load(str(_escrever(tmp_path, doc)))

    def test_raiz_que_nao_e_objeto(self, tmp_path):
        caminho = tmp_path / "lista.json"
        caminho.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(cfg.ConfigError):
            cfg.load(str(caminho))


class TestResumo:
    def test_extrai_o_que_importa(self, tmp_path):
        resumo = cfg.describe(str(_escrever(tmp_path, VALIDO)))
        assert resumo["collection"] == "col_x"
        assert resumo["embedder_model"] == "bge-m3"
        assert resumo["reranker"] is True
        assert resumo["language"] == "pt"

    def test_reranker_ausente_vira_false(self, tmp_path):
        doc = json.loads(json.dumps(VALIDO))
        del doc["config"]["reranker"]
        assert cfg.describe(str(_escrever(tmp_path, doc)))["reranker"] is False
