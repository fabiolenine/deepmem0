"""Comandos do CLI: escopo obrigatório, confirmação de escrita e códigos de saída.

Os códigos de saída são CONTRATO — script que automatize sobre o CLI depende
deles, e mudá-los em silêncio quebraria automação alheia.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mem0.cli import app as cli_main
from mem0.cli import commands


def _parse(argv):
    """Parse + merge das flags globais — o mesmo caminho que `main` percorre."""
    return cli_main._fundir_globais(cli_main.construir_parser().parse_args(argv))

CONFIG = {
    "contract_version": 1,
    "config": {"vector_store": {"provider": "qdrant",
                                "config": {"collection_name": "c", "url": "http://q"}}},
}


@pytest.fixture
def config_path(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(CONFIG), encoding="utf-8")
    return str(p)


@pytest.fixture
def memoria():
    """Substitui o Memory — nenhum teste do CLI toca corpus de verdade."""
    fake = MagicMock()
    fake.search.return_value = {"results": [{"id": "a" * 36, "memory": "texto",
                                            "score": 0.9, "rerank_score": 0.99}]}
    fake.get.return_value = {"id": "a" * 36, "memory": "texto", "user_id": "ana"}
    fake.get_all.return_value = {"results": [{"id": "b" * 36, "memory": "outro"}]}
    fake.history.return_value = [{"event": "ADD", "created_at": "2026-01-01",
                                  "new_memory": "texto"}]
    fake.add.return_value = {"results": [{"id": "c" * 36, "memory": "novo"}]}
    with patch("mem0.Memory.from_config", return_value=fake):
        yield fake


class TestEscopoObrigatorio:
    """Escopo sem default é decisão de segurança: um default silencioso leria —
    ou escreveria — no corpus de outra pessoa."""

    @pytest.mark.parametrize("cmd", ["search", "list", "add"])
    def test_sem_user_o_parser_recusa(self, cmd, capsys):
        argv = {"search": ["search", "q"], "list": ["list"],
                "add": ["add", "texto"]}[cmd]
        with pytest.raises(SystemExit) as exc:
            _parse(argv)
        assert exc.value.code == 2
        assert "--user" in capsys.readouterr().err

    def test_com_user_passa(self):
        args = _parse(["search", "q", "--user", "ana"])
        assert args.user == "ana"


class TestBusca:
    def test_nao_reforca_a_memoria(self, config_path, memoria):
        """Consultar pelo terminal não é re-encontro: contá-lo enviesaria o
        ranking que o próprio comando exibe."""
        args = _parse(
            ["search", "q", "--user", "ana", "--config", config_path])
        commands.cmd_search(args)
        assert memoria.search.call_args.kwargs["reinforce"] is False

    def test_escopo_vai_como_filtro(self, config_path, memoria):
        args = _parse(
            ["search", "q", "--user", "ana", "--agent", "bot", "--config", config_path])
        commands.cmd_search(args)
        filtros = memoria.search.call_args.kwargs["filters"]
        assert filtros == {"user_id": "ana", "agent_id": "bot"}

    def test_json_e_contrato(self, config_path, memoria, capsys):
        args = _parse(
            ["--json", "search", "q", "--user", "ana", "--config", config_path])
        assert commands.cmd_search(args) == commands.EXIT_OK
        saida = json.loads(capsys.readouterr().out)
        assert saida["count"] == 1 and saida["query"] == "q"

    def test_rerank_default_nao_forca_nada(self, config_path, memoria):
        args = _parse(
            ["search", "q", "--user", "ana", "--config", config_path])
        assert memoria.search.call_args is None or True
        commands.cmd_search(args)
        assert memoria.search.call_args.kwargs["rerank"] is None

    def test_no_rerank_desliga(self, config_path, memoria):
        args = _parse(
            ["search", "q", "--user", "ana", "--no-rerank", "--config", config_path])
        commands.cmd_search(args)
        assert memoria.search.call_args.kwargs["rerank"] is False


class TestEscrita:
    def test_sem_yes_NAO_escreve(self, config_path, memoria, capsys):
        """O corpus é de produção; um engano de histórico de shell não pode
        alterá-lo."""
        args = _parse(
            ["add", "texto novo", "--user", "ana", "--config", config_path])
        assert commands.cmd_add(args) == commands.EXIT_PRECISA_CONFIRMAR
        memoria.add.assert_not_called()
        assert "PRÉVIA" in capsys.readouterr().err

    def test_com_yes_escreve(self, config_path, memoria):
        args = _parse(
            ["add", "texto novo", "--user", "ana", "--yes", "--config", config_path])
        assert commands.cmd_add(args) == commands.EXIT_OK
        memoria.add.assert_called_once()
        assert memoria.add.call_args.kwargs["user_id"] == "ana"

    def test_raw_desliga_a_extracao(self, config_path, memoria):
        args = _parse(
            ["add", "t", "--user", "ana", "--yes", "--raw", "--config", config_path])
        commands.cmd_add(args)
        assert memoria.add.call_args.kwargs["infer"] is False

    def test_texto_vazio_e_erro(self, config_path, memoria):
        args = _parse(
            ["add", "   ", "--user", "ana", "--yes", "--config", config_path])
        assert commands.cmd_add(args) == commands.EXIT_ERRO
        memoria.add.assert_not_called()


class TestLeituras:
    def test_get_inexistente_tem_codigo_proprio(self, config_path, memoria):
        memoria.get.return_value = None
        args = _parse(
            ["get", "x" * 36, "--user", "ana", "--config", config_path])
        assert commands.cmd_get(args) == commands.EXIT_NAO_ENCONTRADO

    def test_list_avisa_quando_o_teto_morde(self, config_path, memoria, capsys):
        """`get_all` não tem cursor: atingir o limite pode esconder o resto."""
        memoria.get_all.return_value = {"results": [{"id": str(i)} for i in range(3)]}
        args = _parse(
            ["list", "--user", "ana", "--limit", "3", "--config", config_path])
        commands.cmd_list(args)
        assert "ATINGIDO" in capsys.readouterr().err

    def test_history(self, config_path, memoria, capsys):
        args = _parse(
            ["--json", "history", "a" * 36, "--user", "ana", "--config", config_path])
        assert commands.cmd_history(args) == commands.EXIT_OK
        assert json.loads(capsys.readouterr().out)["events"][0]["event"] == "ADD"


class TestSaidaEErro:
    def test_config_ausente_tem_codigo_proprio(self, tmp_path, capsys):
        rc = cli_main.main(["search", "q", "--user", "ana",
                            "--config", str(tmp_path / "nada.json")])
        assert rc == commands.EXIT_CONFIG
        assert "config export" in capsys.readouterr().err

    def test_sem_verbo_mostra_ajuda(self, capsys):
        assert cli_main.main([]) == commands.EXIT_OK
        assert "deepmem0" in capsys.readouterr().out

    def test_codigos_sao_distintos(self):
        codigos = [commands.EXIT_OK, commands.EXIT_ERRO, commands.EXIT_CONFIG,
                   commands.EXIT_NAO_ENCONTRADO, commands.EXIT_PRECISA_CONFIRMAR]
        assert len(set(codigos)) == len(codigos)


class TestDoctor:
    def test_reporta_pacote_ausente_sem_importar(self, config_path, capsys):
        """`doctor` usa find_spec: importar `ollama` só para diagnosticar
        transformaria uma checagem em dependência."""
        args = _parse(
            ["--json", "doctor", "--config", config_path])
        commands.cmd_doctor(args)
        saida = json.loads(capsys.readouterr().out)
        nomes = {c["check"] for c in saida["checks"]}
        assert {"config", "qdrant_client", "spacy"} <= nomes

    def test_config_invalida_nao_derruba_o_doctor(self, tmp_path, capsys):
        args = _parse(
            ["--json", "doctor", "--config", str(tmp_path / "nada.json")])
        assert commands.cmd_doctor(args) == commands.EXIT_ERRO
        saida = json.loads(capsys.readouterr().out)
        cfg = next(c for c in saida["checks"] if c["check"] == "config")
        assert cfg["ok"] is False and "error" in cfg


class TestVerbosDoPlano:
    """Os verbos que o plano aprovado exigia — e que faltavam na primeira volta."""

    def test_todos_os_verbos_do_plano_existem(self):
        acoes = [a for a in cli_main.construir_parser()._actions
                 if getattr(a, "choices", None)]
        verbos = set(acoes[0].choices)
        # `get_all` é alias de `list`: o plano nomeia o método do core, e `list`
        # é a convenção em CLI. Os dois resolvem.
        assert {"add", "search", "get", "get_all", "list", "history",
                "entities", "delete", "doctor", "config"} <= verbos

    def test_delete_sem_yes_NAO_apaga(self, config_path, memoria, capsys):
        args = _parse(["delete", "a" * 36, "--user", "ana", "--config", config_path])
        assert commands.cmd_delete(args) == commands.EXIT_PRECISA_CONFIRMAR
        memoria.delete.assert_not_called()
        saida = capsys.readouterr().err
        assert "PRÉVIA" in saida
        # a prévia mostra o TEXTO: confirmar um id opaco é confirmar nada
        assert "texto" in saida

    def test_delete_com_yes_apaga(self, config_path, memoria):
        args = _parse(["delete", "a" * 36, "--user", "ana", "--yes", "--config", config_path])
        assert commands.cmd_delete(args) == commands.EXIT_OK
        memoria.delete.assert_called_once_with("a" * 36)

    def test_delete_de_inexistente_nao_confirma_nada(self, config_path, memoria):
        memoria.get.return_value = None
        args = _parse(["delete", "x" * 36, "--user", "ana", "--yes", "--config", config_path])
        assert commands.cmd_delete(args) == commands.EXIT_NAO_ENCONTRADO
        memoria.delete.assert_not_called()

    def test_dry_run_e_yes_juntos_sao_RECUSADOS(self, config_path, memoria):
        """Deixar um vencer o outro em silêncio seria o pior desfecho: quem
        escreveu --dry-run acreditaria estar simulando enquanto grava."""
        args = _parse(["add", "t", "--user", "ana", "--dry-run", "--yes",
                       "--config", config_path])
        with pytest.raises(ValueError, match="contradizem"):
            commands.cmd_add(args)
        memoria.add.assert_not_called()

    def test_dry_run_sozinho_nao_escreve(self, config_path, memoria):
        args = _parse(["add", "t", "--user", "ana", "--dry-run",
                       "--config", config_path])
        assert commands.cmd_add(args) == commands.EXIT_PRECISA_CONFIRMAR
        memoria.add.assert_not_called()

    def test_get_all_e_list_apontam_para_o_mesmo_comando(self, config_path, memoria):
        a = _parse(["list", "--user", "ana", "--config", config_path])
        b = _parse(["get_all", "--user", "ana", "--config", config_path])
        assert a.func is b.func is commands.cmd_list

    def test_entities_lista_e_ordena_por_vinculos(self, config_path, memoria, capsys):
        class P:
            def __init__(self, pid, payload):
                self.id, self.payload = pid, payload

        memoria.entity_store.list.return_value = [
            P("e1", {"data": "Qdrant", "entity_type": "PROPER",
                     "linked_memory_ids": ["m1", "m2", "m3"]}),
            P("e2", {"data": "Fase", "entity_type": "COMPOUND",
                     "linked_memory_ids": ["m1"]}),
        ]
        args = _parse(["--json", "entities", "--user", "ana", "--config", config_path])
        assert commands.cmd_entities(args) == commands.EXIT_OK
        saida = json.loads(capsys.readouterr().out)
        assert [e["data"] for e in saida["entities"]] == ["Qdrant", "Fase"]
        assert saida["entities"][0]["links"] == 3

    def test_entities_filtra_por_contains(self, config_path, memoria, capsys):
        class P:
            def __init__(self, pid, payload):
                self.id, self.payload = pid, payload

        memoria.entity_store.list.return_value = [
            P("e1", {"data": "Qdrant", "data_normalized": "qdrant",
                     "linked_memory_ids": []}),
            P("e2", {"data": "Fase", "data_normalized": "fase",
                     "linked_memory_ids": []}),
        ]
        # Maiúsculas de propósito: o filtro casa contra `data_normalized`, então
        # "QDR" achando "Qdrant" prova que a busca é insensível a caixa.
        args = _parse(["--json", "entities", "--user", "ana", "--contains", "QDR",
                       "--config", config_path])
        commands.cmd_entities(args)
        saida = json.loads(capsys.readouterr().out)
        assert [e["data"] for e in saida["entities"]] == ["Qdrant"]

    def test_entities_ignora_vinculo_envenenado(self, config_path, memoria, capsys):
        """Há linhas legadas gravadas com `set(str)`, que itera caractere a
        caractere — contá-las daria 6 vínculos onde há um id."""
        class P:
            def __init__(self, pid, payload):
                self.id, self.payload = pid, payload

        memoria.entity_store.list.return_value = [
            P("e1", {"data": "X", "linked_memory_ids": "abcdef"}),
        ]
        args = _parse(["--json", "entities", "--user", "ana", "--config", config_path])
        commands.cmd_entities(args)
        assert json.loads(capsys.readouterr().out)["entities"][0]["links"] != 6


class TestEscopoNaLeituraENaExclusao:
    """Um id é só um número: sem conferir posse, ele alcança qualquer corpus.

    `Memory.get` busca por id e não aceita filtro — a conferência tem de ser
    feita pelo chamador, e a ausência dela era uma falha de segurança real.
    """

    @pytest.mark.parametrize("verbo", ["get", "history", "delete"])
    def test_id_sem_user_e_recusado_pelo_parser(self, verbo, capsys):
        with pytest.raises(SystemExit) as exc:
            cli_main.construir_parser().parse_args([verbo, "a" * 36])
        assert exc.value.code == 2
        assert "--user" in capsys.readouterr().err

    def test_get_de_outro_escopo_responde_inexistente(self, config_path, memoria):
        memoria.get.return_value = {"id": "a" * 36, "memory": "alheia",
                                    "user_id": "OUTRA_PESSOA"}
        args = _parse(["get", "a" * 36, "--user", "ana", "--config", config_path])
        assert commands.cmd_get(args) == commands.EXIT_NAO_ENCONTRADO

    def test_delete_de_outro_escopo_NAO_apaga(self, config_path, memoria):
        """O caso que mais importa: apagar memória alheia por id."""
        memoria.get.return_value = {"id": "a" * 36, "memory": "alheia",
                                    "user_id": "OUTRA_PESSOA"}
        args = _parse(["delete", "a" * 36, "--user", "ana", "--yes",
                       "--config", config_path])
        assert commands.cmd_delete(args) == commands.EXIT_NAO_ENCONTRADO
        memoria.delete.assert_not_called()

    def test_history_de_outro_escopo_e_recusado(self, config_path, memoria):
        memoria.get.return_value = {"id": "a" * 36, "user_id": "OUTRA_PESSOA"}
        args = _parse(["history", "a" * 36, "--user", "ana", "--config", config_path])
        assert commands.cmd_history(args) == commands.EXIT_NAO_ENCONTRADO
        memoria.history.assert_not_called()

    def test_alheio_e_inexistente_dao_a_MESMA_resposta(self, config_path, memoria,
                                                       capsys):
        """Distinguir os dois entregaria um oráculo de existência."""
        memoria.get.return_value = {"id": "a" * 36, "user_id": "OUTRA_PESSOA"}
        args = _parse(["get", "a" * 36, "--user", "ana", "--config", config_path])
        commands.cmd_get(args)
        alheio = capsys.readouterr().err

        memoria.get.return_value = None
        args = _parse(["get", "b" * 36, "--user", "ana", "--config", config_path])
        commands.cmd_get(args)
        inexistente = capsys.readouterr().err

        assert alheio.replace("a" * 36, "ID") == inexistente.replace("b" * 36, "ID")

    def test_agent_e_run_tambem_delimitam(self, config_path, memoria):
        memoria.get.return_value = {"id": "a" * 36, "user_id": "ana",
                                    "agent_id": "OUTRO_BOT"}
        args = _parse(["get", "a" * 36, "--user", "ana", "--agent", "bot",
                       "--config", config_path])
        assert commands.cmd_get(args) == commands.EXIT_NAO_ENCONTRADO


class TestEntidadesNaoMentem:
    """Truncar e engolir exceção produzem o mesmo dano: número plausível e errado."""

    class _P:
        def __init__(self, pid, payload):
            self.id, self.payload = pid, payload

    def test_truncamento_e_declarado(self, config_path, memoria, capsys):
        muitas = [self._P(f"e{i}", {"data": f"qdrant {i}", "linked_memory_ids": []})
                  for i in range(commands.POOL_ENTIDADES)]
        memoria.entity_store.list.return_value = muitas
        args = _parse(["--json", "entities", "--user", "ana", "--contains", "qdrant",
                       "--limit", "5", "--config", config_path])
        commands.cmd_entities(args)
        capturado = capsys.readouterr()
        assert "truncada" in capturado.err
        assert json.loads(capturado.out)["truncated"] is True

    def test_sem_truncamento_nao_alarma(self, config_path, memoria, capsys):
        memoria.entity_store.list.return_value = [
            self._P("e1", {"data": "qdrant", "linked_memory_ids": []})]
        args = _parse(["--json", "entities", "--user", "ana", "--contains", "qdrant",
                       "--config", config_path])
        commands.cmd_entities(args)
        capturado = capsys.readouterr()
        assert "truncada" not in capturado.err
        assert json.loads(capturado.out)["truncated"] is False

    def test_vinculo_ilegivel_e_CONTADO_nao_engolido(self, config_path, memoria,
                                                    capsys, monkeypatch):
        """Zero silencioso reordena a lista inteira — a ordenação é por vínculos."""
        def explode(_):
            raise TypeError("payload malformado")

        monkeypatch.setattr(commands, "_normalizar_ids", explode)
        memoria.entity_store.list.return_value = [
            self._P("e1", {"data": "X", "linked_memory_ids": "abc"})]
        args = _parse(["--json", "entities", "--user", "ana", "--config", config_path])
        commands.cmd_entities(args)
        capturado = capsys.readouterr()
        assert "ilegíveis" in capturado.err
        assert json.loads(capturado.out)["unreadable"] == 1

    def test_erro_de_instalacao_SOBE_em_vez_de_virar_zero(self, config_path, memoria,
                                                         monkeypatch):
        """`ImportError` é erro de instalação, não dado malformado: engoli-lo
        transformaria um ambiente quebrado em 'nenhum vínculo'."""
        def sem_core(_):
            raise ImportError("mem0.memory.utils ausente")

        monkeypatch.setattr(commands, "_normalizar_ids", sem_core)
        memoria.entity_store.list.return_value = [
            self._P("e1", {"data": "X", "linked_memory_ids": []})]
        args = _parse(["entities", "--user", "ana", "--config", config_path])
        with pytest.raises(ImportError):
            commands.cmd_entities(args)
