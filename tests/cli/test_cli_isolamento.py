"""O CLI do engine não pode depender do servidor MCP.

Esta é a fronteira que sustenta a visão de três componentes: o engine roda
sozinho. Até aqui isso era uma AFIRMAÇÃO minha; estes testes a tornam
verificável — e falsificável.

A configuração vem de um ARQUIVO gerado pelo MCP, de propósito: dependência de
dado não cria dependência de código, e o sentido do import continua um só.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

CLI_DIR = Path(__file__).resolve().parents[2] / "mem0" / "cli"


def _modulos_do_cli():
    return sorted(p for p in CLI_DIR.glob("*.py"))


class TestImportsEstaticos:
    """Nenhum arquivo do CLI menciona o pacote do MCP — nem lazy, nem em string."""

    @pytest.mark.parametrize("arquivo", _modulos_do_cli(), ids=lambda p: p.name)
    def test_nao_importa_o_mcp(self, arquivo: Path):
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        alvos = set()
        for no in ast.walk(arvore):
            if isinstance(no, ast.Import):
                alvos.update(a.name for a in no.names)
            elif isinstance(no, ast.ImportFrom) and no.module:
                alvos.add(no.module)
        proibidos = {m for m in alvos if m.startswith("mem0_mcp_selfhosted")}
        assert not proibidos, f"{arquivo.name} importa do MCP: {proibidos}"

    @pytest.mark.parametrize("arquivo", _modulos_do_cli(), ids=lambda p: p.name)
    def test_nao_importa_stack_web(self, arquivo: Path):
        """O engine é biblioteca: um CLI que puxasse starlette/jinja2 obrigaria
        quem quer só a biblioteca a carregar servidor web."""
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        alvos = set()
        for no in ast.walk(arvore):
            if isinstance(no, ast.Import):
                alvos.update(a.name.split(".")[0] for a in no.names)
            elif isinstance(no, ast.ImportFrom) and no.module:
                alvos.add(no.module.split(".")[0])
        web = alvos & {"starlette", "jinja2", "uvicorn", "fastapi", "itsdangerous"}
        assert not web, f"{arquivo.name} puxa stack web: {web}"


class TestImportEmTempoDeExecucao:
    """Prova em processo separado: nem o import do CLI nem o `--help` carregam
    o MCP. Verificação estática não pegaria um import dinâmico."""

    def _rodar(self, codigo: str) -> str:
        proc = subprocess.run(
            [sys.executable, "-c", codigo], capture_output=True, text=True, timeout=180,
        )
        assert proc.returncode == 0, proc.stderr[-500:]
        return proc.stdout.strip()

    def test_importar_o_cli_nao_carrega_o_mcp(self):
        saida = self._rodar(
            "import sys; import mem0.cli, mem0.cli.app, mem0.cli.commands;"
            "print(any(m.startswith('mem0_mcp_selfhosted') for m in sys.modules))"
        )
        assert saida == "False"

    def test_help_nao_carrega_o_mcp(self):
        # o veredito vai por STDERR: o --help escreve em stdout, e misturar os
        # dois foi o que quebrou a primeira versão deste teste.
        codigo = (
            "import sys, io, contextlib\n"
            "from mem0.cli.app import main\n"
            "with contextlib.redirect_stdout(io.StringIO()):\n"
            "    try:\n"
            "        main(['--help'])\n"
            "    except SystemExit:\n"
            "        pass\n"
            "carregou = any(m.startswith('mem0_mcp_selfhosted') for m in sys.modules)\n"
            "print(carregou, file=sys.stderr)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", codigo], capture_output=True, text=True, timeout=180,
        )
        assert proc.returncode == 0, proc.stdout[-400:]
        assert proc.stderr.strip().splitlines()[-1] == "False"


class TestContratoCompartilhado:
    """A versão do contrato existe nos dois lados e precisa casar.

    Não dá para importar o do MCP daqui (é justamente o que não pode acontecer),
    então o teste LÊ o valor do outro arquivo quando ele está presente. Onde o
    MCP não existe, o teste se declara inaplicável em vez de passar à toa.
    """

    def test_versao_bate_com_a_do_exportador(self):
        from mem0.cli.config import CONTRACT_VERSION

        candidatos = [
            Path.home() / "mem0-stack" / "mem0-mcp-selfhosted" / "src"
            / "mem0_mcp_selfhosted" / "config_export.py",
        ]
        fonte = next((c for c in candidatos if c.exists()), None)
        if fonte is None:
            pytest.skip("exportador do MCP não está presente nesta máquina")

        arvore = ast.parse(fonte.read_text(encoding="utf-8"))
        do_mcp = next(
            (no.value.value for no in ast.walk(arvore)
             if isinstance(no, ast.Assign)
             and any(getattr(t, "id", "") == "CONTRACT_VERSION" for t in no.targets)
             and isinstance(no.value, ast.Constant)),
            None,
        )
        assert do_mcp == CONTRACT_VERSION, (
            f"contrato divergente: CLI fala v{CONTRACT_VERSION}, "
            f"exportador escreve v{do_mcp} — um dos dois lados ficou para trás"
        )
