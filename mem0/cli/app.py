"""Entrada do CLI do DeepMem0.

`deepmem0 <verbo>`. A configuração vem de um arquivo gerado pelo servidor MCP —
ver `mem0.cli.config` para o porquê.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from mem0.cli import commands
from mem0.cli.config import ConfigError

DESCRICAO = """\
DeepMem0 — memória persistente, pela linha de comando.

A configuração (embedder, vector store, reranker, idioma) vem de um arquivo
gerado pelo servidor MCP:
    mem0-mcp-selfhosted config export --output ~/.deepmem0/config.json
"""


def _escopo(p: argparse.ArgumentParser, *, obrigatorio: bool = True) -> None:
    """Escopo NUNCA tem default: um default silencioso leria ou escreveria no
    corpus de outra pessoa, e o erro pareceria sucesso."""
    p.add_argument("--user", required=obrigatorio, help="user_id (obrigatório)")
    p.add_argument("--agent", help="agent_id (opcional)")
    p.add_argument("--run", help="run_id (opcional)")


def _comuns() -> argparse.ArgumentParser:
    """Flags aceitas DEPOIS do verbo (`deepmem0 search q --config X`)."""
    comum = argparse.ArgumentParser(add_help=False)
    comum.add_argument("--config", help="caminho do config.json (ou DEEPMEM0_CONFIG)")
    comum.add_argument("--json", action="store_true",
                       help="saída em JSON (contrato estável)")
    return comum


def _fundir_globais(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve as flags que podem aparecer antes OU depois do verbo.

    As duas posições são naturais de digitar, e aceitar só uma seria uma
    pegadinha do argparse cobrada do usuário. O merge é MANUAL, com `dest`
    distintos, porque o argparse sobrescreve o valor do nível global com o
    default do subparser ao processar o subcomando — depender desse detalhe
    daria uma flag que funciona numa posição e é ignorada na outra, em silêncio.
    """
    if getattr(args, "config", None) is None:
        args.config = getattr(args, "config_global", None)
    if not getattr(args, "json", False):
        args.json = getattr(args, "json_global", False)
    return args


def construir_parser() -> argparse.ArgumentParser:
    comum = _comuns()
    p = argparse.ArgumentParser(
        prog="deepmem0", description=DESCRICAO,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", dest="config_global", metavar="CONFIG",
                   help="caminho do config.json (ou DEEPMEM0_CONFIG)")
    p.add_argument("--json", dest="json_global", action="store_true",
                   help="saída em JSON (contrato estável)")
    p.set_defaults(config=None, json=False)
    sub = p.add_subparsers(dest="cmd", metavar="<verbo>", parser_class=(
        lambda **kw: argparse.ArgumentParser(parents=[comum], **kw)))

    s = sub.add_parser("search", help="busca semântica no corpus")
    s.add_argument("query")
    _escopo(s)
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--threshold", type=float, default=0.1)
    s.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                   help="força ligar/desligar o reranker (padrão: o da configuração)")
    s.set_defaults(func=commands.cmd_search)

    g = sub.add_parser("get", help="uma memória pelo id")
    g.add_argument("memory_id")
    _escopo(g)
    g.set_defaults(func=commands.cmd_get)

    # `list` é o nome convencional em CLI; `get_all` é o nome do método do core.
    # O alias existe para que quem leu o plano (ou a API) ache o verbo que espera.
    ls = sub.add_parser("list", aliases=["get_all"], help="lista memórias do escopo")
    _escopo(ls)
    ls.add_argument("--limit", type=int, default=50)
    ls.set_defaults(func=commands.cmd_list)

    e = sub.add_parser("entities", help="entidades do escopo e seus vínculos")
    _escopo(e)
    e.add_argument("--contains", help="filtra por trecho do nome")
    e.add_argument("--limit", type=int, default=50)
    e.set_defaults(func=commands.cmd_entities)

    rm = sub.add_parser(
        "delete", help="apaga uma memória (exige confirmação)",
        description="Sem --yes mostra o que seria apagado e sai com 4, sem apagar.")
    rm.add_argument("memory_id")
    _escopo(rm)   # apagar por id SEM escopo apagaria memória de outro corpus
    rm.add_argument("--dry-run", action="store_true",
                    help="explicita a simulação (é o padrão sem --yes)")
    rm.add_argument("--yes", action="store_true", help="confirma a exclusão")
    rm.set_defaults(func=commands.cmd_delete)

    h = sub.add_parser("history", help="histórico de uma memória")
    h.add_argument("memory_id")
    _escopo(h)
    h.set_defaults(func=commands.cmd_history)

    a = sub.add_parser(
        "add", help="grava no corpus (exige --yes)",
        description="Sem --yes mostra a prévia e sai com 4, sem escrever nada.")
    a.add_argument("text", help="texto, ou '-' para ler do stdin")
    _escopo(a)
    a.add_argument("--raw", action="store_true",
                   help="grava o texto como veio, sem extração por LLM")
    # `--dry-run` e `--yes` coexistem de propósito: o plano pedia --dry-run, e o
    # default seguro (não escrever sem confirmação) é mais forte que ele. Assim
    # quem escreve `--dry-run` obtém exatamente o que espera, e quem esquece as
    # duas flags também não escreve. `--dry-run --yes` é contradição e é RECUSADO
    # em vez de uma das duas vencer em silêncio.
    a.add_argument("--dry-run", action="store_true",
                   help="explicita a simulação (é o padrão sem --yes)")
    a.add_argument("--yes", action="store_true", help="confirma a escrita")
    a.set_defaults(func=commands.cmd_add)

    d = sub.add_parser("doctor", help="checa configuração, pacotes e serviços")
    d.set_defaults(func=commands.cmd_doctor)

    c = sub.add_parser("config", help="mostra a configuração em uso")
    c.set_defaults(func=commands.cmd_config_show)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = construir_parser()
    args = _fundir_globais(parser.parse_args(argv))
    if not getattr(args, "func", None):
        parser.print_help()
        return commands.EXIT_OK
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"erro de configuração: {exc}", file=sys.stderr)
        return commands.EXIT_CONFIG
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 — o CLI reporta, não despeja stack
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return commands.EXIT_ERRO


if __name__ == "__main__":
    sys.exit(main())
