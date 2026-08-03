"""CLI do DeepMem0 — a superfície própria do engine.

Existe porque a camada-base não era usável sozinha: até aqui o core só era
alcançável por outro programa (o servidor MCP). Um `deepmem0 search` no terminal
é o que torna a biblioteca um componente, e não apenas uma dependência.

`main` é importado preguiçosamente para que `import mem0.cli` não arraste o
argparse nem os comandos.

O parser vive em `app.py`, e não em `main.py`, porque uma função `main` aqui
sombrearia um módulo de mesmo nome: `from mem0.cli import main` devolveria a
função, e `mem0.cli.main` deixaria de ser importável como módulo. Custou um
teste para descobrir.
"""

from __future__ import annotations


def main(argv=None) -> int:
    from mem0.cli.app import main as _main

    return _main(argv)


__all__ = ["main"]
