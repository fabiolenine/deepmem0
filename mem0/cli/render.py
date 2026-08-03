"""Saída do CLI: texto para humano, JSON para script.

Todo comando aceita `--json`. O formato de texto pode mudar; o JSON é contrato
testado — quem automatiza sobre ele não deve quebrar numa mudança cosmética.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

MAX_LARGURA = 100


def emitir(dado: Any, *, como_json: bool, texto: Optional[str] = None) -> None:
    if como_json:
        json.dump(dado, sys.stdout, ensure_ascii=False, indent=2, default=str)
        sys.stdout.write("\n")
    elif texto is not None:
        print(texto)


def encurtar(texto: str, largura: int = MAX_LARGURA) -> str:
    texto = " ".join((texto or "").split())
    return texto if len(texto) <= largura else texto[: largura - 1] + "…"


def tabela_memorias(itens: List[Dict[str, Any]]) -> str:
    """Uma linha por memória: score (quando houver), id curto e o texto."""
    if not itens:
        return "(nenhuma memória)"
    linhas = []
    for item in itens:
        mid = str(item.get("id", ""))[:8]
        texto = encurtar(item.get("memory") or item.get("data") or "")
        score = item.get("rerank_score")
        if score is None:
            score = item.get("score")
        prefixo = f"{score:.3f}  " if isinstance(score, (int, float)) else ""
        linhas.append(f"{prefixo}{mid}  {texto}")
    return "\n".join(linhas)


def tabela_eventos(eventos: List[Dict[str, Any]]) -> str:
    if not eventos:
        return "(sem histórico)"
    return "\n".join(
        f"{e.get('created_at', '')}  {e.get('event', ''):<10}  "
        f"{encurtar(e.get('new_memory') or e.get('old_memory') or '', 70)}"
        for e in eventos
    )
