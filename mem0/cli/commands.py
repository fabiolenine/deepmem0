"""Os verbos do CLI, sobre a API pública do `Memory`.

Duas regras que valem para todos:

**Escopo é explícito.** Nenhum comando adivinha `user_id`. Um default silencioso
faria o comando ler — ou pior, ESCREVER — no corpus de outra pessoa, e o erro
seria indistinguível de sucesso. `--user` é obrigatório onde o escopo importa.

**Mutação pede confirmação.** `add` só escreve com `--yes`; sem ele, mostra o que
faria e sai com código próprio. O corpus é de produção; um comando de terminal
não deve poder alterá-lo por engano de histórico de shell.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from mem0.cli import render
from mem0.cli.config import ConfigError, describe, load

# Códigos de saída — contrato, testado. Script que automatiza depende deles.
EXIT_OK = 0
EXIT_ERRO = 1
EXIT_CONFIG = 2
EXIT_NAO_ENCONTRADO = 3
EXIT_PRECISA_CONFIRMAR = 4

#: Quantas entidades varrer quando há filtro por substring. O store casa valor
#: exato, não trecho, então o filtro é local — e sem um pool ele veria só as
#: primeiras linhas quaisquer.
POOL_ENTIDADES = 5000


def _memoria(args):
    """Instancia o Memory a partir do arquivo de configuração."""
    from mem0 import Memory

    return Memory.from_config(load(args.config))


def _escopo(args) -> Dict[str, Any]:
    filtros: Dict[str, Any] = {"user_id": args.user}
    if getattr(args, "agent", None):
        filtros["agent_id"] = args.agent
    if getattr(args, "run", None):
        filtros["run_id"] = args.run
    return filtros


def _resultados(saida: Any) -> List[Dict[str, Any]]:
    """Normaliza o retorno do core, que varia entre lista e envelope."""
    if isinstance(saida, dict):
        itens = saida.get("results", [])
    else:
        itens = saida
    return [i for i in (itens or []) if isinstance(i, dict)]


# ------------------------------------------------------------------ leitura


def cmd_search(args) -> int:
    mem = _memoria(args)
    saida = mem.search(
        args.query,
        top_k=args.limit,
        filters=_escopo(args),
        threshold=args.threshold,
        rerank=args.rerank,
        reinforce=False,  # consultar pelo terminal não é re-encontro da memória
    )
    itens = _resultados(saida)
    render.emitir(
        {"query": args.query, "count": len(itens), "results": itens},
        como_json=args.json,
        texto=render.tabela_memorias(itens),
    )
    return EXIT_OK


def _no_escopo(item: Optional[Dict[str, Any]], args) -> bool:
    """O registro pertence ao escopo pedido?

    `Memory.get` busca por id e NÃO aceita filtro, então a conferência é feita
    aqui. Sem ela, um id de outro corpus seria lido — ou apagado — sem
    resistência, o que contradiz a regra de escopo explícito deste arquivo.

    Registro de outro escopo responde como INEXISTENTE, e não "sem permissão":
    a segunda resposta seria um oráculo de existência de graça.
    """
    if not item:
        return False
    return all(item.get(chave) == valor for chave, valor in _escopo(args).items())


def cmd_get(args) -> int:
    mem = _memoria(args)
    item = mem.get(args.memory_id)
    if not _no_escopo(item, args):
        print(f"memória não encontrada: {args.memory_id}", file=sys.stderr)
        return EXIT_NAO_ENCONTRADO
    render.emitir(
        item, como_json=args.json,
        texto=json.dumps(item, ensure_ascii=False, indent=2, default=str),
    )
    return EXIT_OK


def cmd_list(args) -> int:
    mem = _memoria(args)
    itens = _resultados(mem.get_all(filters=_escopo(args), top_k=args.limit))
    render.emitir(
        {"count": len(itens), "results": itens},
        como_json=args.json,
        texto=render.tabela_memorias(itens),
    )
    if len(itens) >= args.limit:
        print(
            f"\n⚠  {args.limit} é o teto pedido e ele foi ATINGIDO — pode haver mais.\n"
            f"   `get_all` não tem cursor; aumente --limit ou filtre melhor.",
            file=sys.stderr,
        )
    return EXIT_OK


def cmd_history(args) -> int:
    mem = _memoria(args)
    if not _no_escopo(mem.get(args.memory_id), args):
        print(f"memória não encontrada: {args.memory_id}", file=sys.stderr)
        return EXIT_NAO_ENCONTRADO
    eventos = mem.history(args.memory_id) or []
    render.emitir(
        {"memory_id": args.memory_id, "events": eventos},
        como_json=args.json,
        texto=render.tabela_eventos(eventos),
    )
    return EXIT_OK


# ------------------------------------------------------------------ escrita


def _confirmado(args) -> bool:
    """`--yes` confirma; `--dry-run` simula; os dois juntos são CONTRADIÇÃO.

    Deixar uma vencer a outra em silêncio seria o pior desfecho: quem escreveu
    `--dry-run` acreditaria estar simulando enquanto o comando escreve.
    """
    if getattr(args, "dry_run", False) and args.yes:
        raise ValueError(
            "--dry-run e --yes se contradizem: um simula, o outro executa. "
            "Escolha um."
        )
    return bool(args.yes)


def cmd_add(args) -> int:
    texto = args.text
    if texto == "-":
        texto = sys.stdin.read()
    if not texto.strip():
        print("nada para adicionar (texto vazio)", file=sys.stderr)
        return EXIT_ERRO

    alvo = {k: v for k, v in _escopo(args).items()}
    if not _confirmado(args):
        print(
            "PRÉVIA — nada foi escrito.\n"
            f"  escopo : {alvo}\n"
            f"  infer  : {not args.raw} (extração por LLM)\n"
            f"  texto  : {render.encurtar(texto, 200)}\n\n"
            "Escrever no corpus exige --yes.",
            file=sys.stderr,
        )
        return EXIT_PRECISA_CONFIRMAR

    mem = _memoria(args)
    saida = mem.add(
        texto,
        user_id=args.user,
        agent_id=getattr(args, "agent", None),
        run_id=getattr(args, "run", None),
        infer=not args.raw,
    )
    itens = _resultados(saida)
    render.emitir(
        {"added": len(itens), "results": itens},
        como_json=args.json,
        texto=(render.tabela_memorias(itens) if itens
               else "nenhum fato novo extraído (o LLM não achou o que guardar)"),
    )
    return EXIT_OK


def cmd_delete(args) -> int:
    """Apaga uma memória. Mostra ANTES o que será apagado.

    Confirmar um id opaco é confirmar nada — a prévia traz o texto, que é o que
    a pessoa de fato reconhece.
    """
    mem = _memoria(args)
    alvo = mem.get(args.memory_id)
    if not _no_escopo(alvo, args):
        # inclui o caso "existe, mas é de outro escopo" — mesma resposta, de
        # propósito: distinguir os dois entregaria um oráculo de existência.
        print(f"memória não encontrada: {args.memory_id}", file=sys.stderr)
        return EXIT_NAO_ENCONTRADO

    texto = alvo.get("memory") or alvo.get("data") or ""
    if not _confirmado(args):
        print(
            "PRÉVIA — nada foi apagado.\n"
            f"  id    : {args.memory_id}\n"
            f"  texto : {render.encurtar(texto, 200)}\n\n"
            "Apagar exige --yes.",
            file=sys.stderr,
        )
        return EXIT_PRECISA_CONFIRMAR

    mem.delete(args.memory_id)
    render.emitir(
        {"deleted": args.memory_id},
        como_json=args.json,
        texto=f"apagada: {args.memory_id}",
    )
    return EXIT_OK


def _normalizar_ids(bruto: Any) -> List[str]:
    """Ids de memória vinculados, pelo normalizador do core.

    Isolado num helper para que o chamador possa capturar ESTREITO: se o import
    do core falhar, isso é erro de instalação e deve subir, não virar "zero
    vínculos".
    """
    from mem0.memory.utils import normalize_linked_memory_ids

    return list(normalize_linked_memory_ids(bruto))


def cmd_entities(args) -> int:
    """Entidades do escopo, com quantos vínculos cada uma tem.

    Lê o entity store pelo próprio core (`mem.entity_store`), sem passar pelo
    MCP — que não tem caminho para isto de qualquer forma.
    """
    mem = _memoria(args)
    store = getattr(mem, "entity_store", None)
    if store is None:
        print("este build do core não expõe entity_store", file=sys.stderr)
        return EXIT_ERRO

    # `list` devolve pontos do vector store (não dicts) e o parâmetro é `top_k`;
    # em alguns backends vem aninhado numa lista de lotes.
    #
    # ⚠️ Com `--contains`, buscar apenas `limit` linhas e filtrar depois devolveria
    # vazio quase sempre: o filtro é por substring e o store não sabe fazê-lo, então
    # as `limit` primeiras linhas quaisquer dificilmente casam. Busca-se um POOL e
    # o corte pelo limite acontece no fim, sobre o que casou.
    pool = max(args.limit, POOL_ENTIDADES) if args.contains else args.limit
    bruto = store.list(filters=_escopo(args), top_k=pool) or []
    if bruto and isinstance(bruto[0], list):
        bruto = bruto[0]
    # O pool cheio significa que a varredura pode ter parado antes do fim: dizer
    # isso é o que separa "não há mais" de "não olhei mais".
    truncado = bool(args.contains) and len(bruto) >= pool

    ilegiveis = 0
    itens = []
    for ponto in bruto:
        payload = getattr(ponto, "payload", None) or {}
        dado = payload.get("data")
        if args.contains:
            alvo = args.contains.casefold()
            comparavel = str(payload.get("data_normalized") or dado or "").casefold()
            if alvo not in comparavel:
                continue
        # Os vínculos passam pelo normalizador do core: há linhas legadas gravadas
        # com `set(str)`, que itera caractere a caractere.
        #
        # ⚠️ A captura é ESTREITA de propósito. Engolir qualquer exceção em
        # `vinculos = []` produziria número plausível e errado — e como a
        # ordenação é POR número de vínculos, um zero silencioso reordena a
        # lista inteira. Linha que não normaliza é CONTADA e reportada.
        try:
            vinculos = list(_normalizar_ids(payload.get("linked_memory_ids")))
        except (TypeError, ValueError, AttributeError):
            ilegiveis += 1
            vinculos = []
        itens.append({
            "id": str(getattr(ponto, "id", "")),
            "data": dado,
            "entity_type": payload.get("entity_type"),
            "links": len(vinculos),
        })
    itens.sort(key=lambda x: (-x["links"], str(x["data"] or "").casefold()))
    total_casou = len(itens)
    itens = itens[: args.limit]     # o corte vem DEPOIS do filtro, não antes
    if truncado:
        print(
            f"\n⚠  varredura truncada em {pool} entidades — pode haver mais\n"
            f"   correspondências fora desse pool. Aumente --limit ou filtre melhor.",
            file=sys.stderr,
        )
    if ilegiveis:
        print(f"⚠  {ilegiveis} entidade(s) com vínculos ilegíveis foram contadas "
              f"como zero — o corpus tem linhas legadas malformadas.", file=sys.stderr)
    render.emitir(
        {"count": len(itens), "matched": total_casou, "scanned": len(bruto),
         "truncated": truncado, "unreadable": ilegiveis, "entities": itens},
        como_json=args.json,
        texto=("\n".join(f"{i['links']:>4}  {i['data']}" for i in itens)
               or "(nenhuma entidade)"),
    )
    return EXIT_OK


# -------------------------------------------------------------- diagnóstico


def cmd_doctor(args) -> int:
    """Sonda o ambiente SEM acrescentar dependência ao core.

    Usa `importlib.util.find_spec` para presença de pacote e `urllib` para
    serviço — nada de importar `ollama` ou `spacy` só para diagnosticar, o que
    transformaria uma checagem em dependência.
    """
    import importlib.util
    import urllib.error
    import urllib.request

    achados: List[Dict[str, Any]] = []

    def pacote(nome: str, papel: str) -> None:
        achados.append({
            "check": nome, "kind": "package", "role": papel,
            "ok": importlib.util.find_spec(nome) is not None,
        })

    def servico(nome: str, url: str, papel: str) -> None:
        estado: Dict[str, Any] = {"check": nome, "kind": "service", "role": papel,
                                  "url": url, "ok": False}
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                estado["ok"] = 200 <= resp.status < 300
                estado["status"] = resp.status
        except urllib.error.HTTPError as exc:
            estado["status"] = exc.code
        except Exception as exc:  # noqa: BLE001 — indisponível é informação
            estado["error"] = f"{type(exc).__name__}"
        achados.append(estado)

    # configuração primeiro: sem ela, o resto é adivinhação
    cfg: Dict[str, Any] = {"check": "config", "kind": "config", "ok": False}
    try:
        resumo = describe(args.config)
        cfg.update(ok=True, **{k: v for k, v in resumo.items() if k != "path"})
        cfg["path"] = resumo["path"]
    except ConfigError as exc:
        cfg["error"] = str(exc).splitlines()[0]
    achados.append(cfg)

    for nome, papel in (("qdrant_client", "vector store"), ("spacy", "entidades"),
                        ("fastembed", "BM25 esparso"),
                        ("sentence_transformers", "reranker")):
        pacote(nome, papel)

    if cfg.get("ok"):
        conf = load(args.config)
        vs = (conf.get("vector_store", {}) or {}).get("config", {}) or {}
        if vs.get("url"):
            servico("qdrant", f"{vs['url'].rstrip('/')}/healthz", "vector store")
        emb = (conf.get("embedder", {}) or {}).get("config", {}) or {}
        base = emb.get("ollama_base_url") or emb.get("base_url")
        if base:
            servico("ollama", f"{base.rstrip('/')}/api/tags", "embedder/LLM")

    ok = all(a.get("ok") for a in achados)
    linhas = []
    for a in achados:
        marca = "ok  " if a.get("ok") else "FALHA"
        detalhe = a.get("error") or a.get("status") or a.get("role") or ""
        linhas.append(f"  [{marca}] {a['check']:<22} {detalhe}")
    render.emitir(
        {"ok": ok, "checks": achados},
        como_json=args.json,
        texto="\n".join(linhas) + ("\n\nTudo pronto." if ok else "\n\nHá pendências acima."),
    )
    return EXIT_OK if ok else EXIT_ERRO


def cmd_config_show(args) -> int:
    resumo = describe(args.config)
    render.emitir(
        resumo, como_json=args.json,
        texto="\n".join(f"  {k:<18} {v}" for k, v in resumo.items()),
    )
    return EXIT_OK
