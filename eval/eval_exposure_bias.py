#!/usr/bin/env python3
"""DeepMem0 v0.2 — the exposure-bias challenge (T3 safety).

`eval_temporal.py` proves the human-memory EFFECT: between two equally-similar
twins, the reinforced one should win. It cannot answer the question that decides
whether search-triggered reinforcement (T3) is safe to enable, because there both
twins already sit on the timeline (the "cold" one carries `reinforced_at=[born]`).

The risk T3 introduces is different and asymmetric. Activation is 0 for a memory
that was never touched and jumps to ~0.54 on the FIRST reinforcement — so the
step that matters is neutral -> touched-once, not touched-once -> touched-often
(1st to 10th hit adds 0.30; 10th to 20th adds 0.046). With T3 on, whatever the
ranker already surfaces crosses that step and gains a lasting edge over memories
that never surfaced: exposure feeding on itself.

So this eval seeds the adversarial case:

    a CORRECT answer that was NEVER retrieved (truly neutral: no timeline at all)
    vs a WEAKER, still-plausible memory that was retrieved ONCE.

The correct one must keep top-1. It is run across the ranking modes so the cost
of each is a number, not an opinion:

    off        dynamics disabled (control)
    tie        weight=0, tie_band=0.002  — bounded post-rerank tie-break only
    fusion     weight=0.15, tie_band=0   — additive term at the fusion stage
    both       weight=0.15, tie_band=0.002

Usage:
  python eval/eval_exposure_bias.py [--collection deepmem0_exposure] [--rerank]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

os.environ.setdefault("MEM0_TELEMETRY", "False")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or os.environ.get("MEM0_QDRANT_API_KEY")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "1024"))
LANGUAGE = os.environ.get("MEM0_LANGUAGE", "pt")
USER_ID = "exposure_demo"

#: `answer` is what the query actually asks for; `exposed` is topically adjacent
#: and plausible, but does NOT answer it. Only `exposed` has been retrieved
#: before — which is exactly what T3 rewards.
CASES = [
    {
        "query": "qual é o limite de memória dos workers do hermes_fx?",
        "answer": "Cada worker do hermes_fx tem limite de 2 GiB de memória imposto pelo orquestrador.",
        "exposed": "Os workers do hermes_fx rodam em containers gerenciados pelo orquestrador interno.",
    },
    {
        "query": "com que frequência rodam os backups completos do Orion?",
        "answer": "Os backups completos do Orion rodam todo domingo às 03h da manhã.",
        "exposed": "Os backups do Orion são armazenados num bucket com retenção de 90 dias.",
    },
    {
        "query": "quem aprova mudanças de schema no projeto Aurora?",
        "answer": "Mudanças de schema no Aurora precisam de aprovação do comitê de dados antes do merge.",
        "exposed": "O projeto Aurora usa migrações versionadas para mudanças de schema.",
    },
    {
        "query": "qual o timeout do healthcheck do boreal_app?",
        "answer": "O timeout do healthcheck do boreal_app é de 5 segundos.",
        "exposed": "O healthcheck do boreal_app roda a cada 30 segundos em todas as instâncias.",
    },
    {
        "query": "qual banco de dados o atlas_ingest usa para eventos brutos?",
        "answer": "O atlas_ingest usa PostgreSQL particionado por dia para os eventos brutos.",
        "exposed": "O atlas_ingest processa eventos brutos em lotes de cinco minutos.",
    },
    {
        "query": "quantas tentativas tem a política de retries do boreal_app?",
        "answer": "A política de retries do boreal_app é de três tentativas com backoff exponencial.",
        "exposed": "O boreal_app registra cada retry no log estruturado com o motivo da falha.",
    },
]

MODES = {
    #             weight, tie_band
    "off": None,
    "tie": (0.0, 0.002),
    # A banda de 0.002 foi calibrada em 2026-07-21 sobre empates cujo gap real
    # media ~0.0002 — 10x de folga. Medir uma banda mais apertada mostra se o
    # dano de exposição vem do MECANISMO ou apenas da largura escolhida.
    "tie_tight": (0.0, 0.0005),
    "fusion": (0.15, 0.0),
    "both": (0.15, 0.002),
}
MODE_ORDER = ("off", "tie", "tie_tight", "fusion", "both")


def build_memory(rerank: bool, mode: str):
    from mem0 import Memory

    dynamics = {"enabled": False}
    if MODES[mode] is not None:
        weight, tie_band = MODES[mode]
        dynamics = {"enabled": True, "weight": weight, "tie_band": tie_band}
    config = {
        "language": LANGUAGE,
        "llm": {
            "provider": "ollama",
            "config": {
                "model": os.environ.get("LLM_MODEL", "llama3.1"),
                "ollama_base_url": OLLAMA_URL,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": ARGS.collection,
                "url": QDRANT_URL,
                "embedding_model_dims": EMBED_DIMS,
                **({"api_key": QDRANT_API_KEY} if QDRANT_API_KEY else {}),
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": EMBED_MODEL,
                "embedding_dims": EMBED_DIMS,
                "ollama_base_url": OLLAMA_URL,
            },
        },
        "dynamics": dynamics,
    }
    if rerank:
        config["reranker"] = {
            "provider": "sentence_transformer",
            "config": {
                "model": os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
                "device": "cpu",
            },
        }
    return Memory.from_config(config)


def seed(memory) -> dict:
    ids = {}
    for case in CASES:
        for key in ("answer", "exposed"):
            result = memory.add(case[key], user_id=USER_ID, infer=False)
            ids[case[key]] = result["results"][0]["id"]
    return ids


def apply_timelines(memory, ids) -> None:
    """Both memories are the same age. Only `exposed` was ever retrieved — ONE
    hit, yesterday, exactly what a single T3 event writes. `answer` gets NO
    timeline at all: it must stay neutral (activation None), not `[born]`."""
    now = datetime.now(timezone.utc)
    born = (now - timedelta(days=30)).isoformat()
    for case in CASES:
        for key in ("answer", "exposed"):
            point = memory.vector_store.get(vector_id=ids[case[key]])
            payload = dict(point.payload)
            payload["created_at"] = born
            payload["updated_at"] = born
            if key == "exposed":
                payload["reinforced_at"] = [born, (now - timedelta(days=1)).isoformat()]
                payload["reinforced_by"] = ["created", "t3"]
                payload["access_count"] = 2
                payload["last_accessed"] = (now - timedelta(days=1)).isoformat()
                payload["last_search_reinforced_at"] = (now - timedelta(days=1)).isoformat()
            else:
                payload.pop("reinforced_at", None)
                payload.pop("reinforced_by", None)
                payload.pop("access_count", None)
            memory.vector_store.update(vector_id=ids[case[key]], payload=payload)


def top_id(memory, query):
    results = memory.search(query, user_id=USER_ID, top_k=5, reinforce=False)["results"]
    return results[0]["id"] if results else None


def main() -> int:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        client.delete_collection(ARGS.collection)
    except Exception:
        pass

    seeder = build_memory(ARGS.rerank, "both")
    ids = seed(seeder)
    apply_timelines(seeder, ids)
    print(f"collection={ARGS.collection} rerank={ARGS.rerank} casos={len(CASES)}")
    print("cada caso: resposta CORRETA nunca recuperada  vs  vizinha plausível "
          "recuperada UMA vez\n")

    results = {}
    winners = {}
    for mode in MODE_ORDER:
        memory = build_memory(ARGS.rerank, mode)
        stolen, per_case = [], {}
        for case in CASES:
            winner = top_id(memory, case["query"])
            per_case[case["query"]] = winner
            # FALHA = a resposta correta perdeu o top-1, seja para quem for. O
            # oráculo anterior só contava quando a vizinha DESIGNADA vencia, e
            # ficava cego a um terceiro candidato roubando o topo.
            if winner != ids[case["answer"]]:
                stolen.append(case["query"])
        results[mode] = stolen
        winners[mode] = per_case
        label = {
            "off": "dynamics OFF (controle)",
            "tie": "tie-break só (weight=0, band=0.002)",
            "tie_tight": "tie-break estreito (band=0.0005)",
            "fusion": "fusão só (weight=0.15, band=0)",
            "both": "fusão + tie-break",
        }[mode]
        status = "ok " if not stolen else "FAIL"
        print(f"  [{status}] {label:38} resposta correta perdeu o top-1 em "
              f"{len(stolen)}/{len(CASES)}")
        for q in stolen:
            who = "a vizinha exposta" if winners[mode][q] == ids[
                next(c["exposed"] for c in CASES if c["query"] == q)] else "OUTRO candidato"
            print(f"          → {q}  (quem levou: {who})")

    try:
        client.delete_collection(ARGS.collection)
    except Exception:
        pass

    # O controle define o piso: se o ranking base já erra um caso, esse erro não
    # é do ACT-R. Só conta como dano de exposição o que PIORA em relação a ele.
    baseline = len(results["off"])
    print(f"\nbaseline (sem dynamics): {baseline}/{len(CASES)} — abaixo disto é o custo do ACT-R")
    # TRANSIÇÕES PAREADAS: um delta agregado zero pode esconder um modo que
    # conserta um caso e estraga outro. O que decide é o dano NOVO, não o saldo.
    verdicts = {}
    for mode in MODE_ORDER[1:]:
        harmed = [c["query"] for c in CASES
                  if winners["off"][c["query"]] == ids[c["answer"]]
                  and winners[mode][c["query"]] != ids[c["answer"]]]
        helped = [c["query"] for c in CASES
                  if winners["off"][c["query"]] != ids[c["answer"]]
                  and winners[mode][c["query"]] == ids[c["answer"]]]
        verdicts[mode] = len(harmed)
        print(f"  {mode:9} dano NOVO {len(harmed)}  |  consertou {len(helped)}  |  "
              f"saldo {len(results[mode]) - baseline:+d}")
    # O GATE é sobre o modo que vai rodar (tie-break only). `fusion`/`both` são
    # MEDIÇÃO, não modos candidatos: o dano que eles mostram é justamente a razão
    # de a fusão ficar em zero. Reportá-los como falha do gate confundiria "a
    # configuração escolhida é segura" com "toda configuração possível é segura".
    print("\n  nota: n=6 e o delta é de UM caso — sinal direcional, não número"
          " calibrado. O que ele mostra é ONDE o viés entra, não o seu tamanho.")
    if verdicts["fusion"] > 0:
        print(f"  → a fusão (weight>0) causa {verdicts['fusion']} dano(s) novo(s): é a"
              " evidência que mantém o peso da fusão em ZERO (decisão D4).")
    ok = verdicts["tie"] <= 0
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"(candidato à Fase 3 = tie-break; dano novo {verdicts['tie']})")
    print("  ⚠️ ESTE EVAL NÃO MEDE FREQUÊNCIA. 1 falha em 6 tem IC95% de ~0,4% a 64%:"
          " ele prova a EXISTÊNCIA de um par em que a banda deixa passar um flip"
          " danoso, não a taxa com que isso ocorre. Os casos foram escritos por"
          " quem conhecia o mecanismo — serve para falsificar, não para calibrar.")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="deepmem0_exposure")
    parser.add_argument("--rerank", action="store_true")
    ARGS = parser.parse_args()
    sys.exit(main())
