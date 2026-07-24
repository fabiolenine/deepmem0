#!/usr/bin/env python3
"""In-process invariant validation for DeepMem0 update versioning (item #7).
Beyond the acceptance eval: proves present=v2, exactly-one-current, bidirectional
lineage, and the critical anti-branching (repeated update on the ORIGINAL id).
Throwaway collection, cleaned at the end."""
import os
from datetime import datetime, timedelta, timezone
import requests

os.environ.setdefault("MEM0_TELEMETRY", "False")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
COLL = "deepmem0_update_versioning_inv"
USER = "versioning_validate"
FAILS = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def build():
    from mem0 import Memory
    return Memory.from_config({
        "language": "pt",
        "llm": {"provider": "ollama", "config": {"model": os.environ.get("LLM_MODEL", "llama3.1"), "ollama_base_url": OLLAMA_URL}},
        "vector_store": {"provider": "qdrant", "config": {"collection_name": COLL, "url": QDRANT_URL, "api_key": os.environ.get("MEM0_QDRANT_API_KEY"), "embedding_model_dims": 1024}},
        "embedder": {"provider": "ollama", "config": {"model": "bge-m3", "embedding_dims": 1024, "ollama_base_url": OLLAMA_URL}},
        "temporality": {"enabled": True},
    })


def payload(m, vid):
    r = m.vector_store.get(vector_id=vid)
    return dict(r.payload) if r is not None else None


def cleanup():
    qh = {"api-key": os.environ.get("MEM0_QDRANT_API_KEY")} if os.environ.get("MEM0_QDRANT_API_KEY") else {}
    for c in (COLL, COLL + "_entities"):
        try:
            requests.delete(f"{QDRANT_URL}/collections/{c}", headers=qh, timeout=15)
        except Exception:
            pass


try:
    m = build()
    # v1
    v1 = m.add("O atlas_ingest usa um banco de dados MySQL.", user_id=USER, infer=False)["results"][0]["id"]
    # backdate v1 to 30d ago
    p = payload(m, v1); t0 = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    p["created_at"] = t0; p["updated_at"] = t0; m.vector_store.update(vector_id=v1, payload=p)

    print("== update #1 (v1 -> v2) ==")
    res1 = m.update(v1, data="O atlas_ingest usa um banco de dados PostgreSQL.")
    v2 = res1.get("id")
    check(v2 and v2 != v1, f"update devolve id novo (v2={v2!r} != v1)")
    check(res1.get("old_id") == v1, "update devolve old_id == v1")
    p1, p2 = payload(m, v1), payload(m, v2)
    check(p1 is not None and p1.get("created_at") == t0, "v1 mantém created_at=T0 (R14/eval premissa)")
    check(p1.get("superseded_by") == v2, "v1.superseded_by == v2")
    check(p1.get("superseded_at") is not None, "v1.superseded_at setado")
    check(p2.get("supersedes") == [v1], "v2.supersedes == [v1] (direção reversa)")
    check(p2.get("superseded_by") in (None, ""), "v2 não é superseded (é vigente)")
    check("mysql" in p1["data"].lower(), "v1 mantém conteúdo antigo (mysql)")
    check("postgresql" in p2["data"].lower(), "v2 tem conteúdo novo (postgresql)")
    check(p2.get("created_at") != t0 and p2.get("created_at") > t0, "v2.created_at = now (> T0)")
    # neutral dynamics on v2
    check("reinforced_at" not in p2 and "access_count" not in p2, "v2 nasce neutro (sem dynamics herdada)")

    print("== present search returns v2, as_of returns v1 ==")
    now_hits = m.search("qual banco o atlas_ingest usa?", user_id=USER, top_k=5)["results"]
    top = now_hits[0] if now_hits else {}
    check(top.get("id") == v2, f"presente: top-1 é v2 (got {top.get('id')})")
    anchor = (datetime.now(timezone.utc) - timedelta(days=15)).date().isoformat()
    as_of_hits = m.search("qual banco o atlas_ingest usa?", user_id=USER, top_k=5, as_of=anchor)["results"]
    ids_asof = [h.get("id") for h in as_of_hits]
    check(v1 in ids_asof and v2 not in ids_asof, "as_of(15d): v1 presente, v2 filtrado")

    print("== exactly one current (non-superseded) in the chain ==")
    sup = {vid: (payload(m, vid) or {}).get("superseded_by") for vid in (v1, v2)}
    currents = [vid for vid, s in sup.items() if not s]
    check(currents == [v2], f"exatamente 1 vigente = v2 (got {currents})")

    print("== update #2 on the ORIGINAL (now superseded) id: must NOT branch ==")
    res2 = m.update(v1, data="O atlas_ingest usa um banco de dados MariaDB.")  # reuse stale v1
    v3 = res2.get("id")
    check(v3 not in (v1, v2), f"update#2 cria v3 novo (got {v3})")
    p1b, p2b, p3 = payload(m, v1), payload(m, v2), payload(m, v3)
    check(p1b.get("superseded_by") == v2, "v1 ainda aponta p/ v2 (first-marking-wins, sem re-marcar)")
    check(p2b.get("superseded_by") == v3, "v2 agora superseded_by v3 (resolveu p/ a cabeça!)")
    check(p3.get("supersedes") == [v2], "v3.supersedes == [v2]")
    check(not p3.get("superseded_by"), "v3 é a nova cabeça vigente")
    sup2 = {vid: (payload(m, vid) or {}).get("superseded_by") for vid in (v1, v2, v3)}
    currents2 = [vid for vid, s in sup2.items() if not s]
    check(currents2 == [v3], f"ainda exatamente 1 vigente = v3 (SEM branching) (got {currents2})")

    print("== history(v1) mostra SUPERSEDED ==")
    h = m.history(v1)
    events = [e.get("event") for e in h]
    check("SUPERSEDED" in events, f"history(v1) tem SUPERSEDED (got {events})")
finally:
    cleanup()

print("\nRESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FALHAS: {FAILS}")
raise SystemExit(1 if FAILS else 0)
