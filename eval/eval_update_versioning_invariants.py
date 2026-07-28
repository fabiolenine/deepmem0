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
    check(p2.get("_mem0_version_prev") == [v1], "v2._mem0_version_prev == [v1] (linhagem reversa)")
    check(p2.get("superseded_by") in (None, ""), "v2 não é superseded (é vigente)")
    check("mysql" in p1["data"].lower(), "v1 mantém conteúdo antigo (mysql)")
    check("postgresql" in p2["data"].lower(), "v2 tem conteúdo novo (postgresql)")
    check(p2.get("created_at") != t0 and p2.get("created_at") > t0, "v2.created_at = now (> T0)")
    # v0.9: herança de ativação — o T2 sobre um head NEUTRO semeia do created_at
    # DO HEAD (t0) e anexa o evento; v2 carrega timeline + âncora, v1 fica
    # intocado (cópia, não transferência; aqui v1 era neutro e segue neutro).
    check(p2.get("reinforced_at", [None])[0] == t0,
          "v2 herda: seed da timeline = created_at do HEAD (t0), nunca o da operação")
    check(p2.get("reinforced_by") == ["created", "t2"], "v2: proveniência [created, t2]")
    check(p2.get("first_seen_at") == t0, "v2.first_seen_at = t0 (âncora de Petrov)")
    check("reinforced_at" not in (p1 or {}), "v1 (neutro) segue neutro — head intocado")

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
    check(p3.get("_mem0_version_prev") == [v2], "v3._mem0_version_prev == [v2]")
    check(not p3.get("superseded_by"), "v3 é a nova cabeça vigente")
    # v0.9: propagação da âncora pela cadeia (contra o store REAL) + janela:
    # o update#2 roda segundos após o #1, então o T2 dele é SUPRIMIDO — v3
    # copia a timeline de v2 sem evento novo.
    check(p3.get("first_seen_at") == t0, "v3.first_seen_at propaga = t0 (v1→v2→v3)")
    check(p3.get("reinforced_at") == p2b.get("reinforced_at"),
          "v3 copia a timeline de v2 (T2 do update#2 suprimido pela janela de 1h)")
    sup2 = {vid: (payload(m, vid) or {}).get("superseded_by") for vid in (v1, v2, v3)}
    currents2 = [vid for vid, s in sup2.items() if not s]
    check(currents2 == [v3], f"ainda exatamente 1 vigente = v3 (SEM branching) (got {currents2})")

    print("== history(v1) mostra SUPERSEDED ==")
    h = m.history(v1)
    events = [e.get("event") for e in h]
    check("SUPERSEDED" in events, f"history(v1) tem SUPERSEDED (got {events})")

    print("== v0.9: crash→purge→retry NÃO perde a timeline (o blocker da transferência) ==")
    # O ciclo real da fila v0.4: worker morre depois do update e antes do
    # mark_done; o retry PURGA a v2 (created_at==submitted_at) e reprocessa.
    # Com CÓPIA, a v1 fica intacta e a v2' re-herda; com transferência, a
    # timeline teria morrido junto com a v2 purgada — é ESTE cenário que
    # derrubou o desenho original no /critic-plan.
    w1 = m.add("O worker Atlas processa 4 lotes em paralelo.", user_id=USER, infer=False)["results"][0]["id"]
    pw = payload(m, w1)
    tw0 = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    pw.update({
        "created_at": tw0, "updated_at": tw0,
        "reinforced_at": [tw0, (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()],
        "reinforced_by": ["created", "t3"], "access_count": 2,
        "reinforce_counts": {"t3": 1},
    })
    m.vector_store.update(vector_id=w1, payload=pw)
    w2 = m.update(w1, data="O worker Atlas processa 8 lotes em paralelo.")["id"]
    check((payload(m, w2) or {}).get("reinforced_by", [])[-1:] == ["t2"], "1ª tentativa: v2 herdou + T2")
    m.vector_store.delete(vector_id=w2)  # o purge-on-retry da fila
    check((payload(m, w1) or {}).get("reinforced_at") is not None,
          "v1 INTACTA após o purge (cópia, não transferência)")
    # retry: mesmo update re-submetido contra o id original; version_next da v1
    # está pendurado (v2 morta) → _resolve_chain_head trata v1 como head.
    w2b = m.update(w1, data="O worker Atlas processa 8 lotes em paralelo.")["id"]
    p_w2b = payload(m, w2b)
    check(w2b not in (w1, w2) and p_w2b is not None, f"retry criou v2' nova ({w2b})")
    check((p_w2b.get("reinforced_at") or [None])[0] == tw0,
          "v2' RE-HERDOU a timeline da v1 intacta — histórico sobreviveu ao ciclo")
    check(p_w2b.get("first_seen_at") == tw0, "v2'.first_seen_at re-ancorado em tw0")

    # === §F (v0.7.2): MATRIZ de as_of com timestamps FIXOS (determinística) =========
    # Boundary é inclusivo (lte em created_at); a penalidade de supersedência é isenta
    # sse superseded_at > as_of. Semeia uma cadeia com created_at/superseded_at EXATOS
    # (não wall-clock) e asseta INCLUSÃO DE CANDIDATO (conjunto, não top-1) + demoção.
    print("== matriz de as_of (timestamps fixos, inclusão de candidato + demoção) ==")
    from mem0.utils.temporality import superseded_penalty_applies, parse_as_of
    for p in m.vector_store.list(filters={"user_id": USER}, top_k=10000)[0]:
        m.vector_store.delete(vector_id=p.id)
    T1, T2 = "2024-03-15T10:00:00+00:00", "2024-06-20T10:00:00+00:00"
    mv1 = m.add("O serviço de pagamentos roda na região us-east-1.", user_id=USER, infer=False)["results"][0]["id"]
    mv2 = m.add("O serviço de pagamentos roda na região sa-east-1.", user_id=USER, infer=False)["results"][0]["id"]
    pp1 = payload(m, mv1); pp1.update({"created_at": T1, "updated_at": T1,
        "superseded_by": mv2, "superseded_at": T2, "_mem0_version_next": mv2})
    m.vector_store.update(vector_id=mv1, payload=pp1)
    pp2 = payload(m, mv2); pp2.update({"created_at": T2, "updated_at": T2, "_mem0_version_prev": [mv1]})
    m.vector_store.update(vector_id=mv2, payload=pp2)

    def asof_ids(anchor):
        res = m.search("onde o serviço de pagamentos roda", user_id=USER, limit=10, as_of=anchor)["results"]
        return {r.get("id") for r in res}

    # (anchor, esperado_incluídos) — INCLUSÃO via filtro de created_at (lte)
    matrix = [
        ("2024-01-01T00:00:00+00:00", set(),        "antes de v1: nenhum"),
        (T1,                          {mv1},        "exatamente v1: só v1 (lte inclusivo, v2 futuro)"),
        ("2024-05-01T00:00:00+00:00", {mv1},        "entre v1/v2: só v1"),
        (T2,                          {mv1, mv2},   "exatamente v2: v1 E v2 (ambos created<=anchor)"),
        ("2024-08-01T00:00:00+00:00", {mv1, mv2},   "depois de v2: v1 E v2"),
        ("2024-03-15",                {mv1},        "date-only do dia de v1: fim-de-dia UTC inclui v1 (10:00)"),
        ("2024-03-14",                set(),        "date-only do dia ANTERIOR: exclui v1 (prova fim-de-dia UTC)"),
    ]
    for anchor, expected, desc in matrix:
        got = asof_ids(anchor)
        check(got == expected, f"{desc} (got {sorted(str(x)[:8] for x in got)})")

    # naive vs tz-aware do MESMO instante -> idêntico
    check(asof_ids("2024-05-01T00:00:00") == asof_ids("2024-05-01T00:00:00+00:00"),
          "as_of naive == tz-aware do mesmo instante")

    # DEMOÇÃO (isenção de penalidade) — asserida DIRETO, separada do score de busca
    p_v1 = payload(m, mv1)
    check(superseded_penalty_applies(p_v1, as_of=parse_as_of(T1)[1]) is False,
          "demoção: as_of em v1 (superseded_at futuro) -> ISENTA")
    check(superseded_penalty_applies(p_v1, as_of=parse_as_of("2024-05-01T00:00:00+00:00")[1]) is False,
          "demoção: as_of entre v1/v2 -> ISENTA")
    check(superseded_penalty_applies(p_v1, as_of=parse_as_of(T2)[1]) is True,
          "demoção: as_of == superseded_at (T2) -> DEMOVIDO (lte)")
    check(superseded_penalty_applies(p_v1, as_of=parse_as_of("2024-08-01T00:00:00+00:00")[1]) is True,
          "demoção: as_of depois de v2 -> DEMOVIDO")

    # as_of inválido -> ValueError (fail-fast, mesmo com temporalidade)
    raised = False
    try:
        m.search("x", user_id=USER, as_of="não-é-data")
    except ValueError:
        raised = True
    check(raised, "as_of inválido levanta ValueError")
finally:
    cleanup()

print("\nRESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FALHAS: {FAILS}")
raise SystemExit(1 if FAILS else 0)
