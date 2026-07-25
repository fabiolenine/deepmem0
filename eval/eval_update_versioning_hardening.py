#!/usr/bin/env python3
"""DeepMem0 v0.7 update-versioning HARDENING checks (roadmap item #7, /critic-results).

Beyond the happy-path acceptance/invariants evals, this exercises the failure-safety
and semantic edges an independent review flagged, ALWAYS enumerating the ENTIRE
collection (not just known ids) to assert: exactly one head, no orphan current, no
dangling superseded_by.

Covers: resolved-head old_id on repeated update; whole-chain delete (no resurrection);
out-of-order (born-superseded) update; in-process concurrency (linear chain, one head);
fault injection after insert / head-mark / verify (compensation leaves one clean head).

Live Qdrant (:6333, api-key) + Ollama bge-m3. Throwaway collection, cleaned at the end.
"""
import os
import threading
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
import requests

os.environ.setdefault("MEM0_TELEMETRY", "False")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
KEY = os.environ.get("QDRANT_API_KEY") or os.environ.get("MEM0_QDRANT_API_KEY")
COLL = "deepmem0_update_versioning_hard"
USER = "versioning_hard"
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
        "vector_store": {"provider": "qdrant", "config": {"collection_name": COLL, "url": QDRANT_URL, "api_key": KEY, "embedding_model_dims": 1024}},
        "embedder": {"provider": "ollama", "config": {"model": "bge-m3", "embedding_dims": 1024, "ollama_base_url": OLLAMA_URL}},
        "temporality": {"enabled": True},
    })


def all_points(m, uid=USER):
    # top_k high enough to enumerate the WHOLE collection in one scroll — the default
    # (100) silently truncates (critic #10). Test collections hold <1k points.
    return m.vector_store.list(filters={"user_id": uid}, top_k=10000)[0]


def heads(pts):
    return [p for p in pts if not (p.payload or {}).get("superseded_by")]


def dangling(m, pts):
    ids = {p.id for p in pts}
    bad = []
    for p in pts:
        sb = (p.payload or {}).get("superseded_by")
        if sb and sb not in ids:
            bad.append(p.id)
    return bad


def add(m, text, uid=USER):
    return m.add(text, user_id=uid, infer=False)["results"][0]["id"]


def wipe(m):
    for p in all_points(m):
        try:
            m.vector_store.delete(vector_id=p.id)
        except Exception:
            pass


def cleanup():
    qh = {"api-key": KEY} if KEY else {}
    for c in (COLL, COLL + "_entities"):
        try:
            requests.delete(f"{QDRANT_URL}/collections/{c}", headers=qh, timeout=15)
        except Exception:
            pass


def scenario_old_id_and_chain(m):
    print("== resolved-head old_id + whole-chain delete + no resurrection ==")
    wipe(m)
    v1 = add(m, "O cache do halcyon usa Redis.")
    r1 = m.update(v1, data="O cache do halcyon usa Memcached.")
    v2 = r1["id"]
    check(r1["old_id"] == v1, "update#1 old_id == v1")
    # update#2 reusing the ORIGINAL (now superseded) id
    r2 = m.update(v1, data="O cache do halcyon usa Hazelcast.")
    v3 = r2["id"]
    check(r2["old_id"] == v2, f"update#2 old_id == v2 (resolved head), got {r2['old_id']}")
    check(v3 not in (v1, v2), "update#2 mints v3")
    pts = all_points(m)
    check(len(heads(pts)) == 1 and heads(pts)[0].id == v3, f"exactly ONE head == v3 across the WHOLE collection (got {[h.id for h in heads(pts)]})")
    check(dangling(m, pts) == [], f"no dangling superseded_by (got {dangling(m, pts)})")
    # delete the WHOLE chain via the original id
    m.delete(v1)
    pts2 = all_points(m)
    chain_left = [p.id for p in pts2 if p.id in (v1, v2, v3)]
    check(chain_left == [], f"delete(v1) removed the WHOLE chain (left {chain_left})")
    present = [h.get("id") for h in m.search("qual cache o halcyon usa?", user_id=USER, top_k=5)["results"]]
    check(all(x not in present for x in (v1, v2, v3)), "present search returns none of the deleted versions")
    try:
        m.update(v1, data="resurrect?")
        resurrected = True
    except ValueError:
        resurrected = False
    check(not resurrected, "update on a deleted id raises (no resurrection)")


def scenario_out_of_order(m):
    print("== out-of-order update is born superseded (head stays current) ==")
    wipe(m)
    head = add(m, "O servidor Vega roda na versão 5.")
    now = datetime.now(timezone.utc)
    p = dict(m.vector_store.get(vector_id=head).payload)
    p["created_at"] = now.isoformat(); p["updated_at"] = p["created_at"]
    m.vector_store.update(vector_id=head, payload=p)
    older = (now - timedelta(days=10)).isoformat()
    r = m.update(head, data="O servidor Vega roda na versão 3.", metadata={"created_at": older})
    check(r["id"] == head, f"current stays the head (got {r['id']})")
    pts = all_points(m)
    hs = heads(pts)
    check(len(hs) == 1 and hs[0].id == head, f"exactly ONE head == the newer fact (got {[h.id for h in hs]})")
    hp = m.vector_store.get(vector_id=head).payload
    check("versão 5" in hp["data"], "head keeps the newer content (v5)")
    older_pts = [pp for pp in pts if pp.id != head]
    check(len(older_pts) == 1 and older_pts[0].payload.get("superseded_by") == head,
          "the older arrival is stored superseded_by the head")
    check(dangling(m, pts) == [], "no dangling")


def scenario_concurrency(m):
    print("== concurrent updates on the same chain -> linear chain, one head ==")
    wipe(m)
    v1 = add(m, "A fila do proteus tem tamanho 10.")
    barrier = threading.Barrier(2)
    results = {}

    def worker(tag, text):
        barrier.wait()  # force overlap
        try:
            results[tag] = m.update(v1, data=text)
        except Exception as e:  # noqa
            results[tag] = {"error": str(e)}

    t1 = threading.Thread(target=worker, args=("A", "A fila do proteus tem tamanho 20."))
    t2 = threading.Thread(target=worker, args=("B", "A fila do proteus tem tamanho 30."))
    t1.start(); t2.start(); t1.join(); t2.join()
    pts = all_points(m)
    hs = heads(pts)
    check(len(hs) == 1, f"exactly ONE head after concurrent updates (got {len(hs)}: {[h.id for h in hs]})")
    check(dangling(m, pts) == [], f"no dangling superseded_by (got {dangling(m, pts)})")
    # a linear chain of 3 (v1 + two updates), each non-head superseded exactly once
    non_heads = [p for p in pts if (p.payload or {}).get("superseded_by")]
    targets = [p.payload["superseded_by"] for p in non_heads]
    check(len(set(targets)) == len(targets), f"no two versions point to the same successor (linear, got targets {targets})")


def _wrap_fail(store, method, should_fail):
    orig = getattr(store, method)

    def wrapped(*a, **k):
        if should_fail(*a, **k):
            raise RuntimeError(f"injected failure in {method}")
        return orig(*a, **k)
    setattr(store, method, wrapped)
    return orig


def scenario_fault_injection(m):
    print("== fault injection: compensation leaves ONE clean head, no orphan/dangling ==")
    # Fault A: insert of the new version fails -> head intact, no orphan.
    wipe(m)
    v1 = add(m, "O token do lyra expira em 30 dias.")
    orig = _wrap_fail(m.vector_store, "insert", lambda *a, **k: True)
    try:
        m.update(v1, data="O token do lyra expira em 60 dias.")
        raised = False
    except Exception:
        raised = True
    finally:
        m.vector_store.insert = orig
    pts = all_points(m)
    check(raised, "[A insert-fail] update raised")
    check([h.id for h in heads(pts)] == [v1], f"[A] head unchanged == v1 (got {[h.id for h in heads(pts)]})")
    check(len(pts) == 1 and dangling(m, pts) == [], f"[A] no orphan/dangling (n={len(pts)})")

    # Fault B: head-mark (vector_store.update setting superseded_by) fails ->
    # the new version is compensated away, head stays clean.
    wipe(m)
    v1 = add(m, "O backup do mira roda às 2h.")
    def is_mark(*a, **k):
        return bool((k.get("payload") or {}).get("superseded_by"))
    orig = _wrap_fail(m.vector_store, "update", is_mark)
    try:
        m.update(v1, data="O backup do mira roda às 4h.")
        raised = False
    except Exception:
        raised = True
    finally:
        m.vector_store.update = orig
    pts = all_points(m)
    check(raised, "[B mark-fail] update raised")
    check([h.id for h in heads(pts)] == [v1], f"[B] head unchanged == v1 (got {[h.id for h in heads(pts)]})")
    check(len(pts) == 1 and dangling(m, pts) == [], f"[B] compensated: no orphan v2/dangling (n={len(pts)})")
    check("às 2h" in m.vector_store.get(vector_id=v1).payload["data"], "[B] head content intact")

    # Fault C: verification fails AFTER the head was marked -> compensation must
    # RESTORE the head's original pointer AND delete the new version.
    wipe(m)
    v1 = add(m, "O limite do rhea é 100.")
    orig_get = m.vector_store.get
    orig_update = m.vector_store.update
    state = {"marked": False, "snap": None}

    def update_c(*a, **k):
        vid = k.get("vector_id") or (a[0] if a else None)
        pl = k.get("payload")
        if pl and pl.get("superseded_by") and vid == v1 and not state["marked"]:
            state["snap"] = dict(orig_get(vector_id=v1).payload)  # pre-mark snapshot
            state["marked"] = True
        return orig_update(*a, **k)

    def get_c(*a, **k):
        vid = k.get("vector_id") or (a[0] if a else None)
        if state["marked"] and vid == v1:
            return SimpleNamespace(id=v1, payload=state["snap"])  # stale -> verify fails
        return orig_get(*a, **k)

    m.vector_store.update = update_c
    m.vector_store.get = get_c
    try:
        m.update(v1, data="O limite do rhea é 200.")
        raised = False
    except Exception:
        raised = True
    finally:
        m.vector_store.update = orig_update
        m.vector_store.get = orig_get
    pts = all_points(m)
    check(raised, "[C verify-fail] update raised")
    check([h.id for h in heads(pts)] == [v1], f"[C] exactly one head == v1, restored (got {[h.id for h in heads(pts)]})")
    check(len(pts) == 1 and dangling(m, pts) == [], f"[C] head restored + new version deleted (n={len(pts)}, dangling={dangling(m, pts)})")
    check(not (m.vector_store.get(vector_id=v1).payload or {}).get("superseded_by"), "[C] head no longer marked superseded")


def scenario_many_to_one_semantic(m):
    # v0.7.1 CORE FIX: delete of a record involved in v0.3 semantic supersedence
    # (one fact supersedes MANY) must NOT over-collect the semantic siblings — only
    # the dedicated update-version lineage is a delete unit.
    print("== many-to-one semantic supersedence: delete does NOT over-delete siblings ==")
    wipe(m)
    a = add(m, "O worker do atlas usa 2 GiB.")
    b = add(m, "O worker do atlas usa 1 CPU.")
    c = add(m, "O worker do atlas usa 4 GiB e 2 CPUs.")  # semantically supersedes A and B
    for x in (a, b):  # semantic marks (NO _mem0_version_* — this is v0.3, not v0.7)
        p = dict(m.vector_store.get(vector_id=x).payload); p["superseded_by"] = c
        m.vector_store.update(vector_id=x, payload=p)
    pc = dict(m.vector_store.get(vector_id=c).payload); pc["supersedes"] = [a, b]
    m.vector_store.update(vector_id=c, payload=pc)
    m.delete(a)  # delete a semantically-superseded record
    left = {p.id for p in all_points(m)}
    check(a not in left, "delete(A) removed A")
    check(b in left and c in left, f"semantic siblings B and C SURVIVE (no over-delete) (left={left})")
    m.delete(c)  # delete the semantic successor
    left2 = {p.id for p in all_points(m)}
    check(c not in left2 and b in left2, "delete(C) removed only C; B still survives")


def scenario_born_superseded_delete(m):
    print("== born-superseded record is collected by delete (not left behind) ==")
    wipe(m)
    head = add(m, "O servidor nyx roda na versao 9.")
    now = datetime.now(timezone.utc)
    p = dict(m.vector_store.get(vector_id=head).payload)
    p["created_at"] = now.isoformat(); p["updated_at"] = p["created_at"]
    m.vector_store.update(vector_id=head, payload=p)
    m.update(head, data="O servidor nyx roda na versao 3.", metadata={"created_at": (now - timedelta(days=10)).isoformat()})
    born = [pt.id for pt in all_points(m) if pt.id != head]
    check(len(born) == 1, f"born-superseded record exists ({born})")
    m.delete(head)  # delete via the current head
    left = {pt.id for pt in all_points(m)}
    check(head not in left and (not born or born[0] not in left),
          f"delete(head) removed head AND the born-superseded record (left={left})")


def scenario_reserved_injection(m):
    print("== reserved lineage fields are stripped from caller metadata (anti-injection) ==")
    from mem0.utils.temporality import FIELD_VERSION_NEXT, FIELD_VERSION_PREV
    wipe(m)
    r = m.add("Fato com injecao de linhagem.", user_id=USER, infer=False,
              metadata={FIELD_VERSION_NEXT: "evil-id", FIELD_VERSION_PREV: ["evil"], "custom": "ok"})
    mid = r["results"][0]["id"]
    pl = m.vector_store.get(vector_id=mid).payload or {}
    check(FIELD_VERSION_NEXT not in pl and FIELD_VERSION_PREV not in pl, "add: injected _mem0_version_* stripped")
    check(pl.get("custom") == "ok", "add: legit custom metadata kept")
    # via legacy/versioned update too
    r2 = m.update(mid, data="Fato atualizado.", metadata={FIELD_VERSION_NEXT: "evil2"})
    cur = r2.get("id", mid)
    check(FIELD_VERSION_NEXT not in (m.vector_store.get(vector_id=cur).payload or {})
          or (m.vector_store.get(vector_id=cur).payload or {}).get(FIELD_VERSION_NEXT) != "evil2",
          "update: caller cannot inject _mem0_version_next")


def scenario_scope_guard(m):
    print("== cross-scope version lineage aborts fail-closed (no mutation) ==")
    wipe(m)
    va = add(m, "Fato do user A.", uid=USER)
    vb = m.add("Fato do user B.", user_id="v07_hard_other", infer=False)["results"][0]["id"]
    # corrupt A's lineage to point across scope
    pa = dict(m.vector_store.get(vector_id=va).payload); pa["_mem0_version_next"] = vb
    m.vector_store.update(vector_id=va, payload=pa)
    raised = False
    try:
        m.delete(va)
    except ValueError:
        raised = True
    check(raised, "delete crossing scope raises (fail-closed)")
    check(m.vector_store.get(vector_id=va) is not None and m.vector_store.get(vector_id=vb) is not None,
          "neither A nor B deleted (no mutation before abort)")
    try:
        m.vector_store.delete(vector_id=vb)
    except Exception:
        pass


def scenario_pagination(m):
    print("== all_points enumerates >100 (pagination) ==")
    import uuid
    wipe(m)
    n = 130
    for i in range(n):
        m.vector_store.insert(vectors=[[0.01] * 1024], ids=[str(uuid.uuid4())],
                              payloads=[{"data": f"pag {i}", "user_id": USER}])
    check(len(all_points(m)) == n, f"all_points returned all {n} (got {len(all_points(m))})")
    wipe(m)


def _pending_ids(m):
    return {p["memory_id"] for p in m.db.list_pending_deletes()}


def scenario_intent_journal(m):
    print("== intent journal: crash em cada ponto -> reconcile converge, tombstone único ==")
    wipe(m)
    # A) crash APÓS pending, ANTES do vetor: dado NÃO se perde; reconcile completa.
    a = add(m, "intent journal fato A")
    orig = _wrap_fail(m.vector_store, "delete", lambda *x, **k: True)  # vetor sempre falha
    m.delete(a)  # caminho versionado -> retorna parcial (não levanta)
    m.vector_store.delete = orig
    check(a in _pending_ids(m), "A: intent PENDING após falha do vetor")
    check(m.vector_store.get(vector_id=a) is not None, "A: vetor VIVO (delete falhou) — sem perda de dado")
    m.reconcile_pending_deletes()
    check(a not in _pending_ids(m), "A: reconcile limpou o pending")
    check(m.vector_store.get(vector_id=a) is None, "A: reconcile removeu o vetor")
    check(m.db.has_delete_tombstone(a), "A: tombstone garantido pelo reconcile")

    # B) crash APÓS vetor+tombstone, ANTES do commit: reconcile idempotente (sem duplicar).
    b = add(m, "intent journal fato B")
    orig_c = m.db.commit_delete
    m.db.commit_delete = lambda *x, **k: (_ for _ in ()).throw(RuntimeError("inject commit"))
    try:
        m.delete(b)
    except Exception:
        pass
    m.db.commit_delete = orig_c
    check(b in _pending_ids(m), "B: intent PENDING (commit falhou)")
    check(m.vector_store.get(vector_id=b) is None, "B: vetor já removido (delete ocorreu)")
    t_before = sum(1 for h in m.db.get_history(b) if h["is_deleted"])
    m.reconcile_pending_deletes()
    t_after = sum(1 for h in m.db.get_history(b) if h["is_deleted"])
    check(b not in _pending_ids(m), "B: reconcile commitou o intent")
    check(t_before == t_after == 1, f"B: tombstone NÃO duplicado (before={t_before} after={t_after})")

    # C) delete concluído é DISTINGUÍVEL de um pendente (autoridade = estado do intent).
    c = add(m, "intent journal fato C")
    m.delete(c)
    check(c not in _pending_ids(m), "C: delete concluído -> nenhum intent pendente")
    check(m.db.has_delete_tombstone(c), "C: tombstone presente")


def scenario_partial_chain_delete(m):
    print("== falha parcial de delete de cadeia: {deleted, remaining}, head vivo, retry converge ==")
    wipe(m)
    v1 = add(m, "cadeia parcial v1")
    m.update(v1, "cadeia parcial v2")
    m.update(v1, "cadeia parcial v3")
    pts = all_points(m)
    head = heads(pts)[0].id
    hist = [p.id for p in pts if (p.payload or {}).get("superseded_by")]  # v1, v2
    # falha SÓ no delete do HEAD (deletado por ÚLTIMO) -> históricos saem, head fica
    orig = _wrap_fail(m.vector_store, "delete", lambda *a, **k: k.get("vector_id") == head)
    res = m.delete(v1)  # delete via v1 -> coleta a cadeia inteira
    m.vector_store.delete = orig
    check(isinstance(res, dict) and res.get("remaining") == [head], f"remaining == [head] (got {res})")
    check(set(res.get("deleted", [])) == set(hist), f"deleted == históricos (got {res.get('deleted')})")
    check(m.vector_store.get(vector_id=head) is not None, "head VIVO após falha parcial")
    for h in hist:
        check(m.vector_store.get(vector_id=h) is None, f"histórico {h[:8]} removido")
    res2 = m.delete(head)  # retry sem falha
    check(m.vector_store.get(vector_id=head) is None, "retry removeu o head")


def scenario_delete_residue(m):
    print("== resíduo pós-delete: tombstones por versão + sem linked_memory_ids pendurado ==")
    wipe(m)
    a1 = add(m, "residuo fato um")
    a2 = add(m, "residuo fato dois")
    # SEMEIA uma entidade determinística (SEM extração LLM) ligada a a1 E a2.
    ent_text = "Entidade Residuo"
    emb = m.embedding_model.embed(ent_text, "add")
    m.entity_store.insert(
        vectors=[emb], ids=["a0000000-0000-4000-8000-000000000001"],  # UUID fixo válido
        payloads=[{"data": ent_text, "entity_type": "concept",
                   "linked_memory_ids": [a1, a2], "user_id": USER}],
    )

    def _linked():
        out = set()
        for e in m.entity_store.list(filters={"user_id": USER}, top_k=10000)[0]:
            for mid in (e.payload or {}).get("linked_memory_ids", []):
                out.add(mid)
        return out

    m.delete(a1)  # a1 sai; entidade SOBREVIVE com a2
    check(m.db.has_delete_tombstone(a1), "tombstone DELETE p/ a1")
    check(a1 not in _linked(), "a1 removido de linked_memory_ids")
    check(a2 in _linked(), "a2 ainda ligado (entidade não deletada cedo)")
    m.delete(a2)  # a2 sai; entidade fica vazia -> deletada
    check(m.db.has_delete_tombstone(a2), "tombstone DELETE p/ a2")
    check(a2 not in _linked(), "a2 removido -> nenhum linked_memory_ids pendurado")


if __name__ == "__main__":
    m = build()
    try:
        scenario_old_id_and_chain(m)
        scenario_out_of_order(m)
        scenario_concurrency(m)
        scenario_fault_injection(m)
        scenario_many_to_one_semantic(m)
        scenario_born_superseded_delete(m)
        scenario_reserved_injection(m)
        scenario_scope_guard(m)
        scenario_pagination(m)
        scenario_intent_journal(m)
        scenario_partial_chain_delete(m)
        scenario_delete_residue(m)
    finally:
        cleanup()
    print("\nRESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FALHAS: {FAILS}")
    raise SystemExit(1 if FAILS else 0)
