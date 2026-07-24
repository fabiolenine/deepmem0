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


def all_points(m):
    return m.vector_store.list(filters={"user_id": USER})[0]


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


if __name__ == "__main__":
    m = build()
    try:
        scenario_old_id_and_chain(m)
        scenario_out_of_order(m)
        scenario_concurrency(m)
        scenario_fault_injection(m)
    finally:
        cleanup()
    print("\nRESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FALHAS: {FAILS}")
    raise SystemExit(1 if FAILS else 0)
