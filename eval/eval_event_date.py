#!/usr/bin/env python3
"""DeepMem0 v0.6 event-date-aware ranking — the event-time proof.

Seeds twin facts whose TEXT does NOT contain the date (so dense/BM25 cannot
discriminate) but whose event_date payloads differ, into a throwaway collection
on the REAL (production-version) Qdrant, and checks:

  A. ANCHORED RANKING — a query that names a date (full date AND month+year
     forms) ranks the correctly-dated twin above the wrong-dated one. Run with
     rerank off (fusion path) AND on (tie-break path).
  B. NO-ANCHOR NO-OP — a date-free query yields an identical ranked-id list with
     event_ranking ON vs OFF (the signal is inert without an anchor).
  C. UNDATED NO-OP — on a corpus with no event_date, an anchored query yields an
     identical ranked-id list ON vs OFF (undated memories are neutral).
  D. EXPLICIT WINDOW FILTER — event_from/event_to (day, month, year, open) return
     only in-window memories, EXCLUDE undated ones, and echo event_filter. Runs
     against the real datetime index (created online on the throwaway collection).
  E. TIE-BREAK GUARD — a query topically decisive for X (event_date far) vs Y
     (event_date == anchor, different topic): X stays rank 1. Proximity never
     overturns a decisive reranker margin (mirror of eval_temporal's guard).

The divisor×penalty interaction, config bounds, parser regression and edge cases
are covered deterministically in tests/deepmem0/test_v06_event_ranking.py. This
harness proves the END-TO-END pipeline (real embeddings, real Qdrant range
filter, real fusion/rerank) and prints per-candidate score traces.

Deterministic (no LLM): facts seeded with infer=False, event_date written to the
payload directly, created_at backdated uniformly so ACT-R cannot confound.

Environment: QDRANT_URL, OLLAMA_URL, EMBED_MODEL (bge-m3), EMBED_DIMS (1024),
MEM0_LANGUAGE (pt), LLM_MODEL (instantiated, never called).

Usage:
  python eval/eval_event_date.py [--collection deepmem0_event_date] [--rerank] [--trace]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

from mem0.configs.base import RERANK_TIE_BAND

os.environ.setdefault("MEM0_TELEMETRY", "False")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "1024"))
LANGUAGE = os.environ.get("MEM0_LANGUAGE", "pt")
# Event tie-break band (roadmap item 16) + fusion weight, both optional config
# overrides. None => use the fork default (event_tie_band=0.008, weight=0.15).
# CLI --event-tie-band wins over the env; resolved in main() after argparse.
EVENT_TIE_BAND = (float(os.environ["MEM0_EVENT_TIE_BAND"])
                  if os.environ.get("MEM0_EVENT_TIE_BAND") else None)
EVENT_WEIGHT = (float(os.environ["MEM0_EVENT_WEIGHT"])
                if os.environ.get("MEM0_EVENT_WEIGHT") else None)
USER_ID = "event_date_demo"
CONTROL_USER_ID = "event_date_demo_ctl"  # undated corpus, separate scope (C)

# Twin pairs: identical topic, date NOT in the text, differing event_date. One
# leg is queried by a FULL date, one by a MONTH+YEAR — exercising both anchor
# forms. The "far" event_date is > window_days (30) from the anchor so proximity
# is 0 there.
PAIRS = [
    {
        "query": "como foi o deploy do faturamento em 17/10/2023?",  # full date
        "right": "O deploy da versão 2.0 do faturamento foi feito numa terça-feira à noite.",
        "right_event": "2023-10-17",
        "wrong": "O deploy da versão 2.0 do faturamento foi feito num sábado de manhã.",
        "wrong_event": "2024-05-03",
    },
    {
        "query": "o incidente do gateway de pagamentos de outubro de 2023",  # month+year
        "right": "O incidente do gateway de pagamentos derrubou o checkout por 40 minutos.",
        "right_event": "2023-10-09",
        "wrong": "O incidente do gateway de pagamentos derrubou o checkout por 12 minutos.",
        "wrong_event": "2022-03-20",
    },
    {
        "query": "a migração do vetor bge-m3 em 06/07/2026",  # full date
        "right": "A migração do embedder para o modelo denso foi concluída sem downtime.",
        "right_event": "2026-07-06",
        "wrong": "A migração do embedder para o modelo denso foi concluída com uma janela de manutenção.",
        "wrong_event": "2025-01-10",
    },
]

# Undated corpus for the C control (no event_date on any of these).
CONTROL = [
    "A Vetorial Labs faz all-hands toda primeira sexta-feira do mês.",
    "O time de plataforma mantém plantão com rotação semanal.",
    "As chaves de API de staging expiram a cada 90 dias.",
]

# For the D explicit-window filter: dated + undated memories under a third scope.
FILTER_USER_ID = "event_date_demo_filter"
FILTER_DATED = [
    ("Contrato de energia assinado no primeiro semestre.", "2023-03-14"),
    ("Renovação do contrato de energia no meio do ano.", "2023-07-22"),
    ("Aditivo do contrato de energia no fim do ano.", "2023-11-30"),
    ("Revisão do contrato de energia no ano seguinte.", "2024-04-05"),
]
FILTER_UNDATED = ["Observação avulsa sobre o contrato de energia, sem data."]

# For the E guard: X is topically decisive for the query but dated far; Y matches
# the anchor date but is a different topic.
GUARD_USER_ID = "event_date_demo_guard"
GUARD_X = "O relatório trimestral de vendas da região sul teve crescimento recorde."
GUARD_X_EVENT = "2020-01-05"  # far from the anchor
GUARD_Y = "A confraternização de fim de ano da equipe foi no salão principal."
GUARD_Y_EVENT = "2023-10-17"  # == anchor
GUARD_QUERY = "como foi o relatório trimestral de vendas da região sul em 17/10/2023?"

BORN_DAYS_AGO = 20  # uniform created_at so ACT-R activation is identical for all


def build_memory(rerank: bool, event_ranking: bool = True, temporality_enabled: bool = True):
    from mem0 import Memory

    # v0.6 / roadmap item 16: the post-rerank event tie-break band is a config
    # knob (event_tie_band), NOT a search param — build_memory used to hardcode
    # the temporality dict and silently pin the fork default (0.008), so
    # `--rerank` could never test a candidate band. Pass it through explicitly
    # (CLI --event-tie-band or MEM0_EVENT_TIE_BAND) and echo the effective value.
    temporality = {"enabled": temporality_enabled, "event_ranking": event_ranking}
    if EVENT_TIE_BAND is not None:
        temporality["event_tie_band"] = EVENT_TIE_BAND
    if EVENT_WEIGHT is not None:
        temporality["event_ranking_weight"] = EVENT_WEIGHT
    config = {
        "language": LANGUAGE,
        "llm": {
            "provider": "ollama",
            "config": {"model": os.environ.get("LLM_MODEL", "llama3.1"), "ollama_base_url": OLLAMA_URL},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": ARGS.collection,
                "url": QDRANT_URL,
                "embedding_model_dims": EMBED_DIMS,
                **({"api_key": os.environ["MEM0_QDRANT_API_KEY"]}
                   if os.environ.get("MEM0_QDRANT_API_KEY") else {}),
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {"model": EMBED_MODEL, "embedding_dims": EMBED_DIMS, "ollama_base_url": OLLAMA_URL},
        },
        "temporality": temporality,
    }
    if rerank:
        config["reranker"] = {
            "provider": "sentence_transformer",
            "config": {"model": os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"), "device": "cpu"},
        }
    return Memory.from_config(config)


def _seed(memory, text, user_id, event_date=None, born=None):
    result = memory.add(text, user_id=user_id, infer=False)
    mem_id = result["results"][0]["id"]
    payload = dict(memory.vector_store.get(vector_id=mem_id).payload)
    if born:
        payload["created_at"] = born
        payload["updated_at"] = born
    if event_date:
        payload["event_date"] = event_date
    memory.vector_store.update(vector_id=mem_id, payload=payload)
    return mem_id


def seed(memory) -> dict:
    born = (datetime.now(timezone.utc) - timedelta(days=BORN_DAYS_AGO)).isoformat()
    ids = {}
    for pair in PAIRS:
        ids[pair["right"]] = _seed(memory, pair["right"], USER_ID, pair["right_event"], born)
        ids[pair["wrong"]] = _seed(memory, pair["wrong"], USER_ID, pair["wrong_event"], born)
    for text in CONTROL:
        ids[text] = _seed(memory, text, CONTROL_USER_ID, None, born)
    for text, ed in FILTER_DATED:
        ids[text] = _seed(memory, text, FILTER_USER_ID, ed, born)
    for text in FILTER_UNDATED:
        ids[text] = _seed(memory, text, FILTER_USER_ID, None, born)
    ids[GUARD_X] = _seed(memory, GUARD_X, GUARD_USER_ID, GUARD_X_EVENT, born)
    ids[GUARD_Y] = _seed(memory, GUARD_Y, GUARD_USER_ID, GUARD_Y_EVENT, born)
    return ids


def search(memory, query, user_id=USER_ID, k=5, explain=False, **kw):
    return memory.search(query, user_id=user_id, top_k=k, explain=explain, **kw)


def ranked_ids(memory, query, user_id=USER_ID, k=5, **kw):
    return [r["id"] for r in search(memory, query, user_id, k, **kw)["results"]]


def trace(memory, query, ids_by_text, label):
    """Print per-candidate score traces (why the ranking came out as it did)."""
    resp = search(memory, query, USER_ID, k=6, explain=True)
    print(f"    [trace] {label}: {query!r}")
    id_to_text = {v: k[:38] for k, v in ids_by_text.items()}
    for rank, r in enumerate(resp["results"]):
        det = r.get("score_details") or {}
        ev = r.get("event_proximity")
        anchor = resp.get("event_anchor")
        print(
            f"      #{rank} {id_to_text.get(r['id'], r['id'][:8]):40} "
            f"score={r.get('score', 0):.4f} eprox={ev} "
            f"ev_boost={det.get('event_boost', 0):.4f} anchor={anchor}"
        )


def cleanup(collection):
    """Verified teardown (mirror eval_update_versioning): delete the collection
    and its entity sibling, then confirm both are gone."""
    import requests

    hdr = ({"api-key": os.environ["MEM0_QDRANT_API_KEY"]}
           if os.environ.get("MEM0_QDRANT_API_KEY") else {})
    ok = True
    for coll in (collection, collection + "_entities"):
        try:
            requests.delete(f"{QDRANT_URL}/collections/{coll}", headers=hdr, timeout=10)
            r = requests.get(f"{QDRANT_URL}/collections/{coll}", headers=hdr, timeout=10)
            if r.status_code != 404:
                print(f"    [cleanup] WARNING: {coll} still present (status {r.status_code})")
                ok = False
        except Exception as e:  # noqa: BLE001
            print(f"    [cleanup] error deleting {coll}: {e}")
            ok = False
    return ok


def main() -> int:
    global ARGS, EVENT_TIE_BAND, EVENT_WEIGHT
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="deepmem0_event_date")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--event-tie-band", type=float, default=None,
                        help="post-rerank event tie-break band (default: fork 0.008). "
                             "Overrides MEM0_EVENT_TIE_BAND. Roadmap item 16 calibration.")
    parser.add_argument("--event-weight", type=float, default=None,
                        help="fusion event_ranking_weight (default: fork 0.15). Overrides MEM0_EVENT_WEIGHT.")
    ARGS = parser.parse_args()
    if ARGS.event_tie_band is not None:
        EVENT_TIE_BAND = ARGS.event_tie_band
    if ARGS.event_weight is not None:
        EVENT_WEIGHT = ARGS.event_weight

    eff_band = EVENT_TIE_BAND if EVENT_TIE_BAND is not None else "0.008(fork default)"
    eff_weight = EVENT_WEIGHT if EVENT_WEIGHT is not None else "0.15(fork default)"
    print(f"collection={ARGS.collection} rerank={ARGS.rerank} embed={EMBED_MODEL} qdrant={QDRANT_URL} "
          f"event_tie_band={eff_band} event_weight={eff_weight}")
    failures = []
    try:
        memory_on = build_memory(ARGS.rerank, event_ranking=True)
        ids = seed(memory_on)
        print(f"seeded {len(ids)} memories ({len(PAIRS)} twin pairs + {len(CONTROL)} undated"
              f" + {len(FILTER_DATED) + len(FILTER_UNDATED)} filter + 2 guard)")

        # --- A. Anchored ranking: right-dated twin above wrong-dated ----------
        # HARD gate on the FUSION path (the event boost is the primary lever and
        # directly promotes the right-dated twin). On the RERANK path this is
        # INFORMATIONAL: at the conservative default event_tie_band (0.008) the
        # post-rerank tie-break only fires on genuine reranker ties, and real
        # near-duplicate twins sit at ~0.0084 in relevance space (measured 0.0021 on
        # the old doubly-sigmoided axis) — just over the near-tie
        # threshold — so the reranker's (noise) preference stands. Strengthening
        # this needs held-out band calibration (roadmap follow-up); it is NOT a
        # failure of the mechanism, which the fusion path proves.
        a_wins = 0
        for pair in PAIRS:
            ranked = ranked_ids(memory_on, pair["query"], k=6)
            rr = ranked.index(ids[pair["right"]]) if ids[pair["right"]] in ranked else 99
            wr = ranked.index(ids[pair["wrong"]]) if ids[pair["wrong"]] in ranked else 99
            if rr < wr:
                a_wins += 1
            elif ARGS.trace:
                trace(memory_on, pair["query"], ids, f"A-miss right_rank={rr} wrong_rank={wr}")
        gate = "informational" if ARGS.rerank else "HARD"
        band_note = f" band={EVENT_TIE_BAND if EVENT_TIE_BAND is not None else 0.008}" if ARGS.rerank else ""
        print(f"[A] correctly-dated twin outranks wrong-dated ({'rerank' if ARGS.rerank else 'fusion'}, "
              f"{gate}{band_note}): {a_wins}/{len(PAIRS)}")
        if not ARGS.rerank and a_wins < len(PAIRS):
            failures.append("A")

        # --- F. HARD rerank-path gate: the tie-break obeys its OWN spec -------
        # [A] cannot fail on the rerank path without widening the band, which is
        # the held-out calibration this project has blocked. But the band VALUE
        # is not the only thing worth gating — the MECHANISM is, and that can be
        # judged without choosing a value: measure each twin pair's actual
        # relevance gap at runtime, then require the two halves of the spec.
        #
        #   gap <  band  ->  a genuine tie: the dated twin MUST be promoted
        #   gap >= band  ->  a decision with margin: the reranker's order STANDS
        #
        # Band-value-agnostic by construction: re-calibrating the band moves
        # which branch each pair lands in, never whether the assertion holds.
        # This is what the critic's "cases bracketing the band" asks for, without
        # the overfit of hand-picking a pair to sit just inside it.
        memory_off = build_memory(ARGS.rerank, event_ranking=False)
        if ARGS.rerank:
            band = EVENT_TIE_BAND if EVENT_TIE_BAND is not None else RERANK_TIE_BAND
            f_ok, f_total, f_detail, f_branch = 0, 0, [], {"tie": 0, "margin": 0}
            for pair in PAIRS:
                resp = search(memory_on, pair["query"], USER_ID, k=6)
                by_id = {r["id"]: r for r in resp["results"]}
                right, wrong = by_id.get(ids[pair["right"]]), by_id.get(ids[pair["wrong"]])
                if not right or not wrong:
                    continue  # out of pool: [A]/[E] own that, not this gate
                rs, ws = right.get("rerank_score"), wrong.get("rerank_score")
                if rs is None or ws is None:
                    continue
                f_total += 1
                gap = abs(rs - ws)
                order = [r["id"] for r in resp["results"]]
                right_first = order.index(right["id"]) < order.index(wrong["id"])
                if gap < band:
                    f_branch["tie"] += 1
                    ok = right_first          # genuine tie -> date decides
                    why = "tie -> dated twin must win"
                else:
                    f_branch["margin"] += 1
                    # Compared against an INDEPENDENTLY captured reference (the
                    # same query with event_ranking OFF), NOT against this
                    # pipeline's own scores: `base` is monotonic in rerank_score,
                    # so `right_first == (rs > ws)` is true by construction
                    # whenever no tie-break fires — it would pass vacuously
                    # exactly when there is nothing to check.
                    off_order = ranked_ids(memory_off, pair["query"], k=6)
                    if right["id"] not in off_order or wrong["id"] not in off_order:
                        f_detail.append("margin: par fora do pool com event_ranking OFF")
                        ok = False
                    else:
                        ok = right_first == (off_order.index(right["id"])
                                             < off_order.index(wrong["id"]))
                    why = "margin -> order must equal the event_ranking=OFF reference"
                f_ok += ok
                if not ok:
                    f_detail.append(f"gap={gap:.6f} band={band} {why}")
            print(f"[F] rerank tie-break obeys its own spec (HARD, band={band}): {f_ok}/{f_total}"
                  f"  [ramos: tie={f_branch['tie']} margin={f_branch['margin']}]")
            for d in f_detail:
                print(f"    [F-miss] {d}")
            if f_total == 0:
                print("    [F] INCONCLUSIVO: nenhum par com rerank_score em ambos os gêmeos")
                failures.append("F")
            elif not (f_branch["tie"] and f_branch["margin"]):
                # A gate whose coverage depends on where the corpus happened to
                # fall is not a gate. With event_tie_band=0 (a legal config)
                # EVERY pair lands in `margin` and the tie half is never tested.
                print(f"    [F] INCONCLUSIVO: os dois ramos precisam de casos "
                      f"(tie={f_branch['tie']} margin={f_branch['margin']})")
                failures.append("F")
            elif f_ok < f_total:
                failures.append("F")

        # --- B. No-anchor no-op: ON == OFF on a date-free query ---------------
        b_ok = 0
        no_date_queries = ["fale sobre o deploy do faturamento", "o que houve com o gateway de pagamentos"]
        for q in no_date_queries:
            if ranked_ids(memory_on, q, k=6) == ranked_ids(memory_off, q, k=6):
                b_ok += 1
        print(f"[B] no-anchor query: event_ranking ON == OFF on {b_ok}/{len(no_date_queries)}")
        if b_ok < len(no_date_queries):
            failures.append("B")

        # --- C. Undated corpus no-op: anchored query, ON == OFF --------------
        c_ok = 0
        control_queries = [
            "quando é o all-hands da Vetorial Labs em 17/10/2023?",
            "o plantão do time de plataforma de outubro de 2023",
        ]
        for q in control_queries:
            on = ranked_ids(memory_on, q, user_id=CONTROL_USER_ID, k=3)
            off = ranked_ids(memory_off, q, user_id=CONTROL_USER_ID, k=3)
            if on == off:
                c_ok += 1
        print(f"[C] undated corpus: anchored query ON == OFF on {c_ok}/{len(control_queries)}")
        if c_ok < len(control_queries):
            failures.append("C")

        # --- D. Explicit window filter ---------------------------------------
        d_checks = []
        # exact day
        r = search(memory_on, "contrato de energia", FILTER_USER_ID, k=10, event_from="2023-07-22", event_to="2023-07-22")
        got = {x["id"] for x in r["results"]}
        d_checks.append(("day", got == {ids[FILTER_DATED[1][0]]}, r.get("event_filter")))
        # month
        r = search(memory_on, "contrato de energia", FILTER_USER_ID, k=10, event_from="2023-07", event_to="2023-07")
        d_checks.append(("month", {x["id"] for x in r["results"]} == {ids[FILTER_DATED[1][0]]}, r.get("event_filter")))
        # whole year 2023 -> the three 2023 dated ones, NOT the 2024 one, NOT undated
        r = search(memory_on, "contrato de energia", FILTER_USER_ID, k=10, event_from="2023", event_to="2023")
        want_2023 = {ids[FILTER_DATED[i][0]] for i in (0, 1, 2)}
        got_2023 = {x["id"] for x in r["results"]}
        d_checks.append(("year", got_2023 == want_2023, r.get("event_filter")))
        # open interval from 2024 -> only the 2024 one
        r = search(memory_on, "contrato de energia", FILTER_USER_ID, k=10, event_from="2024")
        d_checks.append(("open", {x["id"] for x in r["results"]} == {ids[FILTER_DATED[3][0]]}, r.get("event_filter")))
        # Undated exclusion is enforced by the set-equality above: the undated id
        # is never in any expected set, so if the window leaked it, that check fails.
        d_pass = sum(1 for _, ok, _ in d_checks if ok)
        for name, ok, echo in d_checks:
            if not ok:
                print(f"    [D-detail] {name} FAILED (echo={echo})")
        echo_ok = all(c[2] for c in d_checks)  # event_filter echoed on every windowed search
        print(f"[D] explicit window filter (day/month/year/open): {d_pass}/{len(d_checks)}; echo={echo_ok}")
        if d_pass < len(d_checks) or not echo_ok:
            failures.append("D")

        # --- E. Tie-break guard: decisive topic not overturned by proximity ---
        ranked = ranked_ids(memory_on, GUARD_QUERY, user_id=GUARD_USER_ID, k=2)
        e_ok = bool(ranked) and ranked[0] == ids[GUARD_X]
        print(f"[E] decisive-topic X stays rank 1 despite Y matching the anchor date: {e_ok}")
        if ARGS.trace and not e_ok:
            trace(memory_on, GUARD_QUERY, ids, "E-guard")
        if not e_ok:
            failures.append("E")

        print("RESULT:", "PASS" if not failures else f"FAIL {failures}")
        return 1 if failures else 0
    finally:
        cleanup(ARGS.collection)


if __name__ == "__main__":
    sys.exit(main())
