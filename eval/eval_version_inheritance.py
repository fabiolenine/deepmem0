#!/usr/bin/env python3
"""DeepMem0 v0.9 — o gate SEMÂNTICO da herança de ativação no update versionado.

Por que ele existe: o golden de produção roda com weight=0/tie_band=0 — o delta
+0.0000 dele é wiring check, quase tautológico. AQUI o weight é ligado (SÓ no
eval, collection descartável) e o comportamento que o usuário pediu é medido:

  [A] fato REFORÇADO é atualizado → a versão nova SUPERA uma gêmea fresca
      igualmente plausível (a herança + T2 valem no ranking; sem herança a v2
      nasceria neutra e a queixa original voltaria);
  [B] sem double-dip: a versão VELHA (superseded) fica abaixo da nova — a
      máscara zera a ativação dela e a penalidade age sem cancelamento;
  [C] anti-degrau: update de head NEUTRO não cunha boost de nascimento — o
      seed vem do created_at do HEAD (boost ≈ nível de 1 evento genuíno,
      NUNCA os 0.667 do bug que o /critic-plan pegou);
  [D] time-travel: com as_of ANTERIOR ao update, a versão velha rankeia COM a
      ativação dela (cópia preserva a vista histórica; transferência a mataria).

Requer Qdrant + Ollama locais. Uso:
  MEM0_QDRANT_API_KEY=... python eval/eval_version_inheritance.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

os.environ.setdefault("MEM0_TELEMETRY", "False")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
COLL = "deepmem0_version_inherit"
USER = "version_inherit_eval"
WEIGHT = 0.5  # forte DE PROPÓSITO: o eval mede o mecanismo, não calibra produção
FAILS = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def build(dynamics_enabled=True):
    from mem0 import Memory

    return Memory.from_config({
        "language": "pt",
        "llm": {"provider": "ollama", "config": {
            "model": os.environ.get("LLM_MODEL", "llama3.1"),
            "ollama_base_url": OLLAMA_URL}},
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": COLL, "url": QDRANT_URL,
            "api_key": os.environ.get("MEM0_QDRANT_API_KEY"),
            "embedding_model_dims": 1024}},
        "embedder": {"provider": "ollama", "config": {
            "model": "bge-m3", "embedding_dims": 1024,
            "ollama_base_url": OLLAMA_URL}},
        "temporality": {"enabled": True},
        "dynamics": {"enabled": dynamics_enabled, "weight": WEIGHT},
    })


def cleanup():
    qh = ({"api-key": os.environ["MEM0_QDRANT_API_KEY"]}
          if os.environ.get("MEM0_QDRANT_API_KEY") else {})
    for c in (COLL, COLL + "_entities"):
        try:
            requests.delete(f"{QDRANT_URL}/collections/{c}", headers=qh, timeout=15)
        except Exception:
            pass


def payload(m, vid):
    r = m.vector_store.get(vector_id=vid)
    return dict(r.payload) if r is not None else None


def plant(m, vid, *, created_days, reinforced_days=()):
    """Backdate + timeline plantada (forma exata do reinforcement_fields)."""
    p = payload(m, vid)
    t0 = (datetime.now(timezone.utc) - timedelta(days=created_days)).isoformat()
    p["created_at"] = t0
    p["updated_at"] = t0
    if reinforced_days:
        stamps = [t0] + [
            (datetime.now(timezone.utc) - timedelta(days=d)).isoformat()
            for d in sorted(reinforced_days, reverse=True)
        ]
        p["reinforced_at"] = stamps
        p["reinforced_by"] = ["created"] + ["t3"] * len(reinforced_days)
        p["access_count"] = len(stamps)
        p["reinforce_counts"] = {"t3": len(reinforced_days)}
        p["last_accessed"] = stamps[-1]
    m.vector_store.update(vector_id=vid, payload=p)
    return t0


def rank_of(m, query, vid, k=5, as_of=None):
    kw = {"as_of": as_of} if as_of else {}
    hits = m.search(query, user_id=USER, top_k=k, **kw)["results"]
    ids = [h.get("id") for h in hits]
    return ids.index(vid) if vid in ids else None, ids


def main() -> int:
    cleanup()
    m = build()
    add = lambda text: m.add(text, user_id=USER, infer=False)["results"][0]["id"]

    print("== semeadura ==")
    # Par A: alvo (reforçado, será atualizado) vs gêmea fresca igualmente plausível
    a1 = add("O serviço de ingestão Hermes usa fila SQLite com WAL para os jobs.")
    twin = add("O serviço de ingestão Hermes processa jobs em um banco de filas local.")
    # Cenário C: head neutro, será atualizado
    c1 = add("O coletor Boreas publica métricas no gateway a cada 30 segundos.")
    # Cenário D: distrator fresco pré-âncora
    d_dist = add("O painel Boreas mostra métricas de gateway em tempo real no grafana.")

    plant(m, a1, created_days=30, reinforced_days=(20, 10, 2))
    plant(m, twin, created_days=30)
    t0_c = plant(m, c1, created_days=30)
    plant(m, d_dist, created_days=30)

    print("== updates versionados ==")
    a2 = m.update(a1, data="O serviço de ingestão Hermes usa fila SQLite em modo WAL para os jobs.")["id"]
    c2 = m.update(c1, data="O coletor Boreas publica métricas no gateway a cada 45 segundos.")["id"]
    check(a2 != a1 and c2 != c1, "updates criaram versões novas")

    q_hermes = "como o Hermes guarda os jobs de ingestão?"
    q_boreas = "com que frequência o Boreas publica métricas?"

    print("\n[A] herança vale no ranking: v2 do fato reforçado supera a gêmea fresca")
    p_a2 = payload(m, a2)
    check((p_a2.get("reinforced_by") or [])[-1] == "t2", "v2 carrega a timeline + T2")
    r_a2, ids = rank_of(m, q_hermes, a2)
    r_twin, _ = rank_of(m, q_hermes, twin)
    check(r_a2 is not None and r_twin is not None and r_a2 < r_twin,
          f"v2 (herdada) acima da gêmea fresca (v2 rank {r_a2}, gêmea rank {r_twin})")

    print("\n[B] sem double-dip: a versão velha (superseded) fica abaixo da nova")
    r_a1, _ = rank_of(m, q_hermes, a1)
    check(r_a1 is None or r_a1 > r_a2,
          f"v1 mascarada+penalizada abaixo da v2 (v1 rank {r_a1}, v2 rank {r_a2})")

    print("\n[C] anti-degrau: update de head neutro não cunha boost de nascimento")
    from mem0.utils.dynamics import boost_from_payload
    p_c2 = payload(m, c2)
    boost = boost_from_payload(p_c2)
    # esperado: seed no created_at do HEAD (30d => 30^-0.5≈0.183) + T2 agora (1.0)
    expected = (0.183 + 1.0) / (1 + 0.183 + 1.0)
    check(p_c2.get("reinforced_at", [None])[0] == t0_c,
          "seed da v2 = created_at do HEAD (30d atrás), não o instante do update")
    check(abs(boost - expected) < 0.03 and boost < 0.6,
          f"boost bounded ({boost:.3f} ≈ {expected:.3f}; o degrau do bug seria 0.667)")

    print("\n[D] time-travel: as_of antes do update mantém a ativação da versão velha")
    anchor = (datetime.now(timezone.utc) - timedelta(days=5)).date().isoformat()
    r_v1, ids_asof = rank_of(m, q_hermes, a1, as_of=anchor)
    check(r_v1 is not None and a2 not in ids_asof,
          f"as_of: v1 volta, v2 filtrada (ids {ids_asof})")
    m_off = build(dynamics_enabled=False)
    r_v1_off, _ = rank_of(m_off, q_hermes, a1, as_of=anchor)
    check(r_v1 is not None and r_v1_off is not None and r_v1 <= r_v1_off,
          f"ativação histórica conta no as_of (on rank {r_v1} <= off rank {r_v1_off})")

    cleanup()
    print("\nRESULT: " + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        cleanup()
