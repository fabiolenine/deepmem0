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
  [D] time-travel: com as_of ANTERIOR ao update, a versão velha continua
      ELEGÍVEL a ativação (a máscara é isenta pelo mesmo predicado da
      penalidade; transferência a deixaria neutra). ⚠️ ENUNCIADO PRECISO
      (/critic-results): o VALOR da ativação é o de HOJE, sobre a timeline
      completa — as_of gata elegibilidade, não reconstrução histórica (eventos
      pós-âncora contam; reconstruir exato é impossível p/ caudas dobradas).
      Reconstrução histórica = decisão de design futura, não um bug daqui;
  [A2] ABLAÇÃO CAUSAL: mesmo corpus com version_inherits_dynamics=False — se
      a v2 vencesse a gêmea também SEM herança, o [A] seria vitória de
      similaridade disfarçada de herança.

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


def build(dynamics_enabled=True, inherits=True):
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
        "temporality": {"enabled": True, "version_inherits_dynamics": inherits},
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
    return ids.index(vid) if vid in ids else None, ids, hits


def seed_pair(m):
    """Par ADVERSARIAL do cenário A: a query usa o VOCABULÁRIO DA GÊMEA, então
    por similaridade pura a gêmea vence — só a ativação herdada pode virar a
    ordem. (1ª versão do par não era adversarial: a ablação [A2] revelou que a
    v2 vencia por similaridade mesmo SEM herança — a crítica tinha razão.)"""
    add = lambda text: m.add(text, user_id=USER, infer=False)["results"][0]["id"]
    a1 = add("O serviço de ingestão Hermes usa fila SQLite com WAL para os jobs.")
    twin = add("O serviço de ingestão Hermes processa os jobs num banco de filas local dedicado.")
    plant(m, a1, created_days=30, reinforced_days=(20, 10, 2))
    plant(m, twin, created_days=30)
    a2 = m.update(a1, data="O serviço de ingestão Hermes usa fila SQLite em modo WAL para os jobs.")["id"]
    return a1, twin, a2


#: Query com déficit CALIBRADO (sondada nos DOIS braços contra o corpus
#: COMPLETO do eval — corpus mínimo dá veredito diferente, o BM25/idf muda):
#: puxa o vocabulário da gêmea ("processa") o bastante para a gêmea vencer SEM
#: herança, e pouco o bastante para a ativação herdada virar a ordem COM.
#: Duas variantes falharam antes: a fraca (abaixo) deixava a v2 vencer por
#: similaridade nos dois braços; a forte ("...em que banco de filas local?")
#: empilhava BM25 e a gêmea vencia nos dois.
Q_HERMES = "onde o serviço Hermes processa os jobs?"
#: Query do [D]: puxa o vocabulário da PRÓPRIA v1 — o [D] mede a elegibilidade
#: da v1 na vista as_of, não o duelo com a gêmea (com a query adversarial a
#: gêmea domina o as_of e a elegibilidade não move rank — medido).
Q_HERMES_V1 = "como o Hermes guarda os jobs de ingestão?"


def seed_corpus(m):
    """Corpus COMPLETO do eval — usado pelos DOIS braços (o veredito das
    queries muda com o corpus; a ablação num corpus menor não é ablação)."""
    add = lambda text: m.add(text, user_id=USER, infer=False)["results"][0]["id"]
    a1, twin, a2 = seed_pair(m)
    c1 = add("O coletor Boreas publica métricas no gateway a cada 30 segundos.")
    d_dist = add("O painel Boreas mostra métricas de gateway em tempo real no grafana.")
    t0_c = plant(m, c1, created_days=30)
    plant(m, d_dist, created_days=30)
    c2 = m.update(c1, data="O coletor Boreas publica métricas no gateway a cada 45 segundos.")["id"]
    return {"a1": a1, "twin": twin, "a2": a2, "c1": c1, "c2": c2,
            "d_dist": d_dist, "t0_c": t0_c}


def main() -> int:
    cleanup()
    m = build()
    add = lambda text: m.add(text, user_id=USER, infer=False)["results"][0]["id"]

    print("== semeadura (corpus completo, braço COM herança) ==")
    ids_map = seed_corpus(m)
    a1, twin, a2 = ids_map["a1"], ids_map["twin"], ids_map["a2"]
    c1, c2, t0_c = ids_map["c1"], ids_map["c2"], ids_map["t0_c"]
    check(a2 != a1 and c2 != c1, "updates criaram versões novas")

    q_hermes = Q_HERMES

    print("\n[A] herança vale no ranking: v2 do fato reforçado supera a gêmea fresca")
    p_a2 = payload(m, a2)
    check((p_a2.get("reinforced_by") or [])[-1] == "t2", "v2 carrega a timeline + T2")
    r_a2, ids, hits_a = rank_of(m, q_hermes, a2)
    r_twin = ids.index(twin) if twin in ids else None  # MESMO result set (crítica: 2 buscas)
    check(r_a2 is not None and r_twin is not None and r_a2 < r_twin,
          f"v2 (herdada) acima da gêmea fresca (v2 rank {r_a2}, gêmea rank {r_twin})")

    print("\n[B] sem double-dip DE ATIVAÇÃO: a versão velha fica abaixo da nova")
    # escopo honesto (crítica): a máscara remove só o boost de ativação do
    # supersedido — BM25/entidade/slot de candidato são comportamento v0.3.
    r_a1 = ids.index(a1) if a1 in ids else None
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

    print("\n[D] time-travel: as_of antes do update — a v1 segue ELEGÍVEL a ativação")
    # ⚠️ enunciado preciso (/critic-results): a máscara é ISENTA (mesmo
    # predicado da penalidade), mas o VALOR da ativação é o de hoje sobre a
    # timeline completa — as_of não reconstrói o estado ACT-R histórico
    # (eventos pós-âncora contam; decisão de design futura, não bug daqui).
    anchor = (datetime.now(timezone.utc) - timedelta(days=5)).date().isoformat()
    # Query do [D] puxa o vocabulário da PRÓPRIA v1 (medido: com a query
    # adversarial do [A], a gêmea domina o as_of e a elegibilidade não move).
    r_v1, ids_asof, hits_asof = rank_of(m, Q_HERMES_V1, a1, as_of=anchor)
    check(r_v1 is not None and a2 not in ids_asof,
          f"as_of: v1 volta, v2 filtrada (ids {ids_asof})")
    # Oráculo no MECANISMO, não em anotação de doc: `doc["activation"]` só
    # existe no caminho COM reranker (o adjuster pós-rerank é quem anota; este
    # eval é fusão pura — 1º oráculo aqui falhou por assumir o contrário).
    from mem0.utils.temporality import parse_as_of, superseded_penalty_applies
    p_v1 = payload(m, a1)
    _iso, anchor_dt = parse_as_of(anchor)  # devolve (iso_p_filtro, dt_p_penalidade)
    check(not superseded_penalty_applies(p_v1, as_of=anchor_dt),
          "predicado: penalidade/máscara ISENTAS para v1 na âncora (elegível)")
    check(superseded_penalty_applies(p_v1, as_of=None),
          "predicado: no presente a v1 é penalizada E mascarada")
    check(boost_from_payload(p_v1) > 0,
          "v1 mantém timeline própria (cópia, não transferência) => boost > 0 quando elegível")
    m_off = build(dynamics_enabled=False)
    r_v1_off, _ids, _h = rank_of(m_off, Q_HERMES_V1, a1, as_of=anchor)
    check(r_v1 is not None and r_v1_off is not None and r_v1 < r_v1_off,
          f"a elegibilidade MOVE o rank no as_of (on {r_v1} < off {r_v1_off})")

    print("\n[E] recordação histórica (v0.10): ativação inerte, sem reforço, aviso de sucessor")
    import time as _time

    import mem0.memory.main as _main_mod

    # E1: listas de IDS ORDENADOS idênticas com dynamics on vs off (não só o
    # rank de um doc — crítica: invariante completo)
    r_on = m.search(Q_HERMES_V1, user_id=USER, top_k=5, as_of=anchor, historical=True)
    r_off = m_off.search(Q_HERMES_V1, user_id=USER, top_k=5, as_of=anchor, historical=True)
    ids_on = [h.get("id") for h in r_on["results"]]
    ids_off = [h.get("id") for h in r_off["results"]]
    check(ids_on == ids_off and len(ids_on) > 0,
          f"E1: ids ordenados IDÊNTICOS dynamics on vs off ({ids_on})")
    # E2: aviso de sucessor conhecido (v1 é superseded pela v2)
    v1_hit = next((h for h in r_on["results"] if h.get("id") == a1), None)
    check(v1_hit is not None and v1_hit.get("has_newer_version") is True,
          "E2: v1 volta marcada has_newer_version (sucessor explícito)")
    hr = r_on.get("historical_recall") or {}
    check(hr.get("results_with_newer_version", 0) >= 1 and hr.get("as_of"),
          f"E3: echo historical_recall presente ({hr})")
    # E4: recordar NUNCA reforça — mesmo reinforce=True explícito, com T3 LIGADO.
    # Controle positivo primeiro: o MESMO instrumento vê t3 numa busca default.
    m_t3 = build()
    m_t3.config.dynamics.reinforce_on_search = True
    seen_ctrl, seen_hist = [], []
    _main_mod.reinforcement_observer = lambda *a: seen_ctrl.append(a[1])
    try:
        m_t3.search(Q_HERMES_V1, user_id=USER, top_k=5, as_of=anchor, reinforce=True)
        _t0 = _time.time()
        while not seen_ctrl and _time.time() - _t0 < 5:
            _time.sleep(0.2)
    finally:
        _main_mod.reinforcement_observer = None
    check("t3" in seen_ctrl, f"E4-controle: instrumento vê t3 na busca default ({seen_ctrl})")
    _main_mod.reinforcement_observer = lambda *a: seen_hist.append(a[1])
    try:
        m_t3.search(Q_HERMES_V1, user_id=USER, top_k=5, as_of=anchor,
                    historical=True, reinforce=True)
        _time.sleep(2)
    finally:
        _main_mod.reinforcement_observer = None
    check(seen_hist == [],
          f"E4: recordação com reinforce=True NÃO reforça nada ({seen_hist})")
    # E5: kill switch nunca degrada em silêncio
    m_kill = build()
    m_kill.config.temporality.historical_recall = False
    try:
        m_kill.search(Q_HERMES_V1, user_id=USER, top_k=5, as_of=anchor, historical=True)
        check(False, "E5: historical com feature off deveria levantar erro")
    except ValueError:
        check(True, "E5: feature off => erro claro, nunca busca default disfarçada")

    print("\n[A2] ablação causal: MESMO corpus SEM herança (a vitória do [A] é da herança?)")
    cleanup()
    m_no = build(inherits=False)
    ids_no = seed_corpus(m_no)  # corpus IDÊNTICO — ablação em corpus menor não é ablação
    a1n, twinn, a2n = ids_no["a1"], ids_no["twin"], ids_no["a2"]
    p_a2n = payload(m_no, a2n)
    check("reinforced_at" not in (p_a2n or {}),
          "sem a flag, a v2 nasce neutra (pré-v0.9)")
    r_a2n, ids_n, _h = rank_of(m_no, q_hermes, a2n)
    r_twinn = ids_n.index(twinn) if twinn in ids_n else None
    check(r_twinn is not None and (r_a2n is None or r_twinn < r_a2n),
          f"SEM herança a gêmea vence a v2 neutra (gêmea {r_twinn}, v2 {r_a2n}) — "
          f"logo o [A] mede HERANÇA, não similaridade")

    cleanup()
    print("\nRESULT: " + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        cleanup()
