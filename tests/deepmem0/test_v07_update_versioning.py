"""DeepMem0 v0.7 — update versioning (roadmap item #7) unit tests.

Deterministic coverage for the two pure/near-pure building blocks of the
versioned update transition:

- ``_build_version_metadata`` — the field-by-field metadata policy for a new
  version (inherit-by-blacklist; reset bookkeeping/dynamics/provenance;
  re-infer event_date; stamp created_at/supersedes/task_id);
- ``_resolve_chain_head`` — follow ``superseded_by`` to the current head so a
  reused (superseded) id never branches the chain.

The full store-backed transition (mint v2 + supersede v1 + verify + compensate)
and the as_of leak fix are covered end-to-end by ``eval/eval_update_versioning.py``
(the acceptance gate, ``--expect-fixed``) and
``eval/eval_update_versioning_invariants.py`` (structural invariants).
"""

from __future__ import annotations

from types import SimpleNamespace

from mem0.memory.main import _build_version_metadata, _resolve_chain_head
import pytest

from mem0.utils.temporality import (
    FIELD_EVENT_DATE,
    FIELD_SUPERSEDED_AT,
    FIELD_SUPERSEDED_BY,
    FIELD_SUPERSEDES,
    FIELD_VERSION_NEXT,
    FIELD_VERSION_PREV,
)

TS = "2026-07-24T12:00:00+00:00"


def _rec(mem_id, **payload):
    return SimpleNamespace(id=mem_id, payload=payload)


# --------------------------------------------------------------------------- #
# _resolve_chain_head — navigates the DEDICATED _mem0_version_next lineage      #
# (v0.7.1), fail-closed on cross-scope/cycle.                                   #
# --------------------------------------------------------------------------- #

def _getter(store):
    return lambda vid: store.get(vid)


def test_resolve_single_memory_is_its_own_head():
    store = {"a": _rec("a", data="x")}
    head_id, head = _resolve_chain_head(_getter(store), "a")
    assert head_id == "a"
    assert head is store["a"]


def test_resolve_follows_version_next_to_current_head():
    store = {
        "v1": _rec("v1", data="1", **{FIELD_VERSION_NEXT: "v2"}),
        "v2": _rec("v2", data="2", **{FIELD_VERSION_NEXT: "v3"}),
        "v3": _rec("v3", data="3"),
    }
    for start in ("v1", "v2", "v3"):
        head_id, head = _resolve_chain_head(_getter(store), start)
        assert head_id == "v3", f"from {start}"


def test_resolve_ignores_semantic_superseded_by():
    # a record semantically superseded (v0.3) but with NO version lineage IS a head
    # for update/delete — resolve must NOT follow superseded_by.
    store = {"v1": _rec("v1", data="1", **{FIELD_SUPERSEDED_BY: "sem"}), "sem": _rec("sem", data="s")}
    head_id, _ = _resolve_chain_head(_getter(store), "v1")
    assert head_id == "v1"


def test_resolve_dangling_link_treats_last_valid_as_head():
    store = {"v1": _rec("v1", data="1", **{FIELD_VERSION_NEXT: "gone"})}
    head_id, head = _resolve_chain_head(_getter(store), "v1")
    assert head_id == "v1"
    assert head is store["v1"]


def test_resolve_self_reference_is_head_not_infinite():
    store = {"v1": _rec("v1", data="1", **{FIELD_VERSION_NEXT: "v1"})}
    head_id, _ = _resolve_chain_head(_getter(store), "v1")
    assert head_id == "v1"


def test_resolve_cycle_aborts_fail_closed():
    store = {
        "a": _rec("a", **{FIELD_VERSION_NEXT: "b"}),
        "b": _rec("b", **{FIELD_VERSION_NEXT: "a"}),
    }
    with pytest.raises(ValueError):
        _resolve_chain_head(_getter(store), "a")


def test_resolve_cross_scope_edge_aborts_fail_closed():
    store = {
        "v1": _rec("v1", data="1", user_id="alice", **{FIELD_VERSION_NEXT: "v2"}),
        "v2": _rec("v2", data="2", user_id="bob"),   # different owner
    }
    with pytest.raises(ValueError):
        _resolve_chain_head(_getter(store), "v1")


def test_resolve_missing_start_returns_start_and_none():
    head_id, head = _resolve_chain_head(_getter({}), "ghost")
    assert head_id == "ghost"
    assert head is None


# --------------------------------------------------------------------------- #
# _build_version_metadata                                                      #
# --------------------------------------------------------------------------- #

def _head_payload():
    return {
        "data": "old text",
        "hash": "oldhash",
        "text_lemmatized": "old lemma",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "user_id": "alice",
        "agent_id": "agent-7",
        "actor_id": "Alice",
        "role": "user",
        "memory_type": "fact",
        "importance": "high",
        "domain": "infra",
        "tags": ["db"],
        "category": "hobbies",          # arbitrary custom field
        "priority": "high",             # arbitrary custom field
        # bookkeeping / dynamics / provenance that must NOT carry forward:
        FIELD_SUPERSEDED_BY: "someone",
        FIELD_SUPERSEDED_AT: "2026-02-02T00:00:00+00:00",
        FIELD_SUPERSEDES: ["ancestor"],
        FIELD_EVENT_DATE: "2020-01-01",
        "reinforced_at": ["2026-01-01T00:00:00+00:00"],
        "access_count": 5,
        "last_accessed": "2026-01-02T00:00:00+00:00",
        "source_doc": "doc.pdf",
        "page_start": 1,
        "task_id": "old-task",
    }


def test_build_inherits_owner_and_arbitrary_custom_metadata():
    meta = _build_version_metadata(_head_payload(), "new text", None, TS, "v1", False)
    for k in ("user_id", "agent_id", "actor_id", "role", "memory_type",
              "importance", "domain", "tags", "category", "priority"):
        assert meta[k] == _head_payload()[k], k


def test_build_resets_bookkeeping_and_provenance():
    """v0.9: dynamics saíram desta lista — herdar a timeline virou DECISÃO
    (inherit_dynamics), não acidente da blacklist. O resto continua não
    herdando."""
    meta = _build_version_metadata(_head_payload(), "new text", None, TS, "v1", False)
    for k in (FIELD_SUPERSEDED_BY, FIELD_SUPERSEDED_AT, "source_doc",
              "page_start", "hash", "text_lemmatized", "updated_at"):
        assert k not in meta, f"{k} must not carry forward"
    # data is (re)computed by _create_memory downstream, not here
    assert "data" not in meta


def test_build_dynamics_follow_the_inherit_flag():
    """Sem a flag: versão nasce neutra (pré-v0.9 byte a byte). Com a flag: a
    timeline do head COPIA — o head não é tocado por esta função (pura)."""
    off = _build_version_metadata(_head_payload(), "new text", None, TS, "v1", False,
                                  inherit_dynamics=False)
    for k in ("reinforced_at", "access_count", "last_accessed"):
        assert k not in off, f"{k} herdado sem a flag"
    on = _build_version_metadata(_head_payload(), "new text", None, TS, "v1", False,
                                 inherit_dynamics=True)
    assert on["reinforced_at"] == ["2026-01-01T00:00:00+00:00"]
    assert on["access_count"] == 5
    assert on["last_accessed"] == "2026-01-02T00:00:00+00:00"


class TestVersionUpdateInheritanceBehavioral:
    """Exercita Memory._version_update DE VERDADE (self fake, método não-ligado):
    a v2 nasce com a timeline copiada + T2, o head fica INTOCADO além das marcas
    v0.7, e o notify sai DEPOIS do verify. Born-superseded não herda nada."""

    def _run(self, head_payload, *, caller=None, flag=True):
        import threading

        import mem0.memory.main as main_mod
        from mem0.configs.base import MemoryConfig

        store = {"v1": SimpleNamespace(id="v1", payload=dict(head_payload))}
        created, events = {}, []

        def _create(data, emb, metadata=None):
            created["meta"] = dict(metadata or {})
            store["v2"] = SimpleNamespace(id="v2", payload=dict(metadata or {}))
            return "v2"

        def _update(vector_id, vector=None, payload=None):
            store[vector_id] = SimpleNamespace(id=vector_id, payload=dict(payload or {}))

        cfg = MemoryConfig()
        cfg.temporality.version_inherits_dynamics = flag
        fake = SimpleNamespace(
            config=cfg,
            _version_lock=threading.Lock(),
            vector_store=SimpleNamespace(
                get=lambda vector_id: store.get(vector_id),
                update=_update,
            ),
            _create_memory=_create,
            _link_entities_for_memory=lambda *a, **k: None,
            _delete_memory=lambda *a, **k: None,
            db=SimpleNamespace(add_history=lambda *a, **k: None),
        )
        main_mod.reinforcement_observer = (
            lambda *a: events.append((a[0], a[1], a[2])))
        try:
            result = main_mod.Memory._version_update(
                fake, "v1", "texto novo", {}, caller, cfg.temporality)
        finally:
            main_mod.reinforcement_observer = None
        return result, created["meta"], store, events

    def _head(self):
        return {
            "data": "texto velho", "created_at": "2026-01-01T00:00:00+00:00",
            "user_id": "u",
            "reinforced_at": ["2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00"],
            "reinforced_by": ["created", "t3"],
            "access_count": 2, "reinforce_counts": {"t3": 1},
        }

    def test_forward_copies_timeline_adds_t2_and_leaves_head_intact(self):
        (current, old), meta, store, events = self._run(self._head())
        assert (current, old) == ("v2", "v1")
        assert meta["reinforced_at"][0] == "2026-01-01T00:00:00+00:00"
        assert meta["reinforced_by"] == ["created", "t3", "t2"]
        assert meta["reinforce_counts"] == {"t3": 1, "t2": 1}
        assert meta["first_seen_at"] == "2026-01-01T00:00:00+00:00"
        # head: timeline INTACTA (cópia, não transferência) + marcas v0.7
        head = store["v1"].payload
        assert head["reinforced_at"] == self._head()["reinforced_at"]
        assert head["access_count"] == 2
        assert head["superseded_by"] == "v2"
        # T2 notificado no carrier NOVO, depois do verify
        assert events == [("v2", "t2", "applied")]

    def test_flag_off_is_pre_v09_byte_for_byte(self):
        (current, _old), meta, store, events = self._run(self._head(), flag=False)
        assert current == "v2"
        for k in ("reinforced_at", "reinforced_by", "access_count",
                  "reinforce_counts", "first_seen_at"):
            assert k not in meta, f"{k} herdado com a flag OFF"
        assert events == [], "sem herança não há T2 (comportamento pré-v0.9)"

    def test_born_superseded_inherits_nothing(self):
        # caller com created_at ANTERIOR ao head → born-superseded: o head segue
        # atual e É ELE quem carrega a timeline; o recém-chegado nasce neutro.
        (current, _old), meta, store, events = self._run(
            self._head(), caller={"created_at": "2025-06-01T00:00:00+00:00"})
        assert current == "v1", "head continua o atual"
        for k in ("reinforced_at", "first_seen_at", "reinforce_counts"):
            assert k not in meta, f"born-superseded herdou {k}"
        assert store["v1"].payload["reinforced_at"] == self._head()["reinforced_at"]
        assert events == []


def test_build_caller_can_never_forge_dynamics():
    """Anti-forgery: mesmo com inherit ligado, dynamics do CALLER são
    descartados — só o head é fonte de timeline."""
    forged = {"reinforced_at": ["2030-01-01T00:00:00+00:00"], "access_count": 999,
              "reinforce_counts": {"t3": 999}, "first_seen_at": "1999-01-01T00:00:00+00:00"}
    meta = _build_version_metadata(_head_payload(), "new text", forged, TS, "v1", False,
                                   inherit_dynamics=True)
    assert meta["access_count"] == 5, "access_count do caller venceu o do head"
    assert meta["reinforced_at"] == ["2026-01-01T00:00:00+00:00"]
    assert "first_seen_at" not in meta  # quem escreve é o planner, nunca o caller
    off = _build_version_metadata(_head_payload(), "new text", forged, TS, "v1", False,
                                  inherit_dynamics=False)
    for k in forged:
        assert k not in off, f"{k} forjado pelo caller com inherit off"


def test_build_stamps_created_at_and_version_prev():
    meta = _build_version_metadata(_head_payload(), "new text", None, TS, "v1", False)
    assert meta["created_at"] == TS
    assert meta[FIELD_VERSION_PREV] == ["v1"]     # dedicated lineage (not semantic supersedes)
    assert FIELD_SUPERSEDES not in meta           # an update does NOT write semantic supersedes
    assert meta["created_at"] != _head_payload()["created_at"]


def test_build_immutable_scope_cannot_be_overridden_or_added():
    head = {"data": "x", "user_id": "alice"}  # head has user_id, NO agent_id/run_id
    meta = _build_version_metadata(
        head, "new", {"user_id": "mallory", "agent_id": "evil", "run_id": "evil"}, TS, "v1", False)
    assert meta.get("user_id") == "alice"         # caller cannot change ownership
    assert "agent_id" not in meta and "run_id" not in meta  # nor ADD a scope the head lacks


def test_build_strips_reserved_lineage_from_caller():
    meta = _build_version_metadata(
        _head_payload(), "new", {FIELD_VERSION_NEXT: "evil", FIELD_VERSION_PREV: ["evil"]}, TS, "v1", False)
    assert meta.get(FIELD_VERSION_NEXT) is None        # cannot be injected
    assert meta.get(FIELD_VERSION_PREV) == ["v1"]      # only the transition sets it


def test_build_takes_task_id_from_caller_not_head():
    meta = _build_version_metadata(_head_payload(), "new text", {"task_id": "tsk_new"}, TS, "v1", False)
    assert meta["task_id"] == "tsk_new"


def test_build_caller_metadata_overrides_inherited():
    meta = _build_version_metadata(
        _head_payload(), "new text",
        {"priority": "low", "new_field": "value"}, TS, "v1", False,
    )
    assert meta["priority"] == "low"        # override wins
    assert meta["new_field"] == "value"     # new caller field kept
    assert meta["category"] == "hobbies"    # non-overridden inherited


def test_build_caller_cannot_inject_version_bookkeeping():
    meta = _build_version_metadata(
        _head_payload(), "new text",
        {FIELD_SUPERSEDED_BY: "evil", "created_at": "1999-01-01"}, TS, "v1", False,
    )
    assert FIELD_SUPERSEDED_BY not in meta          # caller bookkeeping stripped
    assert meta["created_at"] == TS                 # operation_ts wins, not caller's


def test_build_reinfers_event_date_from_new_text_when_enabled():
    meta = _build_version_metadata(
        _head_payload(), "A migração ocorreu em 17 de outubro de 2023.", None, TS, "v1", True,
    )
    assert meta[FIELD_EVENT_DATE] == "2023-10-17"   # from NEW text, not the old 2020-01-01


def test_build_omits_event_date_when_disabled_or_absent():
    # disabled
    meta = _build_version_metadata(_head_payload(), "sem data aqui", None, TS, "v1", False)
    assert FIELD_EVENT_DATE not in meta
    # enabled but no unambiguous date in the new text
    meta2 = _build_version_metadata(_head_payload(), "sem data alguma", None, TS, "v1", True)
    assert FIELD_EVENT_DATE not in meta2
