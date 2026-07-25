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


def test_build_resets_bookkeeping_dynamics_and_provenance():
    meta = _build_version_metadata(_head_payload(), "new text", None, TS, "v1", False)
    for k in (FIELD_SUPERSEDED_BY, FIELD_SUPERSEDED_AT, "reinforced_at",
              "access_count", "last_accessed", "source_doc", "page_start",
              "hash", "text_lemmatized", "updated_at"):
        assert k not in meta, f"{k} must not carry forward"
    # data is (re)computed by _create_memory downstream, not here
    assert "data" not in meta


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
