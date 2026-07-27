"""
DeepMem0 v0.2 — human-memory dynamics (ACT-R base-level activation).

Every memory lives on an evolving timeline: each re-encounter or use appends a
reinforcement timestamp. Relevance then carries an activation term

    B_i = ln( sum_j  dt_j^(-d) )

over that timeline (dt_j = DAYS since reinforcement j, d ~= 0.5), a single
quantity capturing both frequency (how many reinforcements) and recency (how
recent they are).

The unit of dt only shifts B by a constant, so choosing it fixes the sigmoid's
operating point. Days center it on memory-corpus timescales: "reinforced once,
today" sits exactly at boost 0.5; a month-old untouched fact ~0.15; a fact
reinforced repeatedly over the last week ~0.7+. dt is clamped to >= 1 day, so
sub-day recency is deliberately flat — a brand-new memory cannot out-activate
a genuinely reinforced one, and same-day repetition is frequency (bounded by
the reinforcement window), not recency.

Activation is DERIVED, never stored. What persists is the event history
(``reinforced_at`` timestamps + ``access_count``); the value is computed lazily
at query time, only for the candidates being ranked. There is no batch decay
job and no persisted weight to refresh — as wall-clock time passes every dt
grows and activation falls on its own, with zero writes.

Creation does NOT put a memory on the timeline. A memory is NEUTRAL — no boost,
no penalty — until it is reinforced for the first time; only then does it join
the timeline, retroactively adopting its ``created_at`` as the first encounter.
This keeps a brand-new fact on equal footing with the legacy corpus (activation
measures re-encounters, not first presentations) and means enabling dynamics
never reprices existing memories or biases fresh adds over old ones.

Bounded growth: only the most recent ``max_timestamps`` reinforcements are
retained verbatim; the older tail is folded into the standard ACT-R hybrid
approximation (Petrov, 2006) using the total count and the memory's age, so
payload size stays O(K) regardless of how old or busy a memory is.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FIELD_REINFORCED_AT = "reinforced_at"
FIELD_ACCESS_COUNT = "access_count"
FIELD_LAST_ACCESSED = "last_accessed"
#: Which trigger produced each retained timestamp, positionally aligned with
#: ``reinforced_at``. Without it a timeline is unattributable: a search exposure
#: (T3) cannot be told apart from an explicit write (T1/T2), so neither a
#: trigger-specific window nor a selective rollback of exposure events is
#: possible — only all-or-nothing deletion of the whole history.
FIELD_REINFORCED_BY = "reinforced_by"
#: Last T3 (search exposure) timestamp, tracked apart from the shared timeline so
#: the exposure window is independent of explicit-write reinforcements.
FIELD_LAST_SEARCH_REINFORCED_AT = "last_search_reinforced_at"

DYNAMICS_FIELDS = (
    FIELD_REINFORCED_AT,
    FIELD_ACCESS_COUNT,
    FIELD_LAST_ACCESSED,
    FIELD_REINFORCED_BY,
    FIELD_LAST_SEARCH_REINFORCED_AT,
)

#: Reinforcement triggers. T1/T2 are explicit writes (a fact was re-stated or
#: evolved); T3 is retrieval EXPOSURE — the memory was returned, which is weaker
#: evidence than a write and is rate-limited on its own window.
TRIGGER_DEDUP = "t1"
TRIGGER_UPDATE = "t2"
TRIGGER_SEARCH = "t3"
TRIGGER_CREATED = "created"  # the created_at seed adopted by the first reinforcement

#: Structured outcome of one reinforcement attempt. The previous boolean collapsed
#: "the window suppressed it" with "it blew up", which made the two indistinguishable
#: in telemetry — exactly the blindness this work exists to remove.
OUTCOME_APPLIED = "applied"
OUTCOME_SUPPRESSED = "suppressed"
OUTCOME_MISSING = "missing"
OUTCOME_FAILED = "failed"

# dt is measured in days, clamped to >= 1: within the first day every memory
# is equally "today", so freshness alone cannot dominate real reinforcement.
_MIN_AGE_DAYS = 1.0


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: Any) -> Optional[datetime]:
    """Tolerant ISO-8601 parse; naive datetimes are assumed UTC."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _age_days(ts: datetime, now: datetime) -> float:
    return max((now - ts).total_seconds() / 86400.0, _MIN_AGE_DAYS)


def base_level_activation(
    reinforced_at: Optional[List[Any]],
    access_count: Optional[int] = None,
    *,
    now: Optional[datetime] = None,
    decay: float = 0.5,
    first_seen: Any = None,
) -> Optional[float]:
    """ACT-R base-level activation over a reinforcement timeline.

    Exact sum over the retained timestamps; if ``access_count`` exceeds the
    retained count, the trimmed tail is approximated by spreading the missing
    reinforcements uniformly between ``first_seen`` and the oldest retained
    timestamp (Petrov 2006 hybrid approximation).

    Returns None when there is no usable history (the memory is neutral).
    """
    now = now or utcnow()
    ages = sorted(
        _age_days(ts, now)
        for ts in (_parse_ts(v) for v in (reinforced_at or []))
        if ts is not None
    )
    if not ages:
        return None

    total = sum(age ** -decay for age in ages)

    n = access_count if isinstance(access_count, int) and access_count > 0 else len(ages)
    missing = n - len(ages)
    if missing > 0:
        first = _parse_ts(first_seen)
        t_first = _age_days(first, now) if first is not None else ages[-1]
        t_oldest = ages[-1]
        if t_first > t_oldest and decay != 1.0:
            # integral-mean of t^-d over [t_oldest, t_first]
            tail_mean = (t_first ** (1.0 - decay) - t_oldest ** (1.0 - decay)) / (
                (1.0 - decay) * (t_first - t_oldest)
            )
        else:
            tail_mean = t_oldest ** -decay
        total += missing * tail_mean

    if total <= 0:
        return None
    return math.log(total)


def activation_boost(activation: Optional[float]) -> float:
    """Squash activation (-inf, ~ln n] into (0, 1); None (no history) -> 0."""
    if activation is None:
        return 0.0
    return 1.0 / (1.0 + math.exp(-activation))


def boost_from_payload(
    payload: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    decay: float = 0.5,
) -> float:
    """Activation boost in [0, 1) from a memory payload; 0 when no history."""
    if not payload or not payload.get(FIELD_REINFORCED_AT):
        return 0.0
    activation = base_level_activation(
        payload.get(FIELD_REINFORCED_AT),
        payload.get(FIELD_ACCESS_COUNT),
        now=now,
        decay=decay,
        first_seen=payload.get("created_at"),
    )
    return activation_boost(activation)


def should_reinforce(
    payload: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    window_seconds: int = 3600,
    trigger: str = TRIGGER_DEDUP,
    search_window_seconds: int = 0,
) -> bool:
    """At most one reinforcement per memory per window, across all triggers.

    Inside the window, re-encounters and hits have NO reinforcement effect —
    this absorbs client retries (a timed-out MCP client re-sending an add must
    not double-count) and approximates the ACT-R spacing effect: massed
    repetition adds nothing, spaced repetition does. ``window_seconds <= 0``
    disables the window.

    T3 (search) additionally honors its OWN window (``search_window_seconds``,
    measured from the last T3 event only). Retrieval is exposure, not confirmed
    use: whatever the ranker keeps surfacing would otherwise compound its own
    visibility, and an explicit write must not spend the exposure budget (nor the
    reverse). This is a deliberate exposure rate-limit, NOT the canonical ACT-R
    spacing effect.
    """
    now = now or utcnow()
    if window_seconds > 0:
        history = payload.get(FIELD_REINFORCED_AT) or []
        last = _parse_ts(history[-1]) if history else None
        if last is not None and (now - last).total_seconds() < window_seconds:
            return False
    if trigger == TRIGGER_SEARCH and search_window_seconds > 0:
        last_search = _parse_ts(payload.get(FIELD_LAST_SEARCH_REINFORCED_AT))
        if last_search is not None and (now - last_search).total_seconds() < search_window_seconds:
            return False
    return True


def reinforcement_fields(
    payload: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    max_timestamps: int = 10,
    trigger: str = TRIGGER_DEDUP,
) -> Dict[str, Any]:
    """Updated dynamics fields for one reinforcement event.

    A memory not yet on the timeline (never reinforced, or created before v0.2)
    joins it here, retroactively adopting its ``created_at`` as the first
    encounter so its first reinforcement yields a two-event history. The
    returned dict contains ONLY the dynamics fields, ready to merge into the
    payload.

    ``reinforced_by`` stays positionally aligned with ``reinforced_at`` (same
    append, same trim) so every retained timestamp keeps its provenance.
    """
    now = now or utcnow()
    history = list(payload.get(FIELD_REINFORCED_AT) or [])
    origins = list(payload.get(FIELD_REINFORCED_BY) or [])
    count = payload.get(FIELD_ACCESS_COUNT)
    count = count if isinstance(count, int) and count > 0 else len(history)

    if not history:
        created = payload.get("created_at")
        if created and _parse_ts(created) is not None:
            history = [created]
            origins = [TRIGGER_CREATED]
            count = max(count, 1)

    # A legacy timeline (pre-provenance) is padded so the alignment invariant
    # holds from here on; unknown origins stay explicitly unknown, never guessed.
    if len(origins) < len(history):
        origins = [None] * (len(history) - len(origins)) + origins

    now_iso = now.isoformat()
    history.append(now_iso)
    origins.append(trigger)
    count += 1
    if max_timestamps > 0 and len(history) > max_timestamps:
        history = history[-max_timestamps:]
        origins = origins[-max_timestamps:]

    fields = {
        FIELD_REINFORCED_AT: history,
        FIELD_REINFORCED_BY: origins,
        FIELD_ACCESS_COUNT: count,
        FIELD_LAST_ACCESSED: now_iso,
    }
    if trigger == TRIGGER_SEARCH:
        fields[FIELD_LAST_SEARCH_REINFORCED_AT] = now_iso
    return fields
