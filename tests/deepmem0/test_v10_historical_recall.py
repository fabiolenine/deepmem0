"""DeepMem0 v0.10 — recordação histórica: o caminho EXPLÍCITO de "o que eu
sabia na época". Pure units, no live infrastructure.

Decisão de produto (28/07/2026): dois caminhos de busca. `as_of` sozinho
preserva TUDO como era (filtro record-time + elegibilidade v0.9); só
`historical=True` muda a semântica — recordar não reforça e não usa peso de
uso. A 1ª proposta (as_of implicar o modo) foi derrubada em revisão: migraria
silenciosamente quem já usa as_of.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mem0.configs.base import MemoryConfig, MemoryDynamicsConfig, MemoryTemporalityConfig
from mem0.memory.main import (
    _annotate_known_successors,
    _apply_post_rerank_adjustments,
    _validate_historical,
)

NOW = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def hours_ago(h):
    return (NOW - timedelta(hours=h)).isoformat()


class TestValidateHistorical:
    def test_off_is_inert(self):
        assert _validate_historical(False, None, None) is False
        assert _validate_historical(False, "2026-01-01", MemoryTemporalityConfig()) is False

    def test_requires_anchor(self):
        with pytest.raises(ValueError, match="requires as_of"):
            _validate_historical(True, None, MemoryTemporalityConfig())

    def test_requires_feature_enabled_never_silent(self):
        # desligado NÃO pode degradar silenciosamente para a busca default —
        # o caller pediu recordação; entregar outra coisa é a classe de bug
        # "config ignorada" que este projeto já pagou.
        with pytest.raises(ValueError, match="disabled"):
            _validate_historical(True, "2026-01-01", None)
        with pytest.raises(ValueError, match="disabled"):
            _validate_historical(
                True, "2026-01-01", MemoryTemporalityConfig(historical_recall=False))

    def test_valid_activates(self):
        assert _validate_historical(True, "2026-01-01", MemoryTemporalityConfig()) is True


class TestAnnotateKnownSuccessors:
    def test_flags_only_explicitly_linked(self):
        docs = [
            {"id": "a", "metadata": {"superseded_by": "b"}},
            {"id": "b", "metadata": {}},
            {"id": "c"},  # sem metadata
        ]
        n = _annotate_known_successors(docs)
        assert n == 1
        assert docs[0]["has_newer_version"] is True
        assert "has_newer_version" not in docs[1]
        assert "has_newer_version" not in docs[2]

    def test_empty_and_none_are_safe(self):
        assert _annotate_known_successors([]) == 0
        assert _annotate_known_successors(None) == 0


class TestHistoricalMasksActivation:
    def _docs(self):
        timeline = {"reinforced_at": [hours_ago(48), hours_ago(24)], "access_count": 2}
        return [
            {"id": "a", "rerank_score": 0.9, "created_at": hours_ago(72),
             "metadata": dict(timeline)},
            {"id": "b", "rerank_score": 0.9, "created_at": hours_ago(72),
             "metadata": {}},
        ]

    def test_post_rerank_activation_inert_under_historical(self):
        dyn = MemoryDynamicsConfig()
        normal = _apply_post_rerank_adjustments(self._docs(), dyn=dyn)
        hist = _apply_post_rerank_adjustments(self._docs(), dyn=dyn, historical=True)
        assert any("activation" in d for d in normal), "controle: no default a ativação existe"
        assert not any("activation" in d for d in hist), \
            "recordação: ativação INERTE também no tie-break"

    def test_historical_does_not_touch_penalty_or_event(self):
        # a máscara do modo é SÓ sobre ativação — penalidade de supersedido e
        # event-proximity continuam (são relevância da época, não uso de hoje)
        temp = MemoryTemporalityConfig()
        docs = [{"id": "old", "rerank_score": 0.9, "created_at": hours_ago(72),
                 "metadata": {"superseded_by": "new", "superseded_at": hours_ago(1)}}]
        out = _apply_post_rerank_adjustments(docs, temp=temp, historical=True)
        assert out[0].get("superseded_penalty") == temp.superseded_penalty


class TestSearchWrapperContract:
    """Comportamental no wrapper REAL (self fake): historical força reinforce
    e devolve o echo; sem âncora, erro ANTES de qualquer I/O."""

    def test_missing_anchor_fails_before_any_io(self):
        import mem0.memory.main as main_mod

        # vector_store que EXPLODE se tocado: prova fail-fast pré-I/O
        angry = SimpleNamespace(search=lambda *a, **k: 1 / 0)
        fake = SimpleNamespace(config=MemoryConfig(), vector_store=angry)
        with pytest.raises(ValueError, match="requires as_of"):
            main_mod.Memory.search(fake, "qualquer", filters={"user_id": "u"},
                                   historical=True)

    def test_config_surface_default_on(self):
        assert MemoryConfig().temporality.historical_recall is True
