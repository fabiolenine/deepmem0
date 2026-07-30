"""Per-language model loading, and readiness when a model is missing.

A missing model for a CONFIGURED non-default language used to fall back to
English and log — the default was silence, which is exactly how an English
pipeline ends up scoring another language's corpus. These tests pin the loud
behaviour and, just as importantly, that nothing here touches the network:
`_ensure_model_available` calls `spacy.cli.download`, and a readiness path that
downloads hangs (measured: ten minutes to timeout).
"""
import pytest

from mem0.utils import spacy_models as sm


@pytest.fixture(autouse=True)
def _limpa_cache():
    sm._nlp_full.clear()
    sm._nlp_lemma.clear()
    sm._load_failed_full.clear()
    sm._load_failed_lemma.clear()
    yield
    sm._nlp_full.clear()
    sm._nlp_lemma.clear()
    sm._load_failed_full.clear()
    sm._load_failed_lemma.clear()


@pytest.fixture
def sem_rede(monkeypatch):
    """Bloqueia o download. Sem isto o teste sai para a internet."""
    def _explode(name):
        raise RuntimeError(f"modelo {name} ausente (download bloqueado no teste)")
    monkeypatch.setattr(sm, "_ensure_model_available", _explode)


class TestModelPorIdioma:
    def test_mapa_de_modelos(self):
        assert sm.model_name("pt") == "pt_core_news_sm"
        assert sm.model_name("en") == "en_core_web_sm"
        assert sm.model_name("pt-BR") == "pt_core_news_sm"

    def test_idioma_desconhecido_cai_no_default(self):
        assert sm.model_name("xx") == sm.MODEL_BY_LANGUAGE[sm.DEFAULT_LANGUAGE]

    def test_cache_e_por_idioma(self, monkeypatch):
        feitos = []

        def _fake(name, disable=None):
            feitos.append(name)
            return object()

        monkeypatch.setattr(sm, "_ensure_model_available", lambda n: None)
        import spacy
        monkeypatch.setattr(spacy, "load", _fake)
        a1, a2 = sm.get_nlp_full("pt"), sm.get_nlp_full("pt")
        b = sm.get_nlp_full("en")
        assert a1 is a2, "segunda chamada tem que vir do cache"
        assert a1 is not b, "idiomas diferentes não podem compartilhar instância"
        assert feitos == ["pt_core_news_sm", "en_core_web_sm"]


class TestReadinessSemModelo:
    def test_idioma_nao_default_sem_modelo_LEVANTA(self, sem_rede):
        """O critério: modelo ausente em deployment de outra língua REPROVA."""
        with pytest.raises(RuntimeError) as e:
            sm.get_nlp_full("pt")
        assert "pt_core_news_sm" in str(e.value)

    def test_default_sem_modelo_apenas_avisa(self, sem_rede):
        """Inglês sem modelo é o estado do upstream em instalação limpa."""
        assert sm.get_nlp_full("en") is None

    def test_strict_zero_permite_degradar_explicitamente(self, sem_rede,
                                                         monkeypatch):
        monkeypatch.setenv("MEM0_SPACY_STRICT", "0")
        # cai para o inglês, que também está bloqueado -> None, mas SEM levantar
        assert sm.get_nlp_full("pt") is None

    def test_strict_um_endurece_o_default(self, sem_rede, monkeypatch):
        monkeypatch.setenv("MEM0_SPACY_STRICT", "1")
        with pytest.raises(RuntimeError):
            sm.get_nlp_full("en")


class TestStatusDoPipeline:
    def test_nao_toca_a_rede(self, monkeypatch):
        """`model_available` não pode chamar `_ensure_model_available`."""
        def _proibido(name):
            raise AssertionError("status disparou download")
        monkeypatch.setattr(sm, "_ensure_model_available", _proibido)
        sm.entity_pipeline_status("pt")
        sm.model_available("pt")

    def test_ausente_marca_degraded(self, monkeypatch):
        monkeypatch.setattr(sm, "model_available", lambda language=None: False)
        assert sm.entity_pipeline_status("pt")["degraded"] is True

    def test_default_ausente_nao_marca_degraded(self, monkeypatch):
        monkeypatch.setattr(sm, "model_available", lambda language=None: False)
        assert sm.entity_pipeline_status("en")["degraded"] is False

    def test_idioma_nao_suportado_e_degraded(self):
        """`model_name` cai em inglês para código desconhecido, então perguntar
        só 'o modelo está instalado?' devolvia True para uma língua sem
        pipeline nenhum — o mesmo silêncio, deslocado de língua."""
        st = sm.entity_pipeline_status("xx")
        assert st["supported"] is False and st["degraded"] is True
        assert st["model"] is None


def test_fallback_nao_faz_deadlock(sem_rede, monkeypatch):
    """`_fallback` chama `_load` de dentro do lock. Com um `Lock` simples isso
    trava o processo, e o caminho que dispara é o de degradar com elegância:
    língua configurada sem modelo e `MEM0_SPACY_STRICT=0`. Produção não pegaria,
    porque lá o modelo existe.
    """
    import threading

    monkeypatch.setenv("MEM0_SPACY_STRICT", "0")
    assert isinstance(sm._lock, type(threading.RLock())), \
        "o lock tem que ser reentrante"

    pronto = threading.Event()

    def _roda():
        sm.get_nlp_full("pt")
        pronto.set()

    t = threading.Thread(target=_roda, daemon=True)
    t.start()
    assert pronto.wait(timeout=15), "deadlock no fallback"
