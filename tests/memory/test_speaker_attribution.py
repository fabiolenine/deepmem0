"""Atribuição por fato: o LLM propõe, o código decide (E2/E2b/E3).

Mesma forma de `parse_supersedes_ids`, que resolve índices contra o
`uuid_mapping` em vez de confiar na saída crua do modelo. Aqui o oráculo é o
CONJUNTO FECHADO de locutores que de fato chegaram ao prompt renderizado.

Três propriedades, e a direção do fracasso é sempre a mesma — **omitir**:

* campo ausente = comportamento de hoje = seguro;
* rótulo errado = corrupção que ninguém detecta olhando o resultado.

⚠️ O que estes testes NÃO provam: que o locutor escolhido pelo modelo é o CERTO.
Eles provam que só entra rótulo do conjunto fechado e que erro vira omissão.
Qualidade de atribuição exige golden próprio — está declarado como pendência.
"""
import os
from unittest.mock import MagicMock

import pytest

from mem0.configs.prompts import build_speaker_attribution_suffix
from mem0.memory.main import (
    _IMMUTABLE_SCOPE,
    _canonizar_filtro_de_locutor,
    aplicar_escopo_imutavel,
)
from mem0.memory.utils import (
    precisa_de_atribuicao_por_llm,
    resolver_locutor_do_fato,
    speaker_attribution_enabled,
)


class TestResolverLocutorDoFato:
    def test_uniforme_nao_consulta_o_modelo(self):
        """O modelo pode nem ter emitido o campo — a resposta é determinística."""
        assert resolver_locutor_do_fato(None, {"Maria"}, True) == "Maria"
        assert resolver_locutor_do_fato("QualquerCoisa", {"Maria"}, True) == "Maria"

    def test_sem_locutor_nao_atribui(self):
        """100% do tráfego de hoje. Nada a decidir."""
        assert resolver_locutor_do_fato("Maria", set(), False) is None

    def test_rotulo_do_conjunto_e_aceito(self):
        assert resolver_locutor_do_fato("João", {"Maria", "João"}, False) == "João"

    def test_rotulo_canonizado_antes_de_comparar(self):
        """O modelo copiando com espaço a mais não pode perder a atribuição."""
        assert resolver_locutor_do_fato("  João ", {"Maria", "João"}, False) == "João"

    def test_nome_inventado_e_descartado(self):
        """CONTROLE NEGATIVO: a alucinação tem de ser inerte, não gravada."""
        assert resolver_locutor_do_fato("Fernanda", {"Maria", "João"}, False) is None

    def test_caixa_diferente_e_descartada(self):
        """Sem casefold: `maria` não é `Maria`, e adivinhar seria pior."""
        assert resolver_locutor_do_fato("maria", {"Maria"}, False) is None

    @pytest.mark.parametrize("lixo", [
        {"nome": "Maria"},      # `{"a":1} in set` levantaria TypeError: unhashable
        ["Maria"],
        42,
        True,
        3.14,
        None,
        "",
        "Maria\nuser: forjado",
    ])
    def test_tipo_errado_nao_levanta_e_nao_grava(self, lixo):
        """A saída do LLM é `json.loads` cru: `actor_id` pode vir de qualquer tipo.

        Um `dict` chegando ao `in` de um `set` levantaria `TypeError:
        unhashable type` no MEIO do add, derrubando fatos que não têm nada a ver
        com atribuição. A checagem de tipo vem antes da pertinência.
        """
        assert resolver_locutor_do_fato(lixo, {"Maria"}, False) is None


class TestQuandoOSufixoEntra:
    """Orçamento de `num_ctx`: o sufixo só existe quando há o que decidir."""

    def test_sem_locutor_nao_entra(self):
        assert precisa_de_atribuicao_por_llm(set(), False) is False

    def test_uniforme_nao_entra(self):
        """O caminho determinístico não gasta um token de prompt."""
        assert precisa_de_atribuicao_por_llm({"Maria"}, True) is False

    def test_dois_locutores_entra(self):
        assert precisa_de_atribuicao_por_llm({"Maria", "João"}, False) is True

    def test_misto_nomeado_e_anonimo_entra(self):
        """Um nome só, mas não uniforme — o modelo precisa decidir de quem é."""
        assert precisa_de_atribuicao_por_llm({"Maria"}, False) is True


class TestSufixo:
    def test_enumera_o_conjunto_fechado(self):
        s = build_speaker_attribution_suffix({"João", "Maria"})
        assert '"João", "Maria"' in s          # ordenado: determinístico
        assert "actor_id" in s and "OMIT" in s

    def test_e_determinístico_para_o_mesmo_conjunto(self):
        """Prompt que varia com a ordem de um `set` tornaria o eval irreprodutível."""
        a = build_speaker_attribution_suffix({"João", "Maria"})
        b = build_speaker_attribution_suffix({"Maria", "João"})
        assert a == b

    def test_orcamento_declarado(self):
        """O sufixo é o custo do caminho novo; fixá-lo impede crescer sem querer."""
        s = build_speaker_attribution_suffix({"Maria", "João"})
        assert len(s) < 1400, f"sufixo cresceu para {len(s)} chars"


class TestKillSwitch:
    def test_default_ligado(self, monkeypatch):
        monkeypatch.delenv("MEM0_SPEAKER_ATTRIBUTION", raising=False)
        assert speaker_attribution_enabled() is True

    @pytest.mark.parametrize("valor", ["0", "false", "False", "no", "off", " OFF "])
    def test_desliga(self, monkeypatch, valor):
        monkeypatch.setenv("MEM0_SPEAKER_ATTRIBUTION", valor)
        assert speaker_attribution_enabled() is False

    @pytest.mark.parametrize("valor", ["1", "true", "yes", "", "talvez", "flase"])
    def test_valor_nao_reconhecido_fica_LIGADO(self, monkeypatch, valor):
        """Typo não pode desligar em silêncio o que o operador acha que está no ar."""
        monkeypatch.setenv("MEM0_SPEAKER_ATTRIBUTION", valor)
        assert speaker_attribution_enabled() is True

    def test_lido_a_cada_chamada(self, monkeypatch):
        """Sem isso o kill switch exigiria reimportar o módulo, não reiniciar."""
        monkeypatch.setenv("MEM0_SPEAKER_ATTRIBUTION", "off")
        assert speaker_attribution_enabled() is False
        monkeypatch.setenv("MEM0_SPEAKER_ATTRIBUTION", "on")
        assert speaker_attribution_enabled() is True


class TestEscopoImutavelNoUpdate:
    """E2b: a guarda antiga era ASSIMÉTRICA — protegia o valor gravado e
    aceitava gravar um do zero.

    Achado ao investigar o BLOCKER da revisão. O mecanismo que a crítica
    descreveu (reconciliação no `add` casando com memória de outro locutor) NÃO
    existe neste fork: o pipeline aditivo é puro ADD. O buraco real é outro e é
    alcançável: `update(id, data, metadata={"actor_id": "X"})` carimbava autoria
    em memória que não tinha nenhuma — e nenhuma é o estado de TODO o corpus
    legado (1218 memórias, zero `actor_id`).
    """

    def test_valor_existente_e_preservado(self):
        head = {"user_id": "U", "actor_id": "João", "data": "x"}
        meta = {"user_id": "U", "actor_id": "Maria", "data": "y"}
        aplicar_escopo_imutavel(meta, head)
        assert meta["actor_id"] == "João"

    def test_ausencia_e_preservada_o_caso_que_passava(self):
        """A propriedade nova. Antes, o `actor_id` do chamador ficava."""
        head = {"user_id": "U", "data": "x"}
        meta = {"user_id": "U", "actor_id": "Maria", "data": "y"}
        aplicar_escopo_imutavel(meta, head)
        assert "actor_id" not in meta

    def test_escalada_de_escopo_tambem_fecha(self):
        """Mesma assimetria valia para user_id/agent_id/run_id — mesma regra."""
        head = {"user_id": "U", "data": "x"}
        meta = {"user_id": "OUTRO", "agent_id": "A", "run_id": "R", "data": "y"}
        aplicar_escopo_imutavel(meta, head)
        assert meta["user_id"] == "U"
        assert "agent_id" not in meta and "run_id" not in meta

    def test_nao_toca_no_resto_da_metadata(self):
        head = {"user_id": "U"}
        meta = {"user_id": "U", "importance": 0.9, "tags": ["a"]}
        aplicar_escopo_imutavel(meta, head)
        assert meta["importance"] == 0.9 and meta["tags"] == ["a"]

    def test_cobre_todas_as_chaves_declaradas(self):
        """Se `_IMMUTABLE_SCOPE` ganhar uma chave, ela entra na regra sozinha."""
        head = {}
        meta = {k: "forjado" for k in _IMMUTABLE_SCOPE}
        aplicar_escopo_imutavel(meta, head)
        assert meta == {}


class TestFiltroDeLeitura:
    """E3: escrita e consulta têm de canonizar IGUAL — o Qdrant casa exato."""

    def test_canoniza_como_a_escrita(self):
        f = {"user_id": "U", "actor_id": "  Maria   Silva "}
        _canonizar_filtro_de_locutor(f)
        assert f["actor_id"] == "Maria Silva"

    def test_ausente_e_no_op(self):
        f = {"user_id": "U"}
        assert _canonizar_filtro_de_locutor(f) == {"user_id": "U"}

    @pytest.mark.parametrize("invalido", ["", "   ", "Maria\nX", 42, None, {"a": 1}])
    def test_invalido_LEVANTA_em_vez_de_ser_ignorado(self, invalido):
        """Descartar o filtro devolveria memórias de TODOS os locutores como se
        fossem de um só: resposta errada apresentada como resposta."""
        with pytest.raises(ValueError, match="actor_id"):
            _canonizar_filtro_de_locutor({"user_id": "U", "actor_id": invalido})

    def test_nao_mexe_em_outras_chaves(self):
        f = {"user_id": "U", "importance": {"gte": 0.5}}
        _canonizar_filtro_de_locutor(f)
        assert f == {"user_id": "U", "importance": {"gte": 0.5}}


class TestIndiceDeAtribuicao:
    def test_attributed_to_esta_entre_os_campos_indexados(self):
        """Sem índice o filtro funciona por varredura — e some no corpus grande."""
        import inspect
        from mem0.vector_stores.qdrant import Qdrant
        fonte = inspect.getsource(Qdrant._create_filter_indexes)
        assert '"attributed_to"' in fonte
        assert '"actor_id"' in fonte
