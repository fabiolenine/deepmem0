"""Carregamento da configuração do CLI.

O engine é uma biblioteca: ele não sabe montar a configuração de uma instalação
concreta — isso depende de embedder, vector store, reranker, idioma e ~80 outras
decisões que vivem no ambiente de quem opera o serviço.

Por isso o CLI **lê um arquivo** em vez de reimplementar essa montagem. Quem
gera o arquivo é o servidor MCP (`mem0-mcp-selfhosted config export`), que já é
a fonte única dessas decisões. Assim:

- não há duas implementações para divergir (uma divergência aqui não daria erro,
  daria RESULTADO PLAUSÍVEL E ERRADO — outra collection, reranker desligado);
- o core continua sem importar nada do MCP: a dependência é de DADO, não de
  código, e a fronteira segue num sentido só;
- o CLI funciona numa máquina que não tem o MCP instalado, desde que alguém
  copie o arquivo para lá.

O arquivo carrega `contract_version`. Quando o formato mudar de forma
incompatível, um arquivo velho é RECUSADO com mensagem acionável — nunca lido
pela metade.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

#: Versão do contrato do arquivo. Só sobe em mudança INCOMPATÍVEL.
CONTRACT_VERSION = 1

DEFAULT_PATH = Path.home() / ".deepmem0" / "config.json"
ENV_PATH = "DEEPMEM0_CONFIG"


class ConfigError(RuntimeError):
    """Configuração ausente, ilegível ou incompatível — sempre acionável."""


def resolve_path(explicit: Optional[str] = None) -> Path:
    """Onde procurar o arquivo: --config, depois env, depois o padrão."""
    if explicit:
        return Path(explicit).expanduser()
    from_env = os.environ.get(ENV_PATH, "").strip()
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_PATH


def load(explicit: Optional[str] = None) -> Dict[str, Any]:
    """Configuração pronta para `Memory.from_config`.

    Falha com mensagem que diz o que fazer. Configuração ausente é o erro mais
    provável de um primeiro uso, e "KeyError: 'embedder'" não ajudaria ninguém.
    """
    path = resolve_path(explicit)
    if not path.exists():
        raise ConfigError(
            f"configuração não encontrada em {path}.\n"
            f"  Gere-a no host que roda o servidor MCP:\n"
            f"      mem0-mcp-selfhosted config export --output {path}\n"
            f"  ou aponte para outro arquivo com --config / {ENV_PATH}."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"não foi possível ler {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} não contém um objeto JSON.")

    versao = raw.get("contract_version")
    if versao is None:
        raise ConfigError(
            f"{path} não declara `contract_version` — provavelmente foi escrito à "
            f"mão. Gere-o com `mem0-mcp-selfhosted config export`."
        )
    if versao != CONTRACT_VERSION:
        raise ConfigError(
            f"{path} usa contract_version={versao}, mas este CLI fala "
            f"{CONTRACT_VERSION}. Regere o arquivo com a versão correspondente do "
            f"servidor — ler um contrato incompatível pela metade daria resultado "
            f"errado sem erro."
        )

    config = raw.get("config")
    if not isinstance(config, dict) or not config:
        raise ConfigError(f"{path} não contém a chave `config` com o dicionário.")
    return config


def describe(explicit: Optional[str] = None) -> Dict[str, Any]:
    """Resumo legível do arquivo — o que `deepmem0 config show` imprime."""
    path = resolve_path(explicit)
    config = load(explicit)
    vector = config.get("vector_store", {}).get("config", {}) or {}
    return {
        "path": str(path),
        "contract_version": CONTRACT_VERSION,
        "collection": vector.get("collection_name"),
        "vector_store": config.get("vector_store", {}).get("provider"),
        "embedder": config.get("embedder", {}).get("provider"),
        "embedder_model": config.get("embedder", {}).get("config", {}).get("model"),
        "llm": config.get("llm", {}).get("provider"),
        "llm_model": config.get("llm", {}).get("config", {}).get("model"),
        "reranker": bool(config.get("reranker")),
        "language": config.get("language"),
    }
