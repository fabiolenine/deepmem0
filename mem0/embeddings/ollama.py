import logging
import os
import subprocess
import sys
from typing import Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

logger = logging.getLogger(__name__)

# Teto de itens por requisição ao `/api/embed`.
#
# Era o ÚNICO provider sem teto — openai/azure/gemini fatiam em 100, vertexai em
# 250.
#
# ⚠️ A justificativa NÃO é prevenir falha, e a medição diz por quê: contra um
# bge-m3 real (GPU de 8 GB, 01/08/2026) a chamada NÃO falhou em nenhum tamanho
# tentado — 32768 itens numa única requisição, `ms/item` plano em 20-23 e VRAM do
# modelo constante. ⚠️ A medição PAROU num limiar de wall-time escolhido por mim
# (744 s por chamada), não numa falha: portanto o que se pode afirmar é "nenhuma
# falha ATÉ 32768", não "não há fronteira".
#
# O que cresce é a LATÊNCIA de uma chamada (743,6 s em 32768; 63,8 s em 1024 de
# ~1,8k chars contra 16,0 s em 256) e o RAIO de uma requisição que morre, que
# derruba o lote inteiro para o fallback item a item do chamador.
#
# 256 fica no platô de `ms/item` nos dois perfis medidos (12,0 ms/item em texto
# curto, 62,7 ms/item em ~1,8k chars), então o teto não custa vazão.
DEFAULT_MAX_BATCH = 256


def _max_batch() -> int:
    """Lido por chamada: um teto que só o processo enxerga no boot é um teto que
    não dá para ajustar sem restart."""
    try:
        n = int(os.environ.get("MEM0_EMBED_MAX_BATCH", DEFAULT_MAX_BATCH))
    except (TypeError, ValueError):
        return DEFAULT_MAX_BATCH
    return n if n > 0 else DEFAULT_MAX_BATCH

try:
    from ollama import Client
except ImportError:
    user_input = input("The 'ollama' library is required. Install it now? [y/N]: ")
    if user_input.lower() == "y":
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
            from ollama import Client
        except subprocess.CalledProcessError:
            print("Failed to install 'ollama'. Please install it manually using 'pip install ollama'.")
            sys.exit(1)
    else:
        print("The required 'ollama' library is not installed.")
        sys.exit(1)


class OllamaEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "nomic-embed-text"
        self.config.embedding_dims = self.config.embedding_dims or 512

        self.client = Client(host=self.config.ollama_base_url)
        self._ensure_model_exists()

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    def _ensure_model_exists(self):
        """
        Ensure the specified model exists locally. If not, pull it from Ollama.
        """
        local_models = self.client.list()["models"]
        target = self._normalize_model_name(self.config.model)
        if not any(
            self._normalize_model_name(model.get("name", "")) == target
            or self._normalize_model_name(model.get("model", "")) == target
            for model in local_models
        ):
            self.client.pull(self.config.model)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Ollama.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        response = self.client.embed(model=self.config.model, input=text)
        embeddings = response.get("embeddings") or []
        if not embeddings:
            raise ValueError(f"Ollama embed() returned no embeddings for model '{self.config.model}'")
        return embeddings[0]

    def embed_batch(self, texts, memory_action="add"):
        """Embed multiple texts, chunked at `MEM0_EMBED_MAX_BATCH` per call.

        A ordem de saída acompanha a de entrada: o `/api/embed` do Ollama
        devolve os vetores na ordem do `input`, e os pedaços são concatenados na
        ordem em que foram enviados. Isso importa porque o chamador casa vetor
        com texto por POSIÇÃO — trocar a ordem colaria o embedding de um fato em
        outro, em silêncio.

        A contagem é conferida POR PEDAÇO, não só no fim: com um único total, um
        pedaço curto compensado por outro longo passaria, e o desalinhamento
        seria exatamente o defeito acima.
        """
        if not texts:
            return []
        limite = _max_batch()
        todos = []
        for i in range(0, len(texts), limite):
            pedaco = texts[i:i + limite]
            response = self.client.embed(model=self.config.model, input=pedaco)
            embeddings = response.get("embeddings") or []
            if len(embeddings) != len(pedaco):
                raise ValueError(
                    f"Ollama embed() returned {len(embeddings)} embeddings for "
                    f"{len(pedaco)} texts (chunk {i}..{i + len(pedaco)} of "
                    f"{len(texts)}) using model '{self.config.model}'")
            todos.extend(embeddings)
        if len(todos) != len(texts):
            raise ValueError(
                f"Ollama embed() returned {len(todos)} embeddings for "
                f"{len(texts)} texts using model '{self.config.model}'")
        return todos
