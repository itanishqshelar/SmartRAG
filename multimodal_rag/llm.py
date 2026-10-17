"""
Thin wrapper around the Ollama client for chat generation and embeddings.
"""

import logging
from typing import Dict, Iterator, List

import ollama

from .config import SmartRAGConfig

logger = logging.getLogger(__name__)


class OllamaLLM:
    def __init__(self, config: SmartRAGConfig):
        self.client = ollama.Client(host=config.models.ollama_host)
        self.model = config.models.llm_model
        self.embedding_model = config.models.embedding_model
        gen = config.generation
        self.options = {
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "num_predict": gen.max_tokens,
            "num_ctx": gen.context_window,
        }

    def status(self) -> Dict[str, bool]:
        """Report whether Ollama is reachable and the configured models are pulled."""
        try:
            installed = [m.model for m in self.client.list().models]
        except Exception as e:
            logger.debug(f"Ollama not reachable: {e}")
            return {"ollama": False, "llm": False, "embeddings": False}
        return {
            "ollama": True,
            "llm": _has_model(installed, self.model),
            "embeddings": _has_model(installed, self.embedding_model),
        }

    def is_available(self) -> bool:
        return all(self.status().values())

    def stream_chat(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        for part in self.client.chat(model=self.model, messages=messages, stream=True, options=self.options):
            if part.message.content:
                yield part.message.content

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return "".join(self.stream_chat(messages))

    def embed(self, texts: List[str]) -> List[List[float]]:
        return list(self.client.embed(model=self.embedding_model, input=texts).embeddings)


def _has_model(installed: List[str], name: str) -> bool:
    """Match 'nomic-embed-text' against 'nomic-embed-text:latest', etc."""
    wanted = name if ":" in name else f"{name}:latest"
    return wanted in installed
