from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import openai  # type: ignore
from rich import print
from tiktoken import get_encoding

from llm_service import TokenCostTracker
from abc import ABC, abstractmethod

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingTypes:
    """Enumeration for different types of embedding models."""
    EMBEDDER_TYPE_OPENAI = "openai"
    EMBEDDER_TYPE_HUGGINGFACE = "huggingface"
    EMBEDDER_TYPE_CORPORATE_OPENAI = "corporate-openai"


class Embedder(ABC):
    def __init__(self, model: str, tracker: TokenCostTracker, normalize: bool = True, batch_size: int = 64) -> None:
        self.model = model
        self.tracker = tracker
        self.normalize = normalize
        self.batch_size = max(1, batch_size)

    def _l2norm(self, arr: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return arr
        n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / n

    @abstractmethod
    async def embed_one(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    async def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        pass


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, tracker: TokenCostTracker, normalize: bool = True, batch_size: int = 64) -> None:
        self.client = openai.AsyncOpenAI()
        super().__init__(model=model, tracker=tracker, normalize=normalize, batch_size=batch_size)

    async def embed_one(self, text: str) -> np.ndarray:
        r = await self.client.embeddings.create(model=self.model, input=text)
        self.tracker.note_embed(self.model, getattr(r, "usage", None), fallback_text=text)
        vec = np.asarray(r.data[0].embedding, dtype="float32")[None, :]
        return self._l2norm(vec)

    async def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            r = await self.client.embeddings.create(model=self.model, input=batch)
            self.tracker.note_embed(self.model, getattr(r, "usage", None))
            vecs = np.asarray([d.embedding for d in r.data], dtype="float32")
            vecs = self._l2norm(vecs)
            out.extend([v[None, :] for v in vecs])
        return out
        
    
class HuggingFaceEmbedder(Embedder):
    def __init__(self, model: str, tracker: TokenCostTracker, normalize: bool = True, batch_size: int = 64) -> None:
        embedding_model = HuggingFaceEmbeddings(model_name=model)
        super().__init__(model=embedding_model, tracker=tracker, normalize=normalize, batch_size=batch_size)
        
    async def embed_one(self, text: str) -> np.ndarray:
        embedding_model: HuggingFaceEmbeddings = self.model
        vec = np.asarray(embedding_model.embed_query(text), dtype="float32")[None, :]
        return self._l2norm(vec)
    
    async def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        embedding_model: HuggingFaceEmbeddings = self.model
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = np.asarray(embedding_model.embed_documents(batch), dtype="float32")
            vecs = self._l2norm(vecs)
            out.extend([v[None, :] for v in vecs])
        return out

    
try: 

    from helpers.azure_openai import init_azure_openai_async_client
    
    class CorporateOpenAIEmbedder(Embedder):
        def __init__(self, model: str, tracker: TokenCostTracker, normalize: bool = True, batch_size: int = 64) -> None:
            self.client = init_azure_openai_async_client(model)
            super().__init__(model=model, tracker=tracker, normalize=normalize, batch_size=batch_size)

        async def embed_one(self, text: str) -> np.ndarray:
            r = await self.client.embeddings.create(model=self.model, input=text)
            self.tracker.note_embed(self.model, getattr(r, "usage", None), fallback_text=text)
            vec = np.asarray(r.data[0].embedding, dtype="float32")[None, :]
            return self._l2norm(vec)

        async def embed_many(self, texts: List[str]) -> List[np.ndarray]:
            out: List[np.ndarray] = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                r = await self.client.embeddings.create(model=self.model, input=batch)
                self.tracker.note_embed(self.model, getattr(r, "usage", None))
                vecs = np.asarray([d.embedding for d in r.data], dtype="float32")
                vecs = self._l2norm(vecs)
                out.extend([v[None, :] for v in vecs])
            return out
        
    corporate_openai_embedder_available = True
        
except Exception as e:
    corporate_openai_embedder_available = False
 

# Factory function to create the correct embedder based on configuration
def get_embedder(embedder_type: EmbeddingTypes, model: str, tracker: TokenCostTracker, normalize: bool = True, batch_size: int = 64) -> Embedder:
    """
    Determines which embedder to use based on the environment variable
    and returns an instance of that class.

    Returns:
        An instance of a class that inherits from Embedder.
    
    Raises:
        ValueError: If the EMBEDDER_TYPE is not recognized.
    """    
    if embedder_type == EmbeddingTypes.EMBEDDER_TYPE_OPENAI:
        return OpenAIEmbedder(model, tracker, normalize, batch_size)
    elif embedder_type == EmbeddingTypes.EMBEDDER_TYPE_HUGGINGFACE:
        return HuggingFaceEmbedder(model, tracker, normalize, batch_size)
    elif corporate_openai_embedder_available and embedder_type == EmbeddingTypes.EMBEDDER_TYPE_CORPORATE_OPENAI:
        return CorporateOpenAIEmbedder(model, tracker, normalize, batch_size)
    else:
        raise ValueError(f"Unknown EMBEDDER_TYPE: '{embedder_type}'. Supported types are 'openai', 'huggingface', and 'corporate-openai'.")