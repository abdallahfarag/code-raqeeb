import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from fastembed import SparseTextEmbedding, TextEmbedding


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class Embedder(Protocol):
    @property
    def dense_size(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]: ...

    async def embed_query(self, text: str) -> Embedding: ...


class FastEmbedEmbedder:
    def __init__(self, dense_model: str, sparse_model: str) -> None:
        self._dense = TextEmbedding(model_name=dense_model)
        self._sparse = SparseTextEmbedding(model_name=sparse_model)
        self._dense_size = len(next(iter(self._dense.embed(["size probe"]))))

    @property
    def dense_size(self) -> int:
        return self._dense_size

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return await asyncio.to_thread(self._embed, list(texts), False)

    async def embed_query(self, text: str) -> Embedding:
        embeddings = await asyncio.to_thread(self._embed, [text], True)
        return embeddings[0]

    def _embed(self, texts: list[str], is_query: bool) -> list[Embedding]:
        if is_query:
            dense = list(self._dense.query_embed(texts))
            sparse = list(self._sparse.query_embed(texts))
        else:
            dense = list(self._dense.embed(texts))
            sparse = list(self._sparse.embed(texts))
        return [
            Embedding(
                dense=d.tolist(),
                sparse_indices=s.indices.tolist(),
                sparse_values=s.values.tolist(),
            )
            for d, s in zip(dense, sparse, strict=True)
        ]
