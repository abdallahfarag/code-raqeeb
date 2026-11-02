import math
import re
import zlib
from collections import Counter
from collections.abc import Sequence

from code_raqeeb.rag.embeddings import Embedding

DENSE_SIZE = 16


def _tokens(text: str) -> list[int]:
    return [zlib.crc32(token.encode("utf-8")) for token in re.split(r"\W+", text) if token]


def _embed(text: str) -> Embedding:
    tokens = _tokens(text)
    dense = [0.0] * DENSE_SIZE
    for token in tokens:
        dense[token % DENSE_SIZE] += 1.0
    norm = math.sqrt(sum(value * value for value in dense))
    counts = Counter(tokens)
    return Embedding(
        dense=[value / norm for value in dense] if norm else dense,
        sparse_indices=list(counts),
        sparse_values=[float(count) for count in counts.values()],
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []

    @property
    def dense_size(self) -> int:
        return DENSE_SIZE

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        self.document_batches.append(list(texts))
        return [_embed(text) for text in texts]

    async def embed_query(self, text: str) -> Embedding:
        return _embed(text)
