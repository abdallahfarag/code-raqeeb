from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from code_raqeeb.rag.embeddings import Embedding
from code_raqeeb.rag.models import CodeChunk, CodeHit, RuleHit
from code_raqeeb.rules.models import Category, ReviewRule

DENSE = "dense"
SPARSE = "sparse"
_PREFETCH_FACTOR = 4
_INDEXED = "indexed"


class QdrantStore:
    def __init__(
        self, client: AsyncQdrantClient, code_collection: str, rules_collection: str
    ) -> None:
        self._client = client
        self._code = code_collection
        self._rules = rules_collection

    async def ensure_collections(self, dense_size: int) -> None:
        for name, keyword_fields in ((self._code, ("repo", "sha")), (self._rules, ("category",))):
            if await self._client.collection_exists(name):
                continue
            await self._client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            for field in keyword_fields:
                await self._client.create_payload_index(
                    name, field, models.PayloadSchemaType.KEYWORD
                )

    async def has_snapshot(self, repo: str, sha: str) -> bool:
        indexed = models.Filter(must=[_snapshot_filter(repo, sha), _match(_INDEXED, True)])
        result = await self._client.count(self._code, count_filter=indexed, exact=True)
        return result.count > 0

    async def mark_snapshot_indexed(self, repo: str, sha: str) -> None:
        await self._client.set_payload(
            self._code, payload={_INDEXED: True}, points=_snapshot_filter(repo, sha), wait=True
        )

    async def upsert_chunks(
        self, chunks: Sequence[CodeChunk], embeddings: Sequence[Embedding]
    ) -> None:
        points = [
            _point(f"{c.repo}@{c.sha}:{c.path}:{c.symbol}:{c.start_line}", e, c.model_dump())
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        await self._client.upsert(self._code, points=points, wait=True)

    async def upsert_rules(
        self, rules: Sequence[ReviewRule], embeddings: Sequence[Embedding]
    ) -> None:
        points = [
            _point(f"rule:{r.id}", e, r.model_dump())
            for r, e in zip(rules, embeddings, strict=True)
        ]
        await self._client.upsert(self._rules, points=points, wait=True)

    async def search_code(
        self, repo: str, sha: str, query: Embedding, limit: int = 5
    ) -> list[CodeHit]:
        points = await self._hybrid(self._code, query, limit, _snapshot_filter(repo, sha))
        return [
            CodeHit(chunk=CodeChunk.model_validate(point.payload), score=point.score)
            for point in points
        ]

    async def search_rules(
        self, query: Embedding, limit: int = 5, category: Category | None = None
    ) -> list[RuleHit]:
        query_filter = (
            None if category is None else models.Filter(must=[_match("category", category)])
        )
        points = await self._hybrid(self._rules, query, limit, query_filter)
        return [
            RuleHit(rule=ReviewRule.model_validate(point.payload), score=point.score)
            for point in points
        ]

    async def _hybrid(
        self,
        collection: str,
        query: Embedding,
        limit: int,
        query_filter: models.Filter | None,
    ) -> list[models.ScoredPoint]:
        candidates = limit * _PREFETCH_FACTOR
        response = await self._client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=query.dense, using=DENSE, limit=candidates, filter=query_filter
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query.sparse_indices, values=query.sparse_values
                    ),
                    using=SPARSE,
                    limit=candidates,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points


def _point(key: str, embedding: Embedding, payload: dict[str, Any]) -> models.PointStruct:
    return models.PointStruct(
        id=str(uuid5(NAMESPACE_URL, key)),
        vector={
            DENSE: embedding.dense,
            SPARSE: models.SparseVector(
                indices=embedding.sparse_indices, values=embedding.sparse_values
            ),
        },
        payload=payload,
    )


def _match(field: str, value: str | bool) -> models.FieldCondition:
    return models.FieldCondition(key=field, match=models.MatchValue(value=value))


def _snapshot_filter(repo: str, sha: str) -> models.Filter:
    return models.Filter(must=[_match("repo", repo), _match("sha", sha)])
