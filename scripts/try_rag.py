import asyncio
import sys

import httpx
from qdrant_client import AsyncQdrantClient

from code_raqeeb.config import get_settings
from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.errors import GitHubError
from code_raqeeb.github.service import load_pull_request
from code_raqeeb.rag.embeddings import FastEmbedEmbedder
from code_raqeeb.rag.indexer import RepoTooLargeError, index_rules, index_snapshot
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.rules.loader import get_rules

USAGE = 'Usage: uv run python scripts/try_rag.py <PR URL> "<question>"'


async def main(url: str, question: str) -> int:
    settings = get_settings()
    print("Loading embedding models (the first run downloads them)...")
    embedder = await asyncio.to_thread(
        FastEmbedEmbedder, settings.dense_embedding_model, settings.sparse_embedding_model
    )

    qdrant = AsyncQdrantClient(url=str(settings.qdrant_url))
    store = QdrantStore(qdrant, settings.qdrant_code_collection, settings.qdrant_rules_collection)
    await store.ensure_collections(embedder.dense_size)

    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
            github = GitHubClient(http, settings)
            bundle = await load_pull_request(github, url, settings)
            report = await index_snapshot(
                github,
                store,
                embedder,
                bundle.ref,
                bundle.base_sha,
                max_files=settings.max_index_files,
                max_chunk_lines=settings.max_chunk_lines,
            )
    except (GitHubError, RepoTooLargeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(report.model_dump())
    print(f"Indexed {await index_rules(store, embedder, get_rules())} rules")

    query = await embedder.embed_query(question)
    print("\nTop code chunks:")
    for hit in await store.search_code(bundle.ref.full_name, bundle.base_sha, query):
        chunk = hit.chunk
        print(
            f"  {hit.score:.3f}  {chunk.path}:{chunk.start_line}-{chunk.end_line}  {chunk.symbol}"
        )
    print("\nTop rules:")
    for rule_hit in await store.search_rules(query):
        print(f"  {rule_hit.score:.3f}  {rule_hit.rule.id}  {rule_hit.rule.title}")

    await qdrant.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(USAGE)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1], sys.argv[2])))
