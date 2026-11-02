import asyncio
from itertools import batched

from pydantic import BaseModel

from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.filters import path_skip_reason
from code_raqeeb.github.models import PrRef
from code_raqeeb.rag.chunker import chunk_csharp
from code_raqeeb.rag.embeddings import Embedder
from code_raqeeb.rag.models import CodeChunk
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.rules.models import RuleSet

_EMBED_BATCH = 64


class IndexReport(BaseModel):
    repo: str
    sha: str
    skipped: bool
    files: int = 0
    chunks: int = 0


class RepoTooLargeError(Exception):
    pass


async def index_snapshot(
    github: GitHubClient,
    store: QdrantStore,
    embedder: Embedder,
    ref: PrRef,
    sha: str,
    *,
    max_files: int,
    max_chunk_lines: int,
) -> IndexReport:
    repo = ref.full_name
    if await store.has_snapshot(repo, sha):
        return IndexReport(repo=repo, sha=sha, skipped=True)

    paths = [path for path in await github.list_tree(ref, sha) if path_skip_reason(path) is None]
    if len(paths) > max_files:
        raise RepoTooLargeError(f"{repo} has {len(paths)} C# files; the index limit is {max_files}")

    contents = await github.get_files_content(ref, paths, sha)
    chunks: list[CodeChunk] = []
    for path, source in contents.items():
        file_chunks = await asyncio.to_thread(
            chunk_csharp, path, source, repo=repo, sha=sha, max_lines=max_chunk_lines
        )
        chunks.extend(file_chunks)

    for batch in batched(chunks, _EMBED_BATCH):
        embeddings = await embedder.embed_documents([chunk.embedding_text for chunk in batch])
        await store.upsert_chunks(batch, embeddings)

    return IndexReport(repo=repo, sha=sha, skipped=False, files=len(paths), chunks=len(chunks))


async def index_rules(store: QdrantStore, embedder: Embedder, rules: RuleSet) -> int:
    texts = [f"{rule.id} {rule.title}: {rule.description}" for rule in rules.rules]
    embeddings = await embedder.embed_documents(texts)
    await store.upsert_rules(rules.rules, embeddings)
    return len(rules.rules)
