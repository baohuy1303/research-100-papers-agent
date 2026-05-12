"""
Chroma vector-store retrieval wrapper.

Single Retriever() instance per process. Methods are budget-aware: when
called without an explicit `k`, the default comes from
api.core.budget.profile()["retrieve_k"], so a $1 run retrieves 3 chunks
while a $20 run retrieves 15.

Embeds the query with text-embedding-3-small (same model as the index)
so cosine similarities are meaningful.

Reranker: BGE-reranker-v2 hook is wired but disabled by default; will
be enabled in Phase 6 once we benchmark whether the latency is worth it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from api.core.budget import profile
from api.core.llm import get_openai_client

ROOT = Path(__file__).parent.parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "paper_chunks"
EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE_PER_TOKEN = 0.02 / 1_000_000


class Retriever:
    """Lazy-loaded Chroma client + OpenAI embedding wrapper."""

    def __init__(self, persist_path: Path | str = CHROMA_DIR):
        self.persist_path = Path(persist_path)
        self._client: chromadb.api.ClientAPI | None = None
        self._collection: chromadb.api.Collection | None = None
        self._oai = None  # lazy

    @property
    def collection(self):
        if self._collection is None:
            if not self.persist_path.exists():
                raise FileNotFoundError(
                    f"Chroma store missing: {self.persist_path} — run scripts/build_indexes.py"
                )
            self._client = chromadb.PersistentClient(path=str(self.persist_path))
            self._collection = self._client.get_collection(name=COLLECTION_NAME)
        return self._collection

    @property
    def oai(self):
        if self._oai is None:
            self._oai = get_openai_client()
        return self._oai

    async def _embed(self, query: str) -> tuple[list[float], int]:
        """Embed a single query string. Returns (vector, tokens_used)."""
        response = await self.oai.embeddings.create(model=EMBED_MODEL, input=query)
        return response.data[0].embedding, response.usage.total_tokens

    async def search(
        self,
        query: str,
        k: int | None = None,
        paper_id: str | None = None,
    ) -> dict[str, Any]:
        """Semantic search.

        Args:
            query: Natural-language query string.
            k: How many chunks to return. Defaults to profile()["retrieve_k"].
            paper_id: If set, restricts search to chunks from one paper
                      (used by Tier 1 single-doc factual handler).

        Returns:
            {
              "query": str,
              "chunks": [{paper_id, section_title, char_offset, text, score}, ...],
              "tokens_used": int,
              "cost_usd": float,
            }
        """
        k = k or profile()["retrieve_k"]
        vec, tokens = await self._embed(query)

        where = {"paper_id": paper_id} if paper_id else None
        results = self.collection.query(
            query_embeddings=[vec],
            n_results=k,
            where=where,
        )

        chunks: list[dict] = []
        for md, doc, dist in zip(
            results["metadatas"][0], results["documents"][0], results["distances"][0]
        ):
            # Chroma returns L2-distance for cosine collections; convert to
            # similarity in [0,1]: sim = 1 - dist/2 for unit vectors
            chunks.append({
                "paper_id": md["paper_id"],
                "section_title": md.get("section_title", ""),
                "char_offset": md.get("char_offset", 0),
                "text": doc,
                "score": round(1 - dist / 2, 4),
            })

        cost = tokens * EMBED_PRICE_PER_TOKEN
        return {
            "query": query,
            "chunks": chunks,
            "tokens_used": tokens,
            "cost_usd": cost,
        }

    async def search_in_paper(self, query: str, paper_id: str, k: int | None = None) -> dict[str, Any]:
        """Convenience wrapper for Tier 1 single-doc retrieval."""
        return await self.search(query, k=k, paper_id=paper_id)
