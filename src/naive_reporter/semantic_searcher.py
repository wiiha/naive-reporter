"""Semantic search via IBM Granite embeddings + ChromaDB."""

import logging
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from naive_reporter.config import settings
from naive_reporter.search_protocol import Searcher
from naive_reporter.types import Document, SearchResult

logger = logging.getLogger(__name__)


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Word-based sliding window."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


class SemanticSearcher(Searcher):
    """Dense semantic search using IBM Granite embeddings + ChromaDB."""

    def __init__(
        self,
        data_dir: str | None = None,
        model_name: str | None = None,
        persist_dir: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        client: Any = None,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else Path(settings.data_dir)
        self._model_name = model_name or settings.embedding_model
        resolved_persist = persist_dir or settings.chroma_persist_dir
        self._persist_path = (
            str(Path(resolved_persist))
            if resolved_persist
            else str(self._data_dir / "chroma_db")
        )
        self._client = client if client is not None else chromadb.PersistentClient(
            path=self._persist_path
        )
        self._collection_name = "naive_reporter_docs"
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._stems: set[str] = set()
        self._embedding_fn: SentenceTransformer | None = None

    def _get_embedding_fn(self) -> SentenceTransformer:
        """Lazy init of embedding function (avoids model download at import time)."""
        if self._embedding_fn is None:
            self._embedding_fn = SentenceTransformer(self._model_name)
        return self._embedding_fn

    def index(self, documents: list[Document]) -> None:
        """Build index from documents' full text."""
        self._stems = set()

        # Recreate collection (clean slate per INV-005 / INV-013)
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass

        collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        embed_fn = self._get_embedding_fn()

        for doc in documents:
            txt_path = self._data_dir / "txt" / f"{doc.stem}.txt"
            if not txt_path.exists():
                logger.warning(
                    "Missing text file for %s, skipping indexing", doc.stem
                )
                continue

            try:
                text = txt_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                logger.warning(
                    "Cannot read text file for %s, skipping indexing",
                    doc.stem,
                    exc_info=True,
                )
                continue

            chunks = _chunk_text(text, self._chunk_size, self._chunk_overlap)
            if not chunks:
                continue

            embeddings = embed_fn.encode(chunks)
            ids = [f"{doc.stem}_{i}" for i in range(len(chunks))]
            metadatas = [{"stem": doc.stem} for _ in chunks]

            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas,  # type: ignore[arg-type]
            )
            self._stems.add(doc.stem)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Semantic search the index."""
        self._validate_k(k)
        if not self._stems:
            return []
        if not query.strip():
            return []

        total_docs = len(self._stems)
        top_n = total_docs if k == -1 else min(k, total_docs)
        if top_n == 0:
            return []

        collection = self._client.get_collection(self._collection_name)
        embed_fn = self._get_embedding_fn()
        query_embedding = embed_fn.encode([query])
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_n,
        )

        # Deduplicate by stem, keep best score
        stem_to_score: dict[str, float] = {}
        ids_result = results.get("ids") or []
        distances_result = results.get("distances") or []
        for ids_list, dist_list in zip(ids_result, distances_result):
            if ids_list is None or dist_list is None:
                continue
            for doc_id, dist in zip(ids_list, dist_list):
                stem = doc_id.rsplit("_", 1)[0]
                score = max(0.0, min(1.0, 1.0 - float(dist)))
                if score == 0:
                    continue
                if stem not in stem_to_score or score > stem_to_score[stem]:
                    stem_to_score[stem] = score

        out = sorted(
            [SearchResult(stem=s, score=v) for s, v in stem_to_score.items()],
            key=lambda r: r.score,
            reverse=True,
        )
        return out if k == -1 else out[:top_n]
