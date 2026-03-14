"""
Retriever Module
Bridges FAISS vector search with SQLite metadata for comprehensive retrieval.
Implements hybrid search: semantic (FAISS) + keyword (SQLite).
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RetrievalResult:
    """A single retrieval result with chunk content and metadata."""
    chunk_id: str
    content: str
    section_heading: str
    paper_id: str
    paper_title: str
    summary: str
    keywords: str
    similarity_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0


class Retriever:
    """
    Hybrid retriever combining semantic search (FAISS) and keyword search (SQLite).
    """

    def __init__(self, faiss_store, metadata_db, embedder):
        """
        Initialize the retriever.

        Args:
            faiss_store: An instance of vectorstore.faiss_store.FAISSStore.
            metadata_db: An instance of database.metadata_db.MetadataDB.
            embedder: An instance of embeddings.embedder.Embedder.
        """
        self.faiss_store = faiss_store
        self.metadata_db = metadata_db
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        paper_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval: semantic + keyword search.

        Args:
            query: User query text.
            top_k: Number of results to return.
            semantic_weight: Weight for semantic similarity scores (0-1).
            keyword_weight: Weight for keyword match scores (0-1).
            paper_id: Optional filter to restrict to a specific paper.

        Returns:
            List of RetrievalResult sorted by combined score.
        """
        # 1. Semantic search via FAISS
        query_embedding = self.embedder.embed_text(query)
        semantic_results = self.faiss_store.search(query_embedding, top_k=top_k * 2)

        # 2. Keyword search via SQLite
        query_keywords = self._extract_query_keywords(query)
        keyword_results = []
        if query_keywords:
            keyword_results = self.metadata_db.search_by_keywords(
                query_keywords, limit=top_k * 2
            )

        # 3. Merge and score
        results = self._merge_results(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            paper_id=paper_id
        )

        # 4. Sort by combined score and return top_k
        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]

    def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Perform pure semantic (vector) search.

        Args:
            query: User query text.
            top_k: Number of results.

        Returns:
            List of RetrievalResult sorted by similarity score.
        """
        query_embedding = self.embedder.embed_text(query)
        search_results = self.faiss_store.search(query_embedding, top_k=top_k)

        results = []
        chunk_ids = [cid for cid, _ in search_results]
        chunk_data = self.metadata_db.get_chunks_by_ids(chunk_ids)
        chunk_map = {c["chunk_id"]: c for c in chunk_data}

        for chunk_id, score in search_results:
            meta = chunk_map.get(chunk_id, {})
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=meta.get("content", ""),
                section_heading=meta.get("section_heading", ""),
                paper_id=meta.get("paper_id", ""),
                paper_title=self._get_paper_title(meta.get("paper_id", "")),
                summary=meta.get("summary", ""),
                keywords=meta.get("keywords", ""),
                similarity_score=score,
                combined_score=score
            ))

        return results

    def _merge_results(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float,
        paper_id: Optional[str]
    ) -> List[RetrievalResult]:
        """Merge semantic and keyword results with weighted scoring."""

        # Collect all unique chunk IDs
        all_chunk_ids = set()
        semantic_scores = {}
        keyword_ids = set()

        for cid, score in semantic_results:
            all_chunk_ids.add(cid)
            semantic_scores[cid] = score

        for result in keyword_results:
            cid = result["chunk_id"]
            all_chunk_ids.add(cid)
            keyword_ids.add(cid)

        # Fetch metadata for all chunks
        chunk_data = self.metadata_db.get_chunks_by_ids(list(all_chunk_ids))
        chunk_map = {c["chunk_id"]: c for c in chunk_data}

        results = []
        for cid in all_chunk_ids:
            meta = chunk_map.get(cid, {})

            # Apply paper filter
            if paper_id and meta.get("paper_id") != paper_id:
                continue

            sem_score = semantic_scores.get(cid, 0.0)
            kw_score = 1.0 if cid in keyword_ids else 0.0
            combined = (semantic_weight * sem_score) + (keyword_weight * kw_score)

            results.append(RetrievalResult(
                chunk_id=cid,
                content=meta.get("content", ""),
                section_heading=meta.get("section_heading", ""),
                paper_id=meta.get("paper_id", ""),
                paper_title=self._get_paper_title(meta.get("paper_id", "")),
                summary=meta.get("summary", ""),
                keywords=meta.get("keywords", ""),
                similarity_score=sem_score,
                keyword_score=kw_score,
                combined_score=combined
            ))

        return results

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract simple keywords from a query for SQLite search."""
        import re
        # Common English stopwords to ignore
        stopwords = {
            "what", "is", "the", "how", "does", "do", "are", "was", "were",
            "can", "could", "would", "should", "will", "may", "might",
            "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "about", "this", "that", "these", "those", "and", "or"
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        return [w for w in words if w not in stopwords]

    def _get_paper_title(self, paper_id: str) -> str:
        """Look up paper title from the database."""
        if not paper_id:
            return ""
        papers = self.metadata_db.list_papers()
        for paper in papers:
            if paper["paper_id"] == paper_id:
                return paper.get("title", "")
        return ""
