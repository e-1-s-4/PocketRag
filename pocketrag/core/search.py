"""
PocketRAG - Document Search Module

Features:
- Semantic search with vector embeddings
- Metadata filtering
- Score thresholding
- Context formatting for LLM prompts
"""
import logging
import json
from typing import List, Dict, Any, Optional

from pocketrag.config import config
from pocketrag.core.embedding import EmbeddingEngine
from pocketrag.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a search result with metadata."""

    def __init__(
        self,
        text: str,
        source: str,
        score: float = 0.0,
        metadata: Optional[Dict] = None,
        rank: int = 0,
    ):
        self.text = text
        self.source = source
        self.score = score
        self.metadata = metadata or {}
        self.rank = rank

    def __repr__(self) -> str:
        return f"SearchResult(source={self.source}, score={self.score:.4f}, rank={self.rank})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
            "rank": self.rank,
        }


class Searcher:
    """
    Handles document search and retrieval.

    Features:
    - Semantic search using vector embeddings
    - Configurable result count and score thresholds
    - Context formatting for LLM prompts
    - Source-based filtering
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        metric: str = "cosine",
    ):
        """
        Initialize the searcher.

        Args:
            db_path: Path to the vector database
            embedding_model: Name of the embedding model
            metric: Distance metric for similarity search
        """
        self.db_path = db_path or config.db_path
        self.embedding_model = embedding_model or config.embedding_model
        self.metric = metric

        # Initialize components
        self.embedder = EmbeddingEngine(self.embedding_model)
        self.store = VectorStore(self.db_path, config.table_name, metric=metric)

        logger.info(f"Searcher initialized with DB at {self.db_path}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_source: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for documents matching the query.

        Args:
            query: The search query
            top_k: Number of results to return
            score_threshold: Minimum relevance score (higher is better)
            filter_source: Filter results by source file
            mode: Search mode (vector, fts, hybrid)

        Returns:
            List of SearchResult objects
        """
        if not self.store.exists():
            logger.warning("No indexed documents found. Run 'pocketrag add' first.")
            return []

        # Use config threshold if not specified
        if score_threshold is None:
            score_threshold = config.score_threshold

        # Determine search mode
        if mode is None:
            mode = "hybrid" if config.enable_hybrid_search else "vector"

        query_vector = None
        if mode in ["vector", "hybrid"]:
            query_vector = self.embedder.embed_single(query)

            if len(query_vector) == 0:
                logger.error("Failed to generate query embedding")
                if mode == "vector":
                    return []

        filter_expr = None
        if filter_source:
            filter_expr = self.store.source_filter_expr(filter_source)

        raw_results = self.store.search(
            query_vector=query_vector,
            query_text=query,
            top_k=top_k,
            filter_expr=filter_expr,
            min_score=score_threshold,
            mode=mode,
        )

        results = []
        for i, res in enumerate(raw_results):
            metadata = {}
            if res.get('metadata'):
                try:
                    metadata = json.loads(res['metadata'])
                except (json.JSONDecodeError, TypeError):
                    pass

            results.append(SearchResult(
                text=res.get('text', ''),
                source=res.get('source', ''),
                score=float(res.get('score', 0.0)),
                metadata=metadata,
                rank=i + 1,
            ))

        logger.debug(f"Search returned {len(results)} results")
        return results

    def format_context(
        self,
        results: List[SearchResult],
        include_sources: bool = True,
        include_scores: bool = False,
        max_length: int = 2000,
    ) -> str:
        """Format search results into context text for LLM prompts."""
        if not results:
            return "No relevant context found."

        formatted_parts = []
        current_length = 0

        for i, result in enumerate(results, 1):
            parts = [f"[Document {i}]"]

            if include_sources:
                parts.append(f"Source: {result.source}")

            if include_scores:
                parts.append(f"Relevance: {result.score:.4f}")

            text = result.text
            if len(text) > 500:
                text = text[:500] + "..."

            parts.append(f"Content: {text}")

            part_text = "\n".join(parts)

            if current_length + len(part_text) > max_length:
                break

            formatted_parts.append(part_text)
            current_length += len(part_text)

        return "\n\n---\n\n".join(formatted_parts)

    def search_with_context(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> tuple[List[SearchResult], str]:
        """Search and return both results and formatted context."""
        results = self.search(query, top_k, score_threshold)
        context = self.format_context(results)
        return results, context

    def count(self) -> int:
        """Get the number of indexed documents."""
        return self.store.count()

    def is_indexed(self) -> bool:
        """Check if documents have been indexed."""
        return self.store.exists() and self.count() > 0

    def get_sources(self) -> List[str]:
        """Get list of all indexed sources."""
        return self.store.get_sources()

    def search_by_source(
        self,
        source: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search only within a specific source file."""
        return self.search(query="", top_k=top_k, filter_source=source)
