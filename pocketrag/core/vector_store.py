"""
PocketRAG - Vector Store Module using LanceDB

Features:
- Efficient vector storage and retrieval
- Metadata filtering
- Incremental updates (no full overwrite)
- Multiple distance metrics support
"""
import logging
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector storage and retrieval using LanceDB.

    Features:
    - Persistent vector storage
    - Metadata filtering
    - Incremental document addition
    - Multiple distance metrics (COSINE, L2, DOT)
    """

    def __init__(
        self,
        db_path: str,
        table_name: str = "documents",
        metric: str = "cosine",
    ):
        """
        Initialize the vector store.

        Args:
            db_path: Path to the LanceDB database directory
            table_name: Name of the table to use
            metric: Distance metric for similarity search (cosine, l2, dot)
        """
        self.db_path = db_path
        self.table_name = table_name
        self.metric = metric.lower()

        self._db = None
        self._table = None
        self._dimension = None
        self._fts_ready = False

    @property
    def db(self):
        """Lazy load the LanceDB connection."""
        if self._db is None:
            try:
                import lancedb
                Path(self.db_path).mkdir(parents=True, exist_ok=True)
                self._db = lancedb.connect(self.db_path)
                logger.info(f"Connected to LanceDB at {self.db_path}")
            except ImportError:
                logger.error("lancedb not installed")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to LanceDB: {e}")
                raise
        return self._db

    @property
    def table(self):
        """Get or create the documents table."""
        if self._table is None:
            try:
                # Try to open existing table
                self._table = self.db.open_table(self.table_name)
                logger.info(f"Opened existing table: {self.table_name}")
            except Exception:
                # Table doesn't exist yet - it will be created on first insert
                logger.info(f"Table {self.table_name} will be created on first insert")
        return self._table

    @staticmethod
    def escape_filter_literal(value: str) -> str:
        """Escape a string literal for LanceDB filter expressions."""
        return str(value).replace("'", "''")

    def source_filter_expr(self, source: str) -> str:
        """Build a safe source equality filter expression."""
        escaped = self.escape_filter_literal(source)
        return f"source = '{escaped}'"

    def _create_schema(self, dimension: int) -> List[Dict[str, Any]]:
        """Create table schema with proper types."""
        import pyarrow as pa

        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dimension)),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("metadata", pa.string()),  # JSON string for flexibility
        ])
        return schema

    def _ensure_fts_index(self) -> None:
        """Create FTS index if needed, without rebuilding on every insert."""
        if self._fts_ready or self._table is None:
            return

        try:
            self._table.create_fts_index("text")
            logger.info("FTS index created for 'text' column")
        except Exception as e:
            msg = str(e).lower()
            # If already exists (or backend doesn't support detecting/creating repeatedly), treat as ready.
            if "exist" in msg or "already" in msg:
                logger.debug("FTS index already exists")
            else:
                logger.warning(f"Failed to create FTS index: {e}")
                return

        self._fts_ready = True

    def rebuild_fts_index(self) -> None:
        """Force a rebuild of the FTS index."""
        if self._table is None:
            return
        try:
            self._table.create_fts_index("text", replace=True)
            self._fts_ready = True
            logger.info("FTS index rebuilt")
        except Exception as e:
            logger.warning(f"Failed to rebuild FTS index: {e}")

    def insert(
        self,
        documents: List[Dict[str, Any]],
        auto_id: bool = True,
    ) -> int:
        """
        Insert documents into the vector store incrementally.

        Args:
            documents: List of dicts with 'vector', 'text', and 'source' keys
            auto_id: Automatically generate IDs for documents

        Returns:
            Number of documents inserted
        """
        if not documents:
            return 0

        try:
            import pyarrow as pa

            # Get dimension from first document if not known
            if self._dimension is None and documents:
                self._dimension = len(documents[0].get('vector', []))

            # Prepare data with IDs
            if auto_id:
                start_id = self.count()
                for i, doc in enumerate(documents):
                    doc['id'] = start_id + i

            # Add metadata field if not present
            for doc in documents:
                if 'metadata' not in doc:
                    doc['metadata'] = '{}'
                elif isinstance(doc['metadata'], dict):
                    doc['metadata'] = json.dumps(doc['metadata'])

            # Create table if it doesn't exist
            if self._table is None:
                schema = self._create_schema(self._dimension)
                self._table = self.db.create_table(
                    self.table_name,
                    schema=schema,
                )
                logger.info(f"Created new table: {self.table_name}")

            # Convert to Arrow table and add
            table_data = {
                'id': [doc['id'] for doc in documents],
                'vector': [doc['vector'] for doc in documents],
                'text': [doc['text'] for doc in documents],
                'source': [doc['source'] for doc in documents],
                'metadata': [doc['metadata'] for doc in documents],
            }

            arrow_table = pa.table(table_data)
            self._table.add(arrow_table)
            self._ensure_fts_index()

            logger.info(f"Added {len(documents)} documents to {self.table_name}")
            return len(documents)

        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise

    def _result_relevance(self, row: Dict[str, Any], mode: str) -> float:
        """Convert raw search row to normalized relevance score (higher is better)."""
        if mode == "fts":
            # LanceDB FTS usually returns `_score` where larger is better.
            return float(row.get('_score', 0.0))

        distance = float(row.get('_distance', 1.0))

        if self.metric in {"cosine", "l2", "euclidean"}:
            return 1.0 / (1.0 + max(distance, 0.0))
        if self.metric in {"dot", "inner_product"}:
            return distance

        return 1.0 / (1.0 + max(distance, 0.0))

    def search(
        self,
        query_vector: Optional[Union[List[float], np.ndarray]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        min_score: float = 0.0,
        mode: str = "vector",  # "vector", "fts", or "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_vector: The query embedding vector
            query_text: Full-text query used by FTS/hybrid modes
            top_k: Number of results to return
            filter_expr: Optional SQL-like filter expression
            min_score: Minimum relevance score threshold (higher is better)
            mode: Search mode: vector, fts, or hybrid

        Returns:
            List of matching documents with metadata
        """
        if self._table is None:
            logger.warning("No table available for search")
            return []

        try:
            if mode == "fts" and query_text:
                search_query = self.table.search(query_text, query_type="fts")
            elif mode == "hybrid" and query_vector is not None and query_text:
                if isinstance(query_vector, np.ndarray):
                    query_vector = query_vector.tolist()
                try:
                    search_query = self.table.search(query_vector).text(query_text)
                except Exception as e:
                    logger.warning(f"Hybrid search failed, falling back to vector: {e}")
                    search_query = self.table.search(query_vector)
            elif query_vector is not None:
                if isinstance(query_vector, np.ndarray):
                    query_vector = query_vector.tolist()
                search_query = self.table.search(query_vector)
            else:
                logger.warning("No query provided for search")
                return []

            if filter_expr:
                search_query = search_query.where(filter_expr)

            search_query = search_query.limit(top_k)

            if mode != "fts":
                search_query = search_query.metric(self.metric)

            results = search_query.to_list()

            if min_score > 0:
                filtered = []
                for row in results:
                    score = self._result_relevance(row, mode=mode)
                    if score >= min_score:
                        row['score'] = score
                        filtered.append(row)
                results = filtered
            else:
                for row in results:
                    row['score'] = self._result_relevance(row, mode=mode)

            logger.debug(f"Search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def exists(self) -> bool:
        """Check if the table exists and has data."""
        try:
            self.db.open_table(self.table_name)
            return True
        except Exception:
            return False

    def count(self) -> int:
        """Get the number of documents in the store."""
        if not self.exists():
            return 0
        try:
            return self.table.count_rows()
        except Exception:
            try:
                return len(self.table.to_list())
            except Exception:
                return 0

    def clear(self) -> None:
        """Clear all documents from the store."""
        try:
            self.db.drop_table(self.table_name)
            self._table = None
            self._dimension = None
            self._fts_ready = False
            logger.info(f"Cleared table: {self.table_name}")
        except Exception as e:
            logger.warning(f"Failed to clear table: {e}")

    def delete_by_source(self, source: str) -> int:
        """
        Delete documents from a specific source.

        Args:
            source: The source file path to delete

        Returns:
            Number of documents deleted
        """
        if not self.exists():
            return 0

        try:
            before_count = self.count()
            self.table.delete(self.source_filter_expr(source))
            after_count = self.count()
            deleted = before_count - after_count

            logger.info(f"Deleted {deleted} documents from source: {source}")
            return deleted

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return 0

    def get_sources(self) -> List[str]:
        """Get list of all unique sources in the store."""
        if not self.exists():
            return []

        try:
            all_docs = self.table.to_list()
            sources = list(set(doc.get('source', '') for doc in all_docs))
            return sorted(sources)
        except Exception:
            return []
