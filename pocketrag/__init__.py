"""PocketRAG - Production RAG Engine."""

__version__ = "1.0.0"
__author__ = "PocketRAG Team"

from pocketrag.config import Config, config
from pocketrag.core import (
    TextChunker,
    EmbeddingEngine,
    VectorStore,
    Indexer,
    Document,
    Searcher,
    SearchResult,
    ChatEngine,
    ChatMessage,
)
from pocketrag.utils import DocumentLoader

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Config
    "Config",
    "config",
    # Core components
    "TextChunker",
    "EmbeddingEngine",
    "VectorStore",
    "Indexer",
    "Document",
    "Searcher",
    "SearchResult",
    "ChatEngine",
    "ChatMessage",
    # Utils
    "DocumentLoader",
]
