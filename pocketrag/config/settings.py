"""
PocketRAG - Production Configuration Module
"""
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Application configuration with sensible defaults."""

    config_version: int = 1

    # Database settings
    db_path: str = "./.pocketrag/data"
    table_name: str = "documents"

    # Embedding model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunk_by_sentence: bool = True

    # Search settings
    default_top_k: int = 5
    score_threshold: float = 0.0
    enable_hybrid_search: bool = False

    # LLM settings
    default_model: str = "qwen3.5:0.8b"
    temperature: float = 0.7
    max_tokens: int = 1024

    # Re-ranking settings
    enable_reranking: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Supported file extensions
    supported_extensions: tuple = field(default_factory=lambda: (
        '.pdf', '.txt', '.md', '.py', '.js', '.json',
        '.ts', '.tsx', '.html', '.css', '.java', '.cpp',
        '.c', '.h', '.go', '.rs', '.rb', '.php',
        '.docx', '.csv', '.xml', '.yaml', '.yml'
    ))

    @property
    def db_dir(self) -> Path:
        """Get the database directory path."""
        return Path(self.db_path).parent

    def ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a fully persisted dictionary."""
        return {
            "config_version": self.config_version,
            "db_path": str(self.db_path),
            "table_name": self.table_name,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_by_sentence": self.chunk_by_sentence,
            "default_top_k": self.default_top_k,
            "score_threshold": self.score_threshold,
            "enable_hybrid_search": self.enable_hybrid_search,
            "default_model": self.default_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_reranking": self.enable_reranking,
            "rerank_model": self.rerank_model,
            "rerank_top_k": self.rerank_top_k,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "supported_extensions": list(self.supported_extensions),
        }

    def _coerce_value(self, key: str, value: Any) -> Any:
        """Coerce loaded configuration values to expected runtime types."""
        if key in {"config_version", "embedding_dimension", "chunk_size", "chunk_overlap", "default_top_k", "max_tokens", "rerank_top_k"}:
            return int(value)
        if key in {"score_threshold", "temperature"}:
            return float(value)
        if key in {"chunk_by_sentence", "enable_hybrid_search", "enable_reranking"}:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on", "enable"}
            return bool(value)
        if key == "supported_extensions":
            if not isinstance(value, (list, tuple)):
                raise ValueError("supported_extensions must be a list or tuple")
            return tuple(str(ext) for ext in value)
        if key == "log_file":
            return None if value in (None, "", "null") else str(value)
        if key in {"db_path", "table_name", "embedding_model", "default_model", "rerank_model", "log_level"}:
            return str(value)
        return value

    def save(self, path: Optional[Path] = None) -> None:
        """Save config to a JSON file."""
        if path is None:
            path = self.db_dir / "config.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, path: Optional[Path] = None) -> None:
        """Load config from a JSON file with safe validation/coercion."""
        if path is None:
            path = self.db_dir / "config.json"

        if not path.exists():
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            for key, value in data.items():
                if not hasattr(self, key) or isinstance(getattr(type(self), key, None), property):
                    logger.warning(f"Ignoring unknown/non-persisted config key: {key}")
                    continue

                try:
                    setattr(self, key, self._coerce_value(key, value))
                except Exception as e:
                    logger.warning(f"Invalid config value for '{key}' ({value!r}): {e}")
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")

    def setup_logging(self) -> None:
        """Configure logging for the application."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        handlers: List[logging.Handler] = [logging.StreamHandler()]

        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self.log_file))

        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format=log_format,
            handlers=handlers
        )


# Global config instance
config = Config()
