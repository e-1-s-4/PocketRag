"""
PocketRAG - Document Indexer Module

Features:
- Incremental indexing (only new/modified files)
- Batch processing with progress tracking
- Automatic metadata extraction
- File change detection
"""
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from tqdm import tqdm

from pocketrag.config import config
from pocketrag.core.chunker import TextChunker
from pocketrag.core.embedding import EmbeddingEngine
from pocketrag.core.vector_store import VectorStore
from pocketrag.utils.document_loader import DocumentLoader

logger = logging.getLogger(__name__)


class Document:
    """Represents a document chunk for indexing."""
    
    def __init__(
        self,
        text: str,
        source: str,
        metadata: Optional[Dict] = None,
        chunk_id: Optional[int] = None,
    ):
        self.text = text
        self.source = source
        self.metadata = metadata or {}
        self.chunk_id = chunk_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
        }


class Indexer:
    """
    Handles document indexing pipeline:
    Load -> Chunk -> Embed -> Store
    
    Features:
    - Incremental indexing based on file hashes
    - Batch processing for efficiency
    - Progress tracking
    - Error recovery
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        chunk_by_sentence: bool = True,
    ):
        """
        Initialize the indexer.
        
        Args:
            db_path: Path to the vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Name of the embedding model
            chunk_by_sentence: Use sentence-aware chunking
        """
        self.db_path = db_path or config.db_path
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        self.embedding_model = embedding_model or config.embedding_model
        
        # Initialize components
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            chunk_by_sentence=chunk_by_sentence,
        )
        self.embedder = EmbeddingEngine(self.embedding_model)
        self.store = VectorStore(self.db_path, config.table_name)
        
        # Metadata tracking
        self._metadata_cache: Dict[str, Dict] = {}
        
        logger.info(f"Indexer initialized with DB at {self.db_path}")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of a file for change detection."""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute hash for {file_path}: {e}")
            return ""
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load existing metadata from the store."""
        if not self.store.exists():
            return {}
        
        try:
            # Get only necessary columns and extract metadata
            # Using select to avoid loading vectors and text
            all_docs = self.store.table.search().select(["source", "metadata"]).to_list()
            metadata = {}
            
            for doc in all_docs:
                source = doc.get('source', '')
                if source:
                    if source not in metadata:
                        # Try to extract hash from metadata if available
                        file_hash = ''
                        meta_str = doc.get('metadata', '{}')
                        if meta_str:
                            try:
                                import json
                                meta_dict = json.loads(meta_str)
                                file_hash = meta_dict.get('file_hash', '')
                            except (json.JSONDecodeError, TypeError):
                                pass

                        metadata[source] = {
                            'hash': file_hash,
                            'indexed_at': None,
                            'chunk_count': 0,
                        }
                    metadata[source]['chunk_count'] = metadata[source].get('chunk_count', 0) + 1
            
            return metadata
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return {}
    
    def _needs_reindexing(
        self,
        file_path: Path,
        existing_metadata: Dict[str, Dict],
    ) -> bool:
        """Check if a file needs to be reindexed."""
        source = str(file_path)
        
        # If file was never indexed, needs indexing
        if source not in existing_metadata:
            return True
        
        # Compute current hash
        current_hash = self._compute_file_hash(file_path)
        existing_hash = existing_metadata[source].get('hash', '')
        
        # If hash changed, needs reindexing
        return current_hash != existing_hash
    
    def index_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        incremental: bool = True,
    ) -> Dict[str, int]:
        """
        Index all documents in a directory.
        
        Args:
            directory_path: Path to the directory to index
            recursive: Whether to search subdirectories
            incremental: Only index new/modified files
            
        Returns:
            Dictionary with indexing statistics
        """
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        files = list(path.glob(pattern))
        files = [
            f for f in files 
            if f.is_file() and f.suffix.lower() in self.loader.SUPPORTED_EXTENSIONS
        ]
        
        logger.info(f"Found {len(files)} supported files to process")
        
        # Load existing metadata for incremental indexing
        existing_metadata = self._load_metadata() if incremental else {}
        
        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "files_unchanged": 0,
            "chunks_created": 0,
            "chunks_replaced": 0,
            "errors": 0
        }
        
        documents = []
        files_to_index = []
        
        # Determine which files need indexing
        for file_path in files:
            if incremental and not self._needs_reindexing(file_path, existing_metadata):
                stats["files_unchanged"] += 1
                continue
            files_to_index.append(file_path)
        
        if not files_to_index:
            logger.info("All files are up to date, nothing to index")
            stats["files_skipped"] = len(files)
            return stats

        # Replace stale chunks for modified files before inserting fresh ones.
        if incremental:
            modified_sources = [str(fp) for fp in files_to_index if str(fp) in existing_metadata]
            for source in modified_sources:
                stats["chunks_replaced"] += self.store.delete_by_source(source)

        # Process files
        for file_path in tqdm(files_to_index, desc="Processing files"):
            try:
                content = self.loader.load(file_path)
                
                if content is None:
                    stats["files_skipped"] += 1
                    continue
                
                # Chunk the content
                chunks = self.chunker.chunk(content)
                
                if not chunks:
                    stats["files_skipped"] += 1
                    continue
                
                # Create document objects with metadata
                file_hash = self._compute_file_hash(file_path)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        text=chunk,
                        source=str(file_path),
                        metadata={
                            "file_hash": file_hash,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "indexed_at": datetime.now().isoformat(),
                        },
                        chunk_id=i,
                    )
                    documents.append(doc)
                
                stats["files_processed"] += 1
                stats["chunks_created"] += len(chunks)
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                stats["errors"] += 1
        
        if not documents:
            logger.warning("No documents to index")
            return stats
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(documents)} chunks...")
        texts = [doc.text for doc in documents]
        
        try:
            embeddings = self.embedder.embed(texts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
        
        # Prepare data for storage
        data_to_store = []
        for i, doc in enumerate(documents):
            data_to_store.append({
                "vector": embeddings[i].tolist(),
                "text": doc.text,
                "source": doc.source,
                "metadata": json.dumps(doc.metadata),
            })
        
        # Store in vector database
        self.store.insert(data_to_store)
        
        logger.info(
            f"Indexing complete: {stats['files_processed']} files, "
            f"{stats['chunks_created']} chunks"
        )
        
        return stats
    
    def index_file(self, file_path: str) -> Dict[str, int]:
        """
        Index a single file.
        
        Args:
            file_path: Path to the file to index
            
        Returns:
            Dictionary with indexing statistics
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        # Load content
        content = self.loader.load(path)
        
        if content is None:
            return {"files_processed": 0, "chunks_created": 0}
        
        # Chunk
        chunks = self.chunker.chunk(content)
        
        if not chunks:
            return {"files_processed": 0, "chunks_created": 0}
        
        # Generate embeddings
        embeddings = self.embedder.embed(chunks)
        
        # Prepare data
        data_to_store = []
        file_hash = self._compute_file_hash(path)
        for i, chunk in enumerate(chunks):
            data_to_store.append({
                "vector": embeddings[i].tolist(),
                "text": chunk,
                "source": str(path),
                "metadata": json.dumps({
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "indexed_at": datetime.now().isoformat(),
                }),
            })
        
        # Replace and store
        self.store.delete_by_source(str(path))
        self.store.insert(data_to_store)
        
        return {"files_processed": 1, "chunks_created": len(chunks)}
    
    def remove_file(self, file_path: str) -> bool:
        """
        Remove a file from the index.
        
        Args:
            file_path: Path to the file to remove
            
        Returns:
            True if successfully removed
        """
        try:
            deleted = self.store.delete_by_source(str(file_path))
            logger.info(f"Removed {deleted} chunks from {file_path}")
            return deleted > 0
        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {e}")
            return False
    
    def refresh_file(self, file_path: str) -> Dict[str, int]:
        """
        Refresh a single file in the index (remove and re-add).
        
        Args:
            file_path: Path to the file to refresh
            
        Returns:
            Dictionary with indexing statistics
        """
        # Remove old version
        self.remove_file(file_path)
        
        # Re-index
        return self.index_file(file_path)
    
    def clear(self) -> None:
        """Clear all indexed documents."""
        self.store.clear()
        self._metadata_cache = {}
        logger.info("Cleared all indexed documents")
    
    def count(self) -> int:
        """Get the number of indexed chunks."""
        return self.store.count()
    
    def get_indexed_files(self) -> List[str]:
        """Get list of all indexed file paths."""
        return self.store.get_sources()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed indexing statistics."""
        sources = self.get_indexed_files()
        total_chunks = self.count()
        
        return {
            "total_files": len(sources),
            "total_chunks": total_chunks,
            "average_chunks_per_file": total_chunks / len(sources) if sources else 0,
            "files": sources,
        }
