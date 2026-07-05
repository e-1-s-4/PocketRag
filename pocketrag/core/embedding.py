"""
PocketRAG - Embedding Engine Module

Supports multiple embedding backends:
- Sentence Transformers (default)
- FastEmbed (lightweight alternative)
- Custom embedding functions
"""
import logging
from typing import List, Union, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Handles text embedding using multiple backend options.
    
    Supports:
    - Sentence Transformers (full-featured)
    - FastEmbed (lightweight, faster inference)
    - Custom embedding functions
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: Name of the embedding model to use
            use_gpu: Whether to use GPU acceleration (if available)
            batch_size: Batch size for encoding multiple texts
            normalize_embeddings: Whether to normalize output embeddings
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        self._model = None
        self._backend = None  # 'sentence_transformers' or 'fastembed'
        self._dimension = None
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """Load the embedding model with fallback options."""
        # Try FastEmbed first (lighter, faster)
        try:
            from fastembed import TextEmbedding
            logger.info(f"Loading FastEmbed model: {self.model_name}")
            self._model = TextEmbedding(model_name=self.model_name)
            self._backend = 'fastembed'
            logger.info("Using FastEmbed backend (optimized for speed)")
            return
        except ImportError:
            logger.debug("FastEmbed not available, trying sentence-transformers")
        except Exception as e:
            logger.warning(f"FastEmbed failed: {e}, trying sentence-transformers")
        
        # Fallback to Sentence Transformers
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            
            device = "cuda" if self.use_gpu else "cpu"
            self._model = SentenceTransformer(
                self.model_name,
                device=device,
            )
            self._backend = 'sentence_transformers'
            logger.info(f"Using SentenceTransformer backend on {device}")
        except ImportError:
            logger.error("Neither fastembed nor sentence-transformers installed")
            raise ImportError(
                "Please install an embedding backend:\n"
                "  pip install fastembed  # Recommended (faster)\n"
                "  pip install sentence-transformers  # Alternative"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        try:
            if self._backend == 'fastembed':
                embeddings = list(self.model.embed(texts, batch_size=self.batch_size))
                embeddings = np.array(embeddings)
            else:  # sentence_transformers
                embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=len(texts) > 10,
                    batch_size=self.batch_size,
                    normalize_embeddings=False,
                )

            if self.normalize_embeddings:
                # Normalize uniformly across backends for consistent cosine behavior
                norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norm[norm == 0] = 1
                embeddings = embeddings / norm
            
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Numpy array of the embedding
        """
        embeddings = self.embed(text)
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is not None:
            return self._dimension
        
        # Load model to get dimension if not already loaded
        _ = self.model
        
        if self._backend == 'fastembed':
            # Get dimension from a test embedding
            test_emb = self.embed_single("test")
            self._dimension = len(test_emb)
        else:
            self._dimension = self.model.get_sentence_embedding_dimension()
        
        return self._dimension
    
    def embed_batch(
        self,
        texts: List[str],
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        Generate embeddings in batches with progress callback.
        
        Args:
            texts: List of texts to embed
            callback: Optional callback function(current, total) for progress
            
        Returns:
            Numpy array of all embeddings
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        total = len(texts)
        
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.embed(batch)
            all_embeddings.append(batch_embeddings)
            
            if callback:
                callback(i + len(batch), total)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
