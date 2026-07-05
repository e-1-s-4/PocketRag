"""
PocketRAG - Core Text Chunking Module
"""
import logging
import re
from typing import List, Optional, Iterator

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text into overlapping chunks with intelligent boundary detection.
    
    Supports multiple chunking strategies:
    - Character-based with fixed size
    - Sentence-aware chunking
    - Token-aware chunking (when tiktoken is available)
    """
    
    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')
    
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        chunk_by_sentence: bool = True,
        min_chunk_size: int = 20,
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of overlapping characters between chunks
            chunk_by_sentence: Try to break on sentence boundaries
            min_chunk_size: Minimum chunk size to keep (smaller chunks are discarded)
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_by_sentence = chunk_by_sentence
        self.min_chunk_size = min_chunk_size
        
        logger.debug(
            f"TextChunker initialized: chunk_size={chunk_size}, "
            f"overlap={overlap}, by_sentence={chunk_by_sentence}"
        )
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        if self.chunk_by_sentence:
            chunks = self._chunk_by_sentences(text)
        else:
            chunks = self._chunk_fixed(text)
        
        # Filter out chunks that are too small
        filtered_chunks = [
            c for c in chunks 
            if len(c.strip()) >= self.min_chunk_size
        ]
        
        logger.debug(
            f"Split text into {len(filtered_chunks)} chunks "
            f"(removed {len(chunks) - len(filtered_chunks)} small chunks)"
        )
        return filtered_chunks
    
    def _chunk_fixed(self, text: str) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        if not text:
            return []

        chunks = []
        step = self.chunk_size - self.overlap
        
        if len(text) <= self.chunk_size:
            return [text]

        for i in range(0, len(text), step):
            if i + self.overlap >= len(text) and i > 0:
                break
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks respecting sentence boundaries.
        
        This method tries to break chunks at sentence boundaries when possible,
        which produces more coherent chunks for retrieval.
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk_size
            if current_length + sentence_length > self.chunk_size:
                # If we have accumulated content, save it as a chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Start new chunk with overlap from previous
                    if self.overlap > 0 and current_chunk:
                        # Calculate how much of the last sentences to keep for overlap
                        overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                        current_chunk = [overlap_text] if overlap_text else []
                        current_length = len(overlap_text)
                    else:
                        current_chunk = []
                        current_length = 0
                
                # Add the current sentence to the new chunk
                if sentence_length <= self.chunk_size:
                    current_chunk.append(sentence)
                    current_length = sentence_length
                else:
                    # Sentence is too long, split it
                    long_chunks = self._chunk_fixed(sentence)
                    chunks.extend(long_chunks[:-1])  # Add all but last
                    # Start new chunk with the last part
                    if long_chunks[-1]:
                        current_chunk = [long_chunks[-1]]
                        current_length = len(long_chunks[-1])
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Simple sentence splitting
        sentences = self.SENTENCE_ENDINGS.split(text)
        return sentences
    
    def _get_overlap_text(self, sentences: List[str], overlap_chars: int) -> str:
        """Get the trailing text from sentences for overlap."""
        if not sentences:
            return ""
        
        # Start from the end and accumulate until we have enough characters
        result = []
        count = 0
        
        for sentence in reversed(sentences):
            if count >= overlap_chars:
                break
            result.insert(0, sentence)
            count += len(sentence) + 1
        
        return ' '.join(result)
    
    def chunk_stream(self, text: str) -> Iterator[str]:
        """
        Generator that yields chunks one at a time.
        
        Useful for processing large texts without loading all chunks into memory.
        
        Args:
            text: The text to chunk
            
        Yields:
            Individual text chunks
        """
        for chunk in self.chunk(text):
            yield chunk
