"""
PocketRAG - Chat Engine Module
"""
import logging
from typing import Optional, Generator

from pocketrag.config import config
from pocketrag.core.search import Searcher

logger = logging.getLogger(__name__)


class ChatMessage:
    """Represents a chat message."""
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class ChatEngine:
    """
    Handles RAG-based chat conversations.
    """
    
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful local assistant powered by PocketRAG. "
        "Use the provided context to answer questions accurately. "
        "If the answer is not in the context, say so clearly. "
        "Do not make up information or speculate beyond what's provided."
    )
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        db_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        top_k: int = 5,
    ):
        """
        Initialize the chat engine.
        
        Args:
            model_name: Name of the Ollama model to use
            db_path: Path to the vector database
            system_prompt: Custom system prompt
            top_k: Number of documents to retrieve for context
        """
        self.model_name = model_name or config.default_model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.top_k = top_k
        
        # Initialize searcher
        self.searcher = Searcher(db_path=db_path)
        
        self._conversation_history: list[ChatMessage] = []
        
        logger.info(f"ChatEngine initialized with model: {self.model_name}")
    
    def _get_context(self, query: str) -> str:
        """Retrieve and format context for the query."""
        _, context = self.searcher.search_with_context(query, top_k=self.top_k)
        return context
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the full prompt for the LLM."""
        return f"{context}\n\nQuestion: {query}"
    
    def chat(self, user_input: str) -> str:
        """
        Send a message and get a complete response.
        
        Args:
            user_input: The user's message
            
        Returns:
            The assistant's response
        """
        try:
            import ollama
        except ImportError:
            logger.error("ollama not installed")
            raise ImportError("Please install ollama: pip install ollama")
        
        # Get context from documents
        context = self._get_context(user_input)
        
        # Build prompt
        prompt = self._build_prompt(user_input, context)
        
        # Call Ollama
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
            )
            
            assistant_message = response['message']['content']
            
            # Store in history
            self._conversation_history.append(ChatMessage('user', user_input))
            self._conversation_history.append(ChatMessage('assistant', assistant_message))
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            raise RuntimeError(f"Failed to get response from Ollama: {e}")
    
    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """
        Send a message and stream the response.
        
        Args:
            user_input: The user's message
            
        Yields:
            Chunks of the assistant's response
        """
        try:
            import ollama
        except ImportError:
            logger.error("ollama not installed")
            raise ImportError("Please install ollama: pip install ollama")
        
        # Get context from documents
        context = self._get_context(user_input)
        
        # Build prompt
        prompt = self._build_prompt(user_input, context)
        
        # Stream from Ollama
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                stream=True,
            )
            
            full_response = []
            for chunk in stream:
                content = chunk['message']['content']
                full_response.append(content)
                yield content
            
            # Store in history
            complete_response = ''.join(full_response)
            self._conversation_history.append(ChatMessage('user', user_input))
            self._conversation_history.append(ChatMessage('assistant', complete_response))
            
        except Exception as e:
            logger.error(f"Ollama stream failed: {e}")
            raise RuntimeError(f"Failed to stream response from Ollama: {e}")
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()
        logger.debug("Conversation history cleared")
    
    @property
    def history(self) -> list[ChatMessage]:
        """Get conversation history."""
        return self._conversation_history.copy()
    
    def is_ready(self) -> bool:
        """Check if the chat engine is ready (documents indexed)."""
        return self.searcher.is_indexed()
