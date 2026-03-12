"""
Chat Memory Manager using LlamaIndex.

Replaces the custom ChatMemoryManager with LlamaIndex's
ChatMemoryBuffer which automatically manages conversation history,
token limits, and formatting for LLM prompts.
"""

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from config import settings


class ChatMemoryManager:
    """
    Wrapper around LlamaIndex ChatMemoryBuffer.

    Features
    --------
    - Stores user and assistant messages
    - Automatic token trimming
    - Compatible with LlamaIndex chat engines
    - Simple API similar to previous memory manager
    """

    def __init__(self, token_limit: int = None):
        """
        Initialize memory manager.

        Args:
            token_limit: Maximum tokens to keep in memory
        """
        self.token_limit = token_limit or getattr(settings, "CHAT_MEMORY_TOKEN_LIMIT", 3000)

        # Initialize LlamaIndex memory
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=self.token_limit
        )

    def add_user_message(self, message: str):
        """
        Store user message.

        Args:
            message: User query
        """
        self.memory.put(
            ChatMessage(role="user", content=message)
        )

    def add_assistant_message(self, message: str):
        """
        Store assistant response.

        Args:
            message: LLM response
        """
        self.memory.put(
            ChatMessage(role="assistant", content=message)
        )

    def get_context(self):
        """
        Return conversation history as list of messages.

        Returns:
            List of ChatMessage objects
        """
        return self.memory.get()

    def clear(self):
        """
        Clear memory for a new session.
        """
        self.memory.reset()

    def get_history(self):
        """
        Return full conversation history.

        Returns:
            List of ChatMessage objects
        """
        return self.memory.get()

    def size(self):
        """
        Return number of stored messages.

        Returns:
            int
        """
        return len(self.memory.get())