import threading
from typing import List, Dict
from collections import defaultdict
from loguru import logger


class ChatHistoryService:
    """
    Thread-safe, in-memory repository for managing conversation logs.
    Ensures that conversation history is preserved across consecutive QA queries.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "ChatHistoryService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                # Store chat messages as a list of dicts mapping conversation_id -> list
                cls._instance._history = defaultdict(list)
            return cls._instance

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """
        Appends a new query or answer to the chat history.
        """
        if not conversation_id:
            return

        with self._lock:
            self._history[conversation_id].append({"role": role, "content": content})
        logger.debug(
            f"Added '{role}' message to conversation '{conversation_id}'. Log size: {len(self._history[conversation_id])}"
        )

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Retrieves all past messages for a conversation ID.
        """
        if not conversation_id:
            return []

        with self._lock:
            # Return a copy of the list to prevent external mutation issues
            return list(self._history[conversation_id])

    def clear_history(self, conversation_id: str) -> None:
        """
        Resets and clears the history for a specific conversation ID.
        """
        if not conversation_id:
            return

        with self._lock:
            if conversation_id in self._history:
                del self._history[conversation_id]
        logger.info(f"Cleared chat history for conversation '{conversation_id}'.")
