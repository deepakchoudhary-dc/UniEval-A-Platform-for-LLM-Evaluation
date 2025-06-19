"""
Transparent AI Chatbot Package
"""

__version__ = "1.0.0"
__author__ = "AI Development Team"
__description__ = "AI Chatbot with Memory, Search, and Explainability"

from .core.chatbot import TransparentChatbot
from .core.memory import MemoryManager
from .core.search import SearchEngine

__all__ = ["TransparentChatbot", "MemoryManager", "SearchEngine"]
