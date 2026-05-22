"""
Part 1 - Web Search implementation using LangGraph with Tracing.

This implementation focuses on:
- Setting up web search using Tavily
- Processing search results
- Formatting responses with citations
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface


class WebSearchChat(ChatInterface):
    """Perplexia Journey - Week 2 Part 1 implementation for web search using LangGraph."""
    
    def __init__(self):
        self.llm = None
        self.search_tool = None
        self.graph = None
    
    def initialize(self) -> None:   
        """Initialize components for web search and tracing.
        
        Students should:
        - Initialize the LLM
        - Set up Tavily search tool
        - Create a LangGraph workflow for web search
        """
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using web search and record trace.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response based on web search results
        """
        return "Not implemented yet. Please implement Perplexia Journey - Week 2 Part 1: Web Search with LangGraph."
