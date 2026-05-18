"""Part 3 - Agentic RAG implementation using LangGraph.

This implementation focuses on:
- Loading and chunking PDFs into a Chroma vector store
- Building a ReAct agent with document-search and web-search tools
- Agent autonomously decides retrieval strategy based on system prompt criteria
- Synthesizing final answers grounded in retrieved content
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface


class AgenticRAGChat(ChatInterface):
    """Perplexia Journey - Week 2 Part 3 implementation focusing on Agentic RAG."""
    
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.search_tool = None
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for Agentic RAG.
        
        Students should:
        - Initialize the chat model
        - Set up document vector store with OPM documents
        - Create tools for document retrieval and web search
        - Build a ReAct agent that can autonomously decide which tools to use
        """
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the Agentic RAG system.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response
        """
        return "Not implemented yet. Please implement Perplexia Journey - Week 2 Part 3: Agentic RAG."

