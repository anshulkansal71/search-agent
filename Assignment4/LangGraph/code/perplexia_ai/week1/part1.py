"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface

class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""
    
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
    
    def initialize(self) -> None:
        """Initialize components for query understanding.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        """
        # TODO: Students implement initialization
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding.
        
        Students should:
        - Classify the query type
        - Generate appropriate response
        - Format based on query type
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 1
            
        Returns:
            str: The assistant's response
        """
        # TODO: Students implement query understanding
        return "hello"