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

        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.query_classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", "Classify the following query into a category: {query}"),
            ("user", "Query: {query}"),
        ])
        self.response_prompts = ChatPromptTemplate.from_messages([
            ("system", "Format the following response based on the query type: {query_type}"),
            ("user", "Response: {response}"),
        ])
    
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
        class GraphNode(TypedDict):
            prompt: str

        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate an answer or idea"),
        ])

        def generate_node(state: Node) -> Node:
            chain = generate_prompt | model | parser
            idea = chain.invoke({})
            return {"idea": idea}

        def critique_node(state: Node) -> Node:
            chain = critique_prompt | model | parser
            critique = chain.invoke({"idea": state["idea"]})
            return {"critique": critique}


        builder = StateGraph(Node)

        builder.add_node("generate", generate_node)
        builder.add_node("critique", critique_node)

        builder.add_edge(START, "generate")
        builder.add_edge("generate", "critique")
        builder.add_edge("critique", END)

        graph = builder.compile()
        result = graph.invoke({})


        
        return "hello"