"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory
model = init_chat_model("gpt-4o-mini", model_provider="openai")
parser = StrOutputParser()

QUERY_CLASSIFIER_SYSTEM_PROMPT = """Categorize an input question into exactly one of the following types by outputting only the single best-matching label. Read the input question, use the definitions and examples below to determine the most appropriate category, and respond strictly with the corresponding label—no explanation, justification, or extra text.

Types:
1. **Factual** – The question seeks a specific piece of information, data, or fact.
   - Example: "What is the capital of France?"
2. **Analytical** – The question requires reasoning, interpretation, explanation, or analysis of causes, implications, or mechanisms.
   - Example: "Why did the Roman Empire fall?"
3. **Comparison** – The question explicitly asks for similarities, differences, or evaluation between two or more entities, concepts, events, etc.
   - Example: "How does renewable energy compare to fossil fuels?"
4. **Definition** – The question asks for the meaning or explanation of a term or concept.
   - Example: "What does 'quantum entanglement' mean?"

# Output Format

Respond with a single label only: Factual, Analytical, Comparison, or Definition. Do not include any reasoning, explanation, or other text.

# Examples

- Input: "What causes tides in the ocean?"
  Output: Analytical

- Input: "What is the definition of inflation?"
  Output: Definition

- Input: "Which country has a larger population: India or China?"
  Output: Comparison

- Input: "When did the Second World War end?"
  Output: Factual

# Notes

- Read each question carefully and select the best-fitting label based on the definitions and examples above.
- Respond only with the single category label—no additional text or explanation.
"""

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
            ("system", QUERY_CLASSIFIER_SYSTEM_PROMPT),
            ("user", "{query}")])
        self.factual_prompt = ChatPromptTemplate.from_messages([
            ("system", "answers should be concise and direct"),
            ("user", "{query}")
        ])
        self.analytical_prompt = ChatPromptTemplate.from_messages([
            ("system", "responses should include reasoning steps"),
            ("user", "{query}")
        ])
        self.comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", "should use structured formats (tables, bullet points)"),
            ("user", "{query}")
        ])
        self.definition_prompt = ChatPromptTemplate.from_messages([
            ("system", "should include examples and use cases"),
            ("user", "{query}")
        ])
        # Arize Imports
        from arize.otel import register
        import os 
        from openinference.instrumentation.langchain import LangChainInstrumentor


        tracer_provider = register(
            space_id=os.environ["ARIZE_SPACE_ID"],
            api_key=os.environ["ARIZE_API_KEY"],
            project_name="langGraph-demo", 
        )

        # Instrument LangChain to send traces to the provider we just registered
        LangChainInstrumentor(tracer_provider=tracer_provider).instrument()

        print(f"Instrumentation active. Sending traces to Arize Cloud Project")


    
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
        class Node(TypedDict):
            query_type: str
            output: str

        def classifier_node(state: Node) -> Node:
            chain = self.query_classifier_prompt | self.llm | parser
            query_type = chain.invoke({"query": message}).strip()
            return {"query_type": query_type}

        

        def factual_node(state: Node) -> Node:
            chain = self.factual_prompt | self.llm | parser
            output = chain.invoke({"query": message}).strip()
            return {"output": output}

        def analytical_node(state: Node) -> Node:
            chain = self.analytical_node | self.llm | parser
            output = chain.invoke({"query": message}).strip()
            return {"output": output}

        def comparison_node(state: Node) -> Node:
            chain = self.comparison_node | self.llm | parser
            output = chain.invoke({"query": message}).strip()
            return {"output": output}

        def definition_node(state: Node) -> Node:
            chain = self.definition_node | self.llm | parser
            output = chain.invoke({"query": message}).strip()
            return {"output": output}

        builder = StateGraph(Node)

        builder.add_node("classifier", classifier_node)
        builder.add_edge(START, "classifier")
        builder.add_conditional_edges("classifier",
            path = route_query,
        path_map = {
            "factual_node": "factual_node",
            "analytical_node": "analytical_node",
            "comparison_node": "comparison_node",
            "definition_node": "definition_node",
        })

        builder.add_edge("factual_node", END)
        builder.add_edge("analytical_node", END)
        builder.add_edge("comparison_node", END)
        builder.add_edge("definition_node", END)


        graph = builder.compile()
        result = graph.invoke({})
        
        return result["output"]