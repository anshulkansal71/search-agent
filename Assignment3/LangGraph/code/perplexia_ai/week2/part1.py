"""
Part 1 - Web Search implementation using LangGraph with Tracing.

This implementation focuses on:
- Setting up web search using Tavily
- Processing search results
- Formatting responses with citations
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from arize.otel import register
import os 
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


SEARCH_SYSTEM_PROMPT = """
Return the response for the user query. You can use the available tool calls, if required.
Response should be co-incise and add citations to the response.

Output format - 


Here is one of the example response - 
User: "What are the latest developments in quantum computing?"
Assistant: "Based on recent web search results:

1. Breakthrough in Error Correction
   Researchers at MIT announced a new quantum error correction method that improves qubit stability by 45%.

2. Commercial Quantum Computing Milestones
   IBM's latest quantum processor reached 433 qubits, bringing practical quantum advantage closer.

3. Quantum Machine Learning Applications
   New algorithms demonstrate 10x speedup for specific machine learning tasks.

Sources:
- MIT Technology Review (April 2023) (link)
- IBM Research Blog (March 2023) (link)
- Nature Quantum Information (May 2023) (link)

Use user messages history, if relevant for answering the question. Otherwise, ignore it.
user messages : {messages_history}
"""

@tool
def tavily_tool(query: str) -> str:
    """Call this tool for any questions related to 2026 events"""
    search_result = TavilySearch(max_results=10, search_depth="advanced", include_answer=True).invoke({"query": query})
    return search_result['answer']  

class WebSearchChat(ChatInterface):
    """Perplexia Journey - Week 2 Part 1 implementation for web search using LangGraph."""
    
    def __init__(self):
        self.llm = None
        self.search_tool = None
        self.graph = None
        self.search_prompt_system = None
    
    def initialize(self) -> None:   
        """Initialize components for web search and tracing.
        
        Students should:
        - Initialize the LLM
        - Set up Tavily search tool
        - Create a LangGraph workflow for web search
        """
        plain_model = init_chat_model("gpt-5-mini", model_provider="openai", reasoning_effort="minimal")
        self.llm = plain_model.bind_tools([tavily_tool])
        self.search_prompt_system = SEARCH_SYSTEM_PROMPT
        

        tracer_provider = register(
            space_id=os.environ["ARIZE_SPACE_ID"],
            api_key=os.environ["ARIZE_API_KEY"],
            project_name="langGraph-demo", 
        )

        # Instrument LangChain to send traces to the provider we just registered
        LangChainInstrumentor(tracer_provider=tracer_provider).instrument()

        
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using web search and record trace.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response based on web search results
        """
        prompt = ChatPromptTemplate([
            SystemMessage(content=self.search_prompt_system),
            MessagesPlaceholder(variable_name= "messages_history"),
            HumanMessage(content=f"query: {message}"),
        ])
        chain = prompt | self.llm
        messages_history = convert_to_llm_messages(chat_history)
        llm_result = chain.invoke({"message": message, "messages_history": messages_history})

        if not llm_result.tool_calls:
            return llm_result.content or ""

        tool_messages = []
        for tool_call in llm_result.tool_calls:
            tool_result = tavily_tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

        final_response = self.llm.invoke(messages + [llm_result] + tool_messages)
        return final_response.content or ""

    def convert_to_llm_messages(chat_history: Optional[List[Dict[str, str]]]) -> List[HumanMessage | AIMessage]:
        if not chat_history:
            return []

        for message in chat_history or []:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "assistant":
                messages.append(AIMessage(content=content))
            else :
                messages.append(HumanMessage(content=content))
            
        return messages



