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
Provide a concise, well-cited response to the user's query. When crafting the response, utilize information from the provided user message history, previous LLM outputs, and any tool call responses as relevant. Always prioritize using the most pertinent and up-to-date sources. Cite all sources in the final answer. If certain inputs are not relevant, ignore them.

# Steps

1. Review the user query carefully.
2. If applicable, analyze the provided user messages history {messages_history}, previous LLM output {previous_llm_response}, and tool call response {tool_messages} to extract relevant information.
3. Organize the key information and reasoning steps needed to answer the user's query.
4. Compose a concise response, clearly presenting facts or developments supported by citations.
5. Conclude with a complete list of cited sources (including title, date, and link). If a citation is missing any of these elements, provide as much detail as possible.
6. Exclude unnecessary information, and ignore any input elements not relevant to the user's query.

# Output Format

Your response must be structured into two sections:
- Answer: One to four concise, clearly demarcated bullet points or paragraphs summarizing the main findings or information relevant to the user query, each directly supported by the sources listed.
- Sources: A bullet-point list or numbered list of all sources referenced in your answer, citing the title, date, and (link).

Always present your logical reasoning and synthesis (step-by-step) before producing your final answer.

# Example

User: "What are the latest developments in quantum computing?"

Assistant:

**Answer:**
- Recent advances in quantum error correction have significantly improved qubit stability (see Source 1).
- IBM has released a quantum processor with 433 qubits, moving closer to realistic quantum advantage (see Source 2).
- New quantum machine learning algorithms now achieve up to tenfold speedups for specific tasks (see Source 3).

**Sources:**
1. MIT Technology Review (April 2023) (link)
2. IBM Research Blog (March 2023) (link)
3. Nature Quantum Information (May 2023) (link)

(Real responses should be tailored to the specific user query, and answers can be longer or shorter as needed.)

# Notes

- Ensure that each answer point references at least one source.
- Only use the user message history and tool call responses if they are relevant to the question; otherwise, do not include or reference them.
- The output must always include cited sources as shown in the example. 

Reminder: Your objective is to produce a concise, well-cited answer to the user's query, systematically reasoning through each required information source, and clearly citing all references.
"""

@tool
def tavily_tool(query: str) -> str:
    """Call this tool for any questions related to 2026 events"""
    search_result = TavilySearch(max_results=10, search_depth="advanced", include_answer=True).invoke({"query": query})
    return search_result['answer']  

class WebSearchChat(ChatInterface):
    """Perplexia Journey - Week 2 Part 1 implementation for web search using LangGraph."""
    
    def __init__(self):
        self.plain_model = None
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
        self.plain_model = init_chat_model("gpt-5-mini", model_provider="openai", reasoning_effort="minimal")
        self.llm = self.plain_model.bind_tools([tavily_tool])
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
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.search_prompt_system),
            ("human", "query: {message}"),
        ])
        chain = prompt | self.llm
        messages_history = self.convert_to_llm_messages(chat_history)
        llm_result = chain.invoke({"message": message, "messages_history": str(messages_history), "tool_messages": "", "previous_llm_response": ""})
        print(str(llm_result))
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
        final_chain = prompt | self.plain_model
        final_response = final_chain.invoke({"message": message, "messages_history": str(messages_history),
                "previous_llm_response": str(llm_result),"tool_messages": str(tool_messages)})
        return final_response.content or ""

    def convert_to_llm_messages(self, chat_history: Optional[List[Dict[str, str]]]) -> List[HumanMessage | AIMessage]:
        if not chat_history:
            return []

        messages = []
        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "assistant":
                messages.append(AIMessage(content=content))
            else :
                messages.append(HumanMessage(content=content))
            
        return messages



