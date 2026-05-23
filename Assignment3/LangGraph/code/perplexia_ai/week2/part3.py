"""Part 3 - Agentic RAG implementation using LangGraph.

This implementation focuses on:
- Loading and chunking PDFs into a Chroma vector store
- Building a ReAct agent with document-search and web-search tools
- Agent autonomously decides retrieval strategy based on system prompt criteria
- Synthesizing final answers grounded in retrieved content
"""

from typing import Dict, List, Optional, TypedDict
from perplexia_ai.core.chat_interface import ChatInterface
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

import os


SYSTEM_PROMPT = """
You are a inteligent agent which is used to answer user queries. 
Use tools to retrieve relevant information to the query.
Firstly, use vector_store_retrieval to fetch relevant context. Answer the query using the
fetched context. If there is no relevant context, then use vector_store_retrieval to
search and return the result using the fetched response.
"""


class StateNode(TypedDict):
    query: str
    response: str


class AgenticRAGChat(ChatInterface):
    """Perplexia Journey - Week 2 Part 3 implementation focusing on Agentic RAG."""
    

    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.search_tool = None
        self.graph = None
        self.tools = None

    
    def initialize(self) -> None:
        """Initialize components for Agentic RAG.
        
        Students should:
        - Initialize the chat model
        - Set up document vector store with OPM documents
        - Create tools for document retrieval and web search
        - Build a ReAct agent that can autonomously decide which tools to use
        """
        self.llm = init_chat_model("gpt-5-mini", model_provider="openai", reasoning_effort="minimal")
        self.search_tool = TavilySearch(max_results=10, search_depth="advanced", include_answer=True)
        tracer_provider = register(
            space_id=os.environ["ARIZE_SPACE_ID"],
            api_key=os.environ["ARIZE_API_KEY"],
            project_name="langGraph-demo", 
        )

        # Instrument LangChain to send traces to the provider we just registered
        LangChainInstrumentor(tracer_provider=tracer_provider).instrument()

        print(f"Instrumentation active. Sending traces to Arize Cloud Project")

        
        # Resolve docs relative to this file, not cwd (run.py is often started from repo root)
        directory_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "docs")
        )
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"PDF docs directory not found: {directory_path}")
        loader = PyPDFDirectoryLoader(directory_path)
        pages = loader.load()
        print("Number of pages: " , len(pages))

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db",
            collection_name="opm_documents_1"
        )
        # Add the chunks (pages) from the PDF to the vector store
        self.vector_store.add_documents(pages)

        @tool
        def tavily_tool(query: str) -> str:
            """Call this tool for any questions related to 2026 events"""
            search_result = self.search_tool.invoke({"query": query})
            return str(search_result)
            # return str(search_result['answer'])

        @tool
        def vector_store_retrieval(query: str) -> str:
            """Call this tool to fetch relevant context from the ingested docs. """
            relevant_docs = self.vector_store.similarity_search(query)
            return str(relevant_docs)

        self.tools = [tavily_tool, vector_store_retrieval]

        builder = StateGraph(StateNode)
        builder.add_node("agent", self.run_agent)
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        self.graph = builder.compile()
    
    def run_agent(self, state_node: StateNode):
        agent = create_agent(
            self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT
        )
        result = agent.invoke({"messages": [HumanMessage(state_node["query"])]})
        return {"response": result["messages"][-1].content}

    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the Agentic RAG system.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response
        """
        result = self.graph.invoke({"query": message})
        return result.get("response", "")

