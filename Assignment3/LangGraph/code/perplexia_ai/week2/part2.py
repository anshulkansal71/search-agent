"""Part 2 - Document RAG implementation using LangGraph.

This implementation focuses on:
- Setting up document loading and processing
- Creating vector embeddings and storage
- Implementing retrieval-augmented generation
- Formatting responses with citations from OPM documents
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

import os

PROMPT = """
For the given user query: {user_query} please provide an answer.
Documents content is present here: {doc_content}
Answer should be basis on the documents content provided. 
If answer is not present in content, don't answer that.
Please refer previous interactions if relevant. Otherwise, ignore it.
Previous interactions are provided here: {messages_history}

Output format:
is_answer_present: bool(true/false)
answer: string
"""

#  Previous interactions with users are provided here: {user_history}


def to_langchain_messages(chat_history: Optional[List[Dict[str, str]]]) -> List[HumanMessage | AIMessage]:
    if not chat_history:
        return []
    messages: List[HumanMessage | AIMessage] = []
    for message in chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content))
        else:
            messages.append(HumanMessage(content))
    return messages

class RagResult(BaseModel):
    is_answer_present: bool = Field(description="Return true or false depending on whether relevant context is present in prompt to answer the question.")
    query_response: str = Field(description="Response of the query if is_answer_present is true. Otherwise, leave it blank")



class DocumentRAGChat(ChatInterface):
    """Perplexia Journey - Week 2 Part 2 implementation for document RAG."""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.document_paths = []
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for document RAG.
        
        Students should:
        - Initialize the LLM
        - Set up document loading and processing (e.g., OPM annual reports)
        - Create vector embeddings
        - Build retrieval system
        - Create LangGraph for RAG workflow
        """
        model = init_chat_model("gpt-5-mini", model_provider="openai", reasoning_effort="minimal")
        self.llm = model.with_structured_output(RagResult)
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
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using document RAG.
        
        Should retrieve relevant information from documents and generate responses.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response based on document knowledge
        """
        messages_history = to_langchain_messages(chat_history)

        docs = self.vector_store.similarity_search(message, k=5)
        chatPromptTemplate = ChatPromptTemplate.from_template(PROMPT)

        chain = chatPromptTemplate | self.llm
        response = chain.invoke({"user_query": message, "doc_content": str(docs), "messages_history": str(messages_history)})
        if not response.is_answer_present:
            return "Sorry, we don't have enough information related to this question"
        return response.query_response
