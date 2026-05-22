"""Part 1 - Deep research chat scaffold for Week 3."""

from typing import Dict, List, Optional, TypedDict

from perplexia_ai.core.chat_interface import ChatInterface
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
from .prompts import RESEARCH_REPORT_FORMAT_GENERATION_PROMPT, SECTION_INVESTIGATION_PROMPT, FINALIZER_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_core.messages import HumanMessage
import os
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
import asyncio

class ResearchSection(BaseModel):
    section_name: str = Field(description = "Section name of the research report under detailed analysis")
    brief_overview: str = Field(description = "Brief overview of section.")
    questions: list[str] = Field(description = "List of questions that needs to be researched upon to create research report for that section.")

class ResearchSectionInvestigation(TypedDict):
    section_name: str
    brief_overview: str
    section_analysis: str
    sources: list[str]

class ResearchNode(TypedDict):
    research_topic: str
    research_sections: list[ResearchSection]
    research_section_investigations: list[ResearchSectionInvestigation]
    final_report: str

class ResearchAnalysis(BaseModel):
    section_analysis: str = Field(description = "Analysis of the section.")
    sources: list[str] = Field(description = "Sources of the analysis of the section.")

class ResearchSections(BaseModel):
    research_sections: list[ResearchSection] = Field(description = "List of research sections under detailed analysis.")

class DeepResearchChat(ChatInterface):
    """Week 3 Part 1 scaffold: deep research style agent chat."""

    def __init__(self):
        self.llm = None
        self.graph = None
        self.tools = None

    def initialize(self) -> None:
        """Initialize models, tools, and graph components for deep research."""
        load_dotenv()

        # Arize setup
        tracer_provider = register(
            space_id=os.environ["ARIZE_SPACE_ID"],
            api_key=os.environ["ARIZE_API_KEY"],
            project_name="langGraph-demo", 
        )
        # Instrument LangChain to send traces to the provider we just registered
        LangChainInstrumentor(tracer_provider=tracer_provider).instrument()

        self.tools = self._create_tools()
        self.llm = init_chat_model(
            "gpt-5-mini",
            model_provider="openai",
            reasoning_effort="low",
        )
        
        builder = StateGraph(ResearchNode)
        builder.add_node("research_manager_agent", self.reasearch_manager_agent)
        builder.add_node("specialised_research_agents", self.specialised_research_agents)
        builder.add_node("finalizer_agent", self.finalizer_agent)

        builder.add_edge(START, "research_manager_agent")
        builder.add_edge("research_manager_agent", "specialised_research_agents")
        builder.add_edge("specialised_research_agents", "finalizer_agent")
        builder.add_edge("finalizer_agent", END)
        
        self.graph = builder.compile()

    def _create_tools(self) -> list:
        @tool
        def tavily_tool(query: str) -> str:
            """Call this tool for searching any query"""
            search_result = TavilySearch(max_results=10, search_depth="advanced", include_answer=True).invoke({"query": query})
            return search_result['answer']  
        return [tavily_tool]

    def reasearch_manager_agent(self, state: ResearchNode) -> ResearchNode:
        agent = create_agent(
            self.llm,
            tools=self.tools,
            response_format = ResearchSections,
            system_prompt=RESEARCH_REPORT_FORMAT_GENERATION_PROMPT
        )
        result = agent.invoke({"messages": [HumanMessage(state["research_topic"])]})

        return {"research_sections": result["structured_response"].research_sections}
    
    async def specialised_research_agents(self, state: ResearchNode) -> ResearchNode :
        research_sections = state["research_sections"]
        research_section_investigations : List[ResearchSectionInvestigation] = []
        agent = create_agent(
            self.llm,
            tools=self.tools,
            response_format = ResearchAnalysis,
            system_prompt=SECTION_INVESTIGATION_PROMPT
        )
        research_section_investigations = await asyncio.gather(*(self.execute_investigation(research_section, agent) for research_section in research_sections), return_exceptions=True)
        # TODO: delete the research sections from state as it is no longer required.

        return {"research_section_investigations": research_section_investigations}
        

    async def execute_investigation(self, research_section: ResearchSection, agent) -> ResearchSectionInvestigation:
        user_prompt = (
            "section_name: " + research_section.section_name
            + "\n\nbrief_overview: " + research_section.brief_overview
            + "\n\nquestions: " + ", ".join(research_section.questions)
        )
        result = await agent.ainvoke({
            "messages": [HumanMessage(user_prompt)]
        })
        curr_analysis = result["structured_response"]

        return {
            "section_name": research_section.section_name,
            "brief_overview": research_section.brief_overview,
            "section_analysis": curr_analysis.section_analysis,
            "sources": curr_analysis.sources,
        }

    def finalizer_agent(self, state: ResearchNode) -> ResearchNode:
        prompt = ChatPromptTemplate([
            ("user", FINALIZER_PROMPT)
        ])
        parser = StrOutputParser()
        chain = prompt | self.llm | parser
        final_report = chain.invoke({"research_topic": state["research_topic"], "sections_summary": state["research_section_investigations"]})
        return {"final_report": final_report }

    

    def process_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Process a message using the deep research workflow."""
        final_report = asyncio.run(self.graph.ainvoke({"research_topic": message}))["final_report"]
        return final_report