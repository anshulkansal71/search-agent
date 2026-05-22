import gradio as gr
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

APP_THEME = gr.themes.Soft()


def create_demo(week: int = 1, mode_str: str = "part1", use_solution: bool = False) -> gr.ChatInterface:
    """Create and return a Gradio demo with the specified week and mode.

    Args:
        week: Which week implementation to use (1, 2, or 3)
        mode_str: String representation of the mode ('part1', 'part2', or 'part3')
        use_solution: If True, use solution implementation; otherwise use student code

    Returns:
        gr.ChatInterface: Configured Gradio chat interface
    """
    code_type = "Solution" if use_solution else "Student"

    if week == 1:
        if use_solution:
            from perplexia_ai.solutions.week1.factory import Week1Mode, create_chat_implementation
        else:
            from perplexia_ai.week1.factory import Week1Mode, create_chat_implementation

        mode_map = {
            "part1": Week1Mode.PART1_QUERY_UNDERSTANDING,
            "part2": Week1Mode.PART2_BASIC_TOOLS,
            "part3": Week1Mode.PART3_MEMORY,
        }

        titles = {
            "part1": f"Perplexia AI - Week 1: Query Understanding ({code_type})",
            "part2": f"Perplexia AI - Week 1: Basic Tools ({code_type})",
            "part3": f"Perplexia AI - Week 1: Memory ({code_type})",
        }

        descriptions = {
            "part1": "Your intelligent AI assistant that can understand different types of questions and format responses accordingly.",
            "part2": "Your intelligent AI assistant that can answer questions, perform calculations, and format responses.",
            "part3": "Your intelligent AI assistant that can answer questions, perform calculations, and maintain conversation context.",
        }

        examples = [
            "What is machine learning?",
            "Compare SQL and NoSQL databases",
            "If I have a dinner bill of $120, what would be a 15% tip?",
            "What about 20%?",
        ]

    elif week == 2:
        if use_solution:
            from perplexia_ai.solutions.week2.factory import Week2Mode, create_chat_implementation
        else:
            from perplexia_ai.week2.factory import Week2Mode, create_chat_implementation

        mode_map = {
            "part1": Week2Mode.PART1_WEB_SEARCH,
            "part2": Week2Mode.PART2_DOCUMENT_RAG,
            "part3": Week2Mode.PART3_AGENTIC_RAG,
        }

        titles = {
            "part1": f"Perplexia AI - Week 2: Web Search ({code_type})",
            "part2": f"Perplexia AI - Week 2: Document RAG ({code_type})",
            "part3": f"Perplexia AI - Week 2: Agentic RAG ({code_type})",
        }

        descriptions = {
            "part1": "Your intelligent AI assistant that can search the web for real-time information.",
            "part2": "Your intelligent AI assistant that can retrieve information from OPM documents.",
            "part3": "Your intelligent AI assistant that can use Agentic RAG to answer questions.",
        }

        if mode_str == "part1":
            examples = [
                "What are the latest developments in quantum computing?",
                "Who is the current CEO of SpaceX?",
                "What were the major headlines in tech news this week?",
                "Compare React and Angular frameworks",
            ]
        elif mode_str == "part2":
            examples = [
                "What new customer experience improvements did OPM implement for retirement services in FY 2022?",
                "How did OPM's approach to improving the federal hiring process evolve from FY 2019 through FY 2022?",
                "What were the performance metrics for OPM in 2020? Compare them with 2019.",
                "What strategic goals did OPM outline in the 2022 report?",
            ]
        else:
            examples = [
                "What were OPM's strategic goals in 2022, and how do they compare to current federal workforce trends?",
                "What retirement service improvements did OPM make in FY 2022? Are there any recent updates?",
                "How does OPM's hiring process compare to private sector best practices?",
                "What were OPM's 2020 performance metrics, and what has changed since then?",
            ]
    elif week == 3:
        if use_solution:
            from perplexia_ai.solutions.week3.factory import Week3Mode, create_chat_implementation
        else:
            from perplexia_ai.week3.factory import Week3Mode, create_chat_implementation

        mode_map = {
            "part1": Week3Mode.PART1_DEEP_RESEARCH,
            "part2": Week3Mode.PART2_MCP_TOOLS,
            "part3": Week3Mode.PART3_AGENT_WITH_SKILLS,
        }

        titles = {
            "part1": f"Perplexia AI - Week 3: DeepResearchChat ({code_type})",
            "part2": f"Perplexia AI - Week 3: MCPToolsChat ({code_type})",
            "part3": f"Perplexia AI - Week 3: AgentWithSkillsChat ({code_type})",
        }

        descriptions = {
            "part1": "Build a deep-research workflow that plans, searches, and synthesizes evidence-backed outputs.",
            "part2": "Use MCP-connected tools to retrieve external data and answer user queries with tool calls.",
            "part3": "Build an agent that can leverage reusable skills for structured research and response generation.",
        }

        if mode_str == "part1":
            examples = [
                "Research the latest open-source text-to-SQL trends and summarize in 5 bullets.",
                "Create a deep-research brief on MCP adoption in AI agent frameworks.",
                "Compare two approaches for long-horizon agent planning and list tradeoffs.",
                "Prepare a concise evidence-backed summary of AI coding assistant benchmarks.",
            ]
        elif mode_str == "part2":
            examples = [
                "Use tools to fetch https://docs.langchain.com/oss/python/langchain/mcp and summarize the setup steps.",
                "Use tools to compute (23 + 19) * 7.",
                "Use MCP tools to retrieve a webpage and extract the top 3 key points.",
                "Show how to combine a fetch MCP server and a local math MCP server in one agent.",
            ]
        else:
            examples = [
                "Use available skills to draft a research note on deep research agents.",
                "Create an assignment briefing with findings, risks, and next actions.",
                "Generate a structured weekly update using relevant skills where helpful.",
                "Plan and produce a concise implementation note for an MCP-enabled assistant.",
            ]
    else:
        raise ValueError(f"Unknown week: {week}. Choose from: [1, 2, 3]")

    if mode_str not in mode_map:
        raise ValueError(f"Unknown mode: {mode_str}. Choose from: {list(mode_map.keys())}")

    mode = mode_map[mode_str]
    chat_interface = create_chat_implementation(mode)
    chat_interface.initialize()

    def respond(message: str, history: List[Dict[str, str]]) -> str:
        """Process the message and return a response."""
        return chat_interface.process_message(message, history)

    demo = gr.ChatInterface(
        fn=respond,
        title=titles[mode_str],
        description=descriptions[mode_str],
        examples=examples,
        run_examples_on_click=True,
        chatbot=gr.Chatbot(show_label=False, height=560, layout="bubble"),
        textbox=gr.Textbox(
            show_label=False,
            placeholder="Ask a question, compare ideas, or run a quick calculation...",
            lines=1,
            max_lines=5,
            submit_btn="Send",
            stop_btn="Stop",
        ),
        fill_height=True,
        fill_width=True,
        show_progress="minimal",
    )
    return demo
