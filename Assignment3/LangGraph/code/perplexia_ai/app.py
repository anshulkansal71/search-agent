import gradio as gr
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

APP_THEME = gr.themes.Soft()


def create_demo(week: int = 1, mode_str: str = "part1", use_solution: bool = False) -> gr.ChatInterface:
    """Create and return a Gradio demo with the specified week and mode.

    Args:
        week: Which week implementation to use (1 or 2)
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
    else:
        raise ValueError(f"Unknown week: {week}. Choose from: [1, 2]")

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
