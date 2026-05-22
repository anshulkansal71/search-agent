"""Factory for creating Week 3 chat implementations."""

from enum import Enum

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.week3.part1 import DeepResearchChat
from perplexia_ai.week3.part2 import MCPToolsChat
from perplexia_ai.week3.part3 import AgentWithSkillsChat


class Week3Mode(Enum):
    """Modes corresponding to the three parts of Week 3 assignment."""

    PART1_DEEP_RESEARCH = "part1"
    PART2_MCP_TOOLS = "part2"
    PART3_AGENT_WITH_SKILLS = "part3"


def create_chat_implementation(mode: Week3Mode) -> ChatInterface:
    """Create and return the appropriate chat implementation for Week 3."""

    implementations = {
        Week3Mode.PART1_DEEP_RESEARCH: DeepResearchChat,
        Week3Mode.PART2_MCP_TOOLS: MCPToolsChat,
        Week3Mode.PART3_AGENT_WITH_SKILLS: AgentWithSkillsChat,
    }

    if mode not in implementations:
        raise ValueError(f"Unknown mode: {mode}")

    implementation_class = implementations[mode]
    implementation = implementation_class()
    return implementation

