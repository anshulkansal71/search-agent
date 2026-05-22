"""Part 3 - Agent with skills chat scaffold for Week 3."""

from typing import Dict, List, Optional

from perplexia_ai.core.chat_interface import ChatInterface


class AgentWithSkillsChat(ChatInterface):
    """Week 3 Part 3 scaffold: agent using reusable skills."""

    def __init__(self):
        self.llm = None
        self.agent = None
        self.skills = []

    def initialize(self) -> None:
        """Initialize agent, skills, and supporting graph/workflow components."""
        pass

    def process_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Process a message using an agent augmented with skills."""
        return "not implemented"

