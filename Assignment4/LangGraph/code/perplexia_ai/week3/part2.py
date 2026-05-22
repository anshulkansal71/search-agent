"""Part 2 - MCP tools chat scaffold for Week 3."""

from typing import Dict, List, Optional

from perplexia_ai.core.chat_interface import ChatInterface


class MCPToolsChat(ChatInterface):
    """Week 3 Part 2 scaffold: MCP-powered tools chat."""

    def __init__(self):
        self.llm = None
        self.mcp_client = None
        self.tools = None
        self.graph = None

    def initialize(self) -> None:
        """Initialize MCP clients, tools, and graph/agent components."""
        pass

    def process_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Process a message using MCP tools."""
        return "not implemented"

