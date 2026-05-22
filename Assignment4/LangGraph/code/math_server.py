"""Simple local MCP math server for Assignment 4 notebook demos.

Run with:
    uv run /Users/aish/projects/problem_first_ai/langgraph_2026/math_server.py
"""

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    name="assignment4-math-server",
    instructions=(
        "Use these tools for arithmetic when users ask for precise numeric results."
    ),
)


@mcp.tool(description="Add two numbers.")
def add(a: float, b: float) -> float:
    return a + b


@mcp.tool(description="Subtract b from a.")
def subtract(a: float, b: float) -> float:
    return a - b


@mcp.tool(description="Multiply two numbers.")
def multiply(a: float, b: float) -> float:
    return a * b


@mcp.tool(description="Divide a by b. Raises an error when b is 0.")
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@mcp.tool(description="Raise base to exponent.")
def power(base: float, exponent: float) -> float:
    return base**exponent


if __name__ == "__main__":
    mcp.run(transport="stdio")
