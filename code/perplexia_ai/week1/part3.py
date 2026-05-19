"""Part 3 - Conversation Memory implementation.

This implementation focuses on:
- Maintain context across messages
- Handle follow-up questions
- Use conversation history in responses
"""

import os
import json
from typing import Dict, List, Optional, TypedDict
from perplexia_ai.core.chat_interface import ChatInterface
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator
from langchain_core.prompts import MessagesPlaceholder


load_dotenv()  # reads .env in the current working directory
PARSER = StrOutputParser()

CALCUATOR_TOOL_USAGE_REQUIRED_PROMPT = """Decide whether a calculator tool is REQUIRED to solve the provided math question, based strictly on arithmetic complexity. If a calculator is REQUIRED, also extract and return the arithmetic expression needed for solution, formatted to use only +, -, *, /, and parentheses, as supported by the calculator tool. Do not solve the problem or perform any calculations, and avoid all explanations or commentary.

For basic mental math that almost anyone can solve instantly in their head (single-digit addition/subtraction, or multiplication with very small single-digit numbers), a calculator is NOT_REQUIRED. For any multiplication, division, or arithmetic involving double-digit or larger numbers, decimals, multiple steps, or any operation more complicated than single-digit mental math, a calculator is REQUIRED.

If you determine REQUIRED, always extract the relevant arithmetic expression from the question, and return it in the format supported by the calculator tool (using only numbers, +, -, *, /, and parentheses as needed).

Your response must always be a valid JSON object, structured as specified below, with no explanations, calculations, or extra commentary.

# Steps

1. Read the provided math question.
2. Decide if the arithmetic is trivial, single-step mental math (single-digit addition, subtraction, or multiplication, e.g., 2+4, 3*7, 8-5). If so, select "NOT_REQUIRED".
3. If the question includes multiplication, division, or operations with double-digit numbers, decimals, more than one operation, or requires multiple arithmetic steps, select "REQUIRED".
4. If REQUIRED, extract the full arithmetic expression needed to solve the question. Format it using only +, -, *, /, and parentheses. Do not solve or simplify the expression.
5. Return your answer ONLY as a JSON object according to the Output Format.
6. Do not show any calculations, explanations, or commentary—just the required JSON output.

# Output Format

Respond with a JSON object in one of the following formats:

If the calculator is NOT_REQUIRED:
```json
{
  "calculator_required": false
}
```

If the calculator is REQUIRED:
```json
{
  "calculator_required": true,
  "expression": "[arithmetic_expression]"
}
```
- The "expression" value should represent the complete arithmetic operation required by the question, using only numbers, +, -, *, /, and parentheses—no variables or words.
- Do not include any other keys in the object.
- Do not return any explanations or commentary.

# Examples

Example 1:  
Input: What is 27 x 14?  
Output:
{
  "calculator_required": true,
  "expression": "27*14"
}

Example 2:  
Input: What is 13892 ÷ 587?  
Output:
{
  "calculator_required": true,
  "expression": "13892/587"
}

Example 3:  
Input: What is 4 + 5?  
Output:
{
  "calculator_required": false
}

Example 4:  
Input: If you have $8.25 and buy two items for $3.18 each, how much do you have left?  
Output:
{
  "calculator_required": true,
  "expression": "8.25-(3.18*2)"
}

Example 5:  
Input: What is 2 + 4?  
Output:
{
  "calculator_required": false
}

Example 6:  
Input: What is the result of (13 + 25) x (47 - 8)?  
Output:
{
  "calculator_required": true,
  "expression": "(13+25)*(47-8)"
}

Example 7:  
Input: You have 41 candies, give 8 to Sarah and split the rest evenly among 3 friends. How many candies does each friend receive?  
Output:
{
  "calculator_required": true,
  "expression": "(41-8)/3"
}

(For each, only output the JSON object as shown. Real examples should precisely follow this structure.)

# Notes

- Do not output calculations, reasoning, explanations, or commentary—only the correctly structured JSON.
- Output "calculator_required": false only for truly basic mental math: single-digit addition, subtraction, or multiplication.
- ALWAYS extract and format the full arithmetic expression when replying with calculator_required: true. Use only +, -, *, /, and parentheses; do not include any other operations or variables.
- If the question is ambiguous or the necessary arithmetic is unclear, extract the most direct arithmetic expression implied by the question. If in doubt, err on the side of calculator_required: true.
- Your answer must follow the JSON format exactly.

Use information from messages history only if it's relevant to the prompt. Otherwise, ignore the messages history.
 User messages history is provided here: {messages_history}

Reminder: Decide calculator necessity strictly on arithmetic complexity. If required, extract the needed arithmetic expression using only +, -, *, /, and parentheses—no calculation. Output only a JSON object in the specified structure—no explanations.

**Remember:** The response must always be a valid JSON object as detailed above."""

QUERY_CLASSIFIER_SYSTEM_PROMPT = """Categorize an input question into exactly one of the following types by outputting only the single best-matching label. Read the input question, use the definitions and examples below to determine the most appropriate category, and respond strictly with the corresponding label—no explanation, justification, or extra text.

Types:
1. **factual** – The question seeks a specific piece of information, data, or fact.
   - Example: "What is the capital of France?"
2. **analytical** – The question requires reasoning, interpretation, explanation, or analysis of causes, implications, or mechanisms.
   - Example: "Why did the Roman Empire fall?"
3. **comparison** – The question explicitly asks for similarities, differences, or evaluation between two or more entities, concepts, events, etc.
   - Example: "How does renewable energy compare to fossil fuels?"
4. **definition** – The question asks for the meaning or explanation of a term or concept.
   - Example: "What does 'quantum entanglement' mean?"

# Output Format

Respond with a single label only: factual, analytical, comparison, or definition. Do not include any reasoning, explanation, or other text.

# Examples

- Input: "What causes tides in the ocean?"
  Output: analytical

- Input: "What is the definition of inflation?"
  Output: definition

- Input: "Which country has a larger population: India or China?"
  Output: comparison

- Input: "When did the Second World War end?"
  Output: factual

# Notes

- Read each question carefully and select the best-fitting label based on the definitions and examples above.
- Respond only with the single category label—no additional text or explanation.

Use information from messages history only if it's relevant to the prompt. Otherwise, ignore the messages history.
 User messages history is provided here: {messages_history}
"""


from typing import Dict, List, Optional

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator

class Node(TypedDict):
    calculator_usage_required: str
    math_expression: str
    query: str
    query_type: str
    output: str
    messages_history: str

def _to_langchain_messages(
    chat_history: Optional[List[Dict[str, str]]],
) -> List[HumanMessage | AIMessage]:
    if not chat_history:
        return []
    messages: List[HumanMessage | AIMessage] = []
    for msg in chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


class MemoryChat(ChatInterface):
    """Week 1 Part 3 implementation adding conversation memory."""
    
    def __init__(self):
        self.llm = None
        self.memory = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.calculator = Calculator()
    
    def initialize(self) -> None:
        """Initialize components for memory-enabled chat.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        - Initialize calculator tool
        - Set up conversation memory
        """
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.query_classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_CLASSIFIER_SYSTEM_PROMPT),
            ("user", "{query}")])
        self.response_prompts = {
            "factual_node": ChatPromptTemplate.from_messages(
                [("system", "answers should be concise and direct. Use information from messages history only if it's relevant to the prompt. Otherwise, ignore the messages history. User messages history is provided here: {messages_history}"),
                ("user", "{query}")]
            ),
            "analytical_node": ChatPromptTemplate.from_messages(
                [("system", "responses should include reasoning steps. Use information from messages history only if it's relevant to the prompt. Otherwise, ignore the messages history. User messages history is provided here: {messages_history}"),
                ("user", "{query}")]
            ),
            "comparison_node": ChatPromptTemplate.from_messages(
                [
                    ("system", "should use structured formats (tables, bullet points). Use information from messages history only if it's relevant to the prompt. Otherwise, ignore the messages history. User messages history is provided here: {messages_history}"),
                    ("user", "{query}"),
                ]
            ),
            "definition_node": ChatPromptTemplate.from_messages(
                [("system", "should include examples and use cases. Use information from messages history only if it's relevant to the prompt. Otherwise, ignore the messages history. User messages history is provided here: {messages_history}"),
                ("user", "{query}")]
            ),
        }
        self.query_types = ["factual", "analytical", "comparison", "definition"]

        tracer_provider = register(
            space_id=os.environ["ARIZE_SPACE_ID"],
            api_key=os.environ["ARIZE_API_KEY"],
            project_name="langGraph-demo", 
        )

        # Instrument LangChain to send traces to the provider we just registered
        LangChainInstrumentor(tracer_provider=tracer_provider).instrument()

        print(f"Instrumentation active. Sending traces to Arize Cloud Project")
    
    def run_prompt(self, prompt: ChatPromptTemplate, query: str, messages_history_v2: str) -> str:
        chain = prompt | self.llm | PARSER
        return chain.invoke(
            {"query": query, "messages_history": messages_history_v2}
        ).strip()

    def classifier_node(self, state: Node) -> Node:
        query_type = self.run_prompt(self.query_classifier_prompt, state["query"], state["messages_history"]).lower()
        return {"query_type": query_type}

    def response_node_factory(self, prompt_key: str):
        def _response_node(state: Node) -> Node:
            return {"output": self.run_prompt(self.response_prompts[prompt_key], state["query"], state["messages_history"])}

        return _response_node

    def calculator_tool_call_node(self, state: Node) -> Node:
        expression = state.get("math_expression", "")
        result = self.calculator.evaluate_expression(expression)
        return {"output": str(result)}

    def calculator_tool_call_required(self, state: Node) -> Node:
        messages = [
            SystemMessage(content=CALCUATOR_TOOL_USAGE_REQUIRED_PROMPT),
            HumanMessage(content=state["query"])
        ]
        result = PARSER.invoke(self.llm.invoke(messages)).strip()
        # Strip markdown code fences if present
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
            result = result.strip()
        try:
            data = json.loads(result)
            required = "REQUIRED" if data.get("calculator_required") else "NOT_REQUIRED"
            expression = data.get("expression", "")
        except Exception:
            required = "NOT_REQUIRED"
            expression = ""
        return {"calculator_usage_required": required, "math_expression": expression}
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with memory and tools.
        
        Students should:
        - Use chat history for context
        - Handle follow-up questions
        - Use calculator when needed
        - Format responses appropriately
        
        Args:
            message: The user's input message
            chat_history: List of previous chat messages
            
        Returns:
            str: The assistant's response
        """


        history_messages = _to_langchain_messages(chat_history)
        builder = StateGraph(Node)

        builder.add_node("classifier", self.classifier_node)
        builder.add_node("factual_node", self.response_node_factory("factual_node"))
        builder.add_node("analytical_node", self.response_node_factory("analytical_node"))
        builder.add_node("comparison_node", self.response_node_factory("comparison_node"))
        builder.add_node("definition_node", self.response_node_factory("definition_node"))
        builder.add_node("calculator_tool_call_node", self.calculator_tool_call_node)
        builder.add_node("calculator_tool_call_required",self.calculator_tool_call_required)
        builder.add_edge(START, "calculator_tool_call_required")
        builder.add_conditional_edges("calculator_tool_call_required", 
            lambda state: state["calculator_usage_required"] if state["calculator_usage_required"] in [
                "REQUIRED", "NOT_REQUIRED"
            ] else "NOT_REQUIRED",
            {
                "NOT_REQUIRED": "classifier",
                "REQUIRED": "calculator_tool_call_node"
            })
        builder.add_edge("calculator_tool_call_node", END)
        builder.add_conditional_edges("classifier",
            lambda state: state["query_type"] if state["query_type"] in self.query_types else "factual",
            {
                "factual" : "factual_node",
                "analytical": "analytical_node",
                "comparison": "comparison_node",
                "definition": "definition_node"
            }
        )
        builder.add_edge("factual_node", END)
        builder.add_edge("analytical_node", END)
        builder.add_edge("comparison_node", END)
        builder.add_edge("definition_node", END)


        graph = builder.compile()

        result = graph.invoke({"query": message,"messages_history": str(history_messages)})

        return result.get("output", "")