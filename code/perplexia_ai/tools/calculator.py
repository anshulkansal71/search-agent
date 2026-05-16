import re
from typing import Union

class Calculator:
    """A simple calculator tool for evaluating basic arithmetic expressions."""
    
    @staticmethod
    def evaluate_expression(expression: str) -> Union[float, str]:
        """Evaluate a basic arithmetic expression.
        
        Supports basic arithmetic operations (+, -, *, /), parentheses,
        and the round(...) function.
        Returns an error message if the expression is invalid or cannot be 
        evaluated safely.
        
        Args:
            expression: A string containing a mathematical expression
                       e.g. "5 + 3" or "10 * (2 + 3)"
            
        Returns:
            Union[float, str]: The result of the evaluation, or an error message
                              if the expression is invalid
        
        Examples:
            >>> Calculator.evaluate_expression("5 + 3")
            8.0
            >>> Calculator.evaluate_expression("10 * (2 + 3)")
            50.0
            >>> Calculator.evaluate_expression("15 / 3")
            5.0
        """
        try:
            # Clean up the expression
            expression = expression.strip()
            
            # Allow digits, operators, whitespace, parentheses, commas, dots, and letters.
            if not re.match(r'^[\d\s\+\-\*\/\(\)\.,a-zA-Z]*$', expression):
                return "Error: Invalid characters in expression"

            # Restrict allowed function names to `round` only.
            names = re.findall(r"[A-Za-z_]+", expression)
            if any(name != "round" for name in names):
                return "Error: Invalid expression"
            
            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}, "round": round})
            
            # Convert to float and handle division by zero
            return float(result)
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except (SyntaxError, TypeError, NameError):
            return "Error: Invalid expression"
        except Exception as e:
            return f"Error: {str(e)}"
