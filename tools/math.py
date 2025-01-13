from llama_index.core.tools import FunctionTool


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

def divide(a: float, b: float) -> float:
    """Divide two numbers and returns the quotient"""
    return a / b

divide_tool = FunctionTool.from_defaults(fn=divide)
