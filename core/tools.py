import math
import re

class ToolsRegistry:
    """
    Registry of executable tools for the Cognitive System.
    Prevents "Mental Math" errors by delegating calculations to Python.
    """
    
    @staticmethod
    def calculate_linear_change(start: float, rate: float, time: float) -> str:
        """
        Calculates linear change over time.
        Formula: Start + (Rate * Time)
        """
        try:
            result = start + (rate * time)
            return f"Result: {result} (Formula: {start} + ({rate} * {time}))"
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def calculate_resource_allocation(total: float, requested: float) -> str:
        """
        Calculates remaining resources and checks for overflow.
        """
        try:
            remaining = total - requested
            status = "OK" if remaining >= 0 else "OVERFLOW"
            return f"Status: {status} | Remaining: {remaining} (Total: {total}, Requested: {requested})"
        except Exception as e:
            return f"Error: {str(e)}"
            
    @staticmethod
    def execute_tool_call(call_str: str) -> str:
        """
        Parses and executes a tool call string.
        Format: TOOL_CALL: function_name(arg1=val1, ...)
        """
        try:
            # Extract function name and arguments
            match = re.search(r"TOOL_CALL:\s*(\w+)\((.*)\)", call_str)
            if not match:
                return "Error: Invalid tool call format."
                
            func_name = match.group(1)
            args_str = match.group(2)
            
            # Parse arguments (simple parsing for now)
            kwargs = {}
            if args_str:
                for arg in args_str.split(','):
                    key, val = arg.split('=')
                    kwargs[key.strip()] = float(val.strip())
            
            # Dispatch
            if func_name == "calculate_linear_change":
                return ToolsRegistry.calculate_linear_change(**kwargs)
            elif func_name == "calculate_resource_allocation":
                return ToolsRegistry.calculate_resource_allocation(**kwargs)
            else:
                return f"Error: Unknown tool '{func_name}'"
                
        except Exception as e:
            return f"Error execution tool: {str(e)}"
