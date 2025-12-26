from pydantic import BaseModel, ValidationError
from typing import Any, Optional
import json

# --- Input Schemas ---
class LinearChangeArgs(BaseModel):
    start: float
    rate: float
    time: float
    variable_name: str = "result" # Target variable to update in state

class AllocationArgs(BaseModel):
    total: float
    requested: float
    variable_name: str = "remaining"

class ToolCall(BaseModel):
    tool: str
    arguments: dict

class ToolResult(BaseModel):
    message: str
    state_update: dict[str, Any] = {}

class ToolsRegistry:
    """
    Registry of executable tools with Strict Schema Validation.
    Now returns ToolResult with state updates.
    """
    
    @staticmethod
    def calculate_linear_change(args: LinearChangeArgs) -> ToolResult:
        try:
            result = args.start + (args.rate * args.time)
            msg = f"Result: {result} (Formula: {args.start} + ({args.rate} * {args.time}))"
            return ToolResult(message=msg, state_update={args.variable_name: result})
        except Exception as e:
            return ToolResult(message=f"Math Error: {str(e)}")

    @staticmethod
    def calculate_resource_allocation(args: AllocationArgs) -> ToolResult:
        try:
            remaining = args.total - args.requested
            status = "OK" if remaining >= 0 else "OVERFLOW"
            msg = f"Status: {status} | Remaining: {remaining} (Total: {args.total}, Requested: {args.requested})"
            return ToolResult(message=msg, state_update={args.variable_name: remaining})
        except Exception as e:
            return ToolResult(message=f"Math Error: {str(e)}")
            
    @staticmethod
    def execute_structured_call(json_str: str) -> ToolResult:
        """
        Parses JSON string -> Validates via Pydantic -> Executes.
        Returns ToolResult.
        """
        try:
            # 1. Parse JSON
            data = json.loads(json_str)
            call = ToolCall(**data)
            
            # 2. Dispatch & Validate Arguments
            if call.tool == "calculate_linear_change":
                args = LinearChangeArgs(**call.arguments)
                return ToolsRegistry.calculate_linear_change(args)
                
            elif call.tool == "calculate_resource_allocation":
                args = AllocationArgs(**call.arguments)
                return ToolsRegistry.calculate_resource_allocation(args)
                
            else:
                return ToolResult(message=f"Error: Unknown tool '{call.tool}'")
                
        except json.JSONDecodeError:
            return ToolResult(message="Error: Invalid JSON format.")
        except ValidationError as e:
            return ToolResult(message=f"Schema Validation Error: {e}")
        except Exception as e:
            return ToolResult(message=f"Execution Error: {str(e)}")
