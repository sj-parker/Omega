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

class SearchAndExtractArgs(BaseModel):
    query: str
    target: str = "fact"
    volatility: str = "low"

class VerifyFactArgs(BaseModel):
    fact: str
    original_source: str
    volatility: str = "low"

class GetCurrentTimeArgs(BaseModel):
    # No arguments needed, but keeps schema consistent
    pass

class ToolCall(BaseModel):
    tool: str
    arguments: dict

class ToolResult(BaseModel):
    message: str
    state_update: dict[str, Any] = {}
    
    def to_dict(self) -> dict:
        return self.dict() if hasattr(self, 'dict') else self.model_dump()

class ToolsRegistry:
    """
    Registry of executable tools with Strict Schema Validation.
    Now returns ToolResult with state updates.
    """
    
    _search_engine = None

    @classmethod
    def get_search_engine(cls):
        """Lazy load search engine."""
        if cls._search_engine is None:
            from core.search_engine import SearchEngine
            cls._search_engine = SearchEngine()
        return cls._search_engine

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
    async def search_and_extract(args: SearchAndExtractArgs) -> ToolResult:
        try:
            from datetime import datetime
            now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            engine = ToolsRegistry.get_search_engine()
            res = await engine.search_and_extract(args.query, args.target, volatility=args.volatility)
            msg = f"Fact: {res['fact']} (Retrieved: {now}, Confidence: {res['confidence']}, Source: {res['source']})"
            return ToolResult(message=msg)
        except Exception as e:
            return ToolResult(message=f"Search Error: {str(e)}")

    @staticmethod
    async def verify_fact(args: VerifyFactArgs) -> ToolResult:
        try:
            from datetime import datetime
            now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            engine = ToolsRegistry.get_search_engine()
            res = await engine.verify_fact(args.fact, args.original_source, volatility=args.volatility)
            status = "VERIFIED" if res['verified'] else "UNVERIFIED"
            msg = f"Status: {status} | Cross-check: {res.get('cross_check_source', 'None')} (Verified at: {now})"
            return ToolResult(message=msg)
        except Exception as e:
            return ToolResult(message=f"Verification Error: {str(e)}")

    @staticmethod
    def get_current_time(args: GetCurrentTimeArgs) -> ToolResult:
        from datetime import datetime
        now = datetime.now()
        days_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        days_ru = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        day_en = days_en[now.weekday()]
        day_ru = days_ru[now.weekday()]
        msg = f"Current Time: {now.strftime('%d.%m.%Y %H:%M:%S')}, {day_en} ({day_ru})"
        return ToolResult(message=msg)

    @staticmethod
    async def execute_structured_call(json_str: str) -> ToolResult:
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
            
            elif call.tool == "search_and_extract":
                args = SearchAndExtractArgs(**call.arguments)
                return await ToolsRegistry.search_and_extract(args)
                
            elif call.tool == "verify_fact":
                args = VerifyFactArgs(**call.arguments)
                return await ToolsRegistry.verify_fact(args)
                
            elif call.tool == "get_current_time":
                args = GetCurrentTimeArgs(**call.arguments)
                return ToolsRegistry.get_current_time(args)
                
            else:
                return ToolResult(message=f"Error: Unknown tool '{call.tool}'")
                
        except json.JSONDecodeError:
            return ToolResult(message="Error: Invalid JSON format.")
        except ValidationError as e:
            return ToolResult(message=f"Schema Validation Error: {e}")
        except Exception as e:
            return ToolResult(message=f"Execution Error: {str(e)}")
    
    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """
        Export tool definitions in Ollama format for FunctionGemma.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate_linear_change",
                    "description": "Calculate linear change: result = start + (rate * time). Use for battery drain, resource consumption, etc. Use NEGATIVE rate for consumption/drain.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start": {
                                "type": "number",
                                "description": "Starting value (e.g., initial battery %)"
                            },
                            "rate": {
                                "type": "number", 
                                "description": "Rate of change per time unit. NEGATIVE for consumption/drain, POSITIVE for charging."
                            },
                            "time": {
                                "type": "number",
                                "description": "Duration in time units (e.g., minutes)"
                            },
                            "variable_name": {
                                "type": "string",
                                "description": "Name of state variable to update (default: 'result')"
                            }
                        },
                        "required": ["start", "rate", "time"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_resource_allocation",
                    "description": "Calculate remaining resources: remaining = total - requested. Returns status OK or OVERFLOW.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "total": {
                                "type": "number",
                                "description": "Total available resources"
                            },
                            "requested": {
                                "type": "number",
                                "description": "Amount being requested/consumed"
                            },
                            "variable_name": {
                                "type": "string",
                                "description": "Name of state variable to update (default: 'remaining')"
                            }
                        },
                        "required": ["total", "requested"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_and_extract",
                    "description": "Search the web for real-time information and extract facts (e.g., prices, weather, news).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "target": {
                                "type": "string",
                                "description": "Specific fact to extract (e.g. 'price', 'date', 'rate')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "verify_fact",
                    "description": "Verify a fact by cross-checking with other sources (excludes original source).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact": {
                                "type": "string",
                                "description": "The fact to verify"
                            },
                            "original_source": {
                                "type": "string",
                                "description": "URL of the original source (to exclude from verification)"
                            }
                        },
                        "required": ["fact", "original_source"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get current system date, time and day of the week. Use this when the exact current date/time is needed for calculations or context.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
