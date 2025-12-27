
import asyncio
import sys
sys.path.insert(0, "e:\\agi2")

from models.llm_interface import FunctionGemmaLLM
from core.tools import ToolsRegistry

async def test_live_functiongemma():
    print("=== Testing Live FunctionGemma Tool Selection ===")
    
    fg = FunctionGemmaLLM()
    tools = ToolsRegistry.get_tool_definitions()
    
    # Test 1: Bitcoin price (should use search_and_extract)
    print("\n--- Test 1: 'search current bitcoin price USD' ---")
    result = await fg.call_tool("search current bitcoin price USD", tools)
    print(f"Result: {result}")
    
    if result and result.get("tool") == "search_and_extract":
        print("SUCCESS: FunctionGemma selected search_and_extract!")
    else:
        print(f"FAILURE: Expected search_and_extract, got {result}")
    
    # Test 2: Battery calculation (should use calculate_linear_change)
    print("\n--- Test 2: 'calculate linear change for battery from 83 with -1.4 rate for 12 mins' ---")
    result2 = await fg.call_tool("calculate linear change for battery from 83 with -1.4 rate for 12 mins", tools)
    print(f"Result: {result2}")
    
    if result2 and result2.get("tool") == "calculate_linear_change":
        print("SUCCESS: FunctionGemma selected calculate_linear_change!")
    else:
        print(f"FAILURE: Expected calculate_linear_change, got {result2}")

if __name__ == "__main__":
    asyncio.run(test_live_functiongemma())
