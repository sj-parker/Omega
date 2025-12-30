
import asyncio
import sys
sys.path.insert(0, "e:\\agi2")

from models.llm_interface import FunctionGemmaLLM
from core.tools import ToolsRegistry
import json

async def test_full_search_flow():
    print("=== Testing Full Search Execution Flow ===")
    
    # Step 1: FunctionGemma selects tool
    fg = FunctionGemmaLLM()
    tools = ToolsRegistry.get_tool_definitions()
    
    print("\n--- Step 1: FunctionGemma Tool Selection ---")
    result = await fg.call_tool("search current bitcoin price USD", tools)
    print(f"FunctionGemma result: {result}")
    
    if not result or result.get("tool") != "search_and_extract":
        print("FAILURE: FunctionGemma did not select search_and_extract")
        return
    
    # Step 2: Execute the tool
    print("\n--- Step 2: Execute Tool via ToolsRegistry ---")
    tool_result = await ToolsRegistry.execute_structured_call(json.dumps(result))
    print(f"Tool execution result: {tool_result}")
    
    if "88" in tool_result.message or "87" in tool_result.message or "89" in tool_result.message:
        print("SUCCESS: Search returned current Bitcoin price (~$88k)!")
    else:
        print(f"WARNING: Search result may be outdated or incorrect: {tool_result.message[:100]}")

if __name__ == "__main__":
    asyncio.run(test_full_search_flow())
