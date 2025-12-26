# Test FunctionGemma Tool Calling
# Verifies that FunctionGemma can parse tasks and generate correct tool calls

import asyncio
import sys
sys.path.insert(0, "e:\\agi2")

from models.llm_interface import FunctionGemmaLLM
from core.tools import ToolsRegistry

async def test_functiongemma_basic():
    """Test basic tool calling with FunctionGemma."""
    print("--- FunctionGemma Basic Test ---")
    
    llm = FunctionGemmaLLM()
    tools = ToolsRegistry.get_tool_definitions()
    
    # Test 1: Linear change calculation - be VERY explicit
    print("\n[Test 1] Linear change: 83.4 - (1.4 * 12)")
    result = await llm.call_tool(
        task_description="Use calculate_linear_change with start=83.4, rate=-1.4 (negative because consumption), time=12",
        tools=tools
    )
    print(f"Result: {result}")
    
    if result and result.get("tool") == "calculate_linear_change":
        print("[OK] Correct tool selected")
        args = result.get("arguments", {})
        print(f"  Args: start={args.get('start')}, rate={args.get('rate')}, time={args.get('time')}")
    else:
        print(f"[FAIL] Expected calculate_linear_change, got: {result}")
    
    # Test 2: Resource allocation
    print("\n[Test 2] Resource allocation: 1000 total, consume 200")
    result = await llm.call_tool(
        task_description="Use calculate_resource_allocation with total=1000, requested=200",
        tools=tools
    )
    print(f"Result: {result}")
    
    if result and result.get("tool") == "calculate_resource_allocation":
        print("[OK] Correct tool selected")
    else:
        print(f"[FAIL] Expected calculate_resource_allocation, got: {result}")
    
    print("\n--- Test Complete ---")

async def test_functiongemma_chain():
    """Test multi-step calculation chain."""
    print("\n--- FunctionGemma Chain Test ---")
    
    llm = FunctionGemmaLLM()
    tools = ToolsRegistry.get_tool_definitions()
    
    # Simulate drone battery calculation
    steps = [
        ("Stage 1: Battery starts at 83.4%, drains at 1.4%/min for 12 minutes", 83.4, -1.4, 12),
        ("Stage 2: Battery at 66.6%, drains at 0.3%/min for 7 minutes", 66.6, -0.3, 7),
        ("Stage 3: Battery at 64.5%, drains at 3.8%/min for 4 minutes", 64.5, -3.8, 4),
    ]
    
    for desc, expected_start, expected_rate, expected_time in steps:
        print(f"\n{desc}")
        result = await llm.call_tool(task_description=desc, tools=tools)
        
        if result and result.get("tool") == "calculate_linear_change":
            args = result.get("arguments", {})
            start_ok = abs(args.get("start", 0) - expected_start) < 0.1
            time_ok = args.get("time") == expected_time
            print(f"  Tool: ✓, start: {'✓' if start_ok else '✗'}, time: {'✓' if time_ok else '✗'}")
            print(f"  Args: {args}")
        else:
            print(f"  ✗ Result: {result}")
    
    print("\n--- Chain Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_functiongemma_basic())
    asyncio.run(test_functiongemma_chain())
