
import asyncio
import sys
from unittest.mock import MagicMock

# Ensure we can import core modules
sys.path.insert(0, "e:\\agi2")

from core.experts import ExpertsModule, ExpertResponse
from models.llm_interface import MockLLM, LLMInterface
from models.schemas import WorldState

# Mock PolicySpace
class MockPolicy:
    def __init__(self):
        self.creative_range = (0.2, 0.8)

# "Dumb" Mock LLM that strictly follows queued responses
class DumbMockLLM(LLMInterface):
    def __init__(self):
        self.responses = []
        
    async def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        # print(f"DEBUG: Generating... Queue len: {len(self.responses)}")
        if self.responses:
            return self.responses.pop(0)
        return ""

async def test_guardrail_logic():
    print("--- Test 1: Guardrail intercepts and recovers ---")
    
    # 1. Setup Mock LLM
    bad_response = """tool_code from datetime import datetime from typing import Dict def calculate_linear_change(btc_price: float, robots_count: int, remaining_usd: float) -> Dict[str, float]: ... return { "btc_price": 0, "robots_count": 0, "remaining_usd": 0 } ..."""
    good_response = "NEED_TOOL: calculate linear change for bitcoin robots"
    
    # Critic must output "[FINAL SYNTHESIS] something" to be accepted by _extract_synthesized_response
    # Otherwise it falls back to expert_responses[0].
    critic_response = "[VERIFICATION PHASE]\nQ: Is it 3?\nA: Yes.\n\n[FINAL SYNTHESIS]\nThe calculation result is 3 robots."
    
    mock_llm = DumbMockLLM()
    # Queue: 
    # 1. Expert call 1 -> Bad (intercepted)
    # 2. Expert call 2 -> Good (tool runs)
    # 3. Expert call 3 -> "Based on result..." (The "Main LLM" wrapping up the tool result)
    # 4. Critic call -> critic_response
    
    tool_synthesis_response = "Based on the observation, the result is 3 robots."
    
    mock_llm.responses = [
        bad_response, 
        good_response, 
        tool_synthesis_response, 
        critic_response
    ]
    
    # 2. Setup ExpertsModule
    mock_tool_caller = MagicMock()
    # call_tool must be awaitable
    future = asyncio.Future()
    future.set_result({"output": "Calculation Result: 3 robots"})
    mock_tool_caller.call_tool.return_value = future
    
    mock_policy = MockPolicy()
    
    expert = ExpertsModule(llm=mock_llm, policy=mock_policy, tool_caller=mock_tool_caller)
    world_state = WorldState(data={})
    
    try:
        result = await expert.consult_expert("Calculate robots", "neutral", world_state=world_state)
        print(f"Final Result 1: {result.response}")
        
        if "tool_code" in result.response:
            print("FAILURE: 'tool_code' leaked into final result!")
        elif "3 robots" in result.response:
             print("SUCCESS: Guardrail intercepted code, and system recovered.")
        else:
             print(f"UNKNOWN RESULT: {result.response}")
            
    except Exception as e:
        print(f"Exception during test 1: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Test 2: Stubborn Hallucination (Max Retries) ---")
    
    # LLM keeps outputting code until MAX_REACT_STEPS (5) is reached
    stubborn_llm = DumbMockLLM()
    # 5 bad responses for the loop
    # 1 response for the Critic call (which will analyze the mess)
    # If the Critic ALSO hallucinates code or quotes it...
    stubborn_llm.responses = [bad_response] * 6 
    
    expert_stubborn = ExpertsModule(llm=stubborn_llm, policy=mock_policy, tool_caller=mock_tool_caller)
    
    try:
        result = await expert_stubborn.consult_expert("Calculate robots", "neutral", world_state=world_state)
        print(f"Final Result (Stubborn): {result.response}")
        
        # If the failure mode is that the system returns the code...
        if "tool_code" in result.response:
            print("FAILURE CONFIRMED: 'tool_code' leaked into final result after max retries!")
        elif "SYSTEM ERROR" in result.response:
             print("SUCCESS: Leaked SYSTEM ERROR messages (Better than code, but ugly).")
        else:
             print(f"Unexpected: {result.response[:50]}...")

    except Exception as e:
        print(f"Exception during test 2: {e}")


if __name__ == "__main__":
    asyncio.run(test_guardrail_logic())
