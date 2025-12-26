# Test Integrated Tool Calling
# Verifies ExpertsModule -> FunctionGemma delegation

import asyncio
import sys
import json
sys.path.insert(0, "e:\\agi2")

from models.llm_interface import LLMInterface, FunctionGemmaLLM
from core.experts import ExpertsModule
from models.schemas import PolicySpace, WorldState

# Custom Mock LLM that acts as "Reasoning Model"
class ReasoningMockLLM(LLMInterface):
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        # If history contains observation, finish
        if "[OBSERVATION]" in prompt:
            return "Based on the calculation, the battery is at 66.6%.\nFINAL: 66.6%"
            
        # First turn: Decide to use tool via NEED_TOOL pattern
        # FunctionGemma needs explicit negative rate info usually
        return "I need to calculate the battery drain.\nNEED_TOOL: calculate linear change for battery from 83.4 with -1.4 rate (negative consumption) for 12 minutes"

async def test_integration():
    print("--- Integrated Tool Calling Test ---")
    
    # 1. Main LLM (Reasoning) -> Outputs NEED_TOOL
    main_llm = ReasoningMockLLM()
    
    # 2. Tool Caller (FunctionGemma) -> Real Ollama call
    print("Initializing FunctionGemma...")
    try:
        tool_caller = FunctionGemmaLLM()
        # Verify connectivity
        res = await tool_caller.generate("hello")
    except Exception as e:
        print(f"Skipping test: FunctionGemma not available ({e})")
        return

    # 3. Experts Module with both
    print("Initializing ExpertsModule...")
    experts = ExpertsModule(main_llm, PolicySpace(), tool_caller=tool_caller)
    
    # 4. Run
    print("Running consult_expert...")
    world_state = WorldState(data={})
    response = await experts.consult_expert(
        expert_type="neutral",
        prompt="Calculate drone battery after 12 mins ascent",
        world_state=world_state
    )
    
    print(f"\nResponse:\n{response.response}")
    print(f"\nFinal State:\n{response.world_state}")
    
    # Validation
    # We expect 83.4 - (1.4 * 12) = 66.6
    state_vals = response.world_state.values()
    found = False
    for v in state_vals:
        if abs(v - 66.6) < 0.1:
            found = True
            break
            
    if found:
        print("\n[SUCCESS] State updated correctly to 66.6 via FunctionGemma!")
    else:
        print(f"\n[FAIL] State verify failed. State: {response.world_state}")

if __name__ == "__main__":
    asyncio.run(test_integration())
