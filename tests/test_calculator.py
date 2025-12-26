import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import CognitiveSystem
from models.llm_interface import MockLLM

class CalculatorMockLLM(MockLLM):
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        s_low = (system_prompt or "").lower()
        p_low = prompt.lower()
        
        # Forecaster Mock: Using Tool Call
        if "forecaster" in s_low and "plan battery" in p_low:
             return """TIMELINE LOG:
[12:00] Start: 50%
[Action] Calculate drop.
TOOL_CALL: calculate_linear_change(start=50, rate=-1, time=30)"""
        
        return f"Response to: {prompt[:30]}"
        
    async def generate_fast(self, prompt, system_prompt=None, temperature=0.7, max_tokens=512):
        return "INTENT: complex\nCONFIDENCE: 0.6"

async def test_calculator():
    print("--- Starting Calculator Tool Verification ---")
    
    system = CognitiveSystem(use_ollama=False)
    mock_llm = CalculatorMockLLM()
    # Inject mock into all components
    system.llm = mock_llm
    system.om.llm = mock_llm
    system.om.experts.llm = mock_llm
    system.om.validator.llm = mock_llm
    
    user_id = "calc_user"
    
    print("\n[Step 1] Asking Forecaster to plan battery (Expect Tool Call)...")
    # We call the expert directly to verify the hook in `consult_expert`
    response = await system.om.experts.consult_expert(
        "forecaster", 
        "Plan battery drop. Start 50%, rate -1%/min, time 30m.", 
        context=""
    )
    
    print(f"Expert Response:\n{response.response}")
    
    # Check for System Output
    assert "[SYSTEM TOOL OUTPUT]" in response.response, "System must append tool output"
    assert "Result: 20.0" in response.response, "Tool execution must return correct math (50 - 30 = 20)"
    
    print("\n--- Verification Finished Successfully ---")

if __name__ == "__main__":
    asyncio.run(test_calculator())
