import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import CognitiveSystem
from models.llm_interface import MockLLM
from models.schemas import WorldState

class ReActMockLLM(MockLLM):
    def __init__(self):
        super().__init__()
        self.step_count = 0

    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        s_low = (system_prompt or "").lower()
        p_low = prompt.lower()
        
        # Expert Mock: Conditional Logic
        if "neutral" in s_low:
             # Most recent observation is key
             last_obs = p_low.split("[observation]:")[-1] if "[observation]:" in p_low else ""
             
             if not last_obs:
                 return """I will try to allocate 100 units.
```json
{
  "tool": "calculate_resource_allocation",
  "arguments": {"total": 50, "requested": 100}
}
```
""" 
             elif "overflow" in last_obs:
                 return """I see there was an overflow. I will now try to allocate 40 units instead.
```json
{
  "tool": "calculate_resource_allocation",
  "arguments": {"total": 50, "requested": 40}
}
```
"""
             elif "remaining: 10.0" in last_obs:
                 return "The allocation of 40 units was successful. We have 10 units remaining."

        return f"Step {self.step_count}: Response to prompt"
        
    async def generate_fast(self, prompt, system_prompt=None, temperature=0.7, max_tokens=512):
        return "INTENT: complex\nCONFIDENCE: 0.6"

async def test_react_loop():
    print("--- Starting Multi-Step ReAct Loop Verification ---")
    
    system = CognitiveSystem(use_ollama=False)
    mock_llm = ReActMockLLM()
    # Inject mock
    system.llm = mock_llm
    system.om.llm = mock_llm
    system.om.experts.llm = mock_llm
    system.om.validator.llm = mock_llm
    
    user_id = "react_user"
    
    print("\n[Step 1] Asking Expert to allocate resources (Expect 2 Tool Calls)...")
    response = await system.om.experts.consult_expert(
        "neutral", 
        "Allocate resources. We need 100 if possible, otherwise as much as you can up to 50.",
        world_state=WorldState(data={}),
        context=""
    )
    
    print(f"Final Expert Response:\n{response.response}")
    
    # Check for multiple observations
    observations = response.response.count("[OBSERVATION]")
    print(f"Number of observations found: {observations}")
    
    assert observations >= 2, "System must perform at least 2 ReAct steps"
    assert "Remaining: 10.0" in response.response, "Final successful allocation must be present"
    assert "overflow" in response.response.lower(), "Initial failure must be recorded in history"
    
    print("\n--- Verification Finished Successfully ---")

if __name__ == "__main__":
    asyncio.run(test_react_loop())
