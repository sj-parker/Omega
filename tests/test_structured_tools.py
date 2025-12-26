import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import CognitiveSystem
from models.llm_interface import MockLLM

class StructuredToolMockLLM(MockLLM):
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        s_low = (system_prompt or "").lower()
        
        # Expert Mock: Using JSON Tool Call
        if "neutral" in s_low:
             return """I will calculate the linear change.
```json
{
  "tool": "calculate_linear_change",
  "arguments": {"start": 50, "rate": -1, "time": 30}
}
```
""" 
        return f"Response to: {prompt[:30]}"
        
    async def generate_fast(self, prompt, system_prompt=None, temperature=0.7, max_tokens=512):
        return "INTENT: complex\nCONFIDENCE: 0.6"

async def test_structured_tools():
    print("--- Starting Structured Tool Use Verification ---")
    
    system = CognitiveSystem(use_ollama=False)
    mock_llm = StructuredToolMockLLM()
    # Inject mock
    system.llm = mock_llm
    system.om.llm = mock_llm
    system.om.experts.llm = mock_llm
    system.om.validator.llm = mock_llm
    
    user_id = "struct_user"
    
    print("\n[Step 1] Asking Neutral Expert to calculate (Expect JSON Tool Call)...")
    # We call the expert directly
    response = await system.om.experts.consult_expert(
        "neutral", 
        "Calculate linear change.", 
        context=""
    )
    
    print(f"Expert Response:\n{response.response}")
    
    # Check for System Output
    assert "[SYSTEM TOOL OUTPUT]" in response.response, "System must append tool output"
    assert "Result: 20.0" in response.response, "Tool execution must return correct math (50 - 30 = 20)"
    
    print("\n--- Verification Finished Successfully ---")

if __name__ == "__main__":
    asyncio.run(test_structured_tools())
