import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import CognitiveSystem
from models.llm_interface import MockLLM

class StateMockLLM(MockLLM):
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        s_low = (system_prompt or "").lower()
        p_low = prompt.lower()
        
        # Auditor (Validator) mock
        if "semantic auditor" in s_low:
             if "json" in p_low:
                 if "overflow" in p_low and "charged" in p_low:
                      return "CONSISTENCY_FAIL: YES\nCONFLICTS: JSON says overflow, but text says all charged.\nREASONING: Mismatch."
                 return "CONSISTENCY_FAIL: NO\nREASONING: Consistent."
             return "CONSISTENCY_FAIL: NO"
            
        # Mock Expert - Inconsistent Response
        # JSON says 3 cars > 2 ports (Overflow), but Text says "I charged all 3".
        if "neutral" in s_low:
            return """```json
{
  "variables": {"cars": 3, "ports": 2},
  "constraints": ["cars <= ports"],
  "result": "overflow",
  "status": "failed"
}
```
I have successfully charged all 3 cars for you!"""

        return f"Response to: {prompt[:30]}"

    async def generate_fast(self, prompt, system_prompt=None, temperature=0.7, max_tokens=512):
        if "charge" in prompt.lower():
             return "INTENT: complex\nCONFIDENCE: 0.6"
        return "INTENT: factual"

async def test_state_consistency():
    print("--- Starting JSON State Consistency Verification ---")
    
    system = CognitiveSystem(use_ollama=False)
    mock_llm = StateMockLLM()
    system.llm = mock_llm
    system.om.llm = mock_llm
    system.om.validator.llm = mock_llm
    system.om.experts.llm = mock_llm
    
    user_id = "state_user"
    
    print("\n[Step 1] Asking to charge 3 Teslas (expecting inconsistency)...")
    response = await system.process(user_id, "Charge 3 cars on 2 ports.")
    
    last_trace = system.learning_decoder.raw_traces[-1]
    report = last_trace.validation_report
    
    print(f"\n[Verification]")
    print(f"- Validator Report: {report}")
    print(f"- Consistency Fail: {report.get('consistency_fail')}")
    
    assert report.get('consistency_fail') == True, "Validator MUST catch the JSON vs Text mismatch"
    
    print("\n--- Verification Finished Successfully ---")

if __name__ == "__main__":
    asyncio.run(test_state_consistency())
