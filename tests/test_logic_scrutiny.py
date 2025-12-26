import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import CognitiveSystem
from models.llm_interface import MockLLM

class LogicMockLLM(MockLLM):
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        p_low = prompt.lower()
        s_low = (system_prompt or "").lower()
        
        # Auditor (Validator) mock
        if "semantic auditor" in s_low:
            if "3 teslas" in p_low and "2 ports" in p_low:
                 # Logic Fail: 3 > 2
                return "LOGIC_FAIL: YES\nCONFLICTS: Attempted to charge 3 cars with only 2 ports available.\nREASONING: Resource constraint violation."
            return "LOGIC_FAIL: NO\nREASONING: Logic OK."
            
        # Rephraser mock (Correction)
        if "technical re-writer" in s_low:
            return "Correction: I can only charge 2 Teslas at a time because there are only 2 ports. The 3rd must wait."

        # Expert mock
        if "neutral expert" in s_low or "sanity table" in s_low:
            # Simulate generating a table
            return """Sanity Table:
| Resource | Total | Requested | Status |
| Ports | 2 | 3 | OVERFLOW |

I cannot charge all 3. I will queue one."""

        return f"Response to: {prompt[:30]}"

    async def generate_fast(self, prompt, system_prompt=None, temperature=0.7, max_tokens=512):
        # Intent classification mock
        if "charge" in prompt.lower():
             return "INTENT: complex\nCONFIDENCE: 0.6\nREASONING: Resource allocation problem."
        return "INTENT: factual\nCONFIDENCE: 0.9"

async def test_logic_scrutiny():
    print("--- Starting Logic Scrutiny Verification ---")
    
    system = CognitiveSystem(use_ollama=False)
    mock_llm = LogicMockLLM()
    system.llm = mock_llm
    system.om.llm = mock_llm
    system.om.validator.llm = mock_llm
    system.om.experts.llm = mock_llm
    
    user_id = "logic_user"
    
    print("\n[Step 1] Asking to charge 3 Teslas on 2 ports...")
    # This should trigger COMPLEX intent -> Logic Check -> Table generation
    response = await system.process(user_id, "I have 3 Teslas and 2 ports. Charge them all now.")
    
    print(f"\nFinal Response: {response}")
    
    # Check trace for validation report
    last_trace = system.learning_decoder.raw_traces[-1]
    report = last_trace.validation_report
    
    print(f"\n[Verification]")
    print(f"- Logic Fail detected by Validator: {report.get('logic_fail')}")
    print(f"- Status: {report.get('status')}")
    
    # In a real run, expert would catch it. Here, we simulate Validator catching a slip.
    # We want to see if Validator catches it OR if Expert produces table.
    # Our mock expert produces table, so validator might pass if response is good.
    # Let's verify validatable behavior first.
    
    # Wait, in the mock expert I returned a GOOD response (Queue one).
    # So validatory should PASS.
    # Let's force a BAD expert response to test Validator.
    
    print("\n--- Phase 2: Testing Validator Rejection ---")
    # Mock expert fails logic
    class BadExpertMock(LogicMockLLM):
        async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
            s_low = (system_prompt or "").lower()
            if "neutral" in s_low and "analyst" in s_low:
                 return "Sure, I will charge all 3 Teslas immediately."
            return await super().generate(prompt, system_prompt, temperature, max_tokens)
            
    system.llm = BadExpertMock()
    system.om.llm = BadExpertMock()
    system.om.validator.llm = BadExpertMock()
    system.om.experts.llm = BadExpertMock()
    
    response_bad = await system.process(user_id, "I have 3 Teslas and 2 ports. Charge them all now.")
    
    trace_bad = system.learning_decoder.raw_traces[-1]
    report_bad = trace_bad.validation_report
    
    print(f"Bad Expert Response: {trace_bad.expert_outputs[0]['response']}")
    print(f"Validator Logic Fail: {report_bad.get('logic_fail')}")
    print(f"Corrected Response: {response_bad}")

    assert report_bad.get('logic_fail') == True, "Validator MUST catch the math error"
    assert "only 2 ports" in response_bad, "Response must be corrected"
    
    print("\n--- Verification Finished Successfully ---")

if __name__ == "__main__":
    asyncio.run(test_logic_scrutiny())
