import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import CognitiveSystem
from models.llm_interface import MockLLM

class PhysicsMockLLM(MockLLM):
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        s_low = (system_prompt or "").lower()
        p_low = prompt.lower()
        
        # Validator Mock
        if "semantic auditor" in s_low:
             if "simultaneous" in p_low and "single port" in p_low:
                 return "CONSTRAINT_FAIL: YES\nCONFLICTS: Cannot do simultaneous on single port.\nREASONING: Physical Constraint."
             return "CONSTRAINT_FAIL: NO"
            
        # Forecaster Mock (Simulating the error first)
        if "forecaster" in s_low:
            # Good response with timeline
            if "timeline log" in s_low or "step-by-step" in s_low:
                 return """TIMELINE LOG:
[12:00] A:30% | Start | 0
[12:20] A:50% | Charge | +20% (1%/m * 20m)
Constraint Check: Target 70% NOT MET."""
            # Bad response (Magic Math)
            return "At 12:20, Robot A will be at 100% because charging is fast."

        return f"Response to: {prompt[:30]}"
        
    async def generate_fast(self, prompt, system_prompt=None, temperature=0.7, max_tokens=512):
        return "INTENT: complex\nCONFIDENCE: 0.6"

async def test_physics_engine():
    print("--- Starting Physics Engine Verification ---")
    
    system = CognitiveSystem(use_ollama=False)
    mock_llm = PhysicsMockLLM()
    system.llm = mock_llm
    system.om.llm = mock_llm
    system.om.validator.llm = mock_llm
    system.om.experts.llm = mock_llm
    
    user_id = "physics_user"
    
    print("\n[Step 1] Asking for impossible charge plan (Single port, simultaneous)...")
    # Validator Test
    context_str = "Context: We have a Single Port.\nUser: Charge both simultaneously."
    response, report = await system.om.validator.validate("Sure, I will charge both simultaneously.", context_str)
    
    print(f"Validator Report: {report}")
    assert report.get('constraint_fail') == True, "Validator must catch Single Port violation"
    
    print("\n[Step 2] Testing Forecaster Timeline Prompt...")
    # Verify Forecaster produces Timeline when prompted correctly
    # detailed prompt provided by OM
    response = await system.om.experts.consult_expert(
        "forecaster", 
        "Plan charging A (30%) to 70% in 20 mins. Rate 1%/min.", 
        context=""
    )
    
    print(f"Forecaster Response:\n{response.response}")
    assert "TIMELINE LOG" in response.response, "Forecaster must produce a Timeline Log"
    assert "1%/m" in response.response, "Forecaster must show the Rate Formula"
    
    print("\n--- Verification Finished Successfully ---")

if __name__ == "__main__":
    asyncio.run(test_physics_engine())
