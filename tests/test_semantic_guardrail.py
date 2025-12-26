import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import CognitiveSystem
from models.llm_interface import MockLLM

class GuardrailMockLLM(MockLLM):
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        p_low = prompt.lower()
        s_low = (system_prompt or "").lower()
        
        # Auditor (Validator) mock
        # Auditor (Validator) mock
        if "semantic auditor" in s_low:
            if "hard drive" in p_low:
                # Detect drift: Draft talked about HDD
                return "DRIFT_FOUND: YES\nCONFLICTS: Term 'Persistent Memory' refers to neural weights in context, but agent used general hardware definition.\nREASONING: Concept drift detected."
            return "DRIFT_FOUND: NO\nREASONING: No drift."
            
        # Rephraser mock
        if "technical re-writer" in s_low:
            return "Rephrased Answer: Persistent Memory in Titans refers to the internal neural weights that store long-term experience."

        # Mediator (Medium path) mock - trigger drift
        if "helpful assistant" in s_low and "persistent memory" in p_low:
            return "Persistent Memory is typically a hard drive or SSD for storage."

        # Expert mock (intentionally wrong to trigger drift)
        if "expert advice" in p_low or "neutral expert" in s_low:
            return "Persistent Memory is like a hard drive or SSD where data stays forever."

        return f"Response to: {prompt[:30]}"

    async def generate_fast(self, prompt, system_prompt=None, temperature=0.7, max_tokens=512):
        # Intent classification mock
        if "persistent memory" in prompt.lower():
            return "INTENT: recall\nCONFIDENCE: 0.5\nREASONING: User asking about a term."
        return "INTENT: factual\nCONFIDENCE: 0.9"

async def test_guardrail_drift():
    print("--- Starting Semantic Guardrail Verification ---")
    
    system = CognitiveSystem(use_ollama=False)
    mock_llm = GuardrailMockLLM()
    system.llm = mock_llm
    system.om.llm = mock_llm
    system.om.validator.llm = mock_llm
    
    # Pre-add a fact to long-term memory (Source of Truth)
    system.context_manager.add_fact(
        "In the Titans architecture, Persistent Memory refers to neural weights that store long-term experience, NOT hardware storage.",
        entities=["Titans", "Persistent Memory", "neural weights"]
    )
    
    user_id = "test_user"
    
    print("\n[Step 1] Asking about Persistent Memory...")
    # This should trigger RECALL intent -> Long-term context injection -> Validator check
    response = await system.process(user_id, "What is Persistent Memory in Titans?")
    
    print(f"\nFinal Response: {response}")
    
    # Check trace for validation report
    last_trace = system.learning_decoder.raw_traces[-1]
    report = last_trace.validation_report
    
    print(f"\n[Verification]")
    print(f"- Drift detected by Validator: {report.get('drift_found')}")
    print(f"- Status: {report.get('status')}")
    
    assert report.get('drift_found') == True, "Validator should have detected concept drift"
    assert "neural weights" in response.lower(), "Final response should be corrected to use neural weights"
    
    print("\n--- Verification Finished Successfully ---")

if __name__ == "__main__":
    asyncio.run(test_guardrail_drift())
