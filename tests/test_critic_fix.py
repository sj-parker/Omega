import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.experts import CriticModule, ExpertResponse
from models.llm_interface import LLMInterface
from models.schemas import CriticAnalysis

# Mock LLM for Critic to test prompt logic
class MockLLM(LLMInterface):
    async def generate(self, prompt, system_prompt, temperature=0.7, **kwargs):
        # We simulate a partial LLM response that MIGHT interpret the prompt instructions
        if "CRITIC" in system_prompt or "Judge" in system_prompt:
             return """[VERIFICATION PHASE]
Analysis: The experts have provided a story. No contradictions.

[FINAL SYNTHESIS]
Once upon a time, a man found a portal in his dreams. He learned about himself. The end."""
        return "Generic response"
        
    async def generate_fast(self, prompt, system_prompt, temperature=0.7, **kwargs):
        return "Fast response"

async def test_critic_synthesis():
    print("=== Testing Critic Synthesis Prompt ===")
    
    # 1. Setup
    mock_llm = MockLLM()
    critic = CriticModule(mock_llm)
    
    # 2. Mock Expert Responses (including the "leakage" prone ones)
    experts = [
        ExpertResponse(
            expert_type="creative",
            response="In the creative expert's view, the man flew to Mars.",
            confidence=0.9
        ),
        ExpertResponse(
            expert_type="conservative", 
            response="The conservative perspective warns about oxygen levels.",
            confidence=0.8
        )
    ]
    
    # 3. Simulate analysis call (we rely on seeing the PROMPT content implicitly via the Mock if we printed it, 
    # but here we just check if the logic runs without error and the 'system_prompt' in the real module has been updated.
    # Since we modified the file on disk, we trust the file content check we did.
    # We will mainly use this to verify the pipeline doesn't crash).
    
    # Ideally, we'd check the exact string sent to the LLM, but for now we verify the prompt construction doesn't fail.
    
    analysis = await critic.analyze(experts, "Tell me a story about space.", intent="creative")
    
    if analysis.recommended_response:
        print("[OK] Critic produced a recommendation.")
        print(f"Result: {analysis.recommended_response}")
    else:
        print("[FAIL] Critic failed to produce recommendation.")

if __name__ == "__main__":
    asyncio.run(test_critic_synthesis())
