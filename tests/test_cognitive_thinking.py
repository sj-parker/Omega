import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import CognitiveSystem
from models.llm_interface import MockLLM
from models.schemas import DecisionDepth

class CognitiveMockLLM(MockLLM):
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        p_low = prompt.lower()
        s_low = (system_prompt or "").lower()
        
        # Simulate thinking delay
        # If we parallelize experts, total time should be ~1 delay instead of 3
        if "expert_type" in s_low or "perspective" in p_low:
            await asyncio.sleep(0.5) 
            return f"Expert advice on {prompt[:20]}"
            
        if "reasoning engine" in s_low:
            return "THOUGHT: We must analyze the complexity of the user's request and coordinate experts."
            
        if "synthesis analyst" in s_low:
            return "SYNTHESIZED RESPONSE: This is the best combined answer from all experts. 4. Disagreement: 0.1"

        return f"Generic response to: {prompt[:30]}"

    async def generate_fast(self, prompt, system_prompt=None, temperature=0.7, max_tokens=512):
        # Intent classification mock
        return "INTENT: analytical\nCONFIDENCE: 0.5\nREASONING: Needs deep thought."

async def test_thinking_process():
    print("--- Starting Cognitive Thinking Verification ---")
    
    system = CognitiveSystem(use_ollama=False)
    mock_llm = CognitiveMockLLM()
    system.llm = mock_llm
    system.om.llm = mock_llm
    system.om.experts.llm = mock_llm # Crucial
    system.om.critic.llm = mock_llm   # Crucial
    system.quality_llm = mock_llm
    
    # Force analytical intent to trigger DEEP path
    user_id = "thinker_1"
    
    print("\n[Step 1] Sending analytical query...")
    start_time = time.time()
    response = await system.process(user_id, "Explain the relationship between entropy and information theory.")
    elapsed = time.time() - start_time
    
    print(f"\nSystem Response: {response[:100]}...")
    print(f"Total time elapsed: {elapsed:.2f}s")
    
    # Check trace
    last_trace = system.learning_decoder.raw_traces[-1]
    
    print(f"\n[Verification]")
    print(f"- Thoughts generated: {'THOUGHT' in last_trace.thoughts}")
    print(f"- Experts used: {len(last_trace.expert_outputs)}")
    print(f"- Critic used: {bool(last_trace.critic_output)}")
    
    assert "combined answer" in response, "Response should be synthesized by critic"
    assert len(last_trace.expert_outputs) == 3, "All 3 experts should be consulted"
    assert elapsed < 2.0, f"System too slow ({elapsed:.2f}s). Parallelization failed?"
    
    print("\n--- Verification Finished Successfully ---")

if __name__ == "__main__":
    asyncio.run(test_thinking_process())
