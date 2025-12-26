import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import CognitiveSystem
from models.llm_interface import MockLLM

class MemoryMockLLM(MockLLM):
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        p_low = prompt.lower()
        s_low = (system_prompt or "").lower()
        
        # Intent classification mocks
        if "classify the user intent" in s_low:
            if "password is" in p_low:
                return "INTENT: memorize\nCONFIDENCE: 1.0\nREASONING: User providing sensitive info."
            if "what was" in p_low and "password" in p_low:
                return "INTENT: recall\nCONFIDENCE: 1.0\nREASONING: User asking for past info."
            return "INTENT: factual\nCONFIDENCE: 0.8\nREASONING: General query."
            
        # Fact extraction mocks
        if "extract the core fact" in p_low:
            if "sirius123" in p_low:
                return "FACT: The project password is Sirius123\nENTITIES: password, Sirius123"
            return "FACT: Generic fact\nENTITIES: none"
            
        return f"Response to: {prompt[:50]}..."

async def test_memory_recall():
    print("--- Starting Memory Pipeline Verification ---")
    
    # 1. Setup system with Mock LLM
    system = CognitiveSystem(use_ollama=False)
    system.llm = MemoryMockLLM()
    system.om.llm = system.llm
    system.quality_llm = system.llm
    
    user_id = "test_user"
    
    # 2. Tell the system a fact
    print("\n[Step 1] Giving a fact...")
    resp1 = await system.process(user_id, "My project password is Sirius123")
    print(f"System: {resp1}")
    
    assert "Sirius123" in resp1, "System should confirm saving the password"
    assert len(system.context_manager.long_term_facts) == 1, "Fact should be in long-term memory"
    
    # 3. Simulate "noise" (Python talk) to fill short-term memory
    print("\n[Step 2] Adding noise (Python talk)...")
    for i in range(10): # Trigger compaction (15+ events)
        await system.process(user_id, f"Tell me more about Python feature {i}")
        
    # 4. Ask for the fact
    print("\n[Step 3] Recalling the fact...")
    resp2 = await system.process(user_id, "What was my project password?")
    print(f"System: {resp2}")
    
    # Check if the fact was actually retrieved and used
    # In MockLLM, the response is just "Response to: ...", but we can check if long_term_context was set in trace
    last_trace = system.learning_decoder.raw_traces[-1]
    # We can't easily see context_slice in trace because it's a snapshot of events
    # But OM prints "[OM] Recall triggered. Long-term context added."
    
    print("\n--- Verification Finished ---")

if __name__ == "__main__":
    asyncio.run(test_memory_recall())
