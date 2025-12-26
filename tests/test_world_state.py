import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import CognitiveSystem
from models.llm_interface import MockLLM

MAX_REACT_STEPS = 1

class StatePersistenceMockLLM(MockLLM):
    def __init__(self):
        super().__init__()

    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        s_low = (system_prompt or "").lower()
        p_low = prompt.lower()
        print(f"[DEBUG] Mock LLM: system='{s_low[:50]}...', prompt='{p_low[:50]}...'")
        
        # Expert Mock - match the new "calculator operator" prompt
        if "calculator operator" in s_low:
             # Extract the ACTUAL user query (last Query: line)
             query = p_low.split("query:")[-1].strip() if "query:" in p_low else p_low
             print(f"[DEBUG] Mock: Extracted query: '{query[:50]}'")
             
             # Check specific queries FIRST
             if "set budget to 1000" in query:
                 return "Budget of 1000 set.\n```json\n{\"tool\": \"calculate_resource_allocation\", \"arguments\": {\"total\": 1000, \"requested\": 0, \"variable_name\": \"budget\"}}\n```"
             
             if "buy item for 200" in query:
                 print(f"[DEBUG] Mock: Matched BUY query!")
                 return "Buying item for 200.\n```json\n{\"tool\": \"calculate_resource_allocation\", \"arguments\": {\"total\": 1000.0, \"requested\": 200, \"variable_name\": \"budget\"}}\n```"
             
             if "whats my current budget" in query:
                 if "800" in p_low:
                     return "Your current budget is 800."
                 else:
                     return f"State shows: {p_low[:80]}"
             
             # Handle observations (tool results)
             if "[observation]" in p_low:
                 return "Confirmed. Operation complete."
             
             print(f"[DEBUG] Mock: No match for query: {query[:50]}")
             return "No match."

        print(f"[DEBUG] Mock LLM: No expert matched system prompt '{s_low[:30]}...'")
        return f"Response to prompt: {prompt[:30]}"
        
    async def generate_fast(self, prompt, system_prompt=None, temperature=0.7, max_tokens=512):
        # Always return deep for these tests to trigger experts
        return "INTENT: complex\nCONFIDENCE: 0.1"

async def test_world_state_persistence():
    print("--- Starting World State Persistence Verification ---")
    
    system = CognitiveSystem(use_ollama=False)
    mock_llm = StatePersistenceMockLLM()
    system.llm = mock_llm
    system.om.llm = mock_llm
    system.om.experts.llm = mock_llm
    system.om.validator.llm = mock_llm
    
    user_id = "state_user"
    
    print("\n[Turn 1] Setting budget to 1000...")
    await system.process(user_id, "Set budget to 1000")
    print(f"Current State Store: {system.world_states[user_id].data}")
    assert system.world_states[user_id].data.get("budget") == 1000
    
    print("\n[Turn 2] Buying item for 200...")
    # Get the latest state to be sure
    print(f"Current State before Process: {system.world_states[user_id].data}")
    
    response = await system.process(user_id, "Buy item for 200")
    print(f"Final Response: {response}")
    print(f"Current State Store: {system.world_states[user_id].data}")
    
    # Check if budget is 800
    if system.world_states[user_id].data.get("budget") != 800:
        print(f"ERROR: Budget is {system.world_states[user_id].data.get('budget')} instead of 800")
    
    assert system.world_states[user_id].data.get("budget") == 800
    
    print("\n[Turn 3] Asking for current budget...")
    response = await system.process(user_id, "Whats my current budget?")
    print(f"Final Response: {response}")
    
    assert "800" in response, "System must know the current budget from the state"
    
    print("\n--- Verification Finished Successfully ---")

if __name__ == "__main__":
    asyncio.run(test_world_state_persistence())
