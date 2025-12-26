# Test Real-Time Data Search
# Verifies that Main LLM delegates "current price" queries to search tool

import asyncio
import sys
sys.path.insert(0, "e:\\agi2")

from models.llm_interface import LLMInterface, FunctionGemmaLLM
from core.experts import ExpertsModule
from models.schemas import PolicySpace, WorldState

# Custom Mock LLM for Reasoning
class ReasoningMockLLM(LLMInterface):
    def __init__(self):
        self.attempts = 0
        
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        # Scenario: User asks for BTC price
        # Just check for keyword in user query part, not whole prompt which might have tools instructions
        if "bitcoin price" in prompt.lower():
            
            # Second attempt (after Guardrail error): CORRECT BEHAVIOR
            if "[SYSTEM ERROR]" in prompt:
                return "I apologize. I need to use the search tool.\nNEED_TOOL: search current bitcoin price USD"
            
            # First attempt: BAD BEHAVIOR (Python Code)
            if self.attempts == 0 and "NEED_TOOL" not in prompt: # Careful with this check
                self.attempts += 1
                return """```python
import requests
def get_price():
    return requests.get("api.coindesk.com").json()['rate']
```"""
        
        elif "[OBSERVATION]" in prompt:
            # Scenario: Got Search Result -> Calculate
            if "Fact:" in prompt:
                return "The price is found. Now I calculate robots.\nNEED_TOOL: calculate resource allocation for 15000 USD with 0.05 BTC cost" 
                
            return "Calculation done.\nFINAL: 3 full robots"
            
        return "I don't know the price. I shouldn't guess."

async def test_search_delegation():
    print("--- Test Search Delegation ---")
    
    main_llm = ReasoningMockLLM()
    
    # Initialize FunctionGemma (assume connected)
    try:
        tool_caller = FunctionGemmaLLM()
    except:
        print("Skipping: FunctionGemma not ready")
        return

    experts = ExpertsModule(main_llm, PolicySpace(), tool_caller=tool_caller)
    world_state = WorldState(data={})
    
    prompt = "What is the current Bitcoin price and how many Robot-A (0.05 BTC) can I buy for 15000 USD?"
    
    print(f"Query: {prompt}")
    response = await experts.consult_expert("neutral", prompt, world_state)
    
    print(f"\nResponse:\n{response.response}")
    
    # Check if search tool was triggered (in history or logs)
    # Since we mock the Reasoning LLM, we primarily verify that the system handles the NEED_TOOL delegation flow correctly 
    # and doesn't crash on 'search' tool execution.
    
    if "[OBSERVATION]" in response.response:
        print("[SUCCESS] Tool execution flow occurred.")
    else:
        print("[FAIL] No observation found (maybe decided to answer directly?)")

if __name__ == "__main__":
    asyncio.run(test_search_delegation())
