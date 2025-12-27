
import asyncio
import sys
sys.path.insert(0, "e:\\agi2")

from core.operational_module import OperationalModule
from models.llm_interface import LLMInterface
from models.schemas import ContextSlice, UserIdentity, WorldState

# Dumb Mock LLM that ALWAYS triggers search for realtime queries
class RealtimeTestLLM(LLMInterface):
    def __init__(self):
        self.call_count = 0
        
    async def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        self.call_count += 1
        
        # Intent classifier should recognize realtime_data
        if "INTENT:" in (system_prompt or ""):
            print(f"[Mock] Intent Classification Call #{self.call_count}")
            return """INTENT: realtime_data
CONFIDENCE: 0.95
REASONING: User is asking for current Bitcoin price, which changes in real-time."""
        
        # Expert should trigger search
        if "OMEGA-DISPATCHER" in (system_prompt or ""):
            print(f"[Mock] Expert Call #{self.call_count}")
            return "NEED_TOOL: search current bitcoin price USD"
        
        # Default
        print(f"[Mock] Generic Call #{self.call_count}: {prompt[:50]}...")
        return "[Mock Response]"

async def test_realtime_intent():
    print("=== Testing realtime_data Intent Flow ===")
    
    llm = RealtimeTestLLM()
    om = OperationalModule(llm=llm, tool_caller=None)  # No real tool caller
    
    context = ContextSlice(
        user_input="Какая сейчас цена Bitcoin?",
        user_identity=UserIdentity(user_id="test", trust_level=0.5),
        world_state=WorldState(data={})
    )
    
    try:
        response, decision, trace = await om.process(context)
        print(f"\\n=== RESULTS ===")
        print(f"Intent Detected: {decision.reasoning}")
        print(f"Depth Used: {decision.depth_used}")
        print(f"Response: {response}")
        
        # Verify
        if decision.depth_used.value == "deep":
            print("\\nSUCCESS: realtime_data intent correctly forced DEEP path!")
        else:
            print(f"\\nFAILURE: Expected DEEP, got {decision.depth_used.value}")
            
        if "NEED_TOOL: search" in response:
            print("SUCCESS: Expert correctly triggered search tool!")
        else:
            print("WARNING: Search tool was not triggered in response.")
            
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_realtime_intent())
