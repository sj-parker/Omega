
import asyncio
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, "e:\\agi2")

from core.operational_module import OperationalModule, DecisionDepth
from models.schemas import ContextSlice, UserIdentity, WorldState, PolicySpace

class TestDateAndLogic(unittest.IsolatedAsyncioTestCase):
    async def test_date_injection_fast(self):
        print("\n--- Testing Date Injection in FAST path ---")
        mock_llm = AsyncMock()
        mock_llm.generate_fast.return_value = "Response with date"
        # Mock generate as fallback if generate_fast not called (though code calls generate_fast)
        mock_llm.generate.return_value = "Response with date"
        
        om = OperationalModule(llm=mock_llm, tool_caller=None)
        
        context = ContextSlice(
            user_input="What date is it?",
            user_identity=UserIdentity(user_id="test", trust_level=0.5),
            world_state=WorldState(data={})
        )
        
        await om._fast_response(context)
        
        # Check call arguments
        # If generate_fast was called:
        if mock_llm.generate_fast.called:
            call_args = mock_llm.generate_fast.call_args
        else:
            call_args = mock_llm.generate.call_args
            
        kwargs = call_args.kwargs
        system_prompt = kwargs.get('system_prompt', '')
        
        current_date = datetime.now().strftime("%d.%m.%Y")
        print(f"Checking for date '{current_date}' in system_prompt: '{system_prompt}'")
        self.assertIn(current_date, system_prompt)
        print("SUCCESS: Date found in FAST path prompt.")

    async def test_date_injection_medium(self):
        print("\n--- Testing Date Injection in MEDIUM path ---")
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "Response with date"
        
        om = OperationalModule(llm=mock_llm, tool_caller=None)
        
        context = ContextSlice(
            user_input="What date is it?",
            user_identity=UserIdentity(user_id="test", trust_level=0.5),
            world_state=WorldState(data={})
        )
        
        await om._medium_response(context)
        
        # Check call arguments
        call_args = mock_llm.generate.call_args
        kwargs = call_args.kwargs
        prompt = kwargs.get('prompt', '')
        
        current_date = datetime.now().strftime("%d.%m.%Y")
        print(f"Checking for date '{current_date}' in prompt: '{prompt[:100]}...'")
        self.assertIn(f"TODAY'S DATE: {current_date}", prompt)
        print("SUCCESS: Date found in MEDIUM path prompt.")

    async def test_analytical_routing_to_medium(self):
        print("\n--- Testing Analytical Intent Routing ---")
        mock_llm = AsyncMock()
        om = OperationalModule(llm=mock_llm, tool_caller=None)
        
        # Mock policy
        om.policy = PolicySpace()
        
        context = ContextSlice(
            user_input="Solve this logic puzzle",
            user_identity=UserIdentity(user_id="test", trust_level=0.5),
            world_state=WorldState(data={})
        )
        
        # Test 1: Analytical Intent (High Conf) -> Should be FAST or MEDIUM (Not DEEP)
        depth = om._decide_depth(intent="analytical", confidence=0.9, context=context)
        print(f"Intent='analytical', Confidence=0.9 -> Depth={depth}")
        self.assertIn(depth, [DecisionDepth.FAST, DecisionDepth.MEDIUM])
        self.assertNotEqual(depth, DecisionDepth.DEEP)
        
        # Test 2: Complex Intent (High Conf) -> FAST (due to fast_path_bias override, relies on Escalation)
        depth = om._decide_depth(intent="complex", confidence=0.9, context=context)
        print(f"Intent='complex', Confidence=0.9 -> Depth={depth}")
        self.assertEqual(depth, DecisionDepth.FAST)

        # Test 3: Complex Intent (Low/Medium Conf) -> DEEP (Forced)
        # Assuming fast_path_bias is around 0.8 and expert_threshold around 0.4
        depth = om._decide_depth(intent="complex", confidence=0.5, context=context)
        print(f"Intent='complex', Confidence=0.5 -> Depth={depth}")
        self.assertEqual(depth, DecisionDepth.DEEP)

        print("SUCCESS: Routing logic verified.")

if __name__ == "__main__":
    unittest.main()
