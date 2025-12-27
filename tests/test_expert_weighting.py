
import asyncio
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, "e:\\agi2")

from core.experts import CriticModule, ExpertResponse

class TestExpertWeighting(unittest.IsolatedAsyncioTestCase):
    async def test_weighting_injection(self):
        print("\n--- Testing Dynamic Expert Weighting ---")
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "FINAL SYNTHESIS: Test Response"
        
        critic = CriticModule(llm=mock_llm)
        
        # Expert responses dummy
        responses = [ExpertResponse(expert_type="neutral", response="A", confidence=0.9, temperature_used=0.0)]
        
        # Test 1: Creative Intent
        await critic.analyze(responses, "idea", intent="creative")
        prompt_arg = mock_llm.generate.call_args.kwargs['prompt']
        
        print(f"Checking Creative prompt: {prompt_arg[-200:]}")
        self.assertIn("PRIORITY: Focus on NOVELTY and IDEAS", prompt_arg)
        self.assertIn("Trust the CREATIVE EXPERT", prompt_arg)
        print("SUCCESS: Creative weighting injected.")
        
        # Test 2: Forecasting Intent
        await critic.analyze(responses, "future", intent="forecasting")
        prompt_arg = mock_llm.generate.call_args.kwargs['prompt']
        
        print(f"Checking Forecasting prompt: {prompt_arg[-200:]}")
        self.assertIn("PRIORITY: Focus on FUTURE TRENDS", prompt_arg)
        self.assertIn("Trust the FORECASTER", prompt_arg)
        print("SUCCESS: Forecasting weighting injected.")

        # Test 3: Realtime Data Intent
        await critic.analyze(responses, "price", intent="realtime_data")
        prompt_arg = mock_llm.generate.call_args.kwargs['prompt']
        
        print(f"Checking Realtime prompt: {prompt_arg[-200:]}")
        self.assertIn("PRIORITY: Focus on FACTS", prompt_arg)
        self.assertIn("Trust the [OBSERVATION]", prompt_arg)
        print("SUCCESS: Realtime weighting injected.")

if __name__ == "__main__":
    unittest.main()
