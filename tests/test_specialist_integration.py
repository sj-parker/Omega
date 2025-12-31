import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from core.task_orchestrator import TaskOrchestrator
from core.task_queue import Task
from core.specialist_broker import SpecialistBroker
from core.specialists.identity_specialist import GeneralIdentitySpecialist

class TestSpecialistIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_identity_specialist_selection(self):
        # Mock dependencies
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="I am Omega, an AI assistant.")
        
        intent_router = MagicMock()
        info_broker = MagicMock()
        experts = MagicMock()
        context_manager = MagicMock()
        
        # Initialize Orchestrator
        orchestrator = TaskOrchestrator(llm, intent_router, info_broker, experts, context_manager)
        
        # Verify Broker initialization
        self.assertIsInstance(orchestrator.specialist_broker, SpecialistBroker)
        self.assertIn("general_identity", orchestrator.specialist_broker.specialists)
        
        # Test selection logic directly
        specialist = orchestrator.specialist_broker.get_selection("Who are you?")
        self.assertIsInstance(specialist, GeneralIdentitySpecialist)
        print("\nTest passed: Identified 'Who are you?' correctly.")

        specialist_mars = orchestrator.specialist_broker.get_selection("Are you from Mars?")
        self.assertIsInstance(specialist_mars, GeneralIdentitySpecialist)
        print("Test passed: Identified 'Mars' correctly.")

        # Test fallback
        specialist_none = orchestrator.specialist_broker.get_selection("What is the price of BTC?")
        self.assertIsNone(specialist_none) # Should fallback to legacy experts (or financial specialist if we had one)
        print("Test passed: Ignored unrelated query.")

if __name__ == '__main__':
    unittest.main()
