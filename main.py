# Main Entry Point
# CLI interface for the Cognitive LLM System

import asyncio
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.schemas import PolicySpace
from models.llm_interface import MockLLM, OllamaLLM, LLMRouter
from core.gatekeeper import Gatekeeper, UserHistoryStore
from core.context_manager import ContextManager
from core.operational_module import OperationalModule
from learning.learning_decoder import LearningDecoder
from learning.homeostasis import HomeostasisController
from learning.reflection import ReflectionController


class CognitiveSystem:
    """
    Main cognitive system orchestrator.
    
    Pipeline:
    User Input → Security Gate → Context Manager → Memory Gate → OM → Response
                                                                  ↓
                                                          Learning Decoder
                                                                  ↓
                                                          Homeostasis (async)
    
    Multi-Model Support:
    - FastLLM: phi3:mini for fast_path (simple tasks)
    - MainLLM: qwen2.5:7b for experts, critic, reflection (quality-critical)
    
    Homeostasis measures quality from MainLLM only, preventing
    the system from learning to prefer the faster but lower-quality model.
    """
    
    def __init__(
        self,
        use_ollama: bool = False,
        main_model: str = "qwen2.5:7b",
        fast_model: str = "phi3:mini",
        use_multi_model: bool = True
    ):
        # Initialize LLM(s)
        if use_ollama:
            if use_multi_model:
                # Multi-model setup: fast + main
                fast_llm = OllamaLLM(model=fast_model)
                main_llm = OllamaLLM(model=main_model)
                self.llm = LLMRouter(fast_llm=fast_llm, main_llm=main_llm)
                print(f"[LLM] Multi-model: Fast={fast_model}, Main={main_model}")
            else:
                # Single model
                self.llm = OllamaLLM(model=main_model)
                print(f"[LLM] Single model: {main_model}")
        else:
            self.llm = MockLLM()
        
        # For reflection and learning - ALWAYS use main LLM
        # This prevents learning from preferring fast (low-quality) model
        if hasattr(self.llm, 'main_llm'):
            self.quality_llm = self.llm.main_llm
        else:
            self.quality_llm = self.llm
        
        # Initialize components
        self.policy = PolicySpace()
        self.history_store = UserHistoryStore()
        self.gatekeeper = Gatekeeper(self.history_store)
        self.context_manager = ContextManager()
        self.om = OperationalModule(self.llm, self.policy)
        
        # Learning Decoder and Reflection use quality_llm (main model)
        self.learning_decoder = LearningDecoder(llm=self.quality_llm)
        self.homeostasis = HomeostasisController(self.policy)
        self.reflection = ReflectionController(
            self.learning_decoder,
            self.homeostasis,
            self.quality_llm  # Use main LLM for reflection quality
        )
        
        self._reflection_started = False
    
    async def process(self, user_id: str, message: str) -> str:
        """
        Process a user message through the full pipeline.
        
        Returns the system response.
        """
        # Step 1: Security Gate
        identity = self.gatekeeper.identify(user_id, message)
        
        if identity.risk_flag:
            print(f"[Security] Risk detected for user {user_id}, trust: {identity.trust_level}")
        
        # Step 2: Record input in Context Manager
        self.context_manager.record_user_input(message)
        
        # Step 3: Get context slice (through Memory Gate)
        context_slice = self.context_manager.get_context_slice(message, identity)
        
        # Step 4: Process through Operational Module
        response, decision, trace = await self.om.process(context_slice)
        
        # Step 5: Record response
        self.context_manager.record_system_response(response)
        self.context_manager.record_decision(decision)
        
        # Step 6: Store trace in Learning Decoder
        self.learning_decoder.record_trace(trace)
        
        # Step 7: Create summary for reflection
        await self.learning_decoder.create_summary(trace)
        
        # Print diagnostics
        print(f"[OM] Depth: {decision.depth_used.value}, Confidence: {decision.confidence:.2f}, Cost: {decision.cost['time_ms']}ms")
        
        return response
    
    async def start_reflection(self, interval: float = 30.0):
        """Start background reflection loop."""
        if not self._reflection_started:
            await self.reflection.start_background(interval)
            self._reflection_started = True
            print("[Reflection] Background loop started")
    
    async def stop_reflection(self):
        """Stop background reflection."""
        if self._reflection_started:
            await self.reflection.stop_background()
            self._reflection_started = False
            print("[Reflection] Background loop stopped")
    
    def get_health_report(self) -> dict:
        """Get system health report."""
        return {
            "policy": self.policy.to_dict(),
            "homeostasis": self.homeostasis.get_health_report(),
            "patterns_found": len(self.learning_decoder.get_patterns()),
            "episodes_recorded": len(self.learning_decoder.summaries)
        }


async def interactive_cli():
    """Interactive CLI for testing the system."""
    print("=" * 60)
    print("  Cognitive LLM System - Interactive Mode")
    print("=" * 60)
    print()
    
    # Ask for LLM choice
    use_ollama = input("Use Ollama? (y/n, default: n): ").strip().lower() == 'y'
    
    if use_ollama:
        use_multi_input = input("Use multi-model? (y/n, default: y): ").strip().lower()
        use_multi = use_multi_input != 'n'
        
        # Get main model with validation
        main_input = input("Main model (default: qwen2.5:7b): ").strip()
        # Reject y/n as model names
        if main_input.lower() in ('y', 'n', 'yes', 'no', ''):
            main_model = "qwen2.5:7b"
        else:
            main_model = main_input
        print(f"  → Using main model: {main_model}")
        
        if use_multi:
            fast_input = input("Fast model (default: phi3:mini): ").strip()
            if fast_input.lower() in ('y', 'n', 'yes', 'no', ''):
                fast_model = "phi3:mini"
            else:
                fast_model = fast_input
            print(f"  → Using fast model: {fast_model}")
            
            system = CognitiveSystem(
                use_ollama=True,
                main_model=main_model,
                fast_model=fast_model,
                use_multi_model=True
            )
        else:
            system = CognitiveSystem(
                use_ollama=True,
                main_model=main_model,
                use_multi_model=False
            )
    else:
        print("[Using MockLLM for testing]")
        system = CognitiveSystem(use_ollama=False)
    
    user_id = input("Your user ID (default: user1): ").strip() or "user1"
    
    # Start reflection in background
    await system.start_reflection(interval=60.0)
    
    print()
    print("Type your messages. Commands:")
    print("  /health   - Show system health")
    print("  /policy   - Show current policy")
    print("  /stats    - Show LLM usage stats")
    print("  /memory   - Show memory status")
    print("  /clean    - Clear ALL memory")
    print("  /sanitize - Remove LLM identity mentions")
    print("  /reflect  - Force reflection")
    print("  /quit     - Exit")
    print()
    
    try:
        while True:
            try:
                user_input = input(f"[{user_id}] > ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                if user_input == "/quit":
                    break
                elif user_input == "/health":
                    import json
                    print(json.dumps(system.get_health_report(), indent=2))
                elif user_input == "/policy":
                    import json
                    print(json.dumps(system.policy.to_dict(), indent=2))
                elif user_input == "/stats":
                    if hasattr(system.llm, 'get_stats'):
                        import json
                        print(json.dumps(system.llm.get_stats(), indent=2))
                    else:
                        print("Single model mode - no stats available")
                elif user_input == "/memory":
                    traces = len(system.learning_decoder.raw_traces)
                    summaries = len(system.learning_decoder.summaries)
                    patterns = len(system.learning_decoder.patterns)
                    print(f"Memory status:")
                    print(f"  📝 Raw traces: {traces}")
                    print(f"  📊 Summaries: {summaries} (need 3+ for reflection)")
                    print(f"  🔍 Patterns found: {patterns}")
                    if summaries > 0:
                        metrics = system.learning_decoder.get_metrics_aggregates()
                        print(f"  📈 Avg confidence: {metrics['avg_confidence']:.2f}")
                        print(f"  ⏱️ Avg cost: {metrics['avg_cost_ms']:.0f}ms")
                        print(f"  ✅ Success rate: {metrics['success_rate']:.1%}")
                elif user_input == "/clean":
                    confirm = input("This will DELETE all memory. Are you sure? (yes/no): ")
                    if confirm.lower() == "yes":
                        count = system.learning_decoder.clear_all_memory()
                        print(f"Cleared {count} items from memory")
                    else:
                        print("Cancelled")
                elif user_input == "/sanitize":
                    count = system.learning_decoder.sanitize_memory()
                    print(f"Sanitized {count} items (removed LLM identity mentions)")
                elif user_input == "/reflect":
                    summaries = len(system.learning_decoder.summaries)
                    if summaries < 3:
                        print(f"Need at least 3 episodes for reflection (current: {summaries})")
                    else:
                        pattern = await system.reflection.reflect_once()
                        if pattern:
                            print(f"✅ Pattern found: {pattern.description}")
                        else:
                            print("No significant pattern detected (system working normally)")
                continue
            
            # Process message
            response = await system.process(user_id, user_input)
            print(f"\n[System] {response}\n")
    
    finally:
        await system.stop_reflection()
        print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(interactive_cli())
