# Main Entry Point
# CLI interface for the Cognitive LLM System
# Refactored with Orchestrator, InfoBroker, and Safety Layer

import asyncio
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.schemas import PolicySpace, WorldState
from models.llm_interface import MockLLM, OllamaLLM, LLMRouter
from core.gatekeeper import Gatekeeper, UserHistoryStore
from core.context_manager import ContextManager
from core.operational_module import OperationalModule
from learning.learning_decoder import LearningDecoder
from learning.homeostasis import HomeostasisController
from learning.reflection import ReflectionController
from core.config import config
from core.tracer import tracer

# NEW: Modular Architecture Components
from core.orchestrator import Orchestrator, ModuleInterface, ModuleCapability
from core.info_broker import InfoBroker
from core.sanitizer import ResponseSanitizer
from core.fallback_generator import FallbackGenerator, UncertaintyLevel


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
        use_ollama: bool = None,
        main_model: str = None,
        fast_model: str = None,
        use_multi_model: bool = None
    ):
        # Load from config or use provided values
        self.use_ollama = use_ollama if use_ollama is not None else config.get("models.use_ollama")
        self.main_model = main_model if main_model is not None else config.get("models.main")
        self.fast_model = fast_model if fast_model is not None else config.get("models.fast")
        self.use_multi_model = use_multi_model if use_multi_model is not None else config.get("models.use_multi_model")

        # Initialize LLM(s)
        if self.use_ollama:
            if self.use_multi_model:
                # Multi-model setup: fast + main
                fast_llm = OllamaLLM(model=self.fast_model)
                main_llm = OllamaLLM(model=self.main_model)
                self.llm = LLMRouter(fast_llm=fast_llm, main_llm=main_llm)
                print(f"[LLM] Multi-model: Fast={self.fast_model}, Main={self.main_model}")
            else:
                # Single model
                self.llm = OllamaLLM(model=self.main_model)
                print(f"[LLM] Single model: {self.main_model}")
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
        
        # NEW: Initialize Orchestrator
        self.orchestrator = Orchestrator(default_timeout=30.0)
        
        # NEW: Initialize Safety Layer
        self.sanitizer = ResponseSanitizer(strict_mode=True)
        self.fallback_generator = FallbackGenerator(default_language="ru")
        
        # NEW: Initialize InfoBroker (will be fully configured after OM creation)
        self.info_broker = InfoBroker(
            context_manager=self.context_manager,
            search_engine=None,  # Will be set from OM
            experts=None         # Will be set after OM creation
        )
        
        # Create OperationalModule with new components
        self.om = OperationalModule(
            llm=self.llm,
            policy=self.policy,
            info_broker=self.info_broker,
            sanitizer=self.sanitizer,
            fallback_generator=self.fallback_generator
        )
        
        # Complete InfoBroker setup
        if hasattr(self.om, 'experts'):
            self.info_broker.experts = self.om.experts
        if hasattr(self.om, 'search_engine'):
            self.info_broker.search_engine = self.om.search_engine
        
        # Register modules in Orchestrator
        asyncio.create_task(self._register_modules())
        
        # Learning Decoder and Reflection use quality_llm (main model)
        self.learning_decoder = LearningDecoder(llm=self.quality_llm)
        self.homeostasis = HomeostasisController(self.policy)
        self.reflection = ReflectionController(
            self.learning_decoder,
            self.homeostasis,
            self.quality_llm  # Use main LLM for reflection quality
        )
        
        self._reflection_started = False
        self.world_states: dict[str, WorldState] = {} # user_id -> WorldState
    
    async def process(self, user_id: str, message: str) -> str:
        """
        Process a user message through the full pipeline.
        
        Returns the system response.
        """
        # Start tracing session
        episode_id = str(uuid.uuid4())
        tracer.start_session(episode_id)
        
        # Step 1: Security Gate
        tracer.add_step("gatekeeper", "Identification", f"Verifying user {user_id}", data_in={"user_id": user_id, "message": message})
        identity = self.gatekeeper.identify(user_id, message)
        tracer.add_step("gatekeeper", "Result", f"Trust level: {identity.trust_level}", data_out=identity.to_dict())
        
        if identity.risk_flag:
            print(f"[Security] Risk detected for user {user_id}, trust: {identity.trust_level}")
        
        # Step 2: Record input in Context Manager
        tracer.add_step("context_manager", "Record Input", "Saving user message to short-term store")
        self.context_manager.record_user_input(message)
        
        # Step 3: Get or create world state for user
        if user_id not in self.world_states:
             self.world_states[user_id] = WorldState()
        
        # Step 3.1: Get context slice (through Memory Gate)
        tracer.add_step("context_manager", "Get Context", "Retrieving relevant context slice")
        context_slice = self.context_manager.get_context_slice(message, identity, self.world_states[user_id])
        tracer.add_step("context_manager", "Context Slice", f"Retrieved {len(context_slice.recent_events)} recent events", data_out=context_slice.to_dict())
        
        # Step 4: Process through Operational Module
        tracer.add_step("operational_module", "Process", "Starting central decision making")
        response, decision, trace = await self.om.process(context_slice, self.context_manager)
        
        # Step 4.1: Persist updated world state
        self.world_states[user_id] = context_slice.world_state
        
        # Step 4.2: Apply Response Sanitizer (NEW)
        tracer.add_step("sanitizer", "Sanitize", "Checking for data leakage in response")
        sanitization = self.sanitizer.sanitize(response, context=message)
        if sanitization.was_modified:
            tracer.add_step("sanitizer", "Result", f"Redacted {sanitization.redactions_count} items", data_out=sanitization.to_dict())
            print(f"[Sanitizer] Redacted {sanitization.redactions_count} items: {sanitization.redaction_types}")
            response = sanitization.sanitized_text
        else:
            tracer.add_step("sanitizer", "Result", "No sensitive data found")
        
        # Step 5: Record response
        self.context_manager.record_system_response(response)
        self.context_manager.record_decision(decision)
        
        # Step 6: Store trace in Learning Decoder
        # Attach tracer steps to the RawTrace
        trace.steps = tracer.get_steps()
        trace.episode_id = episode_id
        
        tracer.add_step("learning_decoder", "Record Trace", "Storing episode trace for reflection")
        self.learning_decoder.record_trace(trace)
        
        # Step 7: Create summary for reflection
        await self.learning_decoder.create_summary(trace)
        
        # Step 8: Memory Compaction Trigger (Omega) - Background
        if len(self.context_manager.short_store.events) >= 15:
            asyncio.create_task(self._trigger_memory_compaction())
            
        # Print diagnostics
        print(f"[OM] Depth: {decision.depth_used.value}, Confidence: {decision.confidence:.2f}, Cost: {decision.cost['time_ms']}ms")
        
        # End session
        tracer.end_session()
        
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
    
    async def _trigger_memory_compaction(self):
        """Compact short-term memory into long-term facts."""
        print("[Memory] Compacting short-term memory...")
        recent_events = self.context_manager.short_store.get_recent_events(10)
        
        # Build context for summarization
        context_text = "\n".join([f"[{e.event_type}] {e.content}" for e in recent_events])
        
        compaction_prompt = f"""Review the following recent interactions and extract 1-2 important PERMANENT facts or context updates that should be kept in long-term memory.
Ignore smalltalk. Focus on: names, project details, passwords, user preferences, or major task shifts.

Recent Interactions:
{context_text}

Output format:
FACTS:
- [fact 1]
- [fact 2] (optional)"""

        try:
            response = await self.quality_llm.generate(
                prompt=compaction_prompt,
                system_prompt="You are a context compaction expert.",
                temperature=0.3
            )
            
            for line in response.split('\n'):
                if line.strip().startswith("-"):
                    fact = line.strip()[1:].strip()
                    if fact:
                        self.context_manager.add_fact(fact, importance=0.6)
            
            # Optionally clear some old events from short store? 
            # ShortContextStore uses deque(maxlen=20) so it's already bounded.
        except Exception as e:
            print(f"[Memory] Compaction error: {e}")
    
    async def _register_modules(self):
        """Register all modules in the Orchestrator."""
        try:
            # Register OperationalModule
            await self.orchestrator.register_module(
                name="operational_module",
                module=self.om,
                interface=ModuleInterface(
                    name="operational_module",
                    input_types=["user_query", "context_slice"],
                    output_types=["response", "decision", "trace"],
                    capabilities=[ModuleCapability.REASON, ModuleCapability.GENERATE],
                    priority=100
                )
            )
            
            # Register ContextManager
            await self.orchestrator.register_module(
                name="context_manager",
                module=self.context_manager,
                interface=ModuleInterface(
                    name="context_manager",
                    input_types=["user_input", "fact"],
                    output_types=["context_slice", "facts"],
                    capabilities=[ModuleCapability.REMEMBER],
                    priority=90
                )
            )
            
            # Register Gatekeeper
            await self.orchestrator.register_module(
                name="gatekeeper",
                module=self.gatekeeper,
                interface=ModuleInterface(
                    name="gatekeeper",
                    input_types=["user_message"],
                    output_types=["identity"],
                    capabilities=[ModuleCapability.VALIDATE],
                    priority=95,
                    is_async=False
                )
            )
            
            print(f"[Orchestrator] Registered {len(self.orchestrator.get_all_modules())} modules")
        except Exception as e:
            print(f"[Orchestrator] Registration error: {e}")
    
    def get_orchestrator_health(self) -> dict:
        """Get Orchestrator health report."""
        return self.orchestrator.health_report()


async def interactive_cli():
    """Interactive CLI for testing the system."""
    print("=" * 60)
    print("  Cognitive LLM System - Interactive Mode")
    print("=" * 60)
    print()
    
    # Ask for LLM choice
    use_ollama_default = config.get("models.use_ollama")
    use_ollama_str = input(f"Use Ollama? (y/n, default: {'y' if use_ollama_default else 'n'}): ").strip().lower()
    
    if use_ollama_str == "":
        use_ollama = use_ollama_default
    else:
        use_ollama = use_ollama_str == 'y'
    
    if use_ollama:
        use_multi_input = input("Use multi-model? (y/n, default: y): ").strip().lower()
        use_multi = use_multi_input != 'n'
        
        # Get main model with validation
        main_default = config.get("models.main")
        main_input = input(f"Main model (default: {main_default}): ").strip()
        # Reject y/n as model names
        if main_input.lower() in ('y', 'n', 'yes', 'no', ''):
            main_model = main_default
        else:
            main_model = main_input
        print(f"  → Using main model: {main_model}")
        
        if use_multi:
            fast_default = config.get("models.fast")
            fast_input = input(f"Fast model (default: {fast_default}): ").strip()
            if fast_input.lower() in ('y', 'n', 'yes', 'no', ''):
                fast_model = fast_default
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
