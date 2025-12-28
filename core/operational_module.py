# Operational Module (OM)
# Central decision-making module with Decision Depth Controller
# Refactored with InfoBroker and FallbackGenerator support

from datetime import datetime
from typing import Optional, TYPE_CHECKING
import time

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models.schemas import (
    ContextSlice, DecisionObject, DecisionDepth, PolicySpace,
    RawTrace, ExpertResponse, CriticAnalysis
)
from models.llm_interface import LLMInterface
from core.experts import ExpertsModule, CriticModule
from core.validator import SemanticValidator
from core.ontology import is_internal_query, entity_exists, extract_entity_name, should_block_search, get_ontology_response

if TYPE_CHECKING:
    from core.info_broker import InfoBroker
    from core.sanitizer import ResponseSanitizer
    from core.fallback_generator import FallbackGenerator
    from core.task_decomposer import TaskDecomposer


# Intent classification prompts
INTENT_CLASSIFIER_PROMPT = """Classify the user intent and estimate your confidence.

Categories:
- memorize: New facts, specific data, or instructions to save for future use (e.g. project passwords, user preferences)
- recall: Questions about past facts, specific data mentioned earlier, or conversation history
- smalltalk: casual conversation, greetings, simple social interaction
- factual: requests for general information or facts NOT in previous conversation
- analytical: requires reasoning, comparison, or analysis
- creative: requires generation of content, ideas
- complex: multi-step tasks, unclear requirements
- confirmation: simple acknowledgements, agreements, or announcements of immediate actions (e.g. "ok", "uploading now", "wait")
- realtime_data: requests for CURRENT/LIVE data that changes over time (e.g. stock prices, crypto prices, weather, exchange rates, news headlines). THIS REQUIRES REAL-TIME SEARCH.

Output format:
INTENT: [category]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]"""


class OperationalModule:
    """
    Operational Module (OM).
    
    Central decision-making module.
    
    Functions:
    - Analyze input + context
    - Choose thinking depth (fast/medium/deep)
    - Call experts and critic
    - Synthesize final response
    - Send trace to Learning Decoder
    """
    
    
    def __init__(
        self,
        llm: LLMInterface,
        policy: Optional[PolicySpace] = None,
        tool_caller: Optional[LLMInterface] = None,
        # NEW: Modular components
        info_broker: Optional['InfoBroker'] = None,
        sanitizer: Optional['ResponseSanitizer'] = None,
        fallback_generator: Optional['FallbackGenerator'] = None
    ):
        self.llm = llm
        self.policy = policy or PolicySpace()
        self.experts = ExpertsModule(llm, self.policy, tool_caller)
        self.critic = CriticModule(llm)
        self.validator = SemanticValidator(llm)
        
        # NEW: Store modular components
        self.info_broker = info_broker
        self.sanitizer = sanitizer
        self.fallback_generator = fallback_generator
        
        # NEW: Task Decomposer for complex multi-step problems
        from core.task_decomposer import TaskDecomposer
        self.task_decomposer = TaskDecomposer(llm)
        
        # NEW: Simulation Engine for deterministic calculations
        from core.simulation_engine import SimulationEngine
        self.simulation_engine = SimulationEngine()
        
        # Expose search_engine for InfoBroker setup
        if hasattr(self.experts, 'llm'):
            from core.search_engine import SearchEngine
            self.search_engine = SearchEngine()
        
        # Initialize Intent Router
        from core.intent_router import IntentRouter
        self.intent_router = IntentRouter(llm)
    
    async def process(
        self,
        context_slice: ContextSlice,
        context_manager: Optional['ContextManager'] = None
    ) -> tuple[str, DecisionObject, RawTrace]:
        """
        Process user input and generate response.
        
        Returns:
        - Final response text
        - Decision object
        - Raw trace for Learning Decoder
        """
        start_time = time.time()
        
        # Step 1: Classify intent and determine depth
        intent, confidence = await self.intent_router.classify(context_slice.user_input)
        depth = self._decide_depth(intent, confidence, context_slice)
        
        # Step 1.5: Handle recall intent (RAG)
        if intent == "recall" and context_manager:
            relevant_facts = context_manager.get_context_for_recall(context_slice.user_input)
            context_slice.long_term_context = relevant_facts
            print(f"[OM] Recall triggered. Long-term context added.")
        
        # Step 1.6: Task Decomposition for complex problems (NEW)
        decomposed_problem = None
        structured_context = ""
        simulation_result = None
        
        if self.task_decomposer.is_complex_problem(context_slice.user_input):
            decomposed_problem = self.task_decomposer.decompose(context_slice.user_input)
            structured_context = self.task_decomposer.get_structured_prompt(decomposed_problem)
            print(f"[OM] Complex problem detected:")
            print(f"     - Entities: {decomposed_problem.entities}")
            print(f"     - Given facts: {list(decomposed_problem.given_facts.keys())}")
            print(f"     - Missing data (DO NOT INVENT): {decomposed_problem.missing_facts}")
            print(f"     - Subtasks: {len(decomposed_problem.subtasks)}")
            
            # Inject structured context to prevent hallucination
            context_slice.long_term_context = structured_context + "\n\n" + (context_slice.long_term_context or "")
            
            # Force DEEP path for complex problems
            if depth != DecisionDepth.DEEP:
                print(f"[OM] Escalating to DEEP for complex problem")
                depth = DecisionDepth.DEEP
        
        # Step 1.6b: Try deterministic simulation (FSM) for robot/resource problems
        from core.simulation_engine import SimulationType
        sim_type = self.simulation_engine.detect_simulation_type(context_slice.user_input)
        
        if sim_type == SimulationType.FSM:
            print(f"[OM] FSM Simulation detected - using deterministic solver")
            scenario = self.simulation_engine.parse_robot_scenario(context_slice.user_input)
            
            if scenario.get("entities") and scenario.get("tasks"):
                simulation_result = self.simulation_engine.run_robot_simulation(scenario)
                
                if simulation_result.success:
                    print(f"[OM] Simulation SUCCESS: {simulation_result.final_values}")
                    # Inject simulation result into context
                    context_slice.long_term_context = f"""
## DETERMINISTIC SIMULATION RESULT (use these EXACT values):
{simulation_result.answer_text}

{context_slice.long_term_context or ""}"""
                else:
                    print(f"[OM] Simulation: {simulation_result.error}")
            
        # Step 1.7: Generate internal thoughts for complex tasks
        thoughts = ""
        if depth == DecisionDepth.DEEP:
            thoughts = await self._generate_thoughts(context_slice)
            print(f"[OM] Inner Monologue: {thoughts[:80]}...")
            
        # Step 2: Generate response based on depth
        expert_outputs = []
        critic_output = None
        
        # ONTOLOGY GATE ENFORCEMENT: Block fabrication of non-existent entities
        if intent == "unknown_internal":
            entity_name = extract_entity_name(context_slice.user_input) or "unknown"
            response = get_ontology_response(entity_name)
            print(f"[OM] ONTOLOGY GATE: Refused to fabricate entity '{entity_name}'")
            depth = DecisionDepth.FAST
            
        elif intent == "memorize" and context_manager:
            # Special case for memorization
            response = await self._memorize_and_respond(context_slice, context_manager)
            depth = DecisionDepth.FAST # Memorization is usually fast
        elif depth == DecisionDepth.FAST:
            # Fast path: single LLM call
            response = await self._fast_response(context_slice)
            
            # DEPTH ESCALATION: If FAST response needs tools, escalate to DEEP
            if "NEED_TOOL:" in response:
                print(f"[OM] Escalating from FAST to DEEP (tool required)")
                thoughts = await self._generate_thoughts(context_slice)
                response, expert_outputs, critic_output = await self._deep_response(context_slice, thoughts)
                depth = DecisionDepth.DEEP
            
        elif depth == DecisionDepth.MEDIUM:
            # Medium path: LLM + memory context
            response = await self._medium_response(context_slice)
            
            # DEPTH ESCALATION: If MEDIUM response needs tools, escalate to DEEP
            if "NEED_TOOL:" in response:
                print(f"[OM] Escalating from MEDIUM to DEEP (tool required)")
                thoughts = await self._generate_thoughts(context_slice)
                response, expert_outputs, critic_output = await self._deep_response(context_slice, thoughts)
                depth = DecisionDepth.DEEP
            
        else:  # DEEP
            # Deep path: experts + critic
            response, expert_outputs, critic_output = await self._deep_response(context_slice, thoughts)
        
        # Step 2.6: Handle insufficient information (NEW)
        if self._is_response_insufficient(response, confidence) and self.fallback_generator:
            from core.fallback_generator import UncertaintyLevel
            level = UncertaintyLevel.HIGH if confidence < 0.3 else UncertaintyLevel.MEDIUM
            fallback = self.fallback_generator.admit_uncertainty(
                topic=context_slice.user_input[:50],
                level=level
            )
            print(f"[OM] FallbackGenerator activated: {level.value}")
            response = fallback.text
            
        # Step 2.5: Apply State Updates from Experts
        if expert_outputs:
             for exp in expert_outputs:
                 if exp.expert_type == "neutral" and exp.world_state:
                     context_slice.world_state.update(exp.world_state)
                     break
            
        # Step 3: Semantic/Logic Guardrail
        # CRITICAL: Bypass validator for intermediate tool commands
        validation_report = {}
        if depth in [DecisionDepth.MEDIUM, DecisionDepth.DEEP]:
            if "NEED_TOOL:" in response or "search" in response.lower()[:50]:
                 print("[OM] Skipping validation for Tool Command.")
            else:
                # Use full reasoning context for validation
                validation_context = f"{context_slice.long_term_context or ''}\n\nUser Input: {context_slice.user_input}"
                response, validation_report = await self.validator.validate(response, validation_context)
                if validation_report.get("status") == "failed":
                    print("[OM] Response corrected by Guardrail.")
        
        # Calculate cost
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Create decision object
        decision = DecisionObject(
            action="respond",
            confidence=confidence,
            depth_used=depth,
            cost={"time_ms": elapsed_ms, "experts_used": len(expert_outputs)},
            policy_snapshot=self.policy.to_dict(),
            reasoning=f"Intent: {intent}, Depth: {depth.value}",
            thoughts=thoughts,
            validation_report=validation_report
        )
        
        # Create raw trace
        trace = RawTrace(
            user_input=context_slice.user_input,
            context_snapshot=context_slice.to_dict(),
            expert_outputs=[e.to_dict() for e in expert_outputs],
            critic_output=critic_output.to_dict() if critic_output else {},
            decision=decision.to_dict(),
            final_response=response,
            thoughts=thoughts,
            validation_report=validation_report
        )
        
        return response, decision, trace
    
    async def _classify_intent(self, user_input: str) -> tuple[str, float]:
        """Classify user intent (Delegated to IntentRouter)."""
        return await self.intent_router.classify(user_input)

    
    def _decide_depth(
        self,
        intent: str,
        confidence: float,
        context: ContextSlice
    ) -> DecisionDepth:
        """
        Decision Depth Controller.
        
        Fast path  → 1 LLM call (smalltalk, high confidence)
        Medium path → LLM + memory (moderate uncertainty)
        Deep path  → experts + critic (complex, low confidence)
        """
        
        # Check semantic rules first (pattern-driven routing)
        user_input_lower = context.user_input.lower()
        intent_lower = intent.lower()
        
        for rule_trigger, target_depth in self.policy.semantic_rules.items():
            trigger_lower = rule_trigger.lower()
            
            # Check for matches in intent or user input
            # Supports both exact substring and word-by-word matching
            trigger_words = trigger_lower.split('_')  # e.g., "complex_analytical" -> ["complex", "analytical"]
            
            matched = False
            if trigger_lower in intent_lower:
                matched = True
            elif trigger_lower in user_input_lower:
                matched = True
            elif any(word in intent_lower for word in trigger_words if len(word) > 2):
                matched = True
            elif any(word in user_input_lower for word in trigger_words if len(word) > 2):
                matched = True
                
            if matched:
                try:
                    forced_depth = DecisionDepth(target_depth.lower())
                    print(f"[OM] Semantic rule applied: '{rule_trigger}' -> {forced_depth.value}")
                    return forced_depth
                except ValueError:
                    pass
        
        # ═══════════════════════════════════════════════════════════════
        # NEW INTENTS: Force FAST/MEDIUM to prevent expert calls
        # ═══════════════════════════════════════════════════════════════
        NO_EXPERT_INTENTS = [
            "self_reflection",      # Self-analysis questions
            "internal_query",       # Omega architecture
            # calculation_simple REMOVED - User wants tools for accuracy
            "unknown_internal",     # Non-existent modules
            "philosophical",        # Introspective/ethical questions
            "analytical"            # Logic puzzles
        ]
        if intent in NO_EXPERT_INTENTS:
            print(f"[OM] NO-EXPERT PATH: Intent '{intent}' -> FAST (pure LLM)")
            return DecisionDepth.FAST
        
        # PRIORITY: Realtime data & Calculation ALWAYS requires DEEP path (tools)
        # PRIORITY: Realtime data & Calculation ALWAYS requires DEEP path (tools)
        if intent in ["realtime_data", "calculation", "calculation_simple"]:
            return DecisionDepth.DEEP
        
        # Fast path conditions
        if intent == "smalltalk" and confidence > 0.7:
            return DecisionDepth.FAST
            
        if intent == "confirmation":
            return DecisionDepth.FAST
        
        if confidence > self.policy.fast_path_bias:
            return DecisionDepth.FAST
        
        # Deep path conditions
        if confidence < self.policy.expert_call_threshold:
            return DecisionDepth.DEEP
        
        if intent == "complex":  # Removed "analytical" to allow reasoning via Medium path
            return DecisionDepth.DEEP
        
        # Low trust users get more scrutiny
        if context.user_identity.trust_level < 0.4:
            return DecisionDepth.DEEP
        
        # Default to medium
        return DecisionDepth.MEDIUM
    
    async def _fast_response(self, context: ContextSlice) -> str:
        """Fast path: single LLM call (uses FastLLM if available)."""
        
        # Inject date and self-identity
        from datetime import datetime
        # Inject date and simplified identity for FAST path (to avoid hallucinations in small models)
        from datetime import datetime
        # Simplified identity for phi3:mini
        current_date_str = datetime.now().strftime("%d.%m.%Y")
        system_msg = f"You are Omega, a helpful AI assistant. Today's date: {current_date_str}. Do not hallucinate."
        
        prompt = context.user_input
        
        # Use fast LLM if router is available
        if hasattr(self.llm, 'generate_fast'):
            return await self.llm.generate_fast(
                prompt=prompt,
                system_prompt=system_msg,
                temperature=0.7,
                max_tokens=512  # Shorter for fast responses
            )
        
        return await self.llm.generate(
            prompt=prompt,
            temperature=0.7
        )
    
    async def _medium_response(self, context: ContextSlice) -> str:
        """Medium path: LLM + memory context."""
        
        # Inject date
        from datetime import datetime
        current_date_str = datetime.now().strftime("%d.%m.%Y")
        
        # Import self-identity
        from core.ontology import SELF_IDENTITY
        
        # Build context from recent events
        context_str = ""
        for event in context.recent_events[-5:]:
            context_str += f"[{event.event_type}] {event.content}\n"
        
        # Add long-term context if available
        if context.long_term_context:
            context_str = context.long_term_context + "\n" + context_str

        prompt = f"""[TODAY'S DATE: {current_date_str}]
Context:
{context_str}

Current goal: {context.active_goal or 'None'}
User state: {context.emotional_state}

User message: {context.user_input}"""
        
        return await self.llm.generate(
            prompt=prompt,
            # Add constraint to NOT output internal logs
            system_prompt=f"{SELF_IDENTITY}\nYou are a helpful assistant. Do NOT output internal module logs (like 'OperationalModule:'). Check your response for naturalness.",
            temperature=0.6
        )
    
    async def _deep_response(
        self,
        context: ContextSlice,
        thoughts: str = ""
    ) -> tuple[str, list[ExpertResponse], CriticAnalysis]:
        """Deep path: experts + critic (uses MainLLM for quality)."""
        
        # Build context
        recent_str = "\n".join([
            f"[{e.event_type}] {e.content}"
            for e in context.recent_events[-5:]
        ])
        
        context_str = recent_str
        if context.long_term_context:
            context_str = context.long_term_context + "\n" + recent_str
        
        # Inject thoughts into expert prompt
        if thoughts:
            context_str = f"INTERNAL THOUGHTS FOR FOCUS: {thoughts}\n\n" + context_str

        if hasattr(self.llm, 'set_mode'):
            self.llm.set_mode("main")
            
        # CONTEXT PURGE (Context Pruning):
        # If the task is purely about data retrieval (realtime_data),
        # prevent mathematical hallucinations by scrubbing old calculation results.
        intent, _ = await self._classify_intent(context.user_input)
        if intent == "realtime_data":
             import re
             # Remove "Result: X" and "Formula: Y" AND old "[OBSERVATION]" lines to prevent context contamination
             context_str = re.sub(r"(Result:|Formula:|\[OBSERVATION\]).*?(\n|$)", "", context_str)
             print(f"[OM] Context Purge Active: Scrubbed calculation artifacts and old observations for Search task.")
        
        # Consult all experts
        expert_responses = await self.experts.consult_all(
            prompt=context.user_input,
            world_state=context.world_state,
            context=context_str
        )
        
        # Get critic analysis
        critic_analysis = await self.critic.analyze(
            expert_responses=expert_responses,
            original_query=context.user_input,
            intent=intent
        )
        
        # Select best response based on critic
        if critic_analysis.recommended_response:
            response = critic_analysis.recommended_response
        else:
            # Default to neutral expert
            response = expert_responses[0].response
        
        return response, expert_responses, critic_analysis
    
    async def _memorize_and_respond(self, context: ContextSlice, km: 'ContextManager') -> str:
        """Extract fact from input, store it, and confirm."""
        
        extraction_prompt = f"""Extract the core fact or piece of information to be remembered from this user message.
User message: {context.user_input}

Output format:
FACT: [concise fact]
ENTITIES: [comma separated list of key entities]"""

        response = await self.llm.generate(
            prompt=extraction_prompt,
            system_prompt="You are a memory extraction expert. Be precise.",
            temperature=0.2
        )
        
        fact = ""
        entities = []
        for line in response.split('\n'):
            if line.startswith("FACT:"):
                fact = line.split(":", 1)[1].strip()
            elif line.startswith("ENTITIES:"):
                entities = [e.strip() for e in line.split(":", 1)[1].split(",")]
        
        if fact:
            km.add_fact(fact, entities)
            return f"Understood. I've saved that: {fact}"
        else:
            return "I understood you want me to remember something, but I couldn't identify a specific fact. Could you please clarify?"

    async def _generate_thoughts(self, context: ContextSlice) -> str:
        """Generate internal strategy/thoughts for the current task."""
        
        thoughts_prompt = f"""Analyze the user query and context. Provide an internal strategy for solving this.
What are the keys to a good answer? What should experts focus on?

If this is a RESOURCE allocation or LOGIC problem:
- Explicitly ask experts to create a "JSON STATE BLOCK".
- Identify constraints (e.g. max ports, budget).

User query: {context.user_input}
Recent context: {[e.content[:50] for e in context.recent_events[-3:]]}

Output: Your internal thoughts (1-2 sentences)."""

        return await self.llm.generate(
            prompt=thoughts_prompt,
            system_prompt="You are the system's core reasoning engine. Think before you act.",
            temperature=0.4
        )

    def update_policy(self, updates: dict):
        """Apply policy updates from Homeostasis."""
        self.policy.apply_update(updates)
        # Update experts with new policy
        self.experts.policy = self.policy
    
    def _is_response_insufficient(self, response: str, confidence: float) -> bool:
        """
        Check if the response is insufficient and should use fallback.
        
        Criteria:
        - Empty or very short response
        - Contains "not found" indicators
        - Low confidence with vague response
        - Response is just repeating the question
        """
        if not response or len(response.strip()) < 10:
            return True
        
        response_lower = response.lower()
        
        # Indicators of failed search / no information
        NO_INFO_PATTERNS = [
            "не удалось найти",
            "no information found",
            "i couldn't find",
            "i don't have",
            "no data available",
            "unable to find",
            "нет данных",
            "информация недоступна",
        ]
        
        for pattern in NO_INFO_PATTERNS:
            if pattern in response_lower:
                return True
        
        # Low confidence + short response = likely insufficient
        if confidence < 0.4 and len(response) < 100:
            return True
        
        return False
