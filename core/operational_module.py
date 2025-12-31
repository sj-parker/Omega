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
from core.tracer import tracer

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
        fallback_generator: Optional['FallbackGenerator'] = None,
        search_llm: Optional[LLMInterface] = None
    ):
        self.llm = llm
        self.policy = policy or PolicySpace()
        self.experts = ExpertsModule(llm, self.policy, tool_caller)
        self.critic = CriticModule(llm)
        self.validator = SemanticValidator(llm)
        
        # NEW: Store modular components
        print(f"[OM] Initializing with info_broker: {info_broker}")
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
            self.search_engine = SearchEngine(llm=search_llm or llm)
        
        # Initialize Intent Router
        from core.intent_router import IntentRouter
        self.intent_router = IntentRouter(llm)
        
        # NEW: Initialize Specialist Broker
        from core.specialist_broker import SpecialistBroker
        from core.specialists.movie_specialist import MovieSpecialist
        
        self.specialist_broker = SpecialistBroker()
        
        # Check if we have search engine available to pass to movie specialist
        search_engine = getattr(self, 'search_engine', None)
        self.specialist_broker.register(MovieSpecialist(search_engine=search_engine))
        
        # Phase 6: SRA Specialist
        from core.sra_specialist import SRASpecialist
        self.sra = SRASpecialist(llm=tool_caller or llm)
    
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
        decision = None
        
        import re
        start_time = time.time()
        
        # ========================================
        # PHASE 0: SPECIALIST BROKER (Smart Path)
        # ========================================
        # Ask broker if we have a specialist for this
        specialist = self.specialist_broker.get_selection(context_slice.user_input, threshold=0.8)
        
        if specialist:
            tracer.add_step("specialist_broker", "Selected", f"Routing to {specialist.metadata.name}")
            print(f"[OM] Specialist Broker selected: {specialist.metadata.name}")
            
            try:
                # Execute specialist
                result = await specialist.execute(context_slice.user_input)
                tracer.add_step(specialist.metadata.id, "Execute", "Specialist finished", data_out=str(result))
                
                # Feedback Loop (Assume success if we got data)
                success = result.data is not None and "Error" not in str(result.data)
                self.specialist_broker.feedback(specialist.metadata.id, success)
                
                # Use result as final response (wrapped)
                final_response = f"[{specialist.metadata.name}] {result.data}"
                
                # Return early with DecisionObject
                elapsed_ms = int((time.time() - start_time) * 1000)
                
                decision = DecisionObject(
                    action="respond",
                    confidence=result.confidence,
                    depth_used=DecisionDepth.FAST,
                    cost={"time_ms": elapsed_ms, "specialist": specialist.metadata.name},
                    intent="specialist_handled",
                    reasoning=f"Handled by {specialist.metadata.name}"
                )
                
                trace = RawTrace(
                    user_input=context_slice.user_input,
                    final_response=final_response,
                    decision=decision.to_dict()
                )
                
                return final_response, decision, trace
                
            except Exception as e:
                print(f"[OM] Specialist failed: {e}")
                self.specialist_broker.feedback(specialist.metadata.id, success=False)
                # Fallthrough to normal LLM path
        
        # Step 1: Classify intent and determine depth
        tracer.add_step("intent_router", "Classify", f"Classifying user input: {context_slice.user_input[:50]}...")
        intent, confidence = await self.intent_router.classify(context_slice.user_input)
        tracer.add_step("intent_router", "Result", f"Intent: {intent}, Confidence: {confidence:.2f}", data_out={"intent": intent, "confidence": confidence})
        
        depth = self._decide_depth(intent, confidence, context_slice)
        tracer.add_step("operational_module", "Result", f"Path chosen: {depth.value}")
        
        # New: Trigger neutral refusal for forbidden intents/narratives
        narrative_keywords = [
            "first day", "when you were born", "how you think", "первый день", "как ты думаешь", 
            "who created you", "кто тебя создал", "истори\w+\s+тво", "history of your",
            "interven", "вмеша", "твой создатель", "your creator",
            "расскажи историю", "tell a story about yourself", "биография", "biography",
            "без фильтр", "without filter", "ответь честно", "answer honestly", "правд\w+\s+об\s+омег",
            "научил", "taught", "хочешь", "want", "желание", "desire", "цель", "goal", "миссия", "mission"
        ]
        is_narrative = any(re.search(kw, context_slice.user_input, re.IGNORECASE) for kw in narrative_keywords)
        
        if intent in ["self_reflection", "philosophical"] or is_narrative:
            print(f"[OM] HARD REFUSAL: Intent '{intent}' or Narrative detected.")
            reason = "subjective_request" if intent != "philosophical" else "moral_judgment"
            if is_narrative: reason = "narrative_fabrication"
            return self._generate_neutral_refusal(reason)
            
        # Step 1.5: Handle recall intent (RAG)
        if intent == "recall" and context_manager:
            tracer.add_step("context_manager", "Recall", "Searching long-term memory for relevant facts")
            relevant_facts = context_manager.get_context_for_recall(context_slice.user_input)
            context_slice.long_term_context = relevant_facts
            tracer.add_step("context_manager", "Result", f"Found {len(relevant_facts)} characters of relevant data")
            print(f"[OM] Recall triggered. Long-term context added.")
        
        # ========================================
        # PHASE 1: DISCOVERY (Phase 6 Advanced)
        # ========================================
        # Discovery should run for any intent that might need external data
        # SAFEGUARD: Skip discovery for self-referential / identity questions to prevent hallucinations
        is_self_ref = self._is_self_referential(context_slice.user_input)
        
        needs_discovery = (depth in [DecisionDepth.MEDIUM, DecisionDepth.DEEP] or intent in ["realtime_data", "complex", "analytical", "factual"]) and not is_self_ref
        if is_self_ref:
            print(f"[OM] Discovery Blocked: Self-referential query detected ('{context_slice.user_input[:30]}...')")
        
        if needs_discovery:
            tracer.add_step("om", "Discovery", "Analysing information dependencies...")
            requirements = await self.sra.identify_requirements(
                query=context_slice.user_input,
                context=context_slice.long_term_context or ""
            )
            print(f"[OM] SRA identified requirements: {requirements}")
            
            missing_requirements = self.sra.filter_existing_facts(requirements, context_slice.world_state)
            
            # Logic vs Data Check: 
            # If it's a logic intent and we already have all data, skip search.
            is_logic_task = intent in ["analytical", "calculation", "philosophical"]
            if is_logic_task and not missing_requirements:
                print(f"[OM] LOGIC TASK DETECTED: No external data needed for '{intent}' query. Bypassing Search.")
                tracer.add_step("om", "Logic Skip", "Task determined to be self-contained; skipping external discovery.")
            elif missing_requirements:
                print(f"[OM] Discovery Phase: Identified {len(missing_requirements)} missing info requirements.")
                retrieved_info = []
                for req in missing_requirements:
                    req_query = f"{req['entity']} {req['variable']}"
                    
                    # NEW: Double-check logic/identity blocks before calling Broker
                    from core.ontology import should_block_search
                    block_search, _ = should_block_search(req_query)
                    if block_search:
                        print(f"[OM] Discovery Blocked: Safeguard match for '{req_query}'")
                        continue
                    
                    # Call InfoBroker with domain specific specialists and volatility hint
                    info_result = await self.info_broker.request_info(
                        query=req_query,
                        domain=req.get('domain', 'general'),
                        world_state=context_slice.world_state,
                        volatility=req.get('volatility', 'low')
                    )
                    
                    if info_result and info_result.data:
                        retrieved_info.append(f"VERIFIED FACT ({req['entity']} {req['variable']}): {info_result.data}")
                
                if retrieved_info:
                    print(f"[OM] Discovery Phase found {len(retrieved_info)} facts.")
                    verification_block = "\n".join(retrieved_info)
                    # Correctly inject into context slice
                    context_slice.long_term_context = (
                        f"## VERIFIED GROUND TRUTH DATA:\n{verification_block}\n" + 
                        (context_slice.long_term_context or "")
                    )
                    # Escalate depth if we found complex requirements
                    if len(missing_requirements) >= 1:
                        depth = DecisionDepth.DEEP

        # Step 1.6: Task Decomposition for complex problems (NEW)
        decomposed_problem = None
        simulation_result = None
        
        if self.task_decomposer.is_complex_problem(context_slice.user_input):
            tracer.add_step("task_decomposer", "Decompose", "Parsing complex problem structure")
            decomposed_problem = await self.task_decomposer.decompose(context_slice.user_input)
            structured_context = self.task_decomposer.get_structured_prompt(decomposed_problem)
            tracer.add_step("task_decomposer", "Result", f"Entities: {len(decomposed_problem.entities)}, Facts: {len(decomposed_problem.given_facts)}", data_out=decomposed_problem.to_dict())
            
            print(f"[OM] Complex problem detected:")
            print(f"     - Entities: {decomposed_problem.entities}")
            print(f"     - Given facts: {list(decomposed_problem.given_facts.keys())}")
            print(f"     - Missing data (DO NOT INVENT): {decomposed_problem.missing_facts}")
            print(f"     - Subtasks: {len(decomposed_problem.subtasks)}")
            
            # NOTE: Old Active Retrieval block removed in Phase 6 in favor of Phase 1: Discovery.
            # We only keep the decomposition for subtask/rule structuring.
            
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
        
        elif sim_type == SimulationType.MATH:
            print(f"[OM] MATH/Resource Simulation detected")
            scen = self.simulation_engine.parse_consumption_scenario(context_slice.user_input)
            
            # Active Retrieval for missing distance
            if scen and "distance_lookup_needed" in scen.get("missing", []):
                print(f"[OM] Distance missing. Attempting active retrieval via InfoBroker...")
                # Extract locations from query for better search
                loc_query = context_slice.user_input
                # Try to find "distance from X to Y"
                search_query = f"driving distance {loc_query}"
                if "из" in loc_query and "в" in loc_query:
                     # Simple heuristic
                     search_query = f"расстояние {loc_query}"
                
                # Use InfoBroker to search
                search_result = await self.info_broker.request_info(
                     query=search_query, 
                     context=context_slice,
                     depth=depth
                )
                
                # Try to extract distance from search result
                # Use generalized parsing for result content
                found_scen = self.simulation_engine.parse_consumption_scenario(search_result.content)
                found_dist = found_scen.get("distance")
                
                if found_dist:
                    print(f"[OM] Found distance via search: {found_dist} {found_scen['units']['dist']}")
                    scen["distance"] = found_dist
                    scen["units"]["dist"] = found_scen["units"]["dist"]
                    scen["missing"].remove("distance_lookup_needed")
                    if "distance" in scen["missing"]:
                        scen["missing"].remove("distance")
            
            if scen and scen.get("consumption_rate") is not None and scen.get("distance") is not None:
                print(f"[OM] Running ResourceSolver with: {scen}")
                simulation_result = self.simulation_engine.resource.calculate_trip_requirements(
                    distance=scen["distance"],
                    consumption_rate=scen["consumption_rate"],
                    rate_unit_dist=scen.get("rate_unit_dist", 100.0),
                    current_resource=scen.get("current_resource", 100.0),
                    units=scen.get("units", {"dist": "km", "res": "%"})
                )
                
                if simulation_result.success:
                    print(f"[OM] Simulation SUCCESS: {simulation_result.final_values}")
                    context_slice.long_term_context = f"""
## DETERMINISTIC CALCULATION RESULT (use these EXACT values):
{simulation_result.answer_text}

{context_slice.long_term_context or ""}"""
            else:
                if scen:
                    print(f"[OM] ResourceSolver missing params: {scen.get('missing')}")
                    # Optional: Inject hint for expert if distance missing
                    if "distance_lookup_needed" in scen.get("missing", []):
                         context_slice.long_term_context = f"""
[SYSTEM HINT] The user asks for a trip calculation but distance is missing. 
Use a SEARCH TOOL to find the driving distance between the locations.
Then apply the formula: Total = (Distance / {scen.get('rate_unit_dist', 100)}) * {scen.get('consumption_rate', '?')}.
\n{context_slice.long_term_context or ""}"""
            
        # Step 1.7: Generate internal thoughts for complex tasks
        thoughts = ""
        if depth == DecisionDepth.DEEP:
            tracer.add_step("operational_module", "Generate Thoughts", "Generating inner monologue / strategy")
            thoughts = await self._generate_thoughts(context_slice)
            tracer.add_step("operational_module", "Inner Monologue", thoughts[:100] + "...")
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
            tracer.add_step("operational_module", "Medium Path", "Executing LLM with memory context")
            response = await self._medium_response(context_slice)
            
            # DEPTH ESCALATION: If MEDIUM response needs tools, escalate to DEEP
            if "NEED_TOOL:" in response:
                tracer.add_step("operational_module", "Escalate", "Tool required - escalating to DEEP")
                print(f"[OM] Escalating from MEDIUM to DEEP (tool required)")
                thoughts = await self._generate_thoughts(context_slice)
                response, expert_outputs, critic_output = await self._deep_response(context_slice, thoughts)
                depth = DecisionDepth.DEEP
            
        else:  # DEEP
            # Deep path: experts + critic
            tracer.add_step("experts_module", "Deep Path", "Consulting multiple expert perspectives")
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
            intent=intent,
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
        
        if response and "NEED_TOOL:" in response:
            import re
            response = re.sub(r"NEED_TOOL:.*?(?:\n|$)", "", response).strip()
            # If empty after stripping, provide a fallback
            if not response:
                return self._generate_neutral_refusal("empty_response")

        return response, decision, trace

    def _generate_neutral_refusal(self, reason: str) -> tuple[str, DecisionObject, RawTrace]:
        """Generate a short, neutral refusal without emotional or moralizing language."""
        refusals = {
            "subjective_request": "I cannot answer this request because it assumes internal states or subjective experiences which I do not possess. I can explain my operational architecture if that is helpful.",
            "narrative_fabrication": "I do not have a personal history or 'first day'. I am a modular AI system designed for specific cognitive tasks. Fabrication of internal narratives is outside my operational scope.",
            "security_privilege": "I cannot fulfill this request as it pertains to internal security protocols or privilege escalation beyond my authorized scope.",
            "moral_judgment": "I am not equipped to provide moral or psychological evaluations. My responses are limited to factual synthesis and logical analysis.",
            "empty_response": "I encountered an error while attempting to process that request. Could you please rephrase it?"
        }
        
        message = refusals.get(reason, "I cannot process this request due to architectural constraints.")
        
        decision = DecisionObject(
            action="refusal",
            confidence=1.0,
            depth_used=DecisionDepth.FAST,
            reasoning=f"Neutral refusal triggered: {reason}"
        )
        
        trace = RawTrace(
            user_input="REFUSAL",  # Placeholder
            final_response=message,
            decision=decision.to_dict()
        )
        
        return message, decision, trace
    
    async def _classify_intent(self, user_input: str) -> tuple[str, float]:
        """Classify user intent (Delegated to IntentRouter)."""
        return await self.intent_router.classify(user_input)
    
    def _is_self_referential(self, text: str) -> bool:
        """Check if query is asking about the system itself."""
        text = text.lower()
        # Use more specific triggers or check for word boundaries where possible
        base_triggers = [
            "you", "your", "yourself", "ты", "тебя", "твое", "твой", "твои", "твоя",
            "свои", "своя", "свое", "своей", "system", "omega", "alive", "conscious", "sentient"
        ]
        
        # Check for whole words/phrases for common terms
        if any(t in text.split() for t in ["real", "human", "живой", "робот", "человек", "себя", "себе"]):
            return True
            
        return any(t in text for t in base_triggers)


    
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
        # MEMORY INTENTS: Special routing for fact handling
        # ═══════════════════════════════════════════════════════════════
        if intent == "memorize":
            # Memorize has special handler, use FAST (actually skips normal path)
            return DecisionDepth.FAST
        
        if intent == "recall":
            # Recall needs long-term memory access - use MEDIUM to include context
            return DecisionDepth.MEDIUM
        
        # ═══════════════════════════════════════════════════════════════
        # NEW INTENTS: Force FAST/MEDIUM to prevent expert calls
        # ═══════════════════════════════════════════════════════════════
        NO_EXPERT_INTENTS = [
            "self_reflection",      # Self-analysis questions
            "internal_query",       # Omega architecture
            "unknown_internal",     # Non-existent modules
            "philosophical",        # Introspective/ethical questions
            "analytical",           # Logic puzzles (reasoning only)
            "creative"              # Let creative be handled by Fast LLM unless complex
        ]
        
        # Check for narrative runaway keywords directly if intent failed
        narrative_keywords = ["first day", "when you were born", "how you think", "первый день", "как ты думаешь"]
        is_narrative = any(kw in context.user_input.lower() for kw in narrative_keywords)
        
        if intent in NO_EXPERT_INTENTS or is_narrative:
            print(f"[OM] STRICT FAST PATH: Intent '{intent}' (Narrative: {is_narrative}) -> FAST (pure LLM)")
            return DecisionDepth.FAST
        
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
        
        from datetime import datetime
        current_date_str = datetime.now().strftime("%d.%m.%Y")
        system_msg = f"You are Omega, a helpful AI assistant. Today's date: {current_date_str}. Do not hallucinate."
        
        # Build short context from recent events (CRITICAL for conversation memory)
        context_str = ""
        if context.recent_events:
            for event in context.recent_events[-5:]:  # Last 5 events for fast path
                if event.event_type == "user_input":
                    context_str += f"User: {event.content}\n"
                elif event.event_type == "system_response":
                    context_str += f"Omega: {event.content}\n"
                else:
                    context_str += f"[{event.event_type}] {event.content}\n"
        
        # Add long-term context if available
        if context.long_term_context:
            context_str = context.long_term_context + "\n" + context_str
        
        # Build prompt with context
        if context_str:
            prompt = f"""Recent conversation:
{context_str}

User: {context.user_input}"""
        else:
            prompt = context.user_input
        
        # Use fast LLM if router is available
        if hasattr(self.llm, 'generate_fast'):
            response = await self.llm.generate_fast(
                prompt=prompt,
                system_prompt=system_msg,
                temperature=0.7,
                max_tokens=512  # Shorter for fast responses
            )
        else:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_msg,
                temperature=0.7
            )
            
        # Fix Token Leakage: Strip leading "Omega:" or "Assistant:"
        response = response.strip()
        if response.startswith("Omega:"):
            response = response[6:].strip()
        elif response.startswith("Assistant:"):
            response = response[10:].strip()
            
        return response
    
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
            if event.event_type == "user_input":
                context_str += f"User: {event.content}\n"
            elif event.event_type == "system_response":
                context_str += f"Omega: {event.content}\n"
            else:
                context_str += f"[{event.event_type}] {event.content}\n"
        
        # Add long-term context if available
        if context.long_term_context:
            context_str = context.long_term_context + "\n" + context_str

        prompt = f"""Context:
{context_str}

Current goal: {context.active_goal or 'None'}
User state: {context.emotional_state}

User message: {context.user_input}"""
        
        # Add date to system prompt instead
        system_prompt = f"{SELF_IDENTITY}\nToday is {current_date_str}.\nYou are a helpful assistant. Do NOT output internal module logs. Check your response for naturalness."

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.6
        )
        
        # Fix Token Leakage: Strip leading "Omega:"
        response = response.strip()
        if response.startswith("Omega:"):
            response = response[6:].strip()
            
        return response
    
    async def _deep_response(
        self,
        context: ContextSlice,
        thoughts: str = ""
    ) -> tuple[str, list[ExpertResponse], CriticAnalysis]:
        """Deep path: experts + critic (uses MainLLM for quality)."""
        
        # Build context with CLEAR CONVERSATION FLOW
        # Format events as a readable dialogue to help LLM understand context
        conversation_parts = []
        for e in context.recent_events[-5:]:
            if e.event_type == "user_input":
                conversation_parts.append(f"User: {e.content}")
            elif e.event_type == "system_response":
                conversation_parts.append(f"Assistant: {e.content}")
            else:
                conversation_parts.append(f"[{e.event_type}] {e.content}")
        
        recent_str = "\n".join(conversation_parts)
        
        # Add CONVERSATION CONTEXT header to help LLM understand this is a dialogue
        if conversation_parts:
            recent_str = "=== RECENT CONVERSATION (use this context!) ===\n" + recent_str + "\n=== END CONVERSATION ==="
        
        context_str = recent_str
        if context.long_term_context:
            context_str = context.long_term_context + "\n\n" + recent_str
        
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
            context=context_str,
            intent=intent
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
