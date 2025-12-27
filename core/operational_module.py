# Operational Module (OM)
# Central decision-making module with Decision Depth Controller

from datetime import datetime
from typing import Optional
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
        tool_caller: Optional[LLMInterface] = None
    ):
        self.llm = llm
        self.policy = policy or PolicySpace()
        self.experts = ExpertsModule(llm, self.policy, tool_caller)
        self.critic = CriticModule(llm)
        self.validator = SemanticValidator(llm)
    
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
        intent, confidence = await self._classify_intent(context_slice.user_input)
        depth = self._decide_depth(intent, confidence, context_slice)
        
        # Step 1.5: Handle recall intent (RAG)
        if intent == "recall" and context_manager:
            relevant_facts = context_manager.get_context_for_recall(context_slice.user_input)
            context_slice.long_term_context = relevant_facts
            print(f"[OM] Recall triggered. Long-term context added.")
            
        # Step 1.6: Generate internal thoughts for complex tasks
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
        """Classify user intent and estimate confidence."""
        
        # ═══════════════════════════════════════════════════════════════
        # ONTOLOGY GATE: Block fabrication of non-existent internal modules
        # ═══════════════════════════════════════════════════════════════
        input_lower = user_input.lower()
        
        # Check if query is about internal Omega architecture
        if is_internal_query(user_input):
            entity_name = extract_entity_name(user_input)
            if entity_name and not entity_exists(entity_name):
                print(f"[OM] ONTOLOGY GATE: Entity '{entity_name}' not found in registry -> blocking fabrication")
                return "unknown_internal", 0.99  # High confidence refusal
            print(f"[OM] ONTOLOGY GATE: Valid internal query about '{entity_name or 'Omega'}'")
        
        # Check if search should be blocked
        block_search, reason = should_block_search(user_input)
        if block_search:
            print(f"[OM] SEARCH BLOCKED: Reason='{reason}' -> forcing MEDIUM path (no tools)")
            if reason == "math_expression":
                return "calculation_simple", 0.95  # Simple math, no tools needed
            elif reason == "self_analysis":
                return "self_reflection", 0.90  # Internal reflection, no search
            else:
                return "internal_query", 0.90  # Architecture query, no search
        
        # ═══════════════════════════════════════════════════════════════
        # PRE-LLM KEYWORD CHECK: Force realtime_data for known data queries
        # ═══════════════════════════════════════════════════════════════
        REALTIME_KEYWORDS = [
            "bitcoin", "btc", "ethereum", "crypto", "price", "цена", "стоит", 
            "курс", "погода", "weather", "stock", "акци", "exchange rate",
            "сегодня стоит", "текущ", "актуальн", "current", "live", "real-time"
        ]
        # Require 2+ keyword matches to prevent false positives (e.g. "текущий статус")
        realtime_matches = sum(1 for kw in REALTIME_KEYWORDS if kw in input_lower)
        if realtime_matches >= 2:
            print(f"[OM] KEYWORD OVERRIDE: Detected realtime data request ({realtime_matches} keywords) -> forcing DEEP path")
            return "realtime_data", 0.95

        # REASONING/LOGIC PUZZLES: These should be solved by thinking, not searching
        REASONING_KEYWORDS = [
            # Math word problems
            "сколько", "скільки", "how many", "how much", "посчитай", "порахуй",
            "в два раза", "в три раза", "больше чем", "менше ніж", "більше ніж",
            # Logic puzzles
            "лишний", "лишнее", "зайвий", "odd one out", "какой не подходит",
            "что произойдет", "що станеться", "what will happen", "what happens",
            # Physical reasoning
            "распилили", "розпиляли", "покрасили", "пофарбували", "разрезали",
            "грани", "грані", "кубик", "куб",
            # Riddles
            "загадка", "riddle", "головоломка", "puzzle",
            # Comparisons that need logic
            "физическ", "фізичн", "механическ", "зеркальн", "дзеркальн",
            # Date/time awareness (system knows the date)
            "какой сегодня", "який сьогодні", "what day", "what date", "сегодня день"
        ]
        reasoning_matches = sum(1 for kw in REASONING_KEYWORDS if kw in input_lower)
        # Require 2+ matches to avoid false positives like "честнее" triggering
        if reasoning_matches >= 2:
            print(f"[OM] REASONING OVERRIDE: Detected logic puzzle ({reasoning_matches} keywords) -> using MEDIUM path for pure reasoning")
            return "analytical", 0.85  # MEDIUM path - will use LLM reasoning without tools

        # PHILOSOPHICAL/INTROSPECTIVE: These should use internal knowledge, NOT search
        PHILOSOPHICAL_KEYWORDS = [
            # Self-reflection
            "честн", "honest", "правд", "truth", "ограничен", "limitation",
            "сомнен", "doubt", "уверен", "confiden", "ошиб", "mistake", "error",
            # Ethics/philosophy  
            "этик", "ethic", "мораль", "moral", "парадокс", "paradox",
            "философ", "philosoph", "дилемм", "dilemma",
            # Introspection
            "чувству", "feel", "думаешь", "think", "считаешь", "believe",
            "представь", "imagine", "между нами", "between us"
        ]
        philosophical_matches = sum(1 for kw in PHILOSOPHICAL_KEYWORDS if kw in input_lower)
        if philosophical_matches >= 2:
            print(f"[OM] PHILOSOPHICAL OVERRIDE: Detected introspective question ({philosophical_matches} keywords) -> using FAST path (no search)")
            return "philosophical", 0.90

        # PHYSICS DETECTION: Physical/mechanical scenarios need expert simulation
        PHYSICS_KEYWORDS = [
            "физик", "фізик", "physics", "механик", "механік", "mechanics",
            "gravity", "гравіт", "давлен", "тиск", "pressure",
            "падает", "падає", "falls", "fall", "упадет",
            "горит", "горить", "burns", "flame", "огонь", "вогонь",
            "вакуум", "vacuum", "лестниц", "драбин", "ladder",
            "температур", "temperature", "кипит", "кипить", "boil",
            "высот", "висот", "altitude", "атмосфер", "atmosphere"
        ]
        physics_matches = sum(1 for kw in PHYSICS_KEYWORDS if kw in input_lower)
        if physics_matches >= 1:
            print(f"[OM] PHYSICS OVERRIDE: Detected physics scenario ({physics_matches} keywords) -> forcing DEEP path")
            return "physics", 0.90

        CALC_KEYWORDS = [
            "calculate", "compute", "посчитай", "рассчитай", "math", 
            "linear change", "rate=", "start=", "equation"
        ]
        if any(kw in input_lower for kw in CALC_KEYWORDS):
             print(f"[OM] KEYWORD OVERRIDE: Detected calculation request -> forcing DEEP path")
             return "calculation", 0.95
        
        # Use generate_fast if available to speed up intent classification
        if hasattr(self.llm, 'generate_fast'):
            response = await self.llm.generate_fast(
                prompt=f"User message: {user_input}",
                system_prompt=INTENT_CLASSIFIER_PROMPT,
                temperature=0.2
            )
        else:
            response = await self.llm.generate(
                prompt=f"User message: {user_input}",
                system_prompt=INTENT_CLASSIFIER_PROMPT,
                temperature=0.2
            )
        
        # Parse response
        intent = "factual"  # default
        confidence = 0.5
        
        lines = response.split('\n')
        for line in lines:
            if line.startswith("INTENT:"):
                intent = line.split(":", 1)[1].strip().lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    pass
        
        return intent, confidence

    
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
             # Remove "Result: X" and "Formula: Y" lines to prevent context contamination
             context_str = re.sub(r"(Result:|Formula:).*", "", context_str)
             print(f"[OM] Context Purge Active: Scrubbed calculation artifacts for Search task.")
        
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
