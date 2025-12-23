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


# Intent classification prompts
INTENT_CLASSIFIER_PROMPT = """Classify the user intent and estimate your confidence.

Categories:
- smalltalk: casual conversation, greetings, simple questions
- factual: requests for information or facts
- analytical: requires reasoning, comparison, or analysis
- creative: requires generation of content, ideas
- complex: multi-step tasks, unclear requirements

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
        policy: Optional[PolicySpace] = None
    ):
        self.llm = llm
        self.policy = policy or PolicySpace()
        self.experts = ExpertsModule(llm, self.policy)
        self.critic = CriticModule(llm)
    
    async def process(
        self,
        context_slice: ContextSlice
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
        
        # Step 2: Generate response based on depth
        expert_outputs = []
        critic_output = None
        
        if depth == DecisionDepth.FAST:
            # Fast path: single LLM call
            response = await self._fast_response(context_slice)
            
        elif depth == DecisionDepth.MEDIUM:
            # Medium path: LLM + memory context
            response = await self._medium_response(context_slice)
            
        else:  # DEEP
            # Deep path: experts + critic
            response, expert_outputs, critic_output = await self._deep_response(context_slice)
        
        # Calculate cost
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Create decision object
        decision = DecisionObject(
            action="respond",
            confidence=confidence,
            depth_used=depth,
            cost={"time_ms": elapsed_ms, "experts_used": len(expert_outputs)},
            policy_snapshot=self.policy.to_dict(),
            reasoning=f"Intent: {intent}, Depth: {depth.value}"
        )
        
        # Create raw trace
        trace = RawTrace(
            user_input=context_slice.user_input,
            context_snapshot=context_slice.to_dict(),
            expert_outputs=[e.to_dict() for e in expert_outputs],
            critic_output=critic_output.to_dict() if critic_output else {},
            decision=decision.to_dict(),
            final_response=response
        )
        
        return response, decision, trace
    
    async def _classify_intent(self, user_input: str) -> tuple[str, float]:
        """Classify user intent and estimate confidence."""
        
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
        
        # New: Check semantic rules first (Maximal approach)
        for rule_trigger, target_depth in self.policy.semantic_rules.items():
            if rule_trigger in intent.lower() or rule_trigger in context.user_input.lower():
                try:
                    # Map string depth to Enum
                    forced_depth = DecisionDepth(target_depth.lower())
                    return forced_depth
                except ValueError:
                    pass
        
        # Fast path conditions
        if intent == "smalltalk" and confidence > 0.7:
            return DecisionDepth.FAST
        
        if confidence > self.policy.fast_path_bias:
            return DecisionDepth.FAST
        
        # Deep path conditions
        if confidence < self.policy.expert_call_threshold:
            return DecisionDepth.DEEP
        
        if intent in ["complex", "analytical"]:
            return DecisionDepth.DEEP
        
        # Low trust users get more scrutiny
        if context.user_identity.trust_level < 0.4:
            return DecisionDepth.DEEP
        
        # Default to medium
        return DecisionDepth.MEDIUM
    
    async def _fast_response(self, context: ContextSlice) -> str:
        """Fast path: single LLM call (uses FastLLM if available)."""
        
        prompt = context.user_input
        
        # Use fast LLM if router is available
        if hasattr(self.llm, 'generate_fast'):
            return await self.llm.generate_fast(
                prompt=prompt,
                temperature=0.7,
                max_tokens=512  # Shorter for fast responses
            )
        
        return await self.llm.generate(
            prompt=prompt,
            temperature=0.7
        )
    
    async def _medium_response(self, context: ContextSlice) -> str:
        """Medium path: LLM + memory context."""
        
        # Build context from recent events
        context_str = ""
        for event in context.recent_events[-5:]:
            context_str += f"[{event.event_type}] {event.content}\n"
        
        prompt = f"""Context:
{context_str}

Current goal: {context.active_goal or 'None'}
User state: {context.emotional_state}

User message: {context.user_input}"""
        
        return await self.llm.generate(
            prompt=prompt,
            temperature=0.6
        )
    
    async def _deep_response(
        self,
        context: ContextSlice
    ) -> tuple[str, list[ExpertResponse], CriticAnalysis]:
        """Deep path: experts + critic (uses MainLLM for quality)."""
        
        # Build context
        context_str = "\n".join([
            f"[{e.event_type}] {e.content}"
            for e in context.recent_events[-5:]
        ])
        
        # Ensure we use main LLM for experts (quality-critical)
        if hasattr(self.llm, 'set_mode'):
            self.llm.set_mode("main")
        
        # Consult all experts
        expert_responses = await self.experts.consult_all(
            prompt=context.user_input,
            context=context_str
        )
        
        # Get critic analysis
        critic_analysis = await self.critic.analyze(
            expert_responses=expert_responses,
            original_query=context.user_input
        )
        
        # Select best response based on critic
        if critic_analysis.recommended_response:
            response = critic_analysis.recommended_response
        else:
            # Default to neutral expert
            response = expert_responses[0].response
        
        return response, expert_responses, critic_analysis
    
    def update_policy(self, updates: dict):
        """Apply policy updates from Homeostasis."""
        self.policy.apply_update(updates)
        # Update experts with new policy
        self.experts.policy = self.policy
