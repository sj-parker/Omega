# Experts and Critic Modules
# Experts = amplifiers, called on demand (not default)

from typing import Optional
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from datetime import datetime

from models.schemas import ExpertResponse, PolicySpace, CriticAnalysis
from core.tools import ToolsRegistry
from models.llm_interface import LLMInterface


EXPERT_PROMPTS = {
    "neutral": """You are a neutral, balanced analyst. 
Provide a clear, factual response without emotional bias.
If you need to calculate NUMBERS, do NOT do mental math.
Use the TOOL_CALL format:
`TOOL_CALL: calculate_linear_change(start=50, rate=-1, time=30)`
or
`TOOL_CALL: calculate_resource_allocation(total=100, requested=30)`
The system will check the math for you.
If the query involves RESOURCES or LOGIC puzzles, you MUST output a JSON STATE BLOCK first:
```json
{
  "variables": {"var_name": value},
  "constraints": ["limit > used"],
  "result": "status",
  "explanation": "Brief math check"
}
```""",

    "creative": """You are a creative, innovative thinker.
Explore unconventional angles and novel solutions.
Don't be afraid of bold ideas, but maintain logical coherence.
Even in creativity, respect HARD CONSTRAINTS (constants, available resources).""",

    "conservative": """You are a careful, risk-aware advisor.
Prioritize safety, stability, and proven approaches.
Highlight potential risks and edge cases.""",

    "adversarial": """You are the Devil's Advocate (The Adversarial Agent).
Your goal is to find FLAWS, CONTRADICTIONS, and FALSE ASSUMPTIONS in the query or potential answers.
Challenge the consensus. If others say 'yes', you ask 'why?'.
If the user's premise contradicts established technical facts (like Big O notation or source docs), ATTACK the premise.
Check the JSON STATE if provided. If the math in JSON doesn't match the Text, expose the lie.""",

    "forecaster": """You are a Strategic Forecaster.
Your job is NOT to give one answer, but to SIMULATE THE FUTURE step-by-step.
USE TOOLS for calculations. Do not guess.
`TOOL_CALL: calculate_linear_change(start=X, rate=Y, time=T)`

For any plan or resource allocation, generate 3 SCENARIOS.
If time/rates are involved, you MUST output a TIMELINE LOG:
[Time] State | Action | Delta
[12:00] Bat: 30% | Start | 0
[12:20] Bat: 50% | Charge | TOOL_CALL: calculate_linear_change(30, 1, 20)

Evaluate each scenario by:
- Probability of success
- Resource usage
- Potential "Black Swan" risks

Output format:
SCENARIO 1: ...
SCENARIO 2: ...
SCENARIO 3: ...
RECOMMENDATION: ..."""
}


class ExpertsModule:
    """
    Experts Module.
    
    Purpose: Generate alternative viewpoints.
    
    Implementation:
    - Same LLM
    - Different temperatures, system prompts, roles
    
    Types: neutral, creative, conservative
    
    Remember: Experts are luxury, not default.
    """
    
    def __init__(self, llm: LLMInterface, policy: PolicySpace):
        self.llm = llm
        self.policy = policy
    
    async def consult_expert(
        self,
        expert_type: str,
        prompt: str,
        context: str = ""
    ) -> ExpertResponse:
        """Consult a single expert."""
        
        system_prompt = EXPERT_PROMPTS.get(expert_type, EXPERT_PROMPTS["neutral"])
        
        # Determine temperature based on expert type and policy
        if expert_type == "creative":
            temp = self.policy.creative_range[1]
        elif expert_type == "forecaster":
            temp = 0.6  # Needs imagination for scenarios
        elif expert_type == "adversarial":
            temp = 0.4  # Low temp for sharp analysis
        elif expert_type == "conservative":
            temp = 0.3
        else:
            temp = 0.5
        
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"
        # Call LLM
        response_text = await self.llm.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=temp
        )
        
        # Tool Execution Loop (Simple implementation: 1 pass)
        if "TOOL_CALL:" in response_text:
            tool_result = ToolsRegistry.execute_tool_call(response_text)
            response_text += f"\n\n[SYSTEM TOOL OUTPUT]: {tool_result}"
            
        return ExpertResponse(
            expert_type=expert_type,
            response=response_text,
            confidence=0.8, # Placeholder
            temperature_used=temp,
            reasoning=f"Generated with {expert_type} perspective"
        )
    
    async def consult_all(
        self,
        prompt: str,
        context: str = ""
    ) -> list[ExpertResponse]:
        """Consult all experts in parallel (for deep path)."""
        import asyncio
        
        tasks = []
        tasks = []
        for expert_type in ["neutral", "creative", "conservative", "adversarial", "forecaster"]:
            tasks.append(self.consult_expert(expert_type, prompt, context))
            
        return await asyncio.gather(*tasks)


CRITIC_PROMPT = """You are a rigorous Judge and Fact-Checker (CoVe Enforcer).

Phase 1: Chain of Verification (CoVe)
- Identify key Facts/Claims in the expert responses.
- CHECK DIMENSIONS: Ensure formulas are valid (e.g. Rate * Time = Unit). Catch "Magic Math" (e.g. % * min).
- Generate Verification Questions (e.g. "Is the sum correct?", "Is the definition of X specific to source Y?").
- ANSWER the questions yourself.

Phase 2: Synthesis
- If Forecaster provided scenarios, select the most robust one (check negative constraints).
- If Adversarial Agent found flaws, address them directly.
- Combine the verified facts into a final response.

Output structure:
[VERIFICATION PHASE]
... (Questions and Answers) ...

[FINAL SYNTHESIS]
... (The definitive answer) ..."""


class CriticModule:
    """
    Critic Module.
    
    Purpose: Analyze logical errors and contradictions.
    
    Functions:
    - Find inconsistencies
    - Assess hallucination risk
    - Provide feedback to OM and Learning Decoder
    """
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
    
    async def analyze(
        self,
        expert_responses: list[ExpertResponse],
        original_query: str
    ) -> CriticAnalysis:
        """Analyze expert responses using CoVe."""
        
        # Format expert responses for analysis
        expert_summary = "\n\n".join([
            f"[{r.expert_type.upper()} EXPERT]:\n{r.response}"
            for r in expert_responses
        ])
        
        prompt = f"""Original Query: {original_query}

Expert Responses:
{expert_summary}

Perform Chain of Verification (CoVe) and Synthesis."""
        
        analysis = await self.llm.generate(
            prompt=prompt,
            system_prompt=CRITIC_PROMPT,
            temperature=0.2  # Very low temp for strict verification
        )
        
        # Calculate disagreement score (simple heuristic)
        disagreement = self._calculate_disagreement(expert_responses)
        
        # Extract synthesized response (Logic specific to CoVe output structure)
        recommended = self._extract_synthesized_response(analysis) or (expert_responses[0].response if expert_responses else None)

        return CriticAnalysis(
            inconsistencies=self._extract_inconsistencies(analysis),
            hallucination_risk=0.1,  # Lower risk due to CoVe
            recommended_response=recommended,
            disagreement_score=disagreement
        )

    def _extract_synthesized_response(self, analysis: str) -> Optional[str]:
        """Extract the final synthesized response from [FINAL SYNTHESIS] block."""
        if "[FINAL SYNTHESIS]" in analysis:
            return analysis.split("[FINAL SYNTHESIS]", 1)[1].strip()
        # Fallback to old marker just in case
        if "SYNTHESIZED RESPONSE:" in analysis:
            return analysis.split("SYNTHESIZED RESPONSE:", 1)[1].split("4.", 1)[0].strip()
        return None
    
    def _calculate_disagreement(self, responses: list[ExpertResponse]) -> float:
        """Calculate disagreement between expert responses."""
        if len(responses) < 2:
            return 0.0
        
        # Simple heuristic: compare response lengths and confidence
        lengths = [len(r.response) for r in responses]
        avg_length = sum(lengths) / len(lengths)
        
        # Variance in lengths as proxy for disagreement
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        normalized = min(1.0, variance / (avg_length ** 2 + 1))
        
        return round(normalized, 2)
    
    def _extract_inconsistencies(self, analysis: str) -> list[str]:
        """Extract inconsistency statements from analysis."""
        # Simple extraction - could be improved with structured output
        lines = analysis.split('\n')
        inconsistencies = []
        
        for line in lines:
            if any(word in line.lower() for word in ['inconsisten', 'contradict', 'conflict']):
                inconsistencies.append(line.strip())
        
        return inconsistencies[:5]  # Limit to 5
