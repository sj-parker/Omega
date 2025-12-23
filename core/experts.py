# Experts and Critic Modules
# Experts = amplifiers, called on demand (not default)

from typing import Optional
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models.schemas import ExpertResponse, CriticAnalysis, PolicySpace
from models.llm_interface import LLMInterface


EXPERT_PROMPTS = {
    "neutral": """You are a neutral, balanced analyst. 
Provide a clear, factual response without emotional bias.
Focus on accuracy and completeness.""",

    "creative": """You are a creative, innovative thinker.
Explore unconventional angles and novel solutions.
Don't be afraid of bold ideas, but maintain logical coherence.""",

    "conservative": """You are a careful, risk-aware advisor.
Prioritize safety, stability, and proven approaches.
Highlight potential risks and edge cases."""
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
            temp = self.policy.creative_range[1]  # Higher temp for creative
        elif expert_type == "conservative":
            temp = 0.3  # Lower temp for conservative
        else:
            temp = 0.5  # Medium for neutral
        
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"
        
        response = await self.llm.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=temp
        )
        
        return ExpertResponse(
            expert_type=expert_type,
            response=response,
            confidence=0.7,  # Will be refined by critic
            temperature_used=temp,
            reasoning=f"Generated with {expert_type} perspective"
        )
    
    async def consult_all(
        self,
        prompt: str,
        context: str = ""
    ) -> list[ExpertResponse]:
        """Consult all experts (for deep path)."""
        
        results = []
        for expert_type in ["neutral", "creative", "conservative"]:
            resp = await self.consult_expert(expert_type, prompt, context)
            results.append(resp)
        
        return results


CRITIC_PROMPT = """You are a critical analyst reviewing responses from multiple experts.

Your job is to:
1. Identify logical inconsistencies
2. Detect potential hallucinations or unfounded claims
3. Assess the level of disagreement between experts
4. Recommend the best response or a synthesis

Be constructive but thorough. Output your analysis in a structured way."""


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
        """Analyze expert responses and provide critique."""
        
        # Format expert responses for analysis
        expert_summary = "\n\n".join([
            f"[{r.expert_type.upper()} EXPERT]:\n{r.response}"
            for r in expert_responses
        ])
        
        prompt = f"""Original Query: {original_query}

Expert Responses:
{expert_summary}

Analyze these responses for:
1. Inconsistencies between experts
2. Potential hallucinations or unfounded claims
3. Which response (or combination) is most reliable
4. Overall disagreement level (0-1 scale)"""
        
        analysis = await self.llm.generate(
            prompt=prompt,
            system_prompt=CRITIC_PROMPT,
            temperature=0.3  # Low temp for objective analysis
        )
        
        # Calculate disagreement score (simple heuristic)
        disagreement = self._calculate_disagreement(expert_responses)
        
        return CriticAnalysis(
            inconsistencies=self._extract_inconsistencies(analysis),
            hallucination_risk=0.2,  # Could be refined with more analysis
            recommended_response=expert_responses[0].response if expert_responses else None,
            disagreement_score=disagreement
        )
    
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
