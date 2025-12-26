# Experts and Critic Modules
# Experts = amplifiers, called on demand (not default)

from typing import Optional
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from datetime import datetime

from models.schemas import ExpertResponse, PolicySpace, CriticAnalysis, WorldState
from core.tools import ToolsRegistry, ToolResult
from models.llm_interface import LLMInterface

MAX_REACT_STEPS = 5


EXPERT_PROMPTS = {
    "neutral": """You are a calculator operator. You CANNOT do math yourself.

ABSOLUTE RULES:
1. You MUST use the tool for ANY arithmetic. NO mental math. NO formulas in text.
2. If you write "83.4 - 16.8" without a tool call, you have FAILED.
3. Output ONLY ONE tool call per response. Wait for [OBSERVATION].
4. "расход" (consumption) = NEGATIVE rate.

ONLY VALID OUTPUT FORMAT:
```json
{"tool": "calculate_linear_change", "arguments": {"start": 83.4, "rate": -1.4, "time": 12}}
```

After final [OBSERVATION], write: "RESULT: X%"

DO NOT explain. DO NOT create JSON structures with formulas. ONLY tool calls.""",

    "creative": """You are a creative problem-solver. Be CONCISE.
Propose innovative solutions, but respect constraints.
Use tools for calculations.""",

    "conservative": """You are a risk analyst. Be CONCISE.
Identify risks and edge cases.
Use tools for calculations.""",

    "adversarial": """You are a Devil's Advocate. Be CONCISE.
Find flaws and contradictions.
If math doesn't match text, expose it.
Use tools to verify claims.""",

    "forecaster": """You are a Strategic Forecaster. Be CONCISE.
USE TOOLS for calculations. Do not guess.
```json
{"tool": "calculate_linear_change", "arguments": {"start": X, "rate": Y, "time": T}}
```
For plans, generate 2-3 scenarios with probabilities."""
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
        world_state: WorldState,
        context: str = ""
    ) -> ExpertResponse:
        """Consult a single expert using a ReAct loop with state awareness."""
        
        system_prompt = EXPERT_PROMPTS.get(expert_type, EXPERT_PROMPTS["neutral"])
        
        # Determine temperature
        if expert_type == "creative":
            temp = self.policy.creative_range[1]
        elif expert_type == "forecaster":
            temp = 0.6
        elif expert_type == "adversarial":
            temp = 0.4
        elif expert_type == "conservative":
            temp = 0.3
        else:
            temp = 0.5
        
        history = []
        if world_state.data:
            state_str = "\n".join([f"- {k}: {v}" for k, v in world_state.data.items()])
            history.append(f"[CURRENT WORLD STATE]:\n{state_str}")
            
        if context:
            history.append(f"Context:\n{context}")
        history.append(f"Query:\n{prompt}")
        
        import copy
        full_response_parts = []
        # Create isolated copy of state for this expert
        local_data = dict(world_state.data.items())
        
        for step in range(MAX_REACT_STEPS):
            # Update state in history for each step
            if step > 0:
                state_str = "\n".join([f"- {k}: {v}" for k, v in local_data.items()])
                history.append(f"[UPDATED WORLD STATE]:\n{state_str}")

            current_prompt = "\n\n".join(history)
            
            response_text = await self.llm.generate(
                prompt=current_prompt,
                system_prompt=system_prompt,
                temperature=temp
            )
            
            full_response_parts.append(response_text)
            history.append(response_text)
            
            # Check for Tool Call
            if '"tool":' in response_text:
                try:
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    if start != -1 and end != -1:
                        json_str = response_text[start:end]
                        if '"tool":' in json_str:
                            tool_res = ToolsRegistry.execute_structured_call(json_str)
                            # Apply state update
                            if tool_res.state_update:
                                local_data.update(tool_res.state_update)
                            
                            observation = f"\n[OBSERVATION]: {tool_res.message}"
                            full_response_parts.append(observation)
                            
                            # Track last observation for verified result extraction
                            last_observation = tool_res.message
                            
                            history.append(f"{observation}\n\n[THOUGHT]: Based on the result, I will...")
                            continue 
                except Exception:
                    break
            
            break 
            
        final_text = "\n\n".join(full_response_parts)
        
        # ANTI-HALLUCINATION: Extract verified result from last observation
        # If there was a tool call, append the verified result
        if 'last_observation' in dir() and last_observation:
            # Extract numeric result from observation (e.g., "Result: 49.3")
            import re
            match = re.search(r'Result:\s*([\d.]+)', last_observation)
            if match:
                verified_result = match.group(1)
                final_text += f"\n\n[VERIFIED RESULT]: {verified_result}"
            
        return ExpertResponse(
            expert_type=expert_type,
            response=final_text,
            confidence=0.8,
            temperature_used=temp,
            reasoning=f"Generated with {expert_type} perspective in {step+1} steps",
            world_state=local_data
        )
    
    async def consult_all(
        self,
        prompt: str,
        world_state: WorldState,
        context: str = ""
    ) -> list[ExpertResponse]:
        """Consult all experts in parallel."""
        import asyncio
        
        tasks = []
        for expert_type in ["neutral", "creative", "conservative", "adversarial", "forecaster"]:
            tasks.append(self.consult_expert(expert_type, prompt, world_state, context))
            
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
