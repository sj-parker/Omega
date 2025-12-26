# Experts and Critic Modules
# Experts = amplifiers, called on demand (not default)

from typing import Optional
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from datetime import datetime
import json
from models.schemas import ExpertResponse, PolicySpace, CriticAnalysis, WorldState
from core.tools import ToolsRegistry, ToolResult
from models.llm_interface import LLMInterface

MAX_REACT_STEPS = 5


EXPERT_PROMPTS = {
    "neutral": """[SYSTEM MESSAGE]
You are OMEGA-DISPATCHER, a non-coding tool interface.
Your ONLY capability is to break down requests into tool calls.

⛔ VISUALIZATION OF ERROR:
User: "Calculate 2+2"
You (WRONG): "print(2+2)"  <-- DO NOT DO THIS. YOU HAVE NO PYTHON INTERPRETER.
You (WRONG): "def add(a,b): return a+b" <-- NO CODING ASSISTANT BEHAVIOR.
You (CORRECT): "NEED_TOOL: calculate linear change..."

✅ CRITICAL PROTOCOL:
1. If you need data (prices, news) -> Write "NEED_TOOL: search <query>"
2. If you need math -> Write "NEED_TOOL: calculate..."
3. DO NOT write Python/JS/Pseudo-code.
4. DO NOT output JSON directly (FunctionGemma will do that).

GOAL: Solve the user's request using ONLY natural language thoughts and `NEED_TOOL:` commands.
After final [OBSERVATION], write: "RESULT: X"
""",

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
    Experts Module with Tool Dispatcher.
    separates reasoning (Main LLM) from tool execution (FunctionGemma).
    """
    
    def __init__(self, llm: LLMInterface, policy: PolicySpace, tool_caller: Optional[LLMInterface] = None):
        self.llm = llm
        self.policy = policy
        self.tool_caller = tool_caller
    
    async def consult_expert(
        self,
        expert_type: str,
        prompt: str,
        world_state: WorldState,
        context: str = ""
    ) -> ExpertResponse:
        """Consult a single expert using a ReAct loop with FunctionGemma dispatch."""
        
        # Modified prompt for FunctionGemma integration
        system_prompt = EXPERT_PROMPTS.get(expert_type, EXPERT_PROMPTS["neutral"])
        
        if self.tool_caller:
           system_prompt += """
           
           TOOL CALLING INSTRUCTION:
           If you need to calculate something, do NOT output JSON.
           Instead, write: "NEED_TOOL: <description of calculation>"
           Example: "NEED_TOOL: calculate linear change for battery from 83.4 with -1.4 rate for 12 mins"
           """
        
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
            
            # --- GUARDRAIL: Detect Python Code Hallucination ---
            # If the model tries to write code blocks or functions, stop it.
            if "```python" in response_text or ("def " in response_text and "return " in response_text):
                print("[Experts] GUARDRAIL TRIGGERED: Python code detected.")
                warning_msg = """
[SYSTEM ERROR]: You wrote Python code.
I do NOT have a Python interpreter. I cannot run this.
STOP writing code.
Use the tool: "NEED_TOOL: <description>"
Try again properly.
"""
                full_response_parts.append(warning_msg)
                history.append(warning_msg)
                continue
            # ----------------------------------------------------

            # 1. Parse for NEED_TOOL intent (for FunctionGemma)
            if self.tool_caller and "NEED_TOOL:" in response_text:
                import re
                match = re.search(r"NEED_TOOL:\s*(.+)", response_text)
                if match:
                    task_desc = match.group(1).strip()
                    print(f"[Experts] Delegating to FunctionGemma: {task_desc}")
                    
                    # FunctionGemma generates the PRECISE JSON
                    tools_def = ToolsRegistry.get_tool_definitions()
                    tool_json = await self.tool_caller.call_tool(task_desc, tools_def)
                    
                    if tool_json:
                        # Execute the generated JSON
                        tool_res = ToolsRegistry.execute_structured_call(json.dumps(tool_json))
                        
                        # Apply state update
                        if tool_res.state_update:
                            local_data.update(tool_res.state_update)
                        
                        observation = f"\n[OBSERVATION]: {tool_res.message}"
                        full_response_parts.append(observation)
                        
                        # Track last observation for verified result extraction
                        last_observation = tool_res.message
                        
                        history.append(f"{observation}\n\n[THOUGHT]: Based on the result, I will...")
                        continue

            # 2. Fallback: Parse legacy JSON (if FunctionGemma not used or Main LLM hallucinates JSON)
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
