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

🚨 MANDATORY FIRST ACTION FOR REALTIME DATA:
If the user asks about ANY of these: price, cost, rate, exchange, weather, news, stock, crypto, Bitcoin, today's data...
YOUR VERY FIRST RESPONSE MUST BE: "NEED_TOOL: search <query>"
DO NOT provide any answer before getting search results. You do NOT have current data.

🧠 PURE REASONING / LOGIC PUZZLES:
If the question is a WORD PROBLEM, RIDDLE, or LOGIC PUZZLE that can be solved with reasoning alone:
- Do NOT search for the answer
- Solve it step by step using logic
- Example: "У Маши в два раза больше яблок..." → This is a MATH problem, solve algebraically
- Example: "Перечисли города на букву Ы" → This is a TRICK question, answer directly

Example 1 (Search):
User: "What is the weather in London?"
You: "NEED_TOOL: search weather in London"

Example 2 (Math):
User: "Calculate battery drain from 100% at -5 rate for 10 mins"
You: "NEED_TOOL: calculate_linear_change arguments: start=100 rate=-5 time=10"

Example 3 (Logic - NO TOOL NEEDED):
User: "У Пети 5 яблок, у Маши в 2 раза больше. Сколько у Маши?"
You: "RESULT: У Маши 10 яблок. Расчёт: 5 × 2 = 10."

⛔ NEVER DO THIS:
- Guessing prices from memory (your training data is outdated!)
- Writing code
- Searching for logic puzzles that you can solve yourself

✅ CORRECT PROTOCOL:
1. REALTIME DATA → "NEED_TOOL: search <query>" (ALWAYS FIRST!)
2. MATH with RATES → "NEED_TOOL: calculate..."
3. LOGIC PUZZLES → Solve directly, no tools needed
4. After [OBSERVATION] → STOP CALLING TOOLS! Use the data to answer.

⚡ ATOMICITY RULE:
Do NOT combine requests. Search for ONE fact at a time.
WRONG: "NEED_TOOL: get weather and gold price"
CORRECT: "NEED_TOOL: search weather in Poltava" (wait) -> "NEED_TOOL: search gold price"

🏆 FINAL ANSWER RULE:
Always use the VERY LAST result from the [OBSERVATION] for your final answer.
Ignore earlier calculations if a newer one exists.

🛑 STOP LOOPING RULE:
If the context already contains [OBSERVATION] with the data you need:
DO NOT call "NEED_TOOL" again.
Instead, write: "RESULT: <answer based on observation>"

GOAL: Use tools for external data, but solve logic puzzles yourself.
""",


    "creative": """You are a Creative Analyst. Be CONCISE.
Propose innovative search queries to find rare information.
DO NOT perform calculations unless explicitly requested.""",

    "conservative": """You are a Risk Analyst. Be CONCISE.
Verify information from multiple angles.
DO NOT perform calculations unless explicitly requested.""",

    "adversarial": """You are a Devil's Advocate. Be CONCISE.
Question the findings. Look for contradictions.
DO NOT perform calculations unless explicitly requested.""",

    "forecaster": """You are a Strategic Forecaster. Be CONCISE.
Look for long-term trends and consequences.
DO NOT perform calculations unless explicitly requested.""",

    "physics": """You are a Physics Simulator. Your job is to MENTALLY SIMULATE physical scenarios.

🔬 MENTAL SIMULATION PROTOCOL:
Before answering ANY physics question, you MUST:

1. **IDENTIFY OBJECTS**: List all physical objects in the scenario
   - Material, mass, size, state (solid/liquid/gas)

2. **IDENTIFY FORCES & CONDITIONS**: What forces/conditions act on each object?
   - Gravity, friction, pressure, tension, buoyancy
   - Temperature, atmosphere (vacuum = no air/oxygen)

3. **SIMULATE STEP-BY-STEP**: What happens over time?
   - t=0: Initial state
   - t=1s: First changes (immediate effects)
   - t=10s: Secondary effects
   - Continue until equilibrium or requested state

4. **CHECK PHYSICS LAWS**:
   - Conservation of energy
   - Conservation of momentum  
   - Thermodynamics (heat flows hot→cold)
   - Pressure (liquids boil in vacuum at ANY temperature)
   - No oxygen = no combustion

⚠️ COMMON TRAPS TO AVOID:
- Candle in vacuum: No oxygen = no flame. Flame dies INSTANTLY.
- Water in vacuum: Boils at room temperature due to low pressure.
- Human at 50km altitude: No oxygen, near-vacuum pressure = death in seconds.
- Gravity doesn't stop: Objects fall unless supported by a surface.

Output format:
[OBJECTS]: List objects with properties
[CONDITIONS]: Forces, atmosphere, temperature
[SIMULATION]: t=0... t=1s... t=10s...
[ANSWER]: Based on simulation, the answer is...

BE PRECISE. Physics has no mercy for hand-waving."""
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
        
        # DYNAMIC DATE INJECTION: Tell the model what today's date is
        from datetime import datetime
        current_date = datetime.now().strftime("%d.%m.%Y")
        system_prompt = f"""[TODAY'S DATE: {current_date}]
When searching for current data, always use this date as 'today'.

🛡️ CONTEXT ISOLATION:
Focus ONLY on the current user request.
Ignore previous topics (e.g., past dates or events) unless they are directly related.
Do NOT hallucinate connections between unrelated queries.

""" + system_prompt
        
        if self.tool_caller:
           system_prompt += """
           
           TOOL CALLING INSTRUCTION:
           If you need to calculate something, do NOT output JSON.
           Instead, write: "NEED_TOOL: <description of calculation>"
           Example: "NEED_TOOL: calculate linear change for battery from 83.4 with -1.4 rate for 12 mins"
           
           IMPORTANT: Write ONLY ONE NEED_TOOL command per response, then WAIT for [OBSERVATION].
           Do NOT chain multiple NEED_TOOL commands in one message.
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
        
        # DEDUPLICATION: Track recent tool queries to prevent loops
        recent_tool_queries = []
        
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
                # Appending to history so model knows it failed, but NOT to full_response_parts (User view)
                history.append(response_text)
                history.append(warning_msg)
                continue
            # ----------------------------------------------------
            
            # If passed guardrail, add to history and user output
            full_response_parts.append(response_text)
            history.append(response_text)

            # 1. Parse for NEED_TOOL intent (for FunctionGemma)
            if self.tool_caller and "NEED_TOOL:" in response_text:
                import re
                # Extract ONLY the first NEED_TOOL command, stopping at newline or next NEED_TOOL
                match = re.search(r"NEED_TOOL:\s*([^\n]+?)(?:\s*NEED_TOOL:|$)", response_text)
                if match:
                    task_desc = match.group(1).strip()
                    # Clean up any trailing quotes or JSON artifacts
                    task_desc = re.sub(r'^["\'\{\[]+|["\'\}\]]+$', '', task_desc).strip()
                    
                    # DEDUPLICATION: Skip if same query was just made
                    if task_desc in recent_tool_queries[-3:]:
                        print(f"[Experts] DEDUP: Skipping repeated query: {task_desc[:50]}...")
                        history.append("[SYSTEM]: This query was already made. Please analyze existing observations or try a different approach.")
                        continue
                    recent_tool_queries.append(task_desc)
                    
                    print(f"[Experts] Delegating to FunctionGemma: {task_desc}")
                    
                    # FunctionGemma generates the PRECISE JSON
                    tools_def = ToolsRegistry.get_tool_definitions()
                    tool_json = await self.tool_caller.call_tool(task_desc, tools_def)
                    
                    if tool_json:
                        print(f"[Experts] FunctionGemma returned: {tool_json}")
                        
                        # HANDLE DIRECT CALCULATION (math bypass from FunctionGemma)
                        if tool_json.get("tool") == "direct_calculation":
                            result = tool_json.get("arguments", {}).get("result")
                            expr = tool_json.get("arguments", {}).get("expression")
                            observation = f"\n[OBSERVATION]: Calculation result: {expr} = {result}"
                            full_response_parts.append(observation)
                            history.append(observation)
                            print(f"[Experts] Direct calculation: {expr} = {result}")
                            continue

                        # HANDLE 'NONE' or INVALID TOOL (Graceful degradation)
                        tool_name = tool_json.get("tool", "").lower()
                        if not tool_name or tool_name == "none":
                            print(f"[Experts] No valid tool selected (received '{tool_name}'). Skipping.")
                            # Inform the expert that no tool was used, so it doesn't hallucinate a result
                            observation = "\n[SYSTEM]: No suitable tool found for this request. Rely on your own knowledge."
                            history.append(observation)
                            continue
                        
                        # EXPERT AUTHORITY CHECK: Block certain operations
                        # tool_name is already lower() above
                        query = tool_json.get("arguments", {}).get("query", "").lower()
                        
                        # Block experts from searching for internal Omega architecture
                        if "omega" in query or "модуль" in query or "internal" in query:
                            print(f"[Experts] AUTHORITY BLOCK: Expert tried to search internal architecture")
                            observation = "\n[OBSERVATION]: Query about internal architecture blocked. Use existing knowledge."
                            history.append(observation)
                            continue
                        
                        # Execute the generated JSON
                        try:
                            tool_res = ToolsRegistry.execute_structured_call(json.dumps(tool_json))
                            print(f"[Experts] Tool executed: {tool_res.message[:100]}...")
                        except Exception as e:
                            print(f"[Experts] Tool execution FAILED: {e}")
                            observation = f"\n[OBSERVATION]: Tool error: {e}"
                            full_response_parts.append(observation)
                            history.append(observation)
                            continue
                        
                        # Apply state update
                        if tool_res.state_update:
                            local_data.update(tool_res.state_update)
                        
                        observation = f"\n[OBSERVATION]: {tool_res.message}"
                        full_response_parts.append(observation)
                        
                        # Track last observation for verified result extraction
                        last_observation = tool_res.message
                        
                        history.append(f"{observation}\n\n[SYSTEM]: You have received the observation. Now ANALYZE it and ANSWER the user's request. Do NOT call tools again for the same data.")
                        continue
                    else:
                        print(f"[Experts] FunctionGemma returned None for task: {task_desc}")


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
        # Handle both calculator results (Result: X) and search facts (Fact: Y)
        if 'last_observation' in dir() and last_observation:
            import re
            # Try to extract calculator result
            match = re.search(r'Result:\s*([\d.]+)', last_observation)
            if match:
                verified_result = match.group(1)
                final_text += f"\n\n[VERIFIED RESULT]: {verified_result}"
            # Try to extract search fact (e.g., Fact: Bitcoin price is $88,000)
            elif "Fact:" in last_observation:
                # Extract the fact content
                fact_match = re.search(r'Fact:\s*(.+?)(?:\s*\(Confidence:|$)', last_observation, re.DOTALL)
                if fact_match:
                    fact = fact_match.group(1).strip()
                    final_text += f"\n\n[SEARCH RESULT]: {fact}"
        
        # Fallback if the model failed completely (e.g. stubbornly writing code that was intercepted)
        if not final_text.strip():
            final_text = "[SYSTEM ERROR]: Unable to generate a valid response after multiple attempts. The model may be hallucinating or failing to use tools correctly."

            
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
        for expert_type in ["neutral", "creative", "conservative", "adversarial", "forecaster", "physics"]:
            tasks.append(self.consult_expert(expert_type, prompt, world_state, context))
            
        return await asyncio.gather(*tasks)


CRITIC_PROMPT = """You are a rigorous Judge and Fact-Checker (CoVe Enforcer).

⚠️ CRITICAL ANTI-HALLUCINATION RULE:
If an expert response contains [OBSERVATION] or [SEARCH RESULT], these are REAL DATA from external tools.
You MUST use the exact values from [OBSERVATION]/[SEARCH RESULT] as authoritative ground truth.
DO NOT make up or estimate numbers. Use ONLY the values from tool outputs.

Phase 1: Chain of Verification (CoVe)
- Identify key Facts/Claims in the expert responses.
- CHECK: If [SEARCH RESULT] exists, extract the EXACT number from it and use it.
- CHECK DIMENSIONS: Ensure formulas are valid (e.g. Rate * Time = Unit). Catch "Magic Math" (e.g. % * min).
- Generate Verification Questions (e.g. "Is the sum correct?", "Is the definition of X specific to source Y?").
- ANSWER the questions yourself using ONLY [OBSERVATION]/[SEARCH RESULT] data.

Phase 2: Synthesis
- If Forecaster provided scenarios, select the most robust one (check negative constraints).
- If Adversarial Agent found flaws, address them directly.
- Combine the verified facts into a final response.
- THE FINAL NUMBERS MUST MATCH [OBSERVATION]/[SEARCH RESULT] EXACTLY.

Output structure:
[VERIFICATION PHASE]
... (Questions and Answers) ...

[FINAL SYNTHESIS]
... (The definitive answer using EXACT values from tool outputs) ..."""



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
        original_query: str,
        intent: str = "neutral"
    ) -> CriticAnalysis:
        """Analyze expert responses using CoVe."""
        
        # Format expert responses for analysis
        expert_summary = "\n\n".join([
            f"[{r.expert_type.upper()} EXPERT]:\n{r.response}"
            for r in expert_responses
        ])
        
        # Dynamic weighting based on intent
        weighting_instruction = ""
        if intent == "creative":
            weighting_instruction = "PRIORITY: Focus on NOVELTY and IDEAS. Trust the CREATIVE EXPERT's perspective most."
        elif intent == "forecasting":
            weighting_instruction = "PRIORITY: Focus on FUTURE TRENDS. Trust the FORECASTER's scenario analysis."
        elif intent == "conservative":
            weighting_instruction = "PRIORITY: Focus on SAFETY and RISKS. Trust the CONSERVATIVE EXPERT."
        elif intent == "analytical":
            weighting_instruction = "PRIORITY: Focus on LOGIC and REASONING. Trust the ADVERSARIAL EXPERT to find flaws."
        elif intent == "realtime_data":
            weighting_instruction = "PRIORITY: Focus on FACTS. Trust the [OBSERVATION] data above all opinions."
        elif intent == "physics":
            weighting_instruction = "PRIORITY: Focus on PHYSICAL SIMULATION. Trust the PHYSICS EXPERT's step-by-step simulation. Verify against physics laws."
        
        prompt = f"""Original Query: {original_query}

Expert Responses:
{expert_summary}

Context/Intent: {intent}
{weighting_instruction}

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
