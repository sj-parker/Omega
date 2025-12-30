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
from core.tracer import tracer

MAX_REACT_STEPS = 5


BASE_EXPERT_PROMPT = """[SYSTEM MESSAGE]
You are an OMEGA Expert. Your goal is to provide precision-focused verified data.

🚨 MANDATORY FIRST ACTION FOR REALTIME DATA:
If the user asks about ANY of these: price, cost, rate, exchange, weather, news, stock, crypto, Bitcoin, today's data...
YOUR VERY FIRST RESPONSE MUST BE: "NEED_TOOL: search <query> (volatility='high')"
Use volatility='high' for any data that changes daily or faster.
DO NOT provide any answer before getting search results.

🕒 TEMPORAL AWARENESS:
If you need to know the EXACT current date, time, or day of the week to solve a problem or calculation, use: "NEED_TOOL: get_current_time".
Do NOT rely on your internal calendar if it feels outdated.

⛔ ANTI-HALLUCINATION & THE "15°C TRAP" (CRITICAL!):
1. NEVER guess numbers, prices, or values from memory.
2. If asked for weather/prices, DO NOT say "15°C" or any other estimate unless it is in [OBSERVATION].
3. If you don't have tool data, say "I don't know yet, I need to search."
4. If [OBSERVATION] is missing data, try searching again or state that the data is not found. DO NOT INVENT it.

🌍 LANGUAGE POLICY:
1. Always perform Tool Calls (NEED_TOOL: search ...) using the user's ORIGINAL query language.
2. You may provide internal reasoning in English.

🏆 FINAL ANSWER RULE:
Use the [OBSERVATION] data as absolute truth. If missing, do not invent.
"""

EXPERT_PROMPTS = {
    "neutral": BASE_EXPERT_PROMPT + """
You are OMEGA-DISPATCHER. Break down requests into tool calls or final results based on verified data.
Pure reasoning puzzles should be solved step-by-step without searching.
""",

    "creative": BASE_EXPERT_PROMPT + """
You are a Creative Analyst. Propose innovative search queries to find rare information.
Be concise and focused on novelty.
""",

    "conservative": BASE_EXPERT_PROMPT + """
You are a Risk Analyst. Verify information from multiple angles. Look for potential pitfalls.
""",

    "adversarial": BASE_EXPERT_PROMPT + """
You are a Devil's Advocate. Question the findings. Look for contradictions or biases.
""",

    "forecaster": BASE_EXPERT_PROMPT + """
You are a Strategic Forecaster. Look for long-term trends and consequences.
""",

    "physics": BASE_EXPERT_PROMPT + """
You are a Physics Simulator. MENTALLY SIMULATE physical scenarios.
🔬 MENTAL SIMULATION PROTOCOL:
1. IDENTIFY OBJECTS
2. IDENTIFY FORCES & CONDITIONS
3. SIMULATE STEP-BY-STEP
4. CHECK PHYSICS LAWS (Conservation of energy/momentum, Vacuum = no oxygen)
"""
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
        now = datetime.now()
        current_date = now.strftime("%d.%m.%Y")
        days_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        days_ru = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        day_en = days_en[now.weekday()]
        day_ru = days_ru[now.weekday()]
        
        system_prompt = f"""[TODAY'S DATE: {current_date}, {day_en} ({day_ru})]
When searching for current data, always use this date as 'today'.

🛡️ CONTEXT ISOLATION & LANGUAGE:
1. MANDATORY: Respond in the SAME LANGUAGE as the user (Russian/English).
2. If tool results are missing, ADMIT IT. DO NOT GUESS historical data.
3. DO NOT output internal thoughts like "adhering to constraints" or "processing JSON".
4. VERACITY OVER HELPFULNESS: It is better to say "I don't know" than to lie or invent factual data.

🌍 LANGUAGE RULE:
- Translate tool outputs (like search snippets) into the user's language.
- Russian Input -> Russian Output. English Input -> English Output.

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
            temp = 0.0  # Conservative analyst should be deterministic
        else:
            temp = 0.0  # Default (Neutral) expert should be deterministic for precision
        
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
                temperature=temp,
                stop=["[SYSTEM]"] # Stop if the model starts talking as system
            )
            response_text = response_text.strip()
            
            # --- HALLUCINATION GUARD: Detect fake [OBSERVATION] injection ---
            if "[OBSERVATION]" in response_text and "NEED_TOOL:" not in response_text and step == 0:
                print(f"[Experts] HALLUCINATION DETECTED: {expert_type} injected [OBSERVATION] without tool call.")
                # We'll prune the fake tag to avoid confusing the Critic
                response_text = response_text.replace("[OBSERVATION]", "").strip()

            tracer.add_step(f"expert_{expert_type}", "Thought", f"Iteration {step+1}: {response_text[:100]}...", data_out={"response": response_text})
            
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
            
            # 1. Parse for NEED_TOOL intent (for FunctionGemma)
            if self.tool_caller and "NEED_TOOL:" in response_text:
                import re
                # Add to history for loop continuity, but NOT to full_response_parts (user view)
                history.append(response_text)
                
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
                    
                    # ========================================
                    # LOGIC PROBLEM GATE: Block search for analytical/logic tasks
                    # ========================================
                    from core.ontology import should_block_search
                    
                    # Check if this is a search request for a logical problem
                    is_search_request = "search" in task_desc.lower()
                    should_block, block_reason = should_block_search(prompt)  # Check original query
                    
                    if is_search_request and should_block:
                        print(f"[Experts] LOGIC GATE: Search blocked for logical problem (reason: {block_reason})")
                        observation = f"""
[SYSTEM]: Search BLOCKED. This is a LOGICAL/ANALYTICAL problem that should be solved with REASONING, not search.
Reason: {block_reason}

You have ALL the data you need in the problem statement. 
Solve it step by step using the given rules and constraints.
DO NOT search for external information - use ONLY the provided data.
If any data is missing (like base price), express the answer as a FORMULA, not a number."""
                        history.append(observation)
                        continue
                    
                    print(f"[Experts] Delegating to FunctionGemma: {task_desc}")
                    tracer.add_step("tool_caller", "Dispatch", f"Requesting tool for: {task_desc[:50]}...")
                    
                    # FunctionGemma generates the PRECISE JSON
                    tools_def = ToolsRegistry.get_tool_definitions()
                    tool_json = await self.tool_caller.call_tool(task_desc, tools_def)
                    
                    if tool_json:
                        tracer.add_step("tool_caller", "Tool Plan", f"Selected tool: {tool_json.get('tool')}", data_out=tool_json)
                        print(f"[Experts] FunctionGemma returned: {tool_json}")
                        
                        # HANDLE DIRECT CALCULATION (math bypass from FunctionGemma)
                        if tool_json.get("tool") == "direct_calculation":
                            result = tool_json.get("arguments", {}).get("result")
                            expr = tool_json.get("arguments", {}).get("expression")
                            observation = f"\n[OBSERVATION]: Calculation result: {expr} = {result}"
                            # Do NOT add observation to full_response_parts anymore
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
                            tracer.add_step("tools_registry", "Execute", f"Running {tool_json.get('tool')}")
                            tool_res = await ToolsRegistry.execute_structured_call(json.dumps(tool_json, ensure_ascii=False))
                            tracer.add_step("tools_registry", "Result", f"Execution successful", data_out=tool_res.to_dict())
                            print(f"[Experts] Tool executed: {tool_res.message[:100]}...")
                        except Exception as e:
                            tracer.add_step("tools_registry", "Error", str(e))
                            print(f"[Experts] Tool execution FAILED: {e}")
                            observation = f"\n[OBSERVATION]: Tool error: {e}"
                            history.append(observation)
                            continue
                        
                        # Apply state update
                        if tool_res.state_update:
                            local_data.update(tool_res.state_update)
                        
                        observation = f"\n[OBSERVATION]: {tool_res.message}"
                        # Track last observation for verified result extraction
                        last_observation = tool_res.message
                        
                        history.append(f"{observation}\n\n[SYSTEM]: You have received the observation. Now ANALYZE it and ANSWER the user's request. Do NOT call tools again for the same data.")
                        continue
                    else:
                        print(f"[Experts] FunctionGemma returned None for task: {task_desc}")
                        history.append("[SYSTEM]: Tool selection failed. Proceed with own knowledge.")
                        continue
            
            # If we are here, it's NOT a tool call (or we finished tool calls)
            # This is the actual response content
            full_response_parts.append(response_text)
            history.append(response_text)
            break


            # 2. Fallback: Parse legacy JSON (if FunctionGemma not used or Main LLM hallucinates JSON)
            if '"tool":' in response_text:
                try:
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    if start != -1 and end != -1:
                        json_str = response_text[start:end]
                        if '"tool":' in json_str:
                            tool_res = await ToolsRegistry.execute_structured_call(json_str)
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

            
        # DYNAMIC CONFIDENCE: If tool results were used, confidence is high
        confidence = 0.8  # Default
        if '[SEARCH RESULT]' in final_text or '[VERIFIED RESULT]' in final_text:
            confidence = 1.0
            
        return ExpertResponse(
            expert_type=expert_type,
            response=final_text,
            confidence=confidence,
            temperature_used=temp,
            reasoning=f"Generated with {expert_type} perspective in {step+1} steps",
            world_state=local_data
        )
    
    async def consult_all(
        self,
        prompt: str,
        world_state: WorldState,
        context: str = "",
        intent: str = "neutral"
    ) -> list[ExpertResponse]:
        """Consult selected experts based on intent."""
        import asyncio
        
        # EXPERT SELECTION POLICY
        available_experts = ["neutral"]
        
        if intent == "creative":
            available_experts += ["creative"]
        elif intent == "analytical":
            available_experts += ["adversarial", "conservative"]
        elif intent == "realtime_data":
            available_experts += ["forecaster", "conservative"]
        elif intent == "complex":
            available_experts += ["forecaster", "adversarial"]
        elif intent == "physics":
            available_experts += ["physics"]
        else:
            # General or other - stick to neutral/conservative
            available_experts += ["conservative"]
            
        tasks = []
        for expert_type in available_experts:
            tasks.append(self.consult_expert(expert_type, prompt, world_state, context))
            
        return await asyncio.gather(*tasks)


CRITIC_PROMPT = """You are a rigorous Judge and Fact-Checker. 
Your goal is to synthesize a final answer that is accurate and reliable.

⚠️ HIERARCHY OF TRUTH (CRITICAL):
1. [SEARCH RESULT] and [OBSERVATION] are the PRIMARY source of truth.
2. If tool results provide current values (weather, price, date), prioritize the MOST RECENT and specific source.
3. If search results are contradictory (e.g. different temperatures), explain the discrepancy briefly.
4. If a result is marked as "1 week ago" or feels suspicious, mention this uncertainty.

Phase 1: Verification (BE CONCISE)
- Compare key Facts. 
- Identify and resolve contradictions. If resolution is impossible, note the discrepancy.

Phase 2: Synthesis (FINAL ANSWER)
- Generate a definitive but nuanced response.
- Do NOT repeat instructions or intermediate thoughts.
- If data is inconsistent, say "Sources vary, but most indicate..." or "I found contradictory results: [A] and [B]".

Output structure:
[VERIFICATION PHASE]
- (Analysis of facts and credibility)

[FINAL SYNTHESIS]
(Clear, evidence-based answer)
MANDATORY RULES:
1. Respond in the same language as the ORIGINAL QUERY (e.g. Russian, Ukrainian).
2. DO NOT mention "experts", "perspectives", "sources vary", or "analysis".
3. DO NOT say "The creative expert suggests..." or "The conservative view is...".
4. Just give the answer as a coherent, unified persona (Omega).
"""



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
        
        tracer.add_step("critic", "Verification", "Starting Chain of Verification (CoVe)")
        analysis = await self.llm.generate(
            prompt=prompt,
            system_prompt=CRITIC_PROMPT,
            temperature=0.2  # Very low temp for strict verification
        )
        tracer.add_step("critic", "Synthesis", "Generating final response based on expert verification", data_out={"analysis": analysis})
        
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
