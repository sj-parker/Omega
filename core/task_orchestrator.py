# Task Orchestrator
# Coordinates TaskQueue-based processing with Context Slicing
# Replaces monolithic OperationalModule.process()

import asyncio
import uuid
import time
from typing import Optional, TYPE_CHECKING
from datetime import datetime

from models.schemas import (
    ContextSlice, DecisionObject, DecisionDepth, PolicySpace,
    RawTrace, ContextScope, WorldState
)
from core.task_queue import Task, TaskQueue, TaskResult, Priority
from core.tracer import tracer

if TYPE_CHECKING:
    from core.context_manager import ContextManager
    from learning.reflection import ReflectionController
    from core.task_decomposer import TaskDecomposer
    from models.llm_interface import LLMInterface


class TaskOrchestrator:
    """
    Task Orchestrator - Coordinates TaskQueue-based processing.
    
    This replaces the monolithic OperationalModule.process() with a
    task-based architecture where:
    1. Intent classification determines task types
    2. Tasks are created with appropriate ContextScope
    3. Workers execute tasks with only necessary context
    4. Results are aggregated into final response
    """
    
    def __init__(
        self,
        llm: 'LLMInterface',
        intent_router: 'IntentRouter',
        info_broker: 'InfoBroker',
        experts: 'ExpertsModule',
        context_manager: 'ContextManager',
        reflection: Optional['ReflectionController'] = None,
        decomposer: Optional['TaskDecomposer'] = None,
        policy: Optional[PolicySpace] = None
    ):
        self.llm = llm
        self.intent_router = intent_router
        self.info_broker = info_broker
        self.experts = experts
        self.context_manager = context_manager
        self.reflection = reflection
        self.decomposer = decomposer
        self.policy = policy or PolicySpace()
        
        # Initialize Critic for expert response selection
        from core.experts import CriticModule
        self.critic = CriticModule(llm)
        
        # Task Queue
        self.task_queue = TaskQueue(max_concurrent=3)
        self._register_handlers()
        
        self._stop_event = asyncio.Event()
        self._worker_tasks: list[asyncio.Task] = []
        self._watchdog_task: Optional[asyncio.Task] = None
        
        # Subscribe to status changes for logging
        self.task_queue.on_status_change(self._log_task_status)
    
    def _register_handlers(self):
        """Register task handlers for different task types."""
        
        self.task_queue.register_handler("llm_fast", self._handle_llm_fast)
        self.task_queue.register_handler("llm_medium", self._handle_llm_medium)
        self.task_queue.register_handler("info_search", self._handle_info_search)
        self.task_queue.register_handler("expert_consult", self._handle_expert_consult)
        self.task_queue.register_handler("synthesis", self._handle_synthesis)
        self.task_queue.register_handler("reflection", self._handle_reflection)
    
    async def start_workers(self, num_workers: int = 2):
        """Start background workers and watchdog."""
        self._stop_event.clear()
        for i in range(num_workers):
            self._worker_tasks.append(asyncio.create_task(self.task_queue.run_worker(self._stop_event)))
            
        # Start Watchdog
        self._watchdog_task = asyncio.create_task(self.task_queue.run_watchdog(self._stop_event))
        print(f"[TaskOrchestrator] Started {num_workers} workers and Watchdog")
    
    async def stop_workers(self):
        """Stop background workers."""
        self._stop_event.set()
        print("[TaskOrchestrator] Workers stopped")
    
    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================
    
    async def process(
        self,
        context_slice: ContextSlice
    ) -> tuple[str, DecisionObject, RawTrace]:
        """
        Process user input using TaskQueue architecture.
        
        Flow:
        1. Classify intent
        2. Create tasks based on intent/depth
        3. Execute tasks (workers inject context)
        4. Aggregate results
        5. Return response
        """
        import time
        start_time = time.time()
        
        # Step 1: Classify intent
        tracer.add_step("task_orchestrator", "Classify", f"Classifying: {context_slice.user_input[:50]}...")
        intent, confidence = await self.intent_router.classify(context_slice.user_input)
        print(f"[TaskOrchestrator] Intent: {intent} ({confidence:.2f})")
        
        # Step 2: Determine task plan
        tasks = await self._create_task_plan(context_slice, intent, confidence)
        print(f"[TaskOrchestrator] Created {len(tasks)} tasks")
        
        # Step 3: Inject context and enqueue tasks
        for task in tasks:
            self.task_queue.inject_context(task, self.context_manager)
            await self.task_queue.enqueue(task)
        
        # Step 4: Wait for all tasks to complete
        results = []
        for task in tasks:
            result = await self.task_queue.wait_for(task.task_id, timeout=30.0)
            if result:
                results.append(result)
                # Propagate steps from worker to main trace
                for step_dict in result.steps:
                    ts = step_dict.get("timestamp")
                    if ts and isinstance(ts, str):
                        try:
                            ts = datetime.fromisoformat(ts)
                        except:
                            ts = None
                    
                    tracer.add_step(
                        module=step_dict.get("module", "worker"),
                        name=step_dict.get("name", "Task"),
                        description=step_dict.get("description", ""),
                        data_in=step_dict.get("data_in"),
                        data_out=step_dict.get("data_out"),
                        timestamp=ts
                    )
        
        # Step 5: Aggregate results
        response = await self._aggregate_results(results, context_slice)
        
        # Create decision object
        elapsed_ms = int((time.time() - start_time) * 1000)
        depth = self._determine_depth(intent, confidence)
        
        # Override depth based on actual complexity
        if len(tasks) > 3 or any("expert" in t.task_type for t in tasks):
            depth = DecisionDepth.DEEP
        elif len(tasks) > 1:
            depth = DecisionDepth.MEDIUM
        
        expert_count = len([t for t in tasks if "expert" in t.task_type or t.task_type == "expert_consult"])
        decision = DecisionObject(
            action="respond",
            confidence=confidence,
            depth_used=depth,
            cost={
                "time_ms": elapsed_ms, 
                "tasks_used": len(tasks),
                "experts_used": expert_count
            },
            policy_snapshot=self.policy.to_dict(),
            intent=intent,
            reasoning=f"Intent: {intent}, Tasks: {len(tasks)}"
        )
        
        # Collect expert outputs for trace
        expert_outputs = []
        for r in results:
            if r.source_module and "expert" in r.source_module and isinstance(r.data, dict):
                expert_outputs.append(r.data)

        # Create trace
        trace = RawTrace(
            user_input=context_slice.user_input,
            context_snapshot=context_slice.to_dict(),
            expert_outputs=expert_outputs,
            decision=decision.to_dict(),
            final_response=response
        )
        
        return response, decision, trace
    
    # ============================================================
    # TASK PLANNING
    # ============================================================
    
    async def _create_task_plan(
        self,
        context: ContextSlice,
        intent: str,
        confidence: float
    ) -> list[Task]:
        """
        Create task plan based on intent and confidence.
        
        Key insight: Each task specifies its context_scope,
        so tasks get only the context they need.
        """
        # Dynamic decomposition for complex problems
        if self.decomposer and self.decomposer.is_complex_problem(context.user_input):
            print(f"[TaskOrchestrator] Using dynamic decomposition for: {intent}")
            return await self._plan_dynamic_tasks(context, intent)

        tasks = []
        
        # Fast path intents
        FAST_INTENTS = ["smalltalk", "confirmation", "self_reflection"]
        
        # Intents requiring search
        SEARCH_INTENTS = ["factual"]  # realtime_data moved to EXPERT_INTENTS
        
        # Intents requiring expert consensus (includes realtime_data for tool usage)
        EXPERT_INTENTS = ["complex", "analytical", "creative", "realtime_data", "calculation", "philosophical", "physics", "memorize"]
        
        if intent in FAST_INTENTS:
            # Single fast LLM task with recent context
            tasks.append(Task(
                task_type="llm_fast",
                payload={"query": context.user_input},
                context_scope=ContextScope.RECENT,
                priority=Priority.HIGH,
                source_module="orchestrator",
                target_module="llm"
            ))
        
        elif intent in SEARCH_INTENTS:
            # Search task (no context needed)
            search_task = Task(
                task_type="info_search",
                payload={"query": context.user_input},
                context_scope=ContextScope.NONE,  # Search doesn't need history
                priority=Priority.HIGH,
                source_module="orchestrator",
                target_module="info_broker"
            )
            tasks.append(search_task)
            
            # Synthesis task (needs search result + recent context)
            tasks.append(Task(
                task_type="llm_medium",
                payload={"query": context.user_input},
                context_scope=ContextScope.RECENT,
                priority=Priority.MEDIUM,
                source_module="orchestrator",
                target_module="llm",
                depends_on=[search_task.task_id]  # Wait for search
            ))
        
        elif intent in EXPERT_INTENTS or confidence < 0.4:
            # Expert consultation tasks (each needs full context)
            for expert_type in ["neutral", "creative", "conservative"]:
                tasks.append(Task(
                    task_type="expert_consult",
                    payload={
                        "expert": expert_type, 
                        "query": context.user_input,
                        "world_state": context.world_state  # Pass world_state
                    },
                    context_scope=ContextScope.FULL,  # Experts need full context
                    priority=Priority.MEDIUM,
                    source_module="orchestrator",
                    target_module=f"expert_{expert_type}"
                ))
        
        elif intent == "recall":
            # Memory recall (needs ALL facts and history)
            tasks.append(Task(
                task_type="llm_medium",
                payload={"query": context.user_input},
                context_scope=ContextScope.FULL,  # Upgraded to FULL for deep memory
                context_filter=context.user_input, 
                priority=Priority.HIGH,
                source_module="orchestrator",
                target_module="llm"
            ))
        
        else:
            # Default: medium path
            tasks.append(Task(
                task_type="llm_medium",
                payload={"query": context.user_input},
                context_scope=ContextScope.RECENT,
                priority=Priority.MEDIUM,
                source_module="orchestrator",
                target_module="llm"
            ))
        
        return tasks

    async def _plan_dynamic_tasks(self, context: ContextSlice, intent: str) -> list[Task]:
        """Use TaskDecomposer to create a dynamic plan with multi-intent support."""
        from core.task_decomposer import SubtaskType
        
        # 1. Split query into independent intents ONLY if it's not a single complex goal
        # If it's realtime_data or analytical, keep it together for TaskDecomposer to handle dependencies
        if intent in ["realtime_data", "complex", "analytical", "physics", "calculation", "factual"]:
            sub_queries = [context.user_input]
        else:
            sub_queries = await self.decomposer.split_query(context.user_input)
            
        tasks = []
        all_subtask_ids = [] # Global track for synthesis dependencies
        
        for q_idx, query in enumerate(sub_queries):
            problem = await self.decomposer.decompose(query)
            task_id_map = {} # Subtask ID -> Task ID
            
            for subtask in problem.subtasks:
                # Generate a unique task ID
                new_task_id = f"dyn_{uuid.uuid4().hex[:8]}"
                task_id_map[subtask.task_id] = new_task_id
                
                # Map depends_on using the map
                depends_on = [task_id_map[dep_id] for dep_id in subtask.depends_on if dep_id in task_id_map]
                
                # Determine task type and context scope
                if subtask.task_type == SubtaskType.LOOKUP:
                    task_type = "info_search"
                    scope = ContextScope.NONE
                    target = "info_broker"
                elif subtask.task_type == SubtaskType.CALCULATE:
                    task_type = "llm_medium"
                    scope = ContextScope.RECENT
                    target = "llm"
                    # Injected instruction for calc
                    subtask.description = f"ARITHMETIC_TASK: {subtask.description}. MANDATORY: Show your work. Convert all units to base (e.g. BTC to USD) before final division."
                else:
                    task_type = "llm_medium"
                    scope = ContextScope.RECENT
                    target = "llm"
                    
                # Prepare payload with structured problem data
                payload = {
                    "query": f"Sub-query: {query}\nTask: {subtask.description}",
                    "subtask_id": subtask.task_id,
                    "subtask_type": subtask.task_type.value,
                    "original_query": context.user_input,
                    "description": f"Intent [{query[:30]}...]: {subtask.description}",
                    "is_dynamic": True,
                    "volatility": "high" if getattr(subtask, 'is_volatile', False) else "low",
                    "problem_data": {
                        "entities": problem.entities,
                        "given_facts": problem.given_facts,
                        "missing_facts": problem.missing_facts,
                        "rules": problem.rules
                    }
                }
                
                # Create the task
                tasks.append(Task(
                    task_id=new_task_id,
                    task_type=task_type,
                    payload=payload,
                    context_scope=scope,
                    priority=Priority.MEDIUM,
                    depends_on=depends_on,
                    source_module="decomposer",
                    target_module=target
                ))
                all_subtask_ids.append(new_task_id)
            
        # 3. Append Synthesis Node (depends on ALL subtasks from ALL intents)
        synthesis_id = f"syn_{uuid.uuid4().hex[:8]}"
        tasks.append(Task(
            task_id=synthesis_id,
            task_type="synthesis",
            payload={
                "original_query": context.user_input,
                "goal": "Synthesize multiple intents into a cohesive response",
                "subtask_count": len(tasks)
            },
            context_scope=ContextScope.RECENT,
            priority=Priority.MEDIUM,
            depends_on=all_subtask_ids,
            source_module="orchestrator",
            target_module="llm"
        ))
            
        print(f"[TaskOrchestrator] Decomposed into {len(tasks)} tasks across {len(sub_queries)} intents")
        return tasks
    
    def _determine_depth(self, intent: str, confidence: float) -> DecisionDepth:
        """Map intent + confidence to decision depth."""
        if intent in ["smalltalk", "confirmation"] or confidence > 0.85:
            return DecisionDepth.FAST
        elif intent in ["complex", "analytical"] or confidence < 0.4:
            return DecisionDepth.DEEP
        else:
            return DecisionDepth.MEDIUM
    
    # ============================================================
    # TASK HANDLERS
    # ============================================================
    
    async def _handle_llm_fast(self, task: Task) -> TaskResult:
        """Handle fast LLM response."""
        with tracer.capture_steps() as steps:
            query = task.payload.get("query", "")
            context = task.injected_context or {}
            
            # Build prompt with injected context
            context_str = ""
            if context.get("recent_events"):
                # Increased from 3 to 10 for better continuity
                for event in context["recent_events"][-10:]:
                    context_str += f"[{event.get('event_type', 'unknown')}] {event.get('content', '')}\n"
            
            prompt = f"{context_str}\nUser: {query}" if context_str else query
            
            from datetime import datetime
            current_date_str = datetime.now().strftime("%d.%m.%Y")
            
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=f"You are Omega, a helpful AI assistant. Today's date: {current_date_str}. MANDATORY: Respond in the SAME LANGUAGE as the user.",
                temperature=0.4,
                max_tokens=512
            )
            
            tracer.add_step(
                module="llm_fast",
                name="Generate",
                description="Fast LLM response",
                data_in={"prompt": prompt},
                data_out={"response": response}
            )
            
            # Safety check for immediate repetition loop
            try:
                last_response = next((e.content for e in reversed(context.get("recent_events", [])) if e.get("event_type") == "system_response"), None)
                if last_response and response.strip() == last_response.strip():
                     print(f"[TaskOrchestrator] Repetition loop detected! Modifying response.")
                     response += " (As I mentioned previously)"
            except Exception as e:
                print(f"[TaskOrchestrator] Repetition check failed: {e}")

            return TaskResult(
                task_id=task.task_id,
                success=True,
                data=response,
                source_module="llm",
                steps=[s.to_dict() for s in steps]
            )
    
    async def _handle_llm_medium(self, task: Task) -> TaskResult:
        """Handle medium LLM response with context."""
        with tracer.capture_steps() as steps:
            query = task.payload.get("query", "")
            context = task.injected_context or {}
            
            # Build richer context
            context_parts = []
            
            if context.get("relevant_facts"):
                facts_str = "\n".join([f"- {f.get('content', '')}" for f in context["relevant_facts"]])
                context_parts.append(f"Relevant facts:\n{facts_str}")
            
            if context.get("recent_events"):
                events_str = "\n".join([f"[{e.get('event_type')}] {e.get('content')}" for e in context["recent_events"]])
                context_parts.append(f"Recent conversation:\n{events_str}")
            
            if context.get("all_events"):
                # Handle FULL scope history
                all_events_str = "\n".join([f"[{e.get('event_type')}] {e.get('content')}" for e in context["all_events"][-50:]])
                context_parts.append(f"FULL Conversation History (Last 50 events):\n{all_events_str}")
            
            # Add structured problem data if available
            prob_data = task.payload.get("problem_data", {})
            if prob_data:
                prob_parts = []
                if prob_data.get("entities"):
                    prob_parts.append(f"Entities: {', '.join(prob_data['entities'])}")
                if prob_data.get("given_facts"):
                    facts = [f"- {k}: {v}" for k, v in prob_data["given_facts"].items()]
                    prob_parts.append(f"Given facts:\n" + "\n".join(facts))
                if prob_data.get("rules"):
                    rules = [f"- {r}" for r in prob_data["rules"]]
                    prob_parts.append(f"Rules:\n" + "\n".join(rules))
                
                if prob_parts:
                    context_parts.append("STRUCTURED PROBLEM DATA:\n" + "\n\n".join(prob_parts))

            # Specialized instruction for arithmetic/reasoning logic
            if task.payload.get("subtask_type") in ["calculate", "reason"] or prob_data.get("given_facts"):
                context_parts.append(
                    "\n!!! CRITICAL EXECUTION RULES !!!\n"
                    "1. SOURCE OF TRUTH: Use ONLY data provided in 'DATA FROM PREVIOUS STEPS' or 'STRUCTURED PROBLEM DATA'.\n"
                    "2. DATA VERIFICATION: Check if all variables required (e.g., BTC/USD price, budget) are in the provided context.\n"
                    "3. MISSING DATA ERROR: If a required value (like search result) is missing, respond ONLY with: 'MISSING_DATA: [Name of missing fact]'. DO NOT estimate or hallucinate numbers!\n"
                    "4. UNIT CONVERSION: You MUST convert all units to a common base (e.g. BTC to USD) using the price from search. Never divide Budget(USD) by Cost(BTC) directly.\n"
                    "5. SHOW WORK: Outline your reasoning and conversion steps clearly."
                )

            # Add dependency results from TaskQueue injection
            dep_results = task.payload.get("results", [])
            if dep_results:
                res_str = "\n".join([f"Result from previous step: {r}" for r in dep_results])
                context_parts.append(f"DATA FROM PREVIOUS STEPS (Verified):\n{res_str}")
            
            full_context = "\n\n".join(context_parts)
            prompt = f"{full_context}\n\nUser: {query}" if full_context else query
            
            from datetime import datetime
            current_date_str = datetime.now().strftime("%d.%m.%Y")
            
            system_prompt = f"You are Omega. Use context to answer. Today's date: {current_date_str}. MANDATORY: Respond in the SAME LANGUAGE as the user."
            
            # Anti-hallucination for dynamic subtasks
            if task.payload.get("is_dynamic"):
                system_prompt += "\nCRITICAL: Use ONLY provided context. If data is missing (e.g. current price, distance, facts), respond ONLY with: 'MISSING_DATA: [Name of fact]' and STOP. DO NOT estimate, assume, or hallucinate numbers even for 'example' purposes."
                if task.payload.get("subtask_type") == "calculate":
                    system_prompt += "\nARITHMETIC: Show all conversion steps. Check dimensional units (USD vs BTC)."

            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1 # Lowered for more facts
            )
            
            tracer.add_step(
                module="llm_medium",
                name="Generate",
                description="Medium LLM response with context",
                data_in={"prompt": prompt},
                data_out={"response": response}
            )
            
            # Safety check for immediate repetition loop
            try:
                last_response = next((e.content for e in reversed(context.get("recent_events", [])) if e.get("event_type") == "system_response"), None)
                if last_response and response.strip() == last_response.strip():
                     print(f"[TaskOrchestrator] Repetition loop detected! Modifying response.")
                     response += " (As I mentioned previously)"
            except Exception as e:
                print(f"[TaskOrchestrator] Repetition check failed: {e}")

            return TaskResult(
                task_id=task.task_id,
                success=True,
                data=response,
                source_module="llm",
                steps=[s.to_dict() for s in steps]
            )
    
    async def _handle_info_search(self, task: Task) -> TaskResult:
        """Handle information search (no context needed)."""
        from core.info_broker import InfoSource
        
        with tracer.capture_steps() as steps:
            query = task.payload.get("query", "")
            volatility = task.payload.get("volatility", "low")
            
            # For dynamic lookup tasks, RESTRICT sources to exclude EXPERT
            # We don't want experts hallucinating real-time prices if search fails.
            sources = [InfoSource.CACHE, InfoSource.SPECIALIST, InfoSource.MEMORY, InfoSource.SEARCH]
            
            tracer.add_step("info_broker", "Search Request", f"Searching for: {query} (volatility={volatility})")
            result = await self.info_broker.request_info(query=query, sources=sources, volatility=volatility)
            
            status = "Found" if result.is_sufficient else "Not Found"
            tracer.add_step("info_broker", "Search Result", f"{status} via {result.source.value} (conf={result.confidence:.2f})")
            
            return TaskResult(
                task_id=task.task_id,
                success=result.is_sufficient if result else False,
                data=result.data if (result and result.is_sufficient) else None, # ONLY pass data if sufficient
                source_module="info_broker",
                steps=[s.to_dict() for s in steps]
            )
    
    async def _handle_expert_consult(self, task: Task) -> TaskResult:
        """Handle expert consultation."""
        with tracer.capture_steps() as steps:
            expert_type = task.payload.get("expert", "neutral")
            query = task.payload.get("query", "")
            world_state = task.payload.get("world_state") or WorldState()
            context = task.injected_context or {}
            
            # Build context string from full context
            from datetime import datetime
            current_date_str = datetime.now().strftime("%d.%m.%Y")
            context_str = f"Today's date: {current_date_str}\n"
            
            if context.get("all_events"):
                # Increased from 5 to 15 for experts
                for event in context["all_events"][-15:]:
                    context_str += f"[{event.get('event_type')}] {event.get('content')}\n"
            
            # Call appropriate expert with world_state
            response = await self.experts.consult_expert(
                expert_type=expert_type,
                prompt=query,
                world_state=world_state,
                context=context_str
            )
            
            tracer.add_step(
                module=f"expert_{expert_type}",
                name="Consult",
                description=f"Consulting {expert_type} expert",
                data_in={"query": query},
                data_out={"response": response.response}
            )
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                data={"expert": expert_type, "response": response.response if hasattr(response, 'response') else str(response)},
                source_module=f"expert_{expert_type}",
                steps=[s.to_dict() for s in steps]
            )
    
    async def _handle_synthesis(self, task: Task) -> TaskResult:
        """Synthesize multiple subtask results into a final coherent answer."""
        results = task.payload.get("results", [])
        original_query = task.payload.get("original_query", "")
        
        # Build context from previous results
        results_str = ""
        for idx, res in enumerate(results):
            if res:
                results_str += f"\n--- STEP {idx+1} ---\n{res}\n"
        
        if not results_str:
            return TaskResult(task_id=task.task_id, success=False, data="Missing subtask data", source_module="synthesis")

        # LLM Synthesis Call
        from datetime import datetime
        current_date_str = datetime.now().strftime("%d.%m.%Y")
        
        prompt = f"""Synthesize a final answer to the original user query based on the sub-steps taken below.
        
ORIGNAL QUERY: {original_query}

SUB-STEPS TAKEN:
{results_str}

## INSTRUCTIONS
1. If steps are missing or contradictory, explain the discrepancy briefly.
2. If data like price or facts were found, CITE the source provided in the steps.
3. Ensure the final answer is COHERENT and directly answers the user's question.
4. MANDATORY: Respond in the SAME LANGUAGE as the user (Russian/Ukrainian).
"""
        
        response = await self.llm.generate(
            prompt=prompt,
            system_prompt = (
            f"You are Omega. Today is {current_date_str}. Synthesize a final, accurate answer in the SAME LANGUAGE as the original query.\n"
            "CRITICAL:\n"
            "1. CITE search results (source names).\n"
            "2. IF DATA IS MISSING (or any previous step reported MISSING_DATA), DO NOT GIVE A CALCULATED ANSWER. Instead, clearly explain what fact is missing and why it's needed.\n"
            "3. DO NOT assume or guess distances, prices, or rates if they were not found in search results.\n"
            "4. If multiple steps differ, explain why (e.g. 'Search found X, but expert thought Y')."
        ),
            temperature=0.3
        )
        
        return TaskResult(
            task_id=task.task_id,
            success=True,
            data=response,
            source_module="synthesis"
        )
    
    # ============================================================
    # RESULT AGGREGATION
    # ============================================================
    
    async def _aggregate_results(
        self,
        results: list[TaskResult],
        context: ContextSlice
    ) -> str:
        """
        Aggregate task results into final response.
        
        For single task: return result directly
        For multiple tasks: use Critic for selection
        """
        if not results:
            return "I apologize, but I couldn't process your request."
        
        # Single result - return directly
        if len(results) == 1:
            return str(results[0].data) if results[0].data else ""
        
        # Multiple results - Check for Synthesis first
        synthesis_result = next((r for r in results if r.source_module == "synthesis"), None)
        if synthesis_result:
            return str(synthesis_result.data)
            
        # Multiple results - expert consensus with Critic
        expert_responses = [r for r in results if "expert" in r.source_module]
        if expert_responses:
            return await self._critic_select(expert_responses, context)
        
        # Search + LLM combination
        search_results = [r for r in results if r.source_module == "info_broker"]
        llm_results = [r for r in results if r.source_module == "llm"]
        
        if search_results and llm_results:
            # LLM should have used search results
            return str(llm_results[0].data) if llm_results[0].data else ""
        
        # Fallback: return first successful result
        for r in results:
            if r.success and r.data:
                return str(r.data)
        
        return "I couldn't generate a complete response."
    
    async def _critic_select(
        self,
        expert_results: list[TaskResult],
        context: ContextSlice
    ) -> str:
        """
        Use CriticModule to select best expert response.
        
        Converts TaskResults to ExpertResponses and runs through Critic.
        """
        from models.schemas import ExpertResponse
        
        # Convert TaskResults to ExpertResponses for CriticModule
        expert_responses = []
        for result in expert_results:
            if result.success and result.data:
                data = result.data
                expert_type = data.get("expert", "unknown") if isinstance(data, dict) else "unknown"
                response_text = data.get("response", str(data)) if isinstance(data, dict) else str(data)
                
                expert_responses.append(ExpertResponse(
                    expert_type=expert_type,
                    response=response_text,
                    confidence=0.8,  # Default confidence
                    temperature_used=0.5
                ))
        
        if not expert_responses:
            return "No valid expert responses to analyze."
        
        # If only one expert, return directly
        if len(expert_responses) == 1:
            return expert_responses[0].response
        
        # Run through Critic for verification and synthesis
        tracer.add_step("critic", "Analyze", f"Analyzing {len(expert_responses)} expert responses")
        critic_analysis = await self.critic.analyze(
            expert_responses=expert_responses,
            original_query=context.user_input,
            intent="expert_synthesis"
        )
        tracer.add_step("critic", "Result", f"Selected response (disagreement: {critic_analysis.disagreement_score})")
        
        # Return Critic's recommended response or first expert's response
        if critic_analysis.recommended_response:
            return critic_analysis.recommended_response
        
        return expert_responses[0].response

    async def _handle_reflection(self, task: Task) -> TaskResult:
        """Handle background reflection task."""
        if not self.reflection:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error="ReflectionController not initialized"
            )
            
        tracer.add_step("reflection", "Start", "Starting background reflection task")
        pattern = await self.reflection.reflect_once()
        
        return TaskResult(
            task_id=task.task_id,
            success=True,
            data={
                "pattern_found": pattern is not None,
                "description": pattern.description if pattern else "No patterns found"
            },
            source_module="reflection"
        )

    def _log_task_status(self, task: Task):
        """Log intermediate task status changes."""
        if task.status == "running":
            print(f"[*] Modeling: {task.task_type} (task_id={task.task_id[:8]})")
        elif task.status == "completed":
            pass # Already logged by TaskQueue
            
    async def _handle_synthesis(self, task: Task) -> TaskResult:
        """
        Final synthesis task that combines results of all subtasks.
        """
        with tracer.capture_steps() as steps:
            original_query = task.payload.get("original_query", "")
            goal = task.payload.get("goal", "")
            
            # 1. Collect results of all dependencies
            subtask_results = []
            for dep_id in task.depends_on:
                res = self.task_queue._completed.get(dep_id)
                if res and res.success:
                    # Try to get meaningful description from payload or result
                    desc = "Subtask"
                    dep_task = self.task_queue._tasks.get(dep_id)
                    if dep_task:
                        desc = dep_task.payload.get("description", dep_task.task_type)
                    
                    subtask_results.append(f"### {desc}\nResult: {res.data}")
            
            if not subtask_results:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error="No successful subtask results to synthesize",
                    steps=[s.to_dict() for s in steps]
                )
                
            # 2. Build synthesis prompt
            results_str = "\n\n".join(subtask_results)
            system_prompt = f"""You are a Response Synthesizer. 
Your goal is to provide a unified, coherent answer to the user's original query.
Combine the findings from multiple subtasks into a logical flow.

MANDATORY: Respond in the SAME LANGUAGE as the original query (Russian/English/etc).

Original Query: {original_query}
Goal: {goal}
"""
            prompt = f"Subtask Results:\n\n{results_str}\n\nFinal Synthesis (Answer the original query):"
            
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                data=response,
                source_module="synthesis",
                steps=[s.to_dict() for s in steps]
            )
