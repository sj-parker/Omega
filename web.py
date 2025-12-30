# Web Interface for Cognitive LLM System
# FastAPI backend with REST API

import asyncio
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import time

import sys
import io

# Force UTF-8 for stdout/stderr (Windows fix)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

from models.schemas import PolicySpace, WorldState, ContextScope
from models.llm_interface import MockLLM, OllamaLLM, LLMRouter, FunctionGemmaLLM
from core.gatekeeper import Gatekeeper, UserHistoryStore
from core.context_manager import ContextManager
from core.operational_module import OperationalModule
from core.orchestrator import Orchestrator, ModuleInterface, ModuleCapability
from core.sanitizer import ResponseSanitizer
from core.fallback_generator import FallbackGenerator, UncertaintyLevel
from core.info_broker import InfoBroker
from core.tracer import tracer
from core.task_queue import Task, Priority
from core.task_orchestrator import TaskOrchestrator
from core.intent_router import IntentRouter
from learning.learning_decoder import LearningDecoder
from core.task_decomposer import TaskDecomposer
from learning.homeostasis import HomeostasisController
from learning.reflection import ReflectionController
from core.config import config


# Global system instance
system = None


class CognitiveSystemWeb:
    """Web-enabled cognitive system."""
    
    def __init__(
        self,
        use_ollama: bool = None,
        main_model: str = None,
        fast_model: str = None,
        use_multi_model: bool = None,
        use_task_orchestrator: bool = True  # NEW: Toggle new architecture
    ):
        # Load from config or use provided values
        self.use_ollama = use_ollama if use_ollama is not None else config.get("models.use_ollama")
        self.main_model = main_model if main_model is not None else config.get("models.main")
        self.fast_model = fast_model if fast_model is not None else config.get("models.fast")
        self.use_multi_model = use_multi_model if use_multi_model is not None else config.get("models.use_multi_model")
        self.use_task_orchestrator = use_task_orchestrator  # NEW

        # Initialize LLM(s)
        self.tool_caller = None
        
        if self.use_ollama:
            if self.use_multi_model:
                fast_llm = OllamaLLM(model=self.fast_model)
                main_llm = OllamaLLM(model=self.main_model)
                self.llm = LLMRouter(fast_llm=fast_llm, main_llm=main_llm)
            else:
                self.llm = OllamaLLM(model=self.main_model)
            
            # Initialize Tool Caller (Qwen2.5)
            try:
                self.tool_caller = FunctionGemmaLLM() # Uses Qwen2.5 default now
                print("[System] Tool Caller (Qwen2.5) initialized.")
            except Exception as e:
                print(f"[System] Failed to init Tool Caller: {e}")
        else:
            self.llm = MockLLM()
        
        # Quality LLM for learning
        self.quality_llm = self.llm.main_llm if hasattr(self.llm, 'main_llm') else self.llm
        
        # Initialize components
        self.policy = PolicySpace()
        self.history_store = UserHistoryStore()
        self.gatekeeper = Gatekeeper(self.history_store)
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        self.context_manager = ContextManager(storage_path="data/context_store.json")
        
        # NEW: Initialize Orchestrator
        self.orchestrator = Orchestrator(default_timeout=30.0)
        
        # NEW: Initialize Safety Layer
        self.sanitizer = ResponseSanitizer(strict_mode=True)
        self.fallback_generator = FallbackGenerator(default_language="ru")
        
        # NEW: Initialize InfoBroker (will be fully configured after OM creation)
        self.info_broker = InfoBroker(
            context_manager=self.context_manager,
            search_engine=None,  # Will be set from OM
            experts=None         # Will be set after OM creation
        )

        self.om = OperationalModule(
            llm=self.llm, 
            policy=self.policy, 
            tool_caller=self.tool_caller,
            info_broker=self.info_broker,
            sanitizer=self.sanitizer,
            fallback_generator=self.fallback_generator
        )
        
        # Complete InfoBroker setup
        if hasattr(self.om, 'experts'):
            self.info_broker.experts = self.om.experts
        if hasattr(self.om, 'search_engine'):
            self.info_broker.search_engine = self.om.search_engine

        # Register modules in Orchestrator
        asyncio.create_task(self._register_modules())

        self.learning_decoder = LearningDecoder(llm=self.quality_llm)
        self.homeostasis = HomeostasisController(self.policy)
        self.reflection = ReflectionController(
            self.learning_decoder,
            self.homeostasis,
            self.quality_llm
        )
        self._reflection_task = None
        self.world_states: dict[str, WorldState] = {}  # user_id -> WorldState
        
        # NEW: Initialize TaskOrchestrator (new architecture)
        if self.use_task_orchestrator:
            self.intent_router = IntentRouter(self.llm)
            self.task_orchestrator = TaskOrchestrator(
                llm=self.llm,
                intent_router=self.intent_router,
                info_broker=self.info_broker,
                experts=self.om.experts,
                context_manager=self.context_manager,
                reflection=self.reflection,
                decomposer=TaskDecomposer(llm=self.llm),
                policy=self.policy
            )
            print("[System] TaskOrchestrator initialized with TaskDecomposer")
            print("[System] TaskOrchestrator initialized (new architecture)")
    
    async def start(self):
        """Start background tasks."""
        # NEW: Start TaskOrchestrator workers
        if self.use_task_orchestrator:
            await self.task_orchestrator.start_workers(num_workers=3)
            print(f"[System] {self.__class__.__name__} started workers.")
            
            # Start background reflection via TaskQueue
            self._reflection_task = asyncio.create_task(self._background_reflection_task())
        else:
            # Legacy reflection loop
            interval = config.get("system.reflection_interval", 60.0)
            await self.reflection.start_background(interval_seconds=interval)
    
    async def _background_reflection_task(self):
        """Periodically adds reflection tasks to the TaskQueue."""
        interval = config.get("system.reflection_interval", 60.0)
        print(f"[System] Background reflection tasks enabled (interval: {interval}s)")
        
        while True:
            try:
                # Add reflection task with lowest priority
                reflection_task = Task(
                    task_id=f"reflect_{int(time.time())}",
                    task_type="reflection",
                    payload={},
                    context_scope=ContextScope.NONE,
                    priority=Priority.BACKGROUND,
                    source_module="system_cron"
                )
                await self.task_orchestrator.task_queue.enqueue(reflection_task)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[System] Reflection enqueue error: {e}")
                await asyncio.sleep(10)

    async def stop(self):
        """Stop background tasks."""
        if self.use_task_orchestrator:
            if self._reflection_task:
                self._reflection_task.cancel()
            await self.task_orchestrator.stop_workers()
            print(f"[System] {self.__class__.__name__} stopped workers.")
        else:
            await self.reflection.stop_background()
    
    async def _register_modules(self):
        """Register all modules in the Orchestrator."""
        try:
            # Register OperationalModule
            await self.orchestrator.register_module(
                name="operational_module",
                module=self.om,
                interface=ModuleInterface(
                    name="operational_module",
                    input_types=["user_query", "context_slice"],
                    output_types=["response", "decision", "trace"],
                    capabilities=[ModuleCapability.REASON, ModuleCapability.GENERATE],
                    priority=100
                )
            )
            
            # Register ContextManager
            await self.orchestrator.register_module(
                name="context_manager",
                module=self.context_manager,
                interface=ModuleInterface(
                    name="context_manager",
                    input_types=["user_input", "fact"],
                    output_types=["context_slice", "facts"],
                    capabilities=[ModuleCapability.REMEMBER],
                    priority=90
                )
            )
            
            # Register Gatekeeper
            await self.orchestrator.register_module(
                name="gatekeeper",
                module=self.gatekeeper,
                interface=ModuleInterface(
                    name="gatekeeper",
                    input_types=["user_id", "message"],
                    output_types=["identity"],
                    capabilities=[ModuleCapability.VALIDATE],
                    priority=110
                )
            )
            print("[System] All modules registered in Orchestrator.")
        except Exception as e:
            print(f"[System] Failed to register modules: {e}")

    async def process(self, user_id: str, message: str) -> dict:
        """Process a message and return full response data."""
        # Start tracing session
        episode_id = str(uuid.uuid4())
        tracer.start_session(episode_id)
        
        # Step 1: Security Gate
        tracer.add_step("gatekeeper", "Identification", f"Verifying user {user_id}", data_in={"user_id": user_id, "message": message})
        identity = self.gatekeeper.identify(user_id, message)
        tracer.add_step("gatekeeper", "Result", f"Trust level: {identity.trust_level}", data_out=identity.to_dict())
        
        # Step 2: Record input in Context Manager
        tracer.add_step("context_manager", "Record Input", "Saving user message to short-term store")
        self.context_manager.record_user_input(message)
        
        # Get or create world state for user
        if user_id not in self.world_states:
            self.world_states[user_id] = WorldState()
        
        # Step 3: Get context slice
        tracer.add_step("context_manager", "Get Context", "Retrieving relevant context slice")
        context_slice = self.context_manager.get_context_slice(message, identity, self.world_states[user_id])
        tracer.add_step("context_manager", "Context Slice", f"Retrieved {len(context_slice.recent_events)} recent events", data_out=context_slice.to_dict())
        
        # Step 4: Process through selected architecture
        if self.use_task_orchestrator:
            # NEW: TaskOrchestrator-based processing with Context Slicing
            tracer.add_step("task_orchestrator", "Process", "Starting TaskQueue-based processing")
            response, decision, trace = await self.task_orchestrator.process(context_slice)
        else:
            # LEGACY: Monolithic OperationalModule processing  
            tracer.add_step("operational_module", "Process", "Starting central decision making")
            response, decision, trace = await self.om.process(context_slice, self.context_manager)
        
        # Step 4.1: Persist updated world state
        self.world_states[user_id] = context_slice.world_state
        
        # Step 4.2: Apply Response Sanitizer
        tracer.add_step("sanitizer", "Sanitize", "Checking for data leakage in response")
        sanitization = self.sanitizer.sanitize(response, context=message)
        if sanitization.was_modified:
            tracer.add_step("sanitizer", "Result", f"Redacted {sanitization.redactions_count} items", data_out=sanitization.to_dict())
            response = sanitization.sanitized_text
        else:
            tracer.add_step("sanitizer", "Result", "No sensitive data found")

        self.context_manager.record_system_response(response)
        self.context_manager.record_decision(decision)
        
        # End tracing session
        steps = tracer.end_session()
        trace.steps = steps or []

        self.learning_decoder.record_trace(trace)
        await self.learning_decoder.create_summary(trace)
        
        return {
            "response": response,
            "decision": {
                "depth": decision.depth_used.value,
                "confidence": decision.confidence,
                "cost_ms": decision.cost.get("time_ms", 0),
                "experts_used": decision.cost.get("experts_used", 0)
            },
            "user": {
                "trust_level": identity.trust_level,
                "risk_flag": identity.risk_flag
            }
        }
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        llm_stats = self.llm.get_stats() if hasattr(self.llm, 'get_stats') else {}
        return {
            "memory": {
                "traces": len(self.learning_decoder.raw_traces),
                "summaries": len(self.learning_decoder.summaries),
                "patterns": len(self.learning_decoder.patterns)
            },
            "metrics": self.learning_decoder.get_metrics_aggregates(),
            "policy": self.policy.to_dict(),
            "llm": llm_stats,
            "health": self.homeostasis.get_health_report()
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global system
    system = CognitiveSystemWeb()
    await system.start()
    print("[Web] Cognitive System started")
    yield
    await system.stop()
    print("[Web] Cognitive System stopped")


app = FastAPI(title="Cognitive LLM System", lifespan=lifespan)


# ============================================================
# API Routes
# ============================================================

@app.post("/api/chat")
async def chat(request: Request):
    """Process a chat message."""
    raw_body = await request.body()
    try:
        data = json.loads(raw_body.decode('utf-8'))
    except Exception as e:
        print(f"[Web] JSON Decode Error: {e}")
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        
    user_id = data.get("user_id", "web_user")
    message = data.get("message", "")
    
    # Debug print to verify encoding
    # print(f"[Web] Chat request: user={user_id}, len={len(message)}, content={message[:50]}...")
    
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)
    
    try:
        result = await system.process(user_id, message)
        return JSONResponse(result)
    except Exception as e:
        print(f"Error processing chat: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"response": f"System Error: {str(e)}", "error": True}, 
            status_code=500
        )


@app.get("/api/stats")
async def stats():
    """Get system statistics."""
    return JSONResponse(system.get_stats())


@app.post("/api/reflect")
async def reflect():
    """Trigger manual reflection."""
    pattern = await system.reflection.reflect_once()
    return JSONResponse({
        "pattern": pattern.description if pattern else None,
        "policy": system.policy.to_dict()
    })


@app.post("/api/sanitize")
async def sanitize():
    """Sanitize memory (remove LLM identity mentions)."""
    count = system.learning_decoder.sanitize_memory()
    return JSONResponse({"sanitized": count})


@app.post("/api/clear")
async def clear():
    """Clear all memory."""
    count = system.learning_decoder.clear_all_memory()
    return JSONResponse({"cleared": count})


# ============================================================
# HTML Frontend
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cognitive LLM System</title>
    <style>
        :root {
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        header {
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        header h1 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--accent);
        }
        
        .stats-bar {
            display: flex;
            gap: 1.5rem;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        
        .stat { display: flex; align-items: center; gap: 0.4rem; }
        .stat-value { color: var(--text); font-weight: 500; }
        
        main {
            flex: 1;
            display: flex;
            gap: 1rem;
            padding: 1rem;
            max-height: calc(100vh - 130px);
        }
        
        .chat-container {
            flex: 2;
            display: flex;
            flex-direction: column;
            background: var(--surface);
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .message {
            max-width: 80%;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        
        .message.user {
            align-self: flex-end;
            background: var(--accent);
            color: #fff;
        }
        
        .message.system {
            align-self: flex-start;
            background: var(--border);
        }
        
        .message-meta {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }
        
        .input-area {
            display: flex;
            gap: 0.5rem;
            padding: 1rem;
            border-top: 1px solid var(--border);
        }
        
        .input-area input {
            flex: 1;
            padding: 0.75rem 1rem;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 1rem;
        }
        
        .input-area input:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        button {
            padding: 0.75rem 1.5rem;
            background: var(--accent);
            color: #fff;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: opacity 0.2s;
        }
        
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .sidebar {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .panel {
            background: var(--surface);
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 1rem;
        }
        
        .panel h3 {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .panel-content {
            font-size: 0.9rem;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 0.4rem 0;
            border-bottom: 1px solid var(--border);
        }
        
        .metric-row:last-child { border: none; }
        
        .actions {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .actions button {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
        }
        
        .actions button:hover {
            border-color: var(--accent);
        }
        
        .loading {
            display: none;
            color: var(--text-muted);
            font-style: italic;
        }
        
        .loading.active { display: block; }
    </style>
</head>
<body>
    <header>
        <h1>🧠 Cognitive LLM System</h1>
        <div class="stats-bar">
            <div class="stat">📝 Traces: <span class="stat-value" id="stat-traces">0</span></div>
            <div class="stat">📊 Patterns: <span class="stat-value" id="stat-patterns">0</span></div>
            <div class="stat">⚡ Confidence: <span class="stat-value" id="stat-confidence">0%</span></div>
        </div>
    </header>
    
    <main>
        <div class="chat-container">
            <div class="messages" id="messages"></div>
            <div class="input-area">
                <input type="text" id="input" placeholder="Введите сообщение..." autocomplete="off">
                <button id="send" onclick="sendMessage()">Отправить</button>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="panel">
                <h3>Последнее решение</h3>
                <div class="panel-content" id="decision">
                    <div class="metric-row"><span>Глубина</span><span id="dec-depth">-</span></div>
                    <div class="metric-row"><span>Уверенность</span><span id="dec-confidence">-</span></div>
                    <div class="metric-row"><span>Время</span><span id="dec-time">-</span></div>
                    <div class="metric-row"><span>Эксперты</span><span id="dec-experts">-</span></div>
                </div>
            </div>
            
            <div class="panel">
                <h3>Policy</h3>
                <div class="panel-content" id="policy">
                    <div class="metric-row"><span>Fast Path Bias</span><span id="pol-fast">-</span></div>
                    <div class="metric-row"><span>Expert Threshold</span><span id="pol-expert">-</span></div>
                </div>
            </div>
            
            <div class="panel">
                <h3>Действия</h3>
                <div class="actions">
                    <button onclick="doReflect()">🔍 Рефлексия</button>
                    <button onclick="doSanitize()">🧹 Очистить identity</button>
                    <button onclick="refreshStats()">🔄 Обновить статистику</button>
                </div>
            </div>
        </div>
    </main>
    
    <script>
        const messagesEl = document.getElementById('messages');
        const inputEl = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        
        inputEl.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        async function sendMessage() {
            const message = inputEl.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            inputEl.value = '';
            sendBtn.disabled = true;
            
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({user_id: 'web_user', message})
                });
                const data = await res.json();
                
                addMessage(data.response, 'system', data.decision);
                updateDecision(data.decision);
                refreshStats();
            } catch (err) {
                addMessage('Error: ' + err.message, 'system');
            }
            
            sendBtn.disabled = false;
            inputEl.focus();
        }
        
        function addMessage(text, type, decision = null) {
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.textContent = text;
            
            if (decision) {
                const meta = document.createElement('div');
                meta.className = 'message-meta';
                meta.textContent = `${decision.depth} | ${(decision.confidence * 100).toFixed(0)}% | ${decision.cost_ms}ms`;
                div.appendChild(meta);
            }
            
            messagesEl.appendChild(div);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }
        
        function updateDecision(d) {
            document.getElementById('dec-depth').textContent = d.depth;
            document.getElementById('dec-confidence').textContent = (d.confidence * 100).toFixed(0) + '%';
            document.getElementById('dec-time').textContent = d.cost_ms + 'ms';
            document.getElementById('dec-experts').textContent = d.experts_used;
        }
        
        async function refreshStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                
                document.getElementById('stat-traces').textContent = data.memory.traces;
                document.getElementById('stat-patterns').textContent = data.memory.patterns;
                document.getElementById('stat-confidence').textContent = 
                    (data.metrics.avg_confidence * 100).toFixed(0) + '%';
                
                document.getElementById('pol-fast').textContent = 
                    data.policy.fast_path_bias.toFixed(2);
                document.getElementById('pol-expert').textContent = 
                    data.policy.expert_call_threshold.toFixed(2);
            } catch (err) {
                console.error('Stats error:', err);
            }
        }
        
        async function doReflect() {
            try {
                const res = await fetch('/api/reflect', {method: 'POST'});
                const data = await res.json();
                if (data.pattern) {
                    addMessage('🔍 Pattern found: ' + data.pattern, 'system');
                } else {
                    addMessage('🔍 No pattern detected', 'system');
                }
                refreshStats();
            } catch (err) {
                addMessage('Error: ' + err.message, 'system');
            }
        }
        
        async function doSanitize() {
            try {
                const res = await fetch('/api/sanitize', {method: 'POST'});
                const data = await res.json();
                addMessage('🧹 Sanitized ' + data.sanitized + ' items', 'system');
            } catch (err) {
                addMessage('Error: ' + err.message, 'system');
            }
        }
        
        // Initial stats load
        refreshStats();
    </script>
</body>
</html>
"""


# ============================================================
# NEW: Lifecycle Visualization
# ============================================================

@app.get("/api/lifecycle/latest")
async def lifecycle_latest():
    """Get the full timeline of the most recent request."""
    if not system.learning_decoder.raw_traces:
        return JSONResponse({"error": "No traces yet"}, status_code=404)
        
    trace = system.learning_decoder.raw_traces[-1]
    
    # Construct a timeline
    timeline = []
    
    # 1. User Input
    timeline.append({
        "step": "Input",
        "description": f"User: {trace.user_input[:50]}..." if trace.user_input else "Empty input",
        "status": "info",
        "timestamp": trace.timestamp.isoformat()
    })
    
    # 2. Context/Gatekeeper (inferred from decision)
    timeline.append({
        "step": "Gatekeeper",
        "description": "Identity verified",
        "status": "success",
        "details": trace.context_snapshot.get("user_identity", {})
    })
    
    # 3. Decision
    decision = trace.decision
    depth = decision.get("depth_used", "fast")
    timeline.append({
        "step": "Operational Module",
        "description": f"Decision: {decision.get('action', 'respond')} ({depth})",
        "status": "warning" if depth == "deep" else "success",
        "details": {
            "confidence": decision.get("confidence"),
            "cost": decision.get("cost"),
            "intent": decision.get("intent", "Unknown")
        }
    })
    
    # 4. Execution (Experts/Sim)
    if trace.expert_outputs:
        for exp in trace.expert_outputs:
            timeline.append({
                "step": f"Expert ({exp.get('expert_type', 'unknown')})",
                "description": "Consulted",
                "status": "info",
                "details": {"confidence": exp.get("confidence")}
            })
    
    # 5. Response
    timeline.append({
        "step": "Response",
        "description": "Final response generated",
        "status": "success",
        "details": {"length": len(trace.final_response)}
    })

    return JSONResponse({
        "trace_id": trace.episode_id,
        "timeline": timeline
    })


@app.get("/api/lifecycle/history")
async def get_lifecycle_history():
    """Get list of recent episode summaries."""
    summaries = system.learning_decoder.get_recent_summaries(n=50) # Get last 50
    return JSONResponse([s.to_dict() for s in summaries])


@app.get("/api/lifecycle/{episode_id}")
async def get_lifecycle(episode_id: str):
    """Get the full timeline and steps for a specific episode."""
    # Search in memory traces
    trace = next((t for t in system.learning_decoder.raw_traces if t.episode_id == episode_id), None)
    
    if not trace:
        # Try to load from disk if not in memory
        storage_path = Path("./learning_data")
        trace_file = storage_path / f"trace_{episode_id}.json"
        if trace_file.exists():
            import json
            with open(trace_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return JSONResponse(data)
        return JSONResponse({"error": "Trace not found"}, status_code=404)
        
    return JSONResponse(trace.to_dict())


@app.get("/api/architecture")
async def architecture():
    """Returns the system architecture structure for visualization."""
    return JSONResponse({
        "modules": [
            {"id": "gatekeeper", "name": "Gatekeeper", "type": "security", "description": "Security & Identity Gate"},
            {"id": "context_manager", "name": "ContextManager", "type": "memory", "description": "Memory & Context Layer"},
            {"id": "operational_module", "name": "OperationalModule", "type": "core", "description": "Central Decision Engine"},
            {"id": "intent_router", "name": "IntentRouter", "type": "logic", "description": "Intent Classification"},
            {"id": "task_decomposer", "name": "TaskDecomposer", "type": "logic", "description": "Problem Decomposition"},
            {"id": "simulation_engine", "name": "SimulationEngine", "type": "logic", "description": "Deterministic Solver"},
            {"id": "experts_module", "name": "Experts", "type": "llm", "description": "Multi-expert Reasoning"},
            {"id": "expert_neutral", "name": "Neutral Expert", "type": "llm", "description": "Precision-focused dispatcher"},
            {"id": "expert_creative", "name": "Creative Expert", "type": "llm", "description": "Innovative query analysis"},
            {"id": "expert_conservative", "name": "Conservative Expert", "type": "llm", "description": "Risk & safety verification"},
            {"id": "expert_adversarial", "name": "Adversarial Expert", "type": "llm", "description": "Devil's advocate logic checking"},
            {"id": "expert_forecaster", "name": "Forecaster Expert", "type": "llm", "description": "Trend & consequence analysis"},
            {"id": "expert_physics", "name": "Physics Expert", "type": "llm", "description": "Deterministic physical simulation"},
            {"id": "info_broker", "name": "InfoBroker", "type": "data", "description": "Unified Data Retrieval"},
            {"id": "critic", "name": "Critic", "type": "llm", "description": "Validation & Synthesis"},
            {"id": "sanitizer", "name": "Sanitizer", "type": "security", "description": "Data Leakage Protection"},
            {"id": "learning_decoder", "name": "Learning", "type": "learning", "description": "Experience Storage"},
            {"id": "reflection", "name": "Reflection", "type": "learning", "description": "Pattern Extraction"},
            {"id": "homeostasis", "name": "Homeostasis", "type": "learning", "description": "Policy Tuning"}
        ],
        "connections": [
            {"from": "gatekeeper", "to": "context_manager", "label": "identity"},
            {"from": "context_manager", "to": "operational_module", "label": "context_slice"},
            {"from": "operational_module", "to": "intent_router", "label": "classify"},
            {"from": "operational_module", "to": "task_decomposer", "label": "decompose"},
            {"from": "operational_module", "to": "simulation_engine", "label": "run_sim"},
            {"from": "operational_module", "to": "experts_module", "label": "consult"},
            {"from": "operational_module", "to": "info_broker", "label": "get_data"},
            {"from": "experts_module", "to": "critic", "label": "perspectives"},
            {"from": "critic", "to": "sanitizer", "label": "synthesis"},
            {"from": "sanitizer", "to": "learning_decoder", "label": "trace"},
            {"from": "learning_decoder", "to": "reflection", "label": "summaries"},
            {"from": "reflection", "to": "homeostasis", "label": "patterns"},
            {"from": "homeostasis", "to": "operational_module", "label": "policy_update"}
        ]
    })


@app.get("/", response_class=HTMLResponse)
async def index_fallback():
    """Serve the main HTML page."""
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


if __name__ == "__main__":
    print("=" * 50)
    print("  Cognitive LLM System - Web Interface")
    port = config.get("system.port", 8000)
    print(f"  Open: http://localhost:{port}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=port)
