# Web Interface for Cognitive LLM System
# FastAPI backend with REST API

import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.schemas import PolicySpace, WorldState
from models.llm_interface import MockLLM, OllamaLLM, LLMRouter
from core.gatekeeper import Gatekeeper, UserHistoryStore
from core.context_manager import ContextManager
from core.operational_module import OperationalModule
from learning.learning_decoder import LearningDecoder
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
        use_multi_model: bool = None
    ):
        # Load from config or use provided values
        self.use_ollama = use_ollama if use_ollama is not None else config.get("models.use_ollama")
        self.main_model = main_model if main_model is not None else config.get("models.main")
        self.fast_model = fast_model if fast_model is not None else config.get("models.fast")
        self.use_multi_model = use_multi_model if use_multi_model is not None else config.get("models.use_multi_model")

        # Initialize LLM(s)
        if self.use_ollama:
            if self.use_multi_model:
                fast_llm = OllamaLLM(model=self.fast_model)
                main_llm = OllamaLLM(model=self.main_model)
                self.llm = LLMRouter(fast_llm=fast_llm, main_llm=main_llm)
            else:
                self.llm = OllamaLLM(model=self.main_model)
        else:
            self.llm = MockLLM()
        
        # Quality LLM for learning
        self.quality_llm = self.llm.main_llm if hasattr(self.llm, 'main_llm') else self.llm
        
        # Initialize components
        self.policy = PolicySpace()
        self.history_store = UserHistoryStore()
        self.gatekeeper = Gatekeeper(self.history_store)
        self.context_manager = ContextManager()
        self.om = OperationalModule(self.llm, self.policy)
        self.learning_decoder = LearningDecoder(llm=self.quality_llm)
        self.homeostasis = HomeostasisController(self.policy)
        self.reflection = ReflectionController(
            self.learning_decoder,
            self.homeostasis,
            self.quality_llm
        )
        self._reflection_task = None
        self.world_states: dict[str, WorldState] = {}  # user_id -> WorldState
    
    async def start(self):
        """Start background tasks."""
        interval = config.get("system.reflection_interval", 60.0)
        await self.reflection.start_background(interval_seconds=interval)
    
    async def stop(self):
        """Stop background tasks."""
        await self.reflection.stop_background()
    
    async def process(self, user_id: str, message: str) -> dict:
        """Process a message and return full response data."""
        identity = self.gatekeeper.identify(user_id, message)
        self.context_manager.record_user_input(message)
        
        # Get or create world state for user
        if user_id not in self.world_states:
            self.world_states[user_id] = WorldState()
        
        context_slice = self.context_manager.get_context_slice(message, identity, self.world_states[user_id])
        
        response, decision, trace = await self.om.process(context_slice)
        
        # Persist updated world state
        self.world_states[user_id] = context_slice.world_state
        
        self.context_manager.record_system_response(response)
        self.context_manager.record_decision(decision)
        self.learning_decoder.record_trace(trace)
        await self.learning_decoder.create_summary(trace)
        
        return {
            "response": response,
            "decision": {
                "depth": decision.depth_used.value,
                "confidence": decision.confidence,
                "cost_ms": decision.cost["time_ms"],
                "experts_used": decision.cost["experts_used"]
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
    data = await request.json()
    user_id = data.get("user_id", "web_user")
    message = data.get("message", "")
    
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)
    
    try:
        result = await system.process(user_id, message)
        return JSONResponse(result)
    except Exception as e:
        print(f"Error processing stats: {e}")
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


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    return HTML_PAGE


if __name__ == "__main__":
    print("=" * 50)
    print("  Cognitive LLM System - Web Interface")
    port = config.get("system.port", 8000)
    print(f"  Open: http://localhost:{port}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=port)
