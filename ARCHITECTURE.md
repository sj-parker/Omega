# Omega Architecture Map

> **Last Updated**: 2024-12-28  
> **Purpose**: Quick reference for module interactions and data flow

## 🏗️ System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           OMEGA SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User Input                                                             │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │ Gatekeeper  │───▶│  Context    │───▶│ Operational │                 │
│  │ (Security)  │    │  Manager    │    │   Module    │                 │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                 │
│                                               │                         │
│              ┌────────────────────────────────┼────────────────┐        │
│              │                                │                │        │
│              ▼                                ▼                ▼        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ TaskDecomposer  │  │ SimulationEngine│  │ ExpertsModule   │         │
│  │ (Parse problem) │  │ (Deterministic) │  │ (LLM reasoning) │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│              │                                │                         │
│              └────────────────┬───────────────┘                         │
│                               ▼                                         │
│                      ┌─────────────────┐                                │
│                      │   Sanitizer     │                                │
│                      │ (Anti-leakage)  │                                │
│                      └────────┬────────┘                                │
│                               ▼                                         │
│                          Response                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
agi2/
├── main.py              # Entry point, CognitiveSystem
├── web.py               # FastAPI web interface
├── config.yaml          # Model configuration
│
├── core/                # Core modules
│   ├── operational_module.py   # Central decision maker
│   ├── context_manager.py      # Memory & context
│   ├── gatekeeper.py           # Security & trust
│   ├── experts.py              # LLM experts + tool dispatch
│   ├── intent_router.py        # Intent classification
│   │
│   ├── # NEW ARCHITECTURE
│   ├── orchestrator.py         # Module registry & hot-swap
│   ├── task_queue.py           # Priority queue
│   ├── info_broker.py          # Unified info retrieval
│   ├── task_decomposer.py      # Complex problem parsing
│   ├── simulation_engine.py    # Deterministic calculations
│   ├── sanitizer.py            # Anti-data-leakage
│   ├── fallback_generator.py   # Graceful "I don't know"
│   │
│   ├── # UTILITIES
│   ├── ontology.py             # Self-identity & search blocking
│   ├── tools.py                # Tool definitions
│   ├── search_engine.py        # Web search
│   └── validator.py            # Response validation
│
├── models/              # Data schemas & LLM interface
│   ├── schemas.py              # Pydantic models
│   └── llm_interface.py        # LLM abstraction
│
├── learning/            # Self-learning system
│   ├── reflection.py           # Pattern extraction
│   └── homeostasis.py          # Policy auto-tuning
│
└── config/
    └── intent_rules.yaml       # Intent classification rules
```

---

## 🔗 Module Interactions

### Main Pipeline
```
User Input
    │
    ▼
┌──────────────┐
│  Gatekeeper  │ ─── Identifies user, calculates trust level
└──────┬───────┘
       │ UserIdentity
       ▼
┌──────────────┐
│ ContextMgr   │ ─── Builds context slice (recent events, facts)
└──────┬───────┘
       │ ContextSlice
       ▼
┌──────────────┐
│    O.M.      │ ─── Routes by intent, calls experts/simulation
└──────┬───────┘
       │
       ├──▶ TaskDecomposer ─── Parses complex problems
       │
       ├──▶ SimulationEngine ─── Deterministic FSM/Math
       │
       ├──▶ ExpertsModule ─── LLM reasoning + tools
       │         │
       │         └──▶ ToolsRegistry ─── Execute tools
       │
       ▼
┌──────────────┐
│  Sanitizer   │ ─── Redacts passwords, API keys
└──────┬───────┘
       │
       ▼
   Response
```

### Key Data Flows

| From | To | Data |
|------|----|------|
| Gatekeeper | ContextManager | `UserIdentity` (trust, anomalies) |
| ContextManager | OperationalModule | `ContextSlice` (events, facts, state) |
| IntentRouter | OperationalModule | `(intent, confidence)` |
| TaskDecomposer | OperationalModule | `DecomposedProblem` (entities, rules) |
| SimulationEngine | OperationalModule | `SimulationResult` (deterministic) |
| ExpertsModule | CriticModule | `ExpertResponse[]` |
| Sanitizer | main.py | `SanitizationResult` |

---

## 🆕 New Components (Dec 2024)

| Module | Purpose | Key Methods |
|--------|---------|-------------|
| **Orchestrator** | Module registry, hot-swap | `register_module()`, `dispatch()` |
| **TaskQueue** | Priority task scheduling | `enqueue()`, `dequeue()` |
| **InfoBroker** | Unified info retrieval | `request_info()` with fallback chain |
| **TaskDecomposer** | Parse complex problems | `decompose()` → entities, rules, missing data |
| **SimulationEngine** | Code-based calculations | `FSMSimulator`, `MathSolver` |
| **Sanitizer** | Prevent data leakage | Regex patterns for passwords, keys |
| **FallbackGenerator** | Graceful "I don't know" | Templates for uncertainty |

---

## 🔧 Configuration

### Model Selection (`config.yaml`)
```yaml
models:
  main: "gemma3:12b"    # Main reasoning
  fast: "gemma3:4b"     # Quick responses
  tools: "qwen2.5:7b"   # Tool calling
```

### Intent Rules (`config/intent_rules.yaml`)
```yaml
intents:
  realtime_data:
    keywords: [price, weather, news]
    threshold: 0.7
```

---

## 🚦 Decision Flow

```
Intent Classification
        │
        ├── smalltalk/confirmation ──▶ FAST path (1 LLM call)
        │
        ├── recall/memorize ──▶ MEDIUM path (+ memory)
        │
        └── complex/calculation ──▶ DEEP path
                    │
                    ├── FSM detected? ──▶ SimulationEngine
                    │
                    └── Otherwise ──▶ Experts + Critic
```

---

## 📊 Monitoring

### CLI Commands
- `/health` - System health report
- `/policy` - Current policy parameters
- `/stats` - LLM usage statistics
- `/memory` - Memory status
- `/reflect` - Force reflection

### Key Metrics
- `cost.time_ms` - Response latency
- `cost.experts_used` - Number of experts called
- `sanitizer.redactions_count` - Data leakage blocks
