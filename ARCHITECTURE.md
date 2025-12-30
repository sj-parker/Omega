# Omega Architecture Map

> **Last Updated**: 2025-12-30  
> **Purpose**: Quick reference for module interactions and data flow

## 🏗️ System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OMEGA SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Input                                                                 │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐                 │
│  │ Gatekeeper  │───▶│  Context    │───▶│ TaskOrchestrator│                 │
│  │ (Security)  │    │  Manager    │    │   (Planner)     │                 │
│  └─────────────┘    └─────────────┘    └────────┬────────┘                 │
│                                                  │                          │
│                                          Creates Task[]                     │
│                                          with ContextScope                  │
│                                                  │                          │
│                                          ┌──────▼──────┐                   │
│                                          │  TaskQueue  │                   │
│                                          │ (Priority)  │                   │
│                                          └──────┬──────┘                   │
│                                                 │                          │
│         ┌───────────────────────────────────────┼──────────────────┐       │
│         │                    │                  │                   │       │
│         ▼ ctx:NONE           ▼ ctx:RECENT       ▼ ctx:FULL          │       │
│  ┌────────────┐       ┌────────────┐     ┌────────────────┐        │       │
│  │InfoBroker  │       │ LLM Fast/  │     │ ExpertsModule  │        │       │
│  │  (Search)  │       │ Medium     │     │(LLM reasoning) │        │       │
│  └────────────┘       └────────────┘     └───────┬────────┘        │       │
│                                                   │                 │       │
│                                         ┌─────────┴─────────┐      │       │
│                                         │     Critic        │      │       │
│                                         │  (Verification)   │      │       │
│                                         └─────────┬─────────┘      │       │
│                                                   │                 │       │
│         └─────────────────────────────────────────┼─────────────────┘       │
│                                                   ▼                         │
│                                          Aggregate Results                  │
│                                                   │                         │
│                              ┌────────────────────┘                         │
│                              ▼                                              │
│                     ┌─────────────────┐                                     │
│                     │    Sanitizer    │                                     │
│                     │ (Anti-leakage)  │                                     │
│                     └────────┬────────┘                                     │
│                              │                                              │
│                              ▼                                              │
│                         Response ──────▶ LearningDecoder ──▶ Reflection     │
│                                                                   │         │
│                                                                   ▼         │
│                                                            Homeostasis      │
│                                                          (Policy update)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Complete Directory Structure

```
agi2/
├── main.py                  # Entry point, CognitiveSystem class
├── web.py                   # FastAPI web interface + real-time stats
├── config.yaml              # Model configuration (main, fast, tools)
├── ARCHITECTURE.md          # This file
├── README.md                # Project overview
├── requirements.txt         # Python dependencies
├── run.bat                  # Windows launcher script
│
├── core/                    # ═══ CORE MODULES ═══
│   ├── __init__.py
│   ├── config.py                 # Global config loader
│   │
│   ├── # ─── MAIN PIPELINE ───
│   ├── gatekeeper.py             # Security gate, trust scoring
│   ├── context_manager.py        # Short/long-term memory, facts
│   ├── operational_module.py     # Central decision maker (O.M.)
│   ├── intent_router.py          # Intent classification (fast/LLM)
│   ├── experts.py                # LLM experts + tool dispatch
│   ├── validator.py              # Response semantic validation
│   │
│   ├── # ─── ORCHESTRATION (NEW) ───
│   ├── orchestrator.py           # Module registry, hot-swap
│   ├── task_queue.py             # Priority queue (CRITICAL→BACKGROUND)
│   ├── info_broker.py            # Unified info retrieval + fallback
│   │
│   ├── # ─── PROBLEM SOLVING (NEW) ───
│   ├── task_decomposer.py        # Parse complex problems (GIVEN/MISSING)
│   ├── simulation_engine.py      # Deterministic FSM/Math (code, not LLM)
│   │
│   ├── # ─── SAFETY (NEW) ───
│   ├── sanitizer.py              # Block passwords, API keys, tokens
│   ├── fallback_generator.py     # Graceful "I don't know" templates
│   ├── identity_filter.py        # Remove LLM identity mentions
│   │
│   ├── # ─── UTILITIES ───
│   ├── ontology.py               # Self-identity, search blocking patterns
│   ├── tools.py                  # Tool definitions & registry
│   └── search_engine.py          # Web search (DuckDuckGo)
│
├── learning/                # ═══ SELF-LEARNING SYSTEM ═══
│   ├── __init__.py
│   ├── learning_decoder.py       # Episode processing, pattern extraction
│   ├── reflection.py             # Background reflection loop
│   ├── homeostasis.py            # Policy auto-tuning
│   └── impact_resolver.py        # Pattern → policy change mapping
│
├── models/                  # ═══ DATA SCHEMAS & LLM ═══
│   ├── __init__.py
│   ├── schemas.py                # Pydantic models (30+ schemas)
│   └── llm_interface.py          # LLM abstraction (Ollama, multi-model)
│
├── config/                  # ═══ CONFIGURATION ═══
│   └── intent_rules.yaml         # Keyword-based intent rules
│
├── learning_data/           # ═══ DATA STORAGE (git-ignored) ═══
│   ├── episodes/                 # Raw conversation traces
│   ├── patterns/                 # Extracted patterns
│   └── policies/                 # Policy snapshots
│
└── tests/                   # ═══ TESTS ═══
    ├── test_intent_router.py
    └── ...
```

---

## 🔗 Module Interactions

### Main Request Pipeline
```
User Input
    │
    ▼
┌──────────────┐
│  Gatekeeper  │ ─── identify() → UserIdentity (trust_level, anomalies)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ ContextMgr   │ ─── get_context_slice() → ContextSlice
└──────┬───────┘      (recent_events, long_term_facts, world_state)
       │
       ▼
┌──────────────┐
│    O.M.      │ ─── classify() → decide_depth()
└──────┬───────┘      (Internal IntentRouter + keyword rules)
       │
       │
       ├── FAST ────────▶ Direct LLM response (with Context Injection)
       │                  (Uses last 5 recent events for continuity)
       │
       ├── MEDIUM ──────▶ LLM + memory context (+ LongTerm Memory)
       │                  (Recall/Fact retrieval path)
       │
       └── DEEP ────────┬──▶ TaskDecomposer.decompose()
                        │       └── DecomposedProblem (entities, rules)
                        │
                        ├──▶ SimulationEngine.run_robot_simulation()
                        │       └── SimulationResult (deterministic)
                        │
                        └──▶ ExpertsModule.consult_all()
                                │   └── 6 expert perspectives
                                ▼
                        ┌────────────┐
                        │   Critic   │ ─── analyze() → CoVe verification
                        └─────┬──────┘
                              │
                              ▼
                      ┌────────────┐
                      │ Sanitizer  │ ─── sanitize() → redact sensitive data
                      └─────┬──────┘
                            │
                            ▼
                       Response
```

### Learning Loop (Background)
```
Response + Decision
        │
        ▼
┌────────────────┐
│ LearningDecoder│ ─── add_trace() → Store episode
└───────┬────────┘
        │ (every N interactions)
        ▼
┌────────────────┐
│  Reflection    │ ─── run_reflection() → Extract patterns
└───────┬────────┘      (Needs min 3 episodes to activate)
        │
        ▼
┌────────────────┐
│ImpactResolver  │ ─── resolve() → Pattern → PolicyUpdate
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Homeostasis    │ ─── apply_update() → Adjust thresholds
└────────────────┘
```

---

## 📊 Key Data Schemas

| Schema | File | Purpose |
|--------|------|---------|
| `UserIdentity` | schemas.py | Trust level, session data |
| `ContextSlice` | schemas.py | Current context for O.M. |
| `DecisionObject` | schemas.py | Decision + reasoning trace |
| `ExpertResponse` | schemas.py | Single expert output |
| `CriticAnalysis` | schemas.py | Verification results |
| `PolicySpace` | schemas.py | System parameters |
| `RawTrace` | schemas.py | Full conversation trace |
| `ExtractedPattern` | schemas.py | Learning pattern |
| `SimulationResult` | simulation_engine.py | FSM/Math result |
| `DecomposedProblem` | task_decomposer.py | Parsed problem |

---

## 🆕 New Components (Dec 2024)

### Orchestration Layer
| Module | Purpose | Key Methods |
|--------|---------|-------------|
| `Orchestrator` | Module registry, hot-swap | `register_module()`, `dispatch()`, `replace_module()` |
| `TaskQueue` | Priority scheduling | `enqueue()`, `dequeue()`, `wait_for()` |
| `InfoBroker` | Unified info retrieval | `request_info()` → Cache→Memory→Search→Expert→Fallback |
| `Tracer` | Context-safe tracing | `start_session()`, `add_step()`, `end_session()` |

### Problem Solving Layer
| Module | Purpose | Key Methods |
|--------|---------|-------------|
| `TaskDecomposer` | Parse GIVEN vs MISSING data | `decompose()`, `is_complex_problem()` |
| `SimulationEngine` | Deterministic calculations | `FSMSimulator`, `MathSolver` |
| **Logic Note** | Trip Detection | Automatically identifies "from A to B" patterns and seek distance data through search. |

### Safety Layer
| Module | Purpose | Key Methods |
|--------|---------|-------------|
| `Sanitizer` | Block sensitive data | `sanitize()` → regex for passwords, API keys |
| `Context Gate` | Block noise context | Orchestrator filter: only passes `sufficient` search data to reasoning steps. |
| `Semantic Verify` | Search result validation | InfoBroker check: triggers only if core keywords (distance, price) exist in snippets. |
| `IdentityFilter` | Remove LLM identity leaks | Filter "As an AI", "I'm Gemma" etc. |

---

## ⚙️ Configuration Files

### `config.yaml` - Model Selection
```yaml
models:
  main: "gemma3:12b"      # Main reasoning (deep path)
  fast: "gemma3:4b"       # Quick responses (fast path)
  tools: "qwen2.5:7b"     # Tool calling (FunctionGemma)
  use_ollama: true
```

### `config/intent_rules.yaml` - Intent Classification
```yaml
intents:
  memorize:
    keywords: [запомни, сохрани, remember, save]
    priority: HIGH
  recall:
    keywords: [напомни, вспомни, remind, what was]
    priority: HIGH
  realtime_data:
    keywords: [price, weather, news, stock, crypto]
    threshold: 0.7
  calculation:
    keywords: [calculate, compute, formula]
    threshold: 0.8
```

---

## 🚦 Decision Depth Flow

```
Intent + Confidence
         │
         ├── confidence > 0.85 ────────────▶ FAST (1 LLM call)
         │   └── smalltalk, confirmation
         │
         ├── 0.5 < confidence < 0.85 ──────▶ MEDIUM (LLM + context)
         │   └── recall, factual
         │
         └── confidence < 0.5 OR complex ──▶ DEEP (Experts + Critic)
                  │
                  └── FSM detected? ───┬──▶ SimulationEngine (code)
                                       │
                                       └──▶ Experts (LLM)
```

---

## 🔒 Search Blocking (ontology.py)

Patterns that block web search:
- Math expressions: `\d+\s*[\*\+\-\/]\s*\d+`
- Self-analysis: `себя`, `yourself`, `what are you`
- Priority problems: `правило.*приоритет`
- Conditional rules: `ниже 10%`, `если.*скидка`
- Resource allocation: `порт.*всего \d+`

---

## � Monitoring & CLI Commands

| Command | Purpose |
|---------|---------|
| `/health` | System health report |
| `/policy` | Current PolicySpace values |
| `/stats` | LLM usage statistics |
| `/memory` | Memory store status |
| `/reflect` | Trigger manual reflection |
| `/clean` | Clear all memory |
| `/sanitize` | Remove LLM identity from history |

---

## 🔄 Key Metrics

| Metric | Location | Description |
|--------|----------|-------------|
| `cost.time_ms` | DecisionObject | Response latency |
| `cost.experts_used` | DecisionObject | Number of experts called |
| `sanitizer.redactions_count` | SanitizationResult | Data leakage blocks |
| `trust_level` | UserIdentity | User trust score (0-1) |
| `confidence` | IntentRouter | Intent classification confidence |
