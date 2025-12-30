# Omega: Cognitive LLM System

> 🧠 An autonomous AI architecture with self-learning, deterministic reasoning, and modular design

Omega is an advanced cognitive architecture for Large Language Models (LLMs), designed to move beyond simple "input-output" patterns towards a more autonomous, self-learning, and reflective system.

## ✨ Key Features

### Core Capabilities
- **Multi-Model Orchestration**: Fast path (Gemma3:4b) for routine tasks, Main expert (Gemma3:12b) for complex reasoning
- **Context Gate (Anti-Hallucination)**: Strictly blocks noisy search snippets and enforces a hard stop on missing data
- **Automated Trip Planning**: Built-in geographical detection for "A to B" travel and distance retrieval
- **Deterministic Simulation**: Code-based FSM/Math solver prevents LLM "guessing" in priority problems
- **Self-Learning**: Periodic reflection extracts patterns and auto-tunes cognitive policies

### Architecture Highlights
| Component | Purpose |
|-----------|---------|
| **TaskDecomposer** | Parses complex problems into GIVEN vs MISSING data |
| **SimulationEngine** | FSM simulator for robots, resources, queues |
| **Orchestrator** | Module registry with hot-swap support |
| **InfoBroker** | Unified information retrieval with fallback chain |
| **Sanitizer** | Blocks passwords, API keys, tokens in responses |

## 🏗️ Architecture

```
User Input → Gatekeeper → ContextManager → OperationalModule
                                                 │
                 ┌───────────────────────────────┼───────────────────┐
                 │                               │                   │
                 ▼                               ▼                   ▼
          TaskDecomposer              SimulationEngine          Experts
          (Parse problem)             (Deterministic)        (LLM reasoning)
                 │                               │                   │
                 └───────────────────────────────┼───────────────────┘
                                                 ▼
                                            Sanitizer → Response
```

📖 See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed module interactions.

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **LLM Engine**: Ollama (Gemma3, Qwen2.5)
- **Networking**: Aiohttp for async LLM communication

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. LLM Setup (Ollama)
```bash
ollama pull gemma3:12b
ollama pull gemma3:4b
ollama pull qwen2.5:7b  # For tool calling
```

### 3. Run the System

**Web Interface (Recommended):**
```bash
python web.py
```
Open [http://localhost:8000](http://localhost:8000)

**Interactive CLI:**
```bash
python main.py
```

## 📂 Project Structure

```
agi2/
├── main.py              # CLI entry point
├── web.py               # FastAPI web interface
├── ARCHITECTURE.md      # Module interaction map
│
├── core/                # Core cognitive components
│   ├── operational_module.py   # Central decision maker
│   ├── simulation_engine.py    # Deterministic FSM/Math
│   ├── task_decomposer.py      # Problem parsing
│   ├── orchestrator.py         # Module registry
│   ├── sanitizer.py            # Anti-data-leakage
│   └── ...
│
├── learning/            # Self-improvement
│   ├── reflection.py    # Pattern extraction
│   └── homeostasis.py   # Policy auto-tuning
│
└── models/              # Schemas & LLM interface
```

## 🔧 Configuration

Edit `config.yaml`:
```yaml
models:
  main: "gemma3:12b"    # Main reasoning model
  fast: "gemma3:4b"     # Quick responses
  tools: "qwen2.5:7b"   # Tool calling
  use_ollama: true
```

## 💡 Example Use Cases

### Robot Scheduling (Deterministic)
```
Input: "Robot-A has 10% charge. Task Alpha needs 70% charge, 20 min duration. Start: 12:00"
Output: 
  - Charge time: 60 min
  - Work time: 20 min  
  - Completion: 13:20 ✅
```

### Complex Priority Problems
```
Input: "EV charging station: 2 ports, peak hours, ambulance priority..."
Output: Structured analysis with formulas (not invented numbers)
```

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Made with 🧠 by Omega Team**
