# Omega: Cognitive LLM System

Omega is an advanced cognitive architecture for Large Language Models (LLMs), designed to move beyond simple "input-output" patterns towards a more autonomous, self-learning, and reflective system.

## 🧠 Key Features

- **Multi-Model Orchestration**: Uses a "Fast Path" (e.g., Phi-3 Mini) for routine tasks and a "Main Expert" (e.g., Qwen-2.5 7B) for complex reasoning and reflection.
- **Cognitive Pipeline**:
    - **Security Gatekeeper**: Analyzes user input for risks and identity.
    - **Context Manager**: Handles short-term and long-term memory retrieval.
    - **Operational Module (OM)**: Oracles and experts for multi-step reasoning.
    - **Learning Decoder**: Extracts patterns and summaries from interactions.
- **Homeostasis & Reflection**: An internal loop that periodically audits performance, extracts new "policies," and adjusts system behavior without manual retraining.
- **Web Interface**: A modern FastAPI-based web UI with real-time stats, health monitoring, and manual reflection triggers.

## 🛠 Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **LLM Engine**: Ollama (supports local models like Qwen2.5 and Phi-3)
- **Networking**: Aiohttp for async LLM communication

## 🚀 Quick Start

### 1. Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. LLM Setup (Ollama)

Ensure you have [Ollama](https://ollama.com/) installed and the following models pulled:
```bash
ollama pull qwen2.5:7b
ollama pull phi3:mini
```

### 3. Run the System

**Web Interface (Recommended):**
```bash
python web.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

**Interactive CLI:**
```bash
python main.py
```

## 📂 Project Structure

- `core/`: Fundamental cognitive components (Gatekeeper, Context, OM).
- `learning/`: Self-improvement logic (Reflection, Homeostasis, Decoder).
- `models/`: LLM interface and data schemas.
- `learning_data/`: (Ignored) Local storage for traces and patterns.
- `web.py`: FastAPI backend and frontend.
- `main.py`: CLI entry point.

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
