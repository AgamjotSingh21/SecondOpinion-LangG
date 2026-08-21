# SecondOpinion – AI Debate Decision System

## 🚀 Overview
**SecondOpinion** is a multi-agent AI system designed to help founders and entrepreneurs make informed decisions by simulating a structured, adversarial debate grounded in startup knowledge.

Instead of providing a generic answer, the system:
- Generates strong arguments **FOR** a decision
- Generates critical arguments **AGAINST** it
- Identifies **blind spots** and formulates targeted follow-up questions
- Conducts a second round of debate specifically addressing the blind spots
- Issues a **definitive verdict** (`GO`, `NO-GO`, or `CONDITIONAL GO`) with confidence scores and actionable next steps

The workflow is orchestrated using **LangGraph**, with **ChromaDB** for RAG document retrieval and **Ollama (Llama 3)** for local LLM inference.

---

## 🧠 Architecture
The system employs 4 specialized AI agents:
1. **FOR Agent** – Acts as a passionate startup advocate supporting the decision.
2. **AGAINST Agent** – Acts as a ruthless devil's advocate challenging the decision.
3. **Questioner Agent** – Analyzes gaps and extracts critical blind spots.
4. **Judge Agent** – Evaluates all rounds and delivers a grounded, structured verdict.

```
User Question
     ↓
Initial Retrieval
     ↓
Round 1 – FOR Agent
     ↓
Round 1 – AGAINST Agent
     ↓
Questioner Agent – Blind Spot Detection & Secondary Retrieval
     ↓
Round 2 – FOR Agent on Blind Spots
     ↓
Round 2 – AGAINST Agent on Blind Spots
     ↓
Judge Agent – Final Verdict
```

---

## 📂 Project Structure
```
SecondOpinion-LangG/
│
├── agents.py                     # Agent logic, system prompts, LLM & ChromaDB calls
├── main.py                       # LangGraph state graph and interactive CLI entrypoint
├── build_db.py                   # Ingests JSON/TXT datasets into ChromaDB vector store
├── evaluate_model.py             # Evaluation split preparation, metrics calculation & benchmark runner
├── requirements.txt              # Project dependencies
├── README.md                     # Documentation
│
├── knowledge_base/               # Knowledge base documents and datasets
│   ├── entrepreneurial_decisions.json
│   └── food_delivery.txt
│
└── evaluation/                   # Evaluation dataset splits and output reports
    ├── eval_cases.json
    ├── knowledge_base_train/
    └── results/
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/AgamjotSingh21/SecondOpinion-LangG.git
cd SecondOpinion-LangG
```

### 2. Create and activate a virtual environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup and start Ollama
Download and install Ollama from [ollama.com](https://ollama.com).
Pull and run the Llama 3 model:
```bash
ollama run llama3
```

### 5. Build the Knowledge Vector Database
Build the ChromaDB vector database from the knowledge base:
```bash
python build_db.py
```

---

## ▶️ Running the Debate System

Run the interactive CLI:
```bash
python main.py
```

### Example Question:
> *Should I start a food delivery startup in Ludhiana?*

---

## 📊 Evaluation & Benchmarking

The project includes an evaluation suite (`evaluate_model.py`) to benchmark the decision model against historical entrepreneurial decisions.

### 1. Prepare Evaluation Split
Create a holdout evaluation dataset and a training knowledge base split:
```bash
python evaluate_model.py prepare --source knowledge_base/entrepreneurial_decisions.json --eval-size 100
```

### 2. Build Evaluation Vector Database (No-Leakage)
```bash
# PowerShell
$env:KNOWLEDGE_BASE_DIR="evaluation/knowledge_base_train"
$env:CHROMA_DB_PATH="./chroma_db_eval"
python build_db.py

# Bash (Linux/macOS)
KNOWLEDGE_BASE_DIR="evaluation/knowledge_base_train" CHROMA_DB_PATH="./chroma_db_eval" python build_db.py
```

### 3. Run Benchmark
```bash
# PowerShell
$env:CHROMA_DB_PATH="./chroma_db_eval"
python evaluate_model.py run --eval-cases evaluation/eval_cases.json --limit 50

# Bash (Linux/macOS)
CHROMA_DB_PATH="./chroma_db_eval" python evaluate_model.py run --eval-cases evaluation/eval_cases.json --limit 50
```

Evaluation outputs are saved to `evaluation/results/`:
- `evaluation_summary.json` (Accuracy, Average Confidence, Confusion Matrix)
- `evaluation_results.csv` (Per-case breakdown of expected vs. predicted verdicts)
- `evaluation_results.json` (Full traces and model answers)

---

## 📋 Verdict Output Format

The Judge Agent delivers a structured verdict formatted as:
```text
============================================================
JUDGE AGENT — FINAL VERDICT
============================================================
VERDICT: [GO / NO-GO / CONDITIONAL GO]
CONFIDENCE: [0-100]%


============================================================
```

---

## 🛠 Tech Stack
- **Orchestration**: LangGraph
- **Vector Database**: ChromaDB
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM Engine**: Ollama (Llama 3)
- **Language**: Python 3.10+
