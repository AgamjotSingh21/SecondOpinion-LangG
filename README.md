# SecondOpinion – AI Debate Decision System

## 🚀 Overview
SecondOpinion is a multi-agent AI system designed to help users make better decisions by simulating a structured debate.

Instead of giving a single answer, the system:
- Generates arguments FOR a decision
- Generates arguments AGAINST it
- Identifies blind spots
- Conducts a second round of debate
- Produces a final verdict

The multi-agent workflow is orchestrated using LangGraph, while the existing RAG pipeline and local Ollama-based Llama 3.

## 🧠 Architecture
The system uses 4 AI agents:
1. FOR Agent – Supports the decision  
2. AGAINST Agent – Challenges the decision  
3. Questioner Agent – Finds blind spots  
4. Judge Agent – Gives final verdict  

The workflow is orchestrated using LangGraph:

User Question
↓
Initial Retrieval
↓
Round 1 – FOR Agent
↓
Round 1 – AGAINST Agent
↓
Questioner – Blind Spot Detection & Additional Retrieval
↓
Round 2 – FOR Agent
↓
Round 2 – AGAINST Agent
↓
Judge Agent
↓
Final Verdict

## 📂 Project Structure
```
second_opinion/
│
├── agents.py
├── main.py
├── build_db.py
├── requirements.txt
├── README.md
├── knowledge_base/
│   └── food_delivery.txt
```

## ⚙️ Setup Instructions

### 1. Clone the repository
git clone https://github.com/AgamjotSingh21/SecondOpinion-LangG.git

cd SecondOpinion/second_opinion

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Build knowledge database
python build_db.py

## 🧠 Running the Local LLM (Ollama)

### Install Ollama
https://ollama.com

### Run model
ollama run llama3

## ▶️ Run the Project
python main.py

## 💬 Example Input
Should I start a food delivery startup in Ludhiana?

## 🔥 Features
- RAG using ChromaDB  
- Multi-agent debate  
- LangGraph-based workflow orchestration
- Blind spot detection
- Adaptive knowledge retrieval
- Two-round structured debate
- Fully offline LLM inference using Ollama

## ⚠️ Notes
- Keep Ollama running  
- Do not upload venv, chroma_db, or .env  

## 🛠 Tech Stack
- Python  
- LangGraph
- ChromaDB  
- Sentence Transformers  
- Ollama  
