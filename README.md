# Medical Agentic ChatBOT (MedGuide AI) ✚

An agentic AI medical education and triage assistant built with **LangGraph**, **Ollama** (local LLM & vision), **Pinecone** (vector search knowledge base), and **FastAPI**.

---

## 🚀 Features

- **Agentic Workflow**: Powered by **LangGraph** with stateful conversation memory and dynamic routing between local knowledge base and web search.
- **Multimodal**: Supports text inquiries and medical image inspection with local vision models (e.g. `moondream`).
- **Knowledge Base Retrieval (RAG)**: Fast semantic search over medical literature indexed into **Pinecone** using `nomic-embed-text`.
- **Live Web Research**: Fallback web search via **Tavily** for current or unindexed medical topics.
- **Emergency Triage Filter**: Immediate safety heuristic checks for red-flag symptoms (e.g., chest pain, breathing difficulties, stroke symptoms) that bypass model latency and direct to emergency services.
- **Clean Interactive Web UI**: Modern responsive interface with camera capture, voice input, text-to-speech answer reading, and source citation badges.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic Settings
- **Agent Orchestration**: LangGraph, LangChain Core, LangGraph SQLite Checkpointer
- **LLM & Embeddings**: Ollama (`llama3.1:8b`, `moondream`, `nomic-embed-text`) / Google Gemini API
- **Vector Search**: Pinecone
- **Web Search**: Tavily API
- **Document Processing**: PyMuPDF (`fitz`)

---

## 📋 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/scahin-knight/Medical-Agentic-ChatBOT.git
cd Medical-Agentic-ChatBOT
```

### 2. Create and Activate Environment
```bash
conda create -n project python=3.11 -y
conda activate project
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
Key variables:
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: `medical-agent-demo`
- `TAVILY_API_KEY`: Your Tavily API key
- `API_TOKEN`: Secret token for API authentication
- `OLLAMA_CHAT_MODEL`: `llama3.1:8b`
- `OLLAMA_VISION_MODEL`: `moondream`

### 4. Pull Local Ollama Models
Ensure [Ollama](https://ollama.com) is installed and running:
```bash
ollama pull llama3.1:8b
ollama pull moondream
ollama pull nomic-embed-text
```

### 5. Ingest Knowledge Base (Optional)
Place medical PDFs in `data/` and run:
```bash
python ingest.py
```

### 6. Run the Application
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at **http://127.0.0.1:8000**.

---

## ⚠️ Disclaimer
This application is strictly for **educational and demonstration purposes**. It is **not** a licensed medical device or a substitute for professional clinical advice, diagnosis, or treatment.
