# 📚 Multilingual RAG Chatbot (Bangla + English)

This is a simple **Retrieval-Augmented Generation (RAG)** system built to answer both **Bengali (Bangla)** and **English** questions using content from a Bangla PDF textbook.

🔍 The system reads from `HSC Bangla 1st Paper`, parses MCQs and stories (even from Bijoy fonts), and answers queries using an LLM powered by **Ollama + Gemma3**.

---

##  Features

- ✅ Supports both Bangla and English queries
- ✅ Parses Bijoy-encoded Bangla fonts from PDFs
- ✅ Uses LangChain + Ollama + ChromaDB + Flask
- ✅ Contextual QA using retrieved PDF content
- ✅ Web interface + REST API endpoint (`/api/ask`)
- ✅ SBERT-based Bengali embeddings (`l3cube-pune`)


---

##  Setup Instructions

###  1. Clone the repository

```bash
git clone https://github.com/Afif404/multilingual_rag.git
cd multilingual_rag
```

---
### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```
---
### 3. Install dependencies

```bash
pip install -r requirements.txt
```
---
###  4. Configure environment variables

```bash
TEXT_PATH=data/HSC26-Bangla1st-Paper.pdf
VECTOR_STORE_DIR=vector_store
USE_CUDA=0
```

---
###  5. Install Ollama & Pull the LLM
Windows:Download from https://ollama.com/download.)

```shell (for windows after download)
ollama run gemma3
```
---
for macOS
```for macOS 
brew install ollama
```
---
###  Run The app

```bash
python app.py
```
---

Now visit: http://127.0.0.1:5000

---

## 🧰 Used Tools and Libraries

This project combines modern NLP infrastructure with practical Bangla support for building a multilingual RAG chatbot:

---

### 🔤 Language Processing & Embeddings

- **langchain**: Core framework for managing LLMs, retrievers, chains, prompts, and memory.
- **langchain-ollama**: Integrates local LLMs (like `gemma3`) from Ollama with LangChain.
- **langchain-huggingface**: Loads and uses Hugging Face models (Bangla SBERT).
- **l3cube-pune/bengali-sentence-similarity-sbert**: Embedding model trained for Bangla sentence similarity.

---

### 📚 PDF Parsing & Bangla Support

- **pdfminer.six**: Extracts text from PDFs, supports parsing paragraphs, tables, and footnotes.
- 🔄 **Bijoy → Unicode Converter**: Custom-built logic to normalize corrupted Bangla glyphs from legacy Bijoy fonts.
- **langdetect**: Detects whether a user query is in Bangla or English to switch prompts dynamically.

---

### 🧠 Vector Store & Search

- **chromadb**: Lightweight vector database for storing and retrieving text chunks.
- **langchain-chroma**: LangChain wrapper for ChromaDB.

---

### 🧪 Local LLM Execution

- **ollama**: Runs the `gemma3` LLM locally without external API calls.

---

### 🌐 Web & API

- **flask**: Lightweight web framework for serving both the API and the frontend.
- **flask-cors**: Enables cross-origin requests for frontend testing or external clients.
- **python-dotenv**: Loads `.env` configuration for file paths, model settings, etc.

---

### ⚙️ Other Dependencies

- **sentence-transformers**, **sentencepiece**, **safetensors**, **accelerate**: Required for loading and accelerating Hugging Face embedding models.

---


## 📡 API Documentation

This project also provides a lightweight REST API for external access to the RAG chatbot engine.

---

### 🔹 Endpoint: `/api/ask`

Accepts a Bangla or English query in JSON format and returns a response generated using relevant content from the PDF.

---

### 📥 Request

- **Method:** `POST`
- **URL:** `http://127.0.0.1:5000/api/ask`
- **Headers:** `Content-Type: application/json`
- **Body Example:**

```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

---
## 🧪 Evaluation Summary

### 📄 Text Extraction
- Used `pdfminer.six` for accurate layout parsing.
- Faced significant issues parsing Bangla PDF content due to **Bijoy font encoding**.
- Resolved with custom **Bijoy → Unicode conversion** logic.

### ✂️ Chunking Strategy
- Hybrid method:
  - MCQs split using regex patterns (e.g., `প্রশ্ন \d+`, `\d+\.`).
  - Stories chunked with `RecursiveCharacterTextSplitter` (`chunk_size=800`, `overlap=150`).
- Ensures semantically rich and retrieval-friendly chunks.

### 🧠 Embedding Model
- Used `l3cube-pune/bengali-sentence-similarity-sbert`.
- Chosen for Bangla sentence-level semantic understanding.
- Works well with SBERT-style similarity-based search.

### 🔍 Vector Retrieval
- Stored embeddings in `ChromaDB`.
- Used `MMR` (Max Marginal Relevance) search with `k=3` for balanced, diverse retrieval.
- Chroma was selected for local, persistent, and fast vector search.

### 💬 Query Comparison
- Language detected via `langdetect`, prompt switches accordingly.
- Context + question passed to `gemma3` via LangChain's conversational RAG.
- Handles vague queries reasonably, though fallback logic could improve robustness.

### ✅ Result Quality
- Relevant answers for well-formed Bangla queries (e.g., শুম্ভুনাথ, মামাকে).
- Could improve with:
  - Better OCR for scanned text
  - Larger chunk context
  - More advanced multilingual models

