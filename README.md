# 🧠 DocuMind AI — RAG-Powered Document Q&A Chatbot

> Upload any PDF and chat with it using AI. Built with a full RAG (Retrieval-Augmented Generation) pipeline — text extraction, smart chunking, vector embeddings, similarity search and Gemini AI answers with page citations.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![Gemini](https://img.shields.io/badge/Google-Gemini%20AI-orange.svg)](https://ai.google.dev)
[![RAG](https://img.shields.io/badge/Architecture-RAG%20Pipeline-purple.svg)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 📄 **PDF Upload** | Drag & drop or click to upload any PDF |
| 🧩 **Smart Chunking** | Splits text into overlapping 500-word segments |
| 🔢 **Vector Embeddings** | Gemini text-embedding-004 for semantic search |
| 🔍 **Similarity Search** | Cosine similarity to find most relevant chunks |
| 🤖 **Gemini AI Answers** | Context-aware answers from document only |
| 📍 **Page Citations** | Every answer shows source page numbers |
| 💬 **Multi-turn Chat** | Maintains conversation history |
| 🗺️ **Auto Summary** | AI generates document summary and key topics |
| 💡 **Smart Suggestions** | Auto-generates 4 relevant questions |

---

## 🏗️ RAG Architecture

```
PDF Upload → Text Extraction (PyPDF2)
         → Smart Chunking (500 words, 100 overlap)
         → Vector Embeddings (Gemini text-embedding-004)
         → In-memory Vector Store
         
User Query → Query Embedding
          → Cosine Similarity Search (Top-4 chunks)
          → Context + History → Gemini 2.0 Flash
          → Answer + Page Citations
```

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, REST API
- **AI/LLM:** Google Gemini 2.0 Flash (chat) + text-embedding-004 (embeddings)
- **RAG:** Custom pipeline — chunking, embedding, cosine similarity search
- **PDF:** PyPDF2
- **Vector Math:** NumPy
- **Frontend:** Vanilla JS, HTML5, CSS3 (dark animated UI)
- **Deploy:** Render, Gunicorn

---

## 🚀 Quick Start

```bash
git clone https://github.com/Harsha-636/doc-qa-chatbot.git
cd doc-qa-chatbot
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
python app.py
```

Get free Gemini API key: https://aistudio.google.com/app/apikey

---

## ☁️ Deploy on Render

1. Push to GitHub
2. Connect repo on render.com
3. Add env variable: `GEMINI_API_KEY=your_key`
4. Build: `pip install -r requirements.txt`
5. Start: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`

---

## 👨‍💻 Author

**Sai Harsha Vardhan Reddy Avula** — B.Tech CSE @ KMCE Hyderabad (2027)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/harsha-avula)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Harsha-636)
