from flask import Flask, jsonify, request
from flask_cors import CORS
import os, json, requests, re, uuid, time
from io import BytesIO
import numpy as np

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# In-memory store: session_id -> {chunks, embeddings, metadata, history}
sessions = {}

# ── PDF EXTRACTION ──
def extract_text(file_bytes):
    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text.strip(), "page": i + 1})
        return pages
    except Exception as e:
        return []

def chunk_text(pages, chunk_size=500, overlap=100):
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        words = text.split()
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if len(chunk_text.strip()) > 50:
                chunks.append({
                    "id": str(uuid.uuid4())[:8],
                    "text": chunk_text,
                    "page": page_data["page"],
                    "word_count": len(chunk_words)
                })
            i += chunk_size - overlap
    return chunks

# ── EMBEDDINGS ──
def get_embedding(text):
    if not GEMINI_API_KEY:
        return get_mock_embedding(text)
    try:
        res = requests.post(
            f"{GEMINI_EMBED_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={"model": "models/text-embedding-004", "content": {"parts": [{"text": text[:2000]}]}},
            timeout=15
        )
        res.raise_for_status()
        return res.json()["embedding"]["values"]
    except:
        return get_mock_embedding(text)

def get_mock_embedding(text):
    np.random.seed(hash(text[:100]) % 2**31)
    return np.random.rand(768).tolist()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def retrieve_chunks(query, session_id, top_k=4):
    if session_id not in sessions:
        return []
    session = sessions[session_id]
    query_emb = get_embedding(query)
    scored = []
    for i, chunk in enumerate(session["chunks"]):
        if i < len(session["embeddings"]):
            score = cosine_similarity(query_emb, session["embeddings"][i])
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]

# ── GEMINI CHAT ──
def call_gemini_chat(messages):
    if not GEMINI_API_KEY:
        return "Gemini API key not configured. Please add GEMINI_API_KEY environment variable."
    try:
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        res = requests.post(
            f"{GEMINI_CHAT_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={"contents": contents,
                  "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024}},
            timeout=30
        )
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"Error calling Gemini: {str(e)}"

# ── ROUTES ──
@app.route("/")
def index():
    return open("index.html").read()

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files supported"}), 400
    file_bytes = file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "File too large. Max 10MB"}), 400

    pages = extract_text(file_bytes)
    if not pages:
        return jsonify({"error": "Could not extract text. Make sure PDF is not a scanned image."}), 400

    chunks = chunk_text(pages)
    if not chunks:
        return jsonify({"error": "Could not process document"}), 400

    session_id = str(uuid.uuid4())[:12]
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk["text"])
        embeddings.append(emb)
        time.sleep(0.05)

    total_words = sum(c["word_count"] for c in chunks)
    sessions[session_id] = {
        "chunks": chunks,
        "embeddings": embeddings,
        "history": [],
        "filename": file.filename,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "total_words": total_words,
        "created_at": time.time()
    }

    # Auto-generate document summary
    sample_text = " ".join([c["text"] for c in chunks[:3]])[:2000]
    summary_prompt = f"""Analyze this document and respond in JSON only (no markdown):
{{
  "title": "document title or topic",
  "summary": "2-3 sentence overview",
  "key_topics": ["topic1", "topic2", "topic3", "topic4", "topic5"],
  "document_type": "type of document (textbook/report/manual/research/other)",
  "suggested_questions": ["question1?", "question2?", "question3?", "question4?"]
}}
Document text: {sample_text}"""

    summary_response = call_gemini_chat([{"role": "user", "content": summary_prompt}])
    try:
        clean = summary_response.replace("```json", "").replace("```", "").strip()
        doc_info = json.loads(clean)
    except:
        doc_info = {
            "title": file.filename.replace(".pdf", ""),
            "summary": f"Document with {len(pages)} pages and {total_words} words processed successfully.",
            "key_topics": ["Document Content", "Key Information", "Main Topics"],
            "document_type": "Document",
            "suggested_questions": [
                "What is this document about?",
                "What are the main topics covered?",
                "Can you summarize the key points?",
                "What are the most important findings?"
            ]
        }

    sessions[session_id]["doc_info"] = doc_info

    return jsonify({
        "success": True,
        "session_id": session_id,
        "filename": file.filename,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "total_words": total_words,
        "doc_info": doc_info
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    question = data.get("question", "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session not found. Please upload a document first."}), 400
    if not question:
        return jsonify({"error": "Question is empty"}), 400

    session = sessions[session_id]
    relevant_chunks = retrieve_chunks(question, session_id, top_k=4)

    context = "\n\n---\n\n".join([
        f"[Page {c['page']}]: {c['text']}" for c in relevant_chunks
    ])

    system_prompt = f"""You are an intelligent document assistant. Answer questions based ONLY on the provided document context.

Rules:
- Answer based strictly on the document content
- If the answer is not in the document, say "This information is not found in the document"
- Be clear, concise and helpful
- Mention page numbers when referencing specific content
- Format your response with clear structure when needed

Document: {session['doc_info'].get('title', 'Uploaded Document')}

DOCUMENT CONTEXT:
{context}"""

    messages = [{"role": "user", "content": system_prompt}]
    for h in session["history"][-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": question})

    answer = call_gemini_chat(messages)

    session["history"].append({"role": "user", "content": question})
    session["history"].append({"role": "assistant", "content": answer})

    source_pages = list(set([c["page"] for c in relevant_chunks]))

    return jsonify({
        "answer": answer,
        "source_pages": source_pages,
        "chunks_used": len(relevant_chunks),
        "question": question
    })

@app.route("/api/session/<session_id>", methods=["GET"])
def get_session(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    s = sessions[session_id]
    return jsonify({
        "filename": s["filename"],
        "total_pages": s["total_pages"],
        "total_chunks": s["total_chunks"],
        "total_words": s["total_words"],
        "doc_info": s["doc_info"],
        "history": s["history"]
    })

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "gemini": bool(GEMINI_API_KEY), "sessions": len(sessions)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
