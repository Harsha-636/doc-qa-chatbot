from flask import Flask, jsonify, request
from flask_cors import CORS
import os, json, requests, uuid, time
from io import BytesIO
import numpy as np

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"

sessions = {}

def extract_text(file_bytes):
    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text.strip(), "page": i + 1})
        return pages
    except:
        return []

def chunk_text(pages, chunk_size=400, overlap=80):
    chunks = []
    for page_data in pages:
        words = page_data["text"].split()
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_str = " ".join(chunk_words)
            if len(chunk_str.strip()) > 50:
                chunks.append({
                    "id": str(uuid.uuid4())[:8],
                    "text": chunk_str,
                    "page": page_data["page"],
                    "word_count": len(chunk_words)
                })
            i += chunk_size - overlap
    return chunks

def get_mock_embedding(text):
    np.random.seed(hash(text[:100]) % 2**31)
    return np.random.rand(768).tolist()

def get_embedding(text):
    # Use mock embeddings to save Gemini quota for chat
    return get_mock_embedding(text)

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

def call_gemini_chat(messages):
    if not GEMINI_API_KEY:
        return "Gemini API key not configured."
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
        if res.status_code == 429:
            time.sleep(5)
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
        return f"Error: {str(e)}"

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

    pages = pages[:15]
    chunks = chunk_text(pages)
    if not chunks:
        return jsonify({"error": "Could not process document"}), 400

    chunks = chunks[:20]
    session_id = str(uuid.uuid4())[:12]
    embeddings = [get_embedding(c["text"]) for c in chunks]

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

    sample_text = " ".join([c["text"] for c in chunks[:2]])[:1500]
    summary_prompt = f"""Analyze this document and respond in JSON only (no markdown, no backticks):
{{"title":"document title","summary":"2-3 sentence overview","key_topics":["topic1","topic2","topic3","topic4"],"document_type":"textbook/report/manual/research/other","suggested_questions":["question1?","question2?","question3?","question4?"]}}
Document: {sample_text}"""

    summary_response = call_gemini_chat([{"role": "user", "content": summary_prompt}])
    try:
        clean = summary_response.replace("```json","").replace("```","").strip()
        doc_info = json.loads(clean)
    except:
        doc_info = {
            "title": file.filename.replace(".pdf",""),
            "summary": f"Document with {len(pages)} pages processed successfully.",
            "key_topics": ["Document Content","Key Information","Main Topics","Key Findings"],
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

    system_prompt = f"""You are a helpful document assistant. Answer ONLY from the document context below.
If the answer is not in the document, say "This information is not found in the document."
Mention page numbers when referencing content. Be clear and concise.

Document: {session['doc_info'].get('title','Uploaded Document')}

CONTEXT:
{context}"""

    messages = [{"role": "user", "content": system_prompt}]
    for h in session["history"][-4:]:
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

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "gemini": bool(GEMINI_API_KEY), "sessions": len(sessions)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
