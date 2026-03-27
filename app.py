import os
import markdown
import anthropic
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()
app = Flask(__name__)

anthropic_key = os.getenv("ANTHROPIC_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
index_name = "muis_chatbot"

if not anthropic_key or not pinecone_key:
    raise ValueError("api keys missing")

claude = anthropic.Anthropic(api_key=anthropic_key)
pc = Pinecone(api_key=pinecone_key)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

def chunk_text(text):
    words = text.split()
    chunks, current, length = [], [], 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= 500:
            chunks.append(" ".join(current))
            current, length = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks

@app.route("/")
def index_route():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("file")
    all_chunks = []
    for f in files:
        reader = PdfReader(f)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        all_chunks.extend(chunk_text(text))
    
    if not all_chunks:
        return jsonify({"success": False})

    index.delete(delete_all=True)
    vectors = []
    for i, chunk in enumerate(all_chunks):
        embedding = embedder.encode(chunk).tolist()
        vectors.append({"id": f"c_{i}", "values": embedding, "metadata": {"text": chunk}})
    
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])
    
    return jsonify({"success": True, "chunks": len(all_chunks)})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    msg = data.get("message", "")
    hist = data.get("history", [])
    
    context = ""
    if data.get("has_index"):
        q_emb = embedder.encode(msg).tolist()
        res = index.query(vector=q_emb, top_k=3, include_metadata=True)
        context = "\n".join([m["metadata"]["text"] for m in res["matches"]])

    sys_prompt = f"Та МУИС-ийн оюутны туслах. Монголоор хариулна. Контекст: {context}"
    
    messages = [{"role": m["role"], "content": m["content"]} for m in hist[-6:]]
    messages.append({"role": "user", "content": msg})

    resp = claude.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        system=sys_prompt,
        messages=messages
    )
    
    html_answer = markdown.markdown(resp.content[0].text)
    return jsonify({"answer": html_answer})

if __name__ == "__main__":
    app.run(debug=True)