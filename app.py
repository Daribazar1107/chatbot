import os
import markdown
import anthropic
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
load_dotenv()
# API keys
# API keys - No longer hardcoded
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_KEY  = os.getenv("PINECONE_API_KEY")
INDEX_NAME    = "muis-chatbot"

# Safety check: Ensure keys are loaded
if not ANTHROPIC_KEY or not PINECONE_KEY:
    raise ValueError("API keys not found. Check your .env file!")
# Clients
claude   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
pc       = Pinecone(api_key=PINECONE_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
CHUNK_SIZE = 500

def chunk_text(text: str) -> list:
    words = text.split()
    chunks, current, length = [], [], 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= CHUNK_SIZE:
            chunks.append(" ".join(current))
            current, length = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks

def index_pdfs(pdf_files) -> int:
    all_chunks = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        all_chunks.extend(chunk_text(text))
    if not all_chunks:
        return 0
    index.delete(delete_all=True)
    vectors = []
    for i, chunk in enumerate(all_chunks):
        embedding = embedder.encode(chunk).tolist()
        vectors.append({"id": f"chunk-{i}", "values": embedding, "metadata": {"text": chunk}})
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])
    return len(all_chunks)

def search_context(query: str, top_k: int = 5) -> str:
    query_embedding = embedder.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    chunks = [m["metadata"]["text"] for m in results["matches"]]
    return "\n\n".join(chunks)

def build_system_prompt(context: str) -> str:
    if not context:
        return """Та МУИС-ийн оюутнуудад туслах чатбот юм.
Одоогоор ямар ч баримт бичиг хуулаагүй байна.
Хэрэглэгчид PDF файл хуулахыг хүсч байгааг мэдэгдэнэ үү.
Монгол хэлээр хариулна уу."""

    return f"""Та МУИС-ийн оюутнуудад туслах чатбот юм.

ДҮРЭМ:
- Зөвхөн доор өгсөн баримт бичгийн мэдээлэлд тулгуурлан хариулна
- Баримт бичигт байхгүй мэдээллийг өөрөөс нэмж хэлэхгүй
- Хэрэв баримт бичигт хариулт байхгүй бол: "Уучлаарай, энэ мэдээлэл баримт бичигт байхгүй байна." гэж хэлнэ
- Монгол хэлээр хариулна уу

БАРИМТ БИЧИГ:
{context}"""

# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.route("/")
def index_route():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Файл олдсонгүй"}), 400
    files = request.files.getlist("file")
    try:
        count = index_pdfs(files)
        return jsonify({"chunks": count, "success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data         = request.json
    user_message = data.get("message", "")
    history      = data.get("history", [])
    has_index    = data.get("has_index", False)

    try:
        context = search_context(user_message) if has_index else ""
        messages = []
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=build_system_prompt(context),
            messages=messages,
        )
        raw_text  = response.content[0].text
        html_text = markdown.markdown(raw_text)
        return jsonify({"answer": html_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
