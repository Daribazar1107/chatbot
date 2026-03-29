import os
import markdown
import anthropic
from flask import Flask, render_template, request, jsonify
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── ТОХИРГОО ────────────────────────────────────────────
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_KEY  = os.getenv("PINECONE_API_KEY")
INDEX_NAME    = "muis-chatbot"

claude   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
pc       = Pinecone(api_key=PINECONE_KEY)
index    = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── ХАЙЛТ ───────────────────────────────────────────────
def search_context(query: str, top_k: int = 10) -> list[dict]:
    """Pinecone-ээс хамааралтай chunk-уудыг score-тэй хамт буцаана."""
    try:
        vec     = embedder.encode(query).tolist()
        results = index.query(vector=vec, top_k=top_k, include_metadata=True)
        matches = []
        for m in results["matches"]:
            score = m.get("score", 0)
            if score < 0.25:   # хамааралгүй chunk хасна
                continue
            matches.append({
                "text":   m["metadata"].get("text", ""),
                "source": m["metadata"].get("source", ""),
                "score":  round(score, 3),
            })
        return matches
    except Exception as e:
        print(f"Search error: {e}")
        return []


def build_context_block(matches: list[dict]) -> str:
    """Chunk-уудыг эх сурвалжтай нь нэгтгэж систем промт руу оруулна."""
    if not matches:
        return ""
    parts = []
    for m in matches:
        parts.append(f"[Эх сурвалж: {m['source']}]\n{m['text']}")
    return "\n\n---\n\n".join(parts)


# ── СИСТЕМ ПРОМТ ────────────────────────────────────────
def build_system_prompt(context: str, user_query: str) -> str:
    if not context:
        return """Та МУИС-ийн оюутны туслах чатбот юм.
Одоогоор мэдээллийн санд холбогдсон баримт бичиг байхгүй байна.
Монгол хэлээр хариулна уу."""

    return f"""Та МУИС-ийн оюутны туслах чатбот юм. Доор өгсөн баримт бичгийн мэдээлэлд тулгуурлан хариулна уу.

═══════════════════════════════
ХЭРЭГЛЭГЧИЙН АСУУЛТ: {user_query}
═══════════════════════════════

ДҮРЭМ:
1. Зөвхөн доор өгсөн БАРИМТ БИЧГИЙН мэдээллээр хариулна. Өөрийн мэдлэгийг нэмж болохгүй.
2. Хариулт баримт бичигт байгаа бол ДЭЛГЭРЭНГҮЙ, ОЙЛГОМЖТОЙгоор тайлбарлана.
3. Хэрэв олон хүн/зүйл олдвол жагсааж харуулаад "Алийг нь дэлгэрэнгүй мэдэхийг хүсэж байна вэ?" гэж асуу.
4. Баримт бичигт хариулт БАЙХГҮЙ бол: "Уучлаарай, энэ мэдээлэл баримт бичигт байхгүй байна." гэж хариулна.
5. Хариултыг Монгол хэлээр цэгцтэй, товч бөгөөд бүрэн бичнэ.
6. Тоо, огноо, нэр зэрэг баримтуудыг ЯНДАРГҮЙ оруулна.

БАРИМТ БИЧИГ:
{context}"""


# ── ROUTE-УУД ───────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data         = request.json
    user_message = data.get("message", "").strip()
    history      = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Хоосон асуулт"}), 400

    try:
        # 1. Pinecone-ээс хамааралтай мэдээлэл хайна
        matches = search_context(user_message)
        context = build_context_block(matches)

        # 2. Яриын түүхийг бэлдэнэ (сүүлийн 6 мессеж)
        messages = []
        for msg in history[-6:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({
                    "role":    msg["role"],
                    "content": msg["content"],
                })
        messages.append({"role": "user", "content": user_message})

        # 3. Claude-д дуудна
        response = claude.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 1500,
            system     = build_system_prompt(context, user_message),
            messages   = messages,
        )

        raw_text       = response.content[0].text
        formatted_html = markdown.markdown(
            raw_text,
            extensions=["tables", "nl2br"],
        )

        # Эх сурвалжийн мэдээллийг хариулттай хамт буцаана (debug)
        sources = list({m["source"] for m in matches}) if matches else []

        return jsonify({
            "answer":  formatted_html,
            "sources": sources,
        })

    except anthropic.APIStatusError as e:
        code = e.status_code
        if code == 429:
            return jsonify({"error": "⏳ Хэтэрхий олон хүсэлт. Хэсэг хүлээгээд дахин оролдоно уу."}), 429
        elif code == 401:
            return jsonify({"error": "❌ Claude API key буруу байна."}), 401
        else:
            return jsonify({"error": f"❌ Claude алдаа ({code})"}), 500

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": f"❌ Алдаа: {str(e)}"}), 500


@app.route("/reload", methods=["POST"])
def reload():
    """data/ хавтасны файлуудыг дахин индексжүүлнэ."""
    try:
        import subprocess
        result = subprocess.run(
            ["python", "ingest.py"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            return jsonify({"success": True, "log": result.stdout})
        else:
            return jsonify({"success": False, "log": result.stderr}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)