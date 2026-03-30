"""
app.py — МУИС chatbot Flask backend

Embedder: paraphrase-multilingual-mpnet-base-v2 (Монгол хэлд тохирсон)
Model:    claude-haiku-4-5-20251001
"""

import os, markdown, anthropic
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
EMBED_MODEL   = "paraphrase-multilingual-mpnet-base-v2"  # ingest.py-тэй ИЖИЛ байх ёстой

# Монгол хэлний синоним — хайлтыг баяжуулна
MN_SYNONYMS = {
    "кредит":    "багц цаг кредит",
    "курс":      "түвшин курс дугаар",
    "хэддүгээр": "дугаар түвшин",
    "голч":      "голч дүн оноо GPA",
    "дүн":       "үнэлгээ дүн оноо тэмдэглэгээ",
    "чөлөө":     "чөлөө авах суралцагч журам",
    "хасах":     "сургуулиас хасах шалтгаан",
    "төлбөр":    "сургалтын төлбөр хураамж",
    "шалгалт":   "шалгалт явцын улирлын",
    "бүртгэл":   "хичээл бүртгүүлэх сонгох",
    "тэтгэлэг":  "тэтгэлэг зээл санхүүгийн дэмжлэг",
    "төгсөлт":   "төгсөлт дүүргэх шаардлага диплом",
    "дадлага":   "дадлага мэргэжлийн магистр",
}

# Хайлтыг хүчитгэх түлхүүр үгс
BOOST_KEYWORDS = {
    "W", "WF", "NR", "NA", "CA", "CR", "RC", "GPA",
    "ГОЛЧ", "ҮНЭЛГЭЭ", "ТЭМДЭГЛЭГЭЭ", "КРЕДИТ", "БАГЦ",
    "ТҮВШИН", "КУРС", "ХАСАХ", "ЧӨЛӨӨ", "ТӨГСӨЛТ",
}

# ── КЛИЕНТҮҮД (нэг удаа ачаална) ────────────────────────
print(f"🔧 Embedder ачааллаж байна: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedder бэлэн.")

claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
pc     = Pinecone(api_key=PINECONE_KEY)
index  = pc.Index(INDEX_NAME)

# ── RAG ХАЙЛТ ───────────────────────────────────────────
def expand_query(query: str) -> str:
    """Монгол синонимоор асуултыг баяжуулна."""
    q_lower = query.lower()
    extras  = []
    for mn, exp in MN_SYNONYMS.items():
        if mn in q_lower:
            extras.append(exp)
    if extras:
        return f"{query} {' '.join(extras)}"
    return query


def search_context(query: str, top_k: int = 12) -> list[dict]:
    """Pinecone-ээс хамааралтай chunk хайна."""
    try:
        if any(kw in query.upper() for kw in BOOST_KEYWORDS):
            top_k = 20

        expanded = expand_query(query)
        vector   = embedder.encode(expanded).tolist()

        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )

        matches = []
        for m in results["matches"]:
            score = m.get("score", 0)
            if score < 0.10:
                continue
            meta = m.get("metadata", {})
            matches.append({
                "text":   meta.get("text", ""),
                "source": meta.get("source", "МУИС-ийн баримт бичиг"),
                "score":  round(score, 3),
            })

        return matches

    except Exception as e:
        print(f"❌ Search error: {e}")
        return []


def build_context_block(matches: list[dict]) -> str:
    """Chunk-уудыг эх сурвалжтай нэгтгэнэ."""
    if not matches:
        return ""
    parts = []
    for m in matches:
        parts.append(f"[{m['source']}]\n{m['text']}")
    return "\n\n---\n\n".join(parts)

# ── СИСТЕМ ПРОМТ ────────────────────────────────────────
SYSTEM_BASE = """Та бол МУИС-ийн "МУИС-Туслах" чатбот юм.

ХЭЛНИЙ ЗААВАР:
- Хэрэглэгч Монгол хэлний АЛИВАА хэлбэрээр асуусан байсан утгаар нь ойлго.
- "кредит" = "багц цаг", "курс" = "түвшин", "хэддүгээр" = "дугаар түвшин",
  "голч" = "голч дүн / GPA", "дүн" = "үнэлгээ", "хасах" = "сургуулиас хасах"
- Үг яг таарахгүй байсан ч утгаар нь хариул.

АЖИЛЛАХ ДҮРЭМ:
1. Зөвхөн доор өгсөн КОНТЕКСТ-ийн мэдээллийг ашигла.
2. Контекстэд байхгүй мэдээллийг өөрийн мэдлэгээр нэмж хэлэхийг ХОРИГЛОНО.
3. Хариулт контекстэд байхгүй бол:
   "Уучлаарай, миний мэдээллийн санд энэ тухай мэдээлэл байхгүй байна." гэж хариул.
4. Үсгэн тэмдэглэгээ (W, WF, I, R, F, CA, NR, NA, CR, RC гэх мэт) асуувал
   тус бүрийн тайлбарыг дэлгэрэнгүй өг.
5. Тоон мэдээлэл (багц цаг, түвшин, хувь) яг зөв дамжуул.
6. Монгол хэлээр товч, цэгцтэй хариул.
7. "Багшаар уулз", "захиргаанд ханд" гэх мэт ерөнхий зөвлөгөө өгөхгүй —
   контекстэд байгаа тодорхой мэдээллийг л хэлэх."""


def build_system_prompt(context: str) -> str:
    ctx = context if context else "Мэдээллийн санд холбогдох баримт олдсонгүй."
    return f"{SYSTEM_BASE}\n\nКОНТЕКСТ:\n{ctx}"

# ── ROUTE-УУД ───────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data         = request.json or {}
    user_message = data.get("message", "").strip()
    history      = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Хоосон асуулт"}), 400

    try:
        # 1. RAG хайлт
        matches      = search_context(user_message)
        context_text = build_context_block(matches)

        # 2. Яриын түүх (сүүлийн 6 мессеж)
        messages = []
        for m in history[-6:]:
            role    = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})

        # 3. Claude дуудах
        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            temperature=0,
            system=build_system_prompt(context_text),
            messages=messages,
        )

        raw_text  = response.content[0].text
        html_text = markdown.markdown(
            raw_text,
            extensions=["tables", "nl2br", "fenced_code"],
        )

        sources = list({m["source"] for m in matches})

        return jsonify({
            "answer":  html_text,
            "sources": sources,
        })

    except anthropic.APIStatusError as e:
        code = e.status_code
        if code == 429:
            msg = "⏳ Хэтэрхий олон хүсэлт. Хэсэг хүлээгээд дахин оролдоно уу."
        elif code == 401:
            msg = "❌ Claude API key буруу байна."
        elif code == 529:
            msg = "❌ Claude сервер ачаалалтай байна. Дахин оролдоно уу."
        else:
            msg = f"❌ Claude алдаа ({code})"
        return jsonify({"error": msg}), 500

    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({"error": f"Алдаа гарлаа: {str(e)}"}), 500


@app.route("/health")
def health():
    """Серверийн байдал шалгах endpoint."""
    try:
        stats = index.describe_index_stats()
        return jsonify({
            "status":       "ok",
            "vector_count": stats.total_vector_count,
            "embed_model":  EMBED_MODEL,
        })
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)