"""
app.py — МУИС chatbot Flask backend
Redis Cache + Query Classifier нэмсэн хувилбар

Embedder: paraphrase-multilingual-mpnet-base-v2
Model:    claude-haiku-4-5-20251001
"""

import os, markdown, anthropic
from flask import Flask, render_template, request, jsonify
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from cache import get_cached, set_cached, stats as cache_stats
from query_rewriter import rewrite as rewrite_query
from retrieval import bm25_index, hybrid_search

load_dotenv()

app = Flask(__name__)

# ── ТОХИРГОО ────────────────────────────────────────────
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_KEY  = os.getenv("PINECONE_API_KEY")
INDEX_NAME    = "muis-chatbot2"
EMBED_MODEL   = "paraphrase-multilingual-mpnet-base-v2"

RELEVANCE_THRESHOLD = 0.60

TRUSTED_SOURCES = {
    "журам.json",
    "chuluu.json",
    "teachers.json",
    "grading.json",
    "level.json",
    "schedule.json",
    "courses.json",
}
REJECT_MESSAGE = (
    "Уучлаарай,асуултаа тодорхой тавина уу, "
    "би зөвхөн МУИС-ийн журам, дүрэм, "
    "академик бодлоготой холбоотой асуултад хариулах боломжтой. "
    "Таны асуулт энэ хүрээнд хамаарахгүй байна. 🎓"
)

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

BOOST_KEYWORDS = {
    "W", "WF", "NR", "NA", "CA", "CR", "RC", "GPA",
    "ГОЛЧ", "ҮНЭЛГЭЭ", "ТЭМДЭГЛЭГЭЭ", "КРЕДИТ", "БАГЦ",
    "ТҮВШИН", "КУРС", "ХАСАХ", "ЧӨЛӨӨ", "ТӨГСӨЛТ",
}

MUИС_KEYWORDS = [
    "журам", "дүрэм", "бодлого", "оюутан", "элсэлт", "төгсөлт",
    "шалгалт", "дипломын", "зэрэг", "тэнхим", "сургууль", "бүртгэл",
    "суралцах", "хичээл", "кредит", "gpa", "оноо", "стипенди",
    "дотуур", "байр", "сургалтын", "хураамж", "чөлөө", "хасагдах",
    "шагнал", "тэтгэлэг", "практик", "дадлага", "хамгаалалт",
    "голч", "дүн", "үнэлгээ", "кредит", "курс", "түвшин",
    # Үсгэн тэмдэглэгээ
    " w ", "wf", " f ", " i ", " r ", "cr", "ca", "nr", "na", "rc",
    " g ", "grade", "grading", "тэмдэглэгээ", "үнэлгээний",
]

# ── КЛИЕНТҮҮД ────────────────────────────────────────────
print(f"🔧 Embedder ачааллаж байна: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedder бэлэн.")

claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
pc     = Pinecone(api_key=PINECONE_KEY)
index  = pc.Index(INDEX_NAME)


def _build_bm25():
    try:
        stats  = index.describe_index_stats()
        total  = stats.total_vector_count
        print(f"🔧 BM25 index үүсгэж байна ({total} vector)...")
        dummy  = embedder.encode("МУИС журам").tolist()
        result = index.query(vector=dummy, top_k=min(total, 8000), include_metadata=True)
        chunks = [
            {"text": m.metadata.get("text", ""), "source": m.metadata.get("source", "")}
            for m in result.matches if m.metadata.get("text")
        ]
        bm25_index.build(chunks)
    except Exception as e:
        print(f"⚠️  BM25 index үүсгэж чадсангүй: {e}")

_build_bm25()


# ── QUERY CLASSIFIER ─────────────────────────────────────

def classify_query(query: str) -> dict:
    query = query.strip()
    if len(query) < 3:
        return {"is_relevant": False, "score": 0.0, "method": "empty"}

    # Шат 1: Keyword — хурдан pass
    q_lower = query.lower()
    if any(kw in q_lower for kw in MUИС_KEYWORDS):
        return {"is_relevant": True, "score": 1.0, "method": "keyword_pass"}

    # Шат 2: Embedding + trusted source
    try:
        expanded = expand_query(query)
        vector   = embedder.encode(expanded).tolist()
        results  = index.query(vector=vector, top_k=5, include_metadata=True)
        if not results["matches"]:
            return {"is_relevant": False, "score": 0.0, "method": "no_matches"}
        best_score = results["matches"][0]["score"]
        for m in results["matches"]:
            src   = m.get("metadata", {}).get("source", "")
            score = m.get("score", 0)
            if src in TRUSTED_SOURCES and score >= RELEVANCE_THRESHOLD:
                return {"is_relevant": True, "score": round(score, 3), "method": "trusted_source"}
        return {"is_relevant": False, "score": round(best_score, 3), "method": "no_trusted_source"}

    except Exception as e:
        print(f"⚠️ Classifier error: {e}")
        # Шат 3: Keyword fallback
        return {"is_relevant": any(kw in q_lower for kw in MUИС_KEYWORDS), "score": 0.0, "method": "keyword_fallback"}


# ── RAG ХАЙЛТ ────────────────────────────────────────────

def expand_query(query: str) -> str:
    q_lower = query.lower()
    extras  = [exp for mn, exp in MN_SYNONYMS.items() if mn in q_lower]
    return f"{query} {' '.join(extras)}" if extras else query


def search_context(query: str, top_k: int = 12) -> list[dict]:
    try:
        if any(kw in query.upper() for kw in BOOST_KEYWORDS):
            top_k = 20

        # 1. Query Rewrite: HyDE + Expand
        rewritten   = rewrite_query(query, use_hyde=True, use_expand=True)
        hyde_text   = rewritten["hyde"]
        expand_text = rewritten["expanded"]

        # 2. Dense search (Pinecone) — HyDE vector
        hyde_vector = embedder.encode(expand_query(hyde_text)).tolist()
        res         = index.query(vector=hyde_vector, top_k=top_k, include_metadata=True)
        dense_matches = []
        for m in res.matches:
            score = m.score
            if score < 0.10:
                continue
            dense_matches.append({
                "text":   m.metadata.get("text", ""),
                "source": m.metadata.get("source", "МУИС-ийн баримт бичиг"),
                "score":  round(score, 3),
                "method": "dense",
            })

        # 3. Hybrid: BM25 + RRF + Rerank
        final = hybrid_search(expand_text, dense_matches, top_k=6)
        return final if final else dense_matches[:6]

    except Exception as e:
        print(f"❌ Search error: {e}")
        return []


def build_context_block(matches: list[dict]) -> str:
    if not matches:
        return ""
    return "\n\n---\n\n".join(f"[{m['source']}]\n{m['text']}" for m in matches)


# ── СИСТЕМ ПРОМТ ─────────────────────────────────────────
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


# ── ROUTE-УУД ─────────────────────────────────────────────
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

    # ── 1. Classify ──────────────────────────────────────
    classification = classify_query(user_message)
    if not classification["is_relevant"]:
        print(f"🚫 Rejected [{classification['method']}] score={classification['score']}: {user_message[:60]}")
        return jsonify({"answer": REJECT_MESSAGE, "sources": [], "cached": False})

    # ── 2. Cache шалгах ───────────────────────────────────
    # Зөвхөн шинэ яриа (history хоосон) үед кэшлэнэ.
    # Яриын дундаас асуувал өмнөх context нөлөөлдөг тул кэш ашиглахгүй.
    use_cache = len(history) == 0
    if use_cache:
        cached = get_cached(user_message)
        if cached:
            return jsonify({**cached, "cached": True})

    try:
        # ── 3. RAG хайлт ─────────────────────────────────
        matches      = search_context(user_message)
        context_text = build_context_block(matches)

        # ── 4. Яриын түүх ────────────────────────────────
        messages = []
        for m in history[-6:]:
            role, content = m.get("role", ""), m.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})

        # ── 5. Claude дуудах ─────────────────────────────
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

        # ── 6. Cache-д хадгалах ───────────────────────────
        if use_cache:
            set_cached(user_message, html_text, sources)

        return jsonify({"answer": html_text, "sources": sources, "cached": False})

    except anthropic.APIStatusError as e:
        code = e.status_code
        msgs = {
            429: "⏳ Хэтэрхий олон хүсэлт. Хэсэг хүлээгээд дахин оролдоно уу.",
            401: "❌ Claude API key буруу байна.",
            529: "❌ Claude сервер ачаалалтай байна. Дахин оролдоно уу.",
        }
        return jsonify({"error": msgs.get(code, f"❌ Claude алдаа ({code})")}), 500

    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({"error": f"Алдаа гарлаа: {str(e)}"}), 500


@app.route("/health")
def health():
    try:
        stats = index.describe_index_stats()
        return jsonify({
            "status":       "ok",
            "vector_count": stats.total_vector_count,
            "embed_model":  EMBED_MODEL,
            "cache":        cache_stats(),
        })
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/admin/cache/flush", methods=["POST"])
def flush_cache():
    from cache import flush_all
    ok = flush_all()
    return jsonify({"flushed": ok})


if __name__ == "__main__":
    app.run(debug=True, port=5000)