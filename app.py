"""
app.py — МУИС chatbot Flask backend (v3)
"""
import os, time, markdown, anthropic
from flask import Flask, render_template, request, jsonify
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from cache import get_cached, set_cached, stats as cache_stats
from query_rewriter import rewrite as rewrite_query
from retrieval import bm25_index, hybrid_search

load_dotenv()
app = Flask(__name__)

ANTHROPIC_KEY       = os.getenv("ANTHROPIC_API_KEY")
PINECONE_KEY        = os.getenv("PINECONE_API_KEY")
INDEX_NAME          = "muis-chatbot"
EMBED_MODEL         = "paraphrase-multilingual-mpnet-base-v2"
RELEVANCE_THRESHOLD = 0.62

TRUSTED_SOURCES = {
    "журам.json","chuluu.json","teachers.json",
    "courses.json","grading.json","level.json",
    "schedule.json","tuition.json",
}

REJECT_MESSAGE = (
    "Уучлаарай, би зөвхөн МУИС-ийн журам, дүрэм, "
    "академик бодлоготой холбоотой асуултад хариулах боломжтой. "
    "Таны асуулт энэ хүрээнд хамаарахгүй байна. 🎓"
)

REJECT_TOPICS = [
    "жор","буз","buuz","хоол хийх","жигнэх","чанах","шарах","хийх арга",
    "гурил","тос","давс","амттан","рецепт","recipe",
    "футбол","basketball","тоглоом","game","chess","шатар",
    "цаг агаар","weather","бороо","цас","температур",
    "кино","дуу","movie","music","netflix","youtube",
    "ерөнхийлөгч","засгийн газар","эмч","эмнэлэг",
    "хайрлах","нөхөр","найз охин","love",
]

MUИС_KEYWORDS = [
    "журам","дүрэм","бодлого","оюутан","суралцагч",
    "элсэлт","төгсөлт","шалгалт","диплом",
    "тэнхим","сургууль","муис","num",
    "хичээл","бүртгэл","бүртгүүл","кредит","багц цаг","багц",
    "курс","түвшин","хэддүгээр",
    "gpa","голч","дүн","үнэлгээ","оноо","тэмдэглэгээ",
    "стипенди","шагнал","тэтгэлэг",
    " w ","wf"," f "," i "," r ","cr","ca","nr","na","rc","grading",
    "төлбөр","хураамж","данс","сургалтын төлбөр",
    "чөлөө","хасагдах","хасах","хасалт",
    "дадлага","практик","хамгаалалт",
    "багш","email","и-мэйл","утас",
    "дотуур байр","байр",
]

MN_SYNONYMS = {
    "кредит":   "багц цаг кредит",
    "курс":     "түвшин курс дугаар",
    "хэддүгээр":"дугаар түвшин",
    "голч":     "голч дүн оноо GPA",
    "дүн":      "үнэлгээ дүн оноо тэмдэглэгээ",
    "чөлөө":    "чөлөө авах суралцагч журам",
    "хасах":    "сургуулиас хасах шалтгаан",
    "төлбөр":   "сургалтын төлбөр хураамж данс",
    "шалгалт":  "шалгалт явцын улирлын",
    "бүртгэл":  "хичээл бүртгүүлэх сонгох",
    "тэтгэлэг": "тэтгэлэг зээл санхүүгийн дэмжлэг",
    "төгсөлт":  "төгсөлт дүүргэх шаардлага диплом",
    "дадлага":  "дадлага мэргэжлийн магистр",
    "түвшин":   "түвшин курс хэддүгээр багц цаг",
}

BOOST_KEYWORDS = {
    "W","WF","NR","NA","CA","CR","RC","GPA",
    "ГОЛЧ","ҮНЭЛГЭЭ","ТЭМДЭГЛЭГЭЭ","КРЕДИТ","БАГЦ",
    "ТҮВШИН","КУРС","ХАСАХ","ЧӨЛӨӨ","ТӨГСӨЛТ",
}

print(f"🔧 Embedder ачааллаж байна: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedder бэлэн.")

claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
pc     = Pinecone(api_key=PINECONE_KEY)
index  = pc.Index(INDEX_NAME)


def _build_bm25():
    if not bm25_index.load():
        print("⚠️  BM25 ажиллахгүй — эхлээд python ingest.py ажиллуулна уу")

_build_bm25()


def expand_query(query: str) -> str:
    q_lower = query.lower()
    extras  = [exp for mn, exp in MN_SYNONYMS.items() if mn in q_lower]
    return f"{query} {' '.join(extras)}" if extras else query


def _dense_search(query_text: str, top_k: int) -> list[dict]:
    try:
        vector = embedder.encode(query_text).tolist()
        res    = index.query(vector=vector, top_k=top_k, include_metadata=True)
        result = []
        for m in res.matches:
            if m.score < 0.10:
                continue
            result.append({
                "text":   m.metadata.get("text",""),
                "source": m.metadata.get("source","МУИС-ийн баримт бичиг"),
                "score":  round(m.score, 3),
                "method": "dense",
            })
        return result
    except Exception as e:
        print(f"⚠️ Dense search error: {e}")
        return []


def classify_and_fetch(query: str, top_k: int = 12) -> dict:
    """
    Classify + Dense retrieval нэг Pinecone query-д нэгтгэсэн.
    Өмнө: classify Pinecone (3s) + retrieval Pinecone (3s) = 6s
    Одоо: нэг query → ~3s хэмнэнэ.
    """
    query   = query.strip()
    q_lower = query.lower()

    if len(query) < 3:
        return {"is_relevant": False, "method": "too_short", "matches": []}

    # Шат 1: Reject blacklist — embedder-т хүрэхгүй (< 1ms)
    if any(tok in q_lower for tok in REJECT_TOPICS):
        print(f"🚫 Blacklist reject: {query[:60]}")
        return {"is_relevant": False, "method": "blacklist", "matches": []}

    # Шат 2: МУИС keyword pass → dense search хийж буцаана
    if any(kw in q_lower for kw in MUИС_KEYWORDS):
        print(f"✅ Keyword pass: {query[:60]}")
        matches = _dense_search(expand_query(query), top_k)
        return {"is_relevant": True, "method": "keyword_pass", "matches": matches}

    # Шат 3: Embedding similarity — classify + retrieval нэгэн зэрэг
    try:
        vector = embedder.encode(expand_query(query)).tolist()
        res    = index.query(vector=vector, top_k=top_k, include_metadata=True)

        if not res["matches"]:
            return {"is_relevant": False, "method": "no_matches", "matches": []}

        best_trusted = 0.0
        dense_matches = []

        for m in res.matches:
            src   = m.get("metadata", {}).get("source", "")
            score = m.score
            if score < 0.10:
                continue
            dense_matches.append({
                "text":   m.metadata.get("text",""),
                "source": src,
                "score":  round(score, 3),
                "method": "dense",
            })
            if src in TRUSTED_SOURCES and score > best_trusted:
                best_trusted = score

        if best_trusted >= RELEVANCE_THRESHOLD:
            print(f"✅ Embedding pass ({best_trusted:.3f}): {query[:60]}")
            return {"is_relevant": True,  "method": f"emb({best_trusted:.3f})", "matches": dense_matches}

        print(f"🚫 Low score ({best_trusted:.3f}): {query[:60]}")
        return {"is_relevant": False, "method": f"low({best_trusted:.3f})", "matches": []}

    except Exception as e:
        print(f"⚠️ Classify error: {e}")
        return {"is_relevant": False, "method": "error", "matches": []}


def enrich_and_rerank(query: str, dense_matches: list[dict], top_k: int = 6) -> list[dict]:
    """HyDE өмнөх алхамд хийгдсэн тул expand + BM25 + Rerank л хийнэ."""
    try:
        rewritten   = rewrite_query(query, use_hyde=False, use_expand=True)
        expand_text = rewritten["expanded"]
        final       = hybrid_search(expand_text, dense_matches, top_k=top_k)
        return final if final else dense_matches[:top_k]
    except Exception as e:
        print(f"⚠️ Enrich error: {e}")
        return dense_matches[:top_k]


def build_context_block(matches: list[dict]) -> str:
    if not matches:
        return ""
    parts = []
    for i, m in enumerate(matches, 1):
        src   = m.get("source","МУИС баримт")
        score = m.get("rerank_score", m.get("score", 0))
        parts.append(f"[Эх сурвалж {i}: {src} | score={score:.3f}]\n{m['text']}")
    return "\n\n---\n\n".join(parts)


def faithfulness_check_fast(answer: str, context: str) -> bool:
    if not context or not answer:
        return True
    aw = set(w for w in answer.lower().split() if len(w) > 2)
    cw = set(w for w in context.lower().split() if len(w) > 2)
    if not aw:
        return True
    ratio = len(aw & cw) / len(aw)
    if ratio < 0.12:
        print(f"⚠️  Faithfulness suspect: {ratio:.2%}")
        return False
    return True


SYSTEM_BASE = """Та бол МУИС-ийн "МУИС-Туслах" чатбот юм.

ХЭЛНИЙ ЗААВАР:
- "кредит"="багц цаг", "курс"="түвшин", "хэддүгээр"="дугаар түвшин"
- "голч"="голч дүн/GPA", "дүн"="үнэлгээ", "хасах"="сургуулиас хасах"
- Монгол хэлний ямар ч хэлбэрийг утгаар нь ойлго.

АЖИЛЛАХ ДҮРЭМ:
1. Зөвхөн доор өгсөн КОНТЕКСТ-ийн мэдээллийг ашигла.
2. Контекстэд байхгүй мэдээллийг өөрийн мэдлэгээр нэмэхийг ХОРИГЛОНО.
3. Контекстэд хариулт байгаа бол заавал тодорхой, бүрэн хариул.
4. Контекстэд огт хариулт байхгүй бол:
   "Уучлаарай, миний мэдээллийн санд энэ тухай мэдээлэл байхгүй байна." гэж хариул.
5. Тоон мэдээлэл (багц цаг, түвшин, хувь, оноо, төлбөр) яг зөв дамжуул.
6. Үсгэн тэмдэглэгээ (W, WF, I, R, F, CA, NR) асуувал тус бүрийн тайлбарыг өг.
7. Багшийн мэдээлэл асуувал email, утас, тэнхимийг тодорхой хэлэх.
8. Монгол хэлээр товч, цэгцтэй хариул.

БҮТЦИЙН ЗААВАР:
- Хүснэгт хэлбэрийн мэдээлэл → Markdown хүснэгт ашигла
- Заалтын дугаар байвал → дурдаж хариул
- "Багшаар уулз","захиргаанд ханд" гэх ерөнхий зөвлөгөө өгөхгүй"""


def build_system_prompt(context: str) -> str:
    ctx = context if context else "Мэдээллийн санд холбогдох баримт олдсонгүй."
    return f"{SYSTEM_BASE}\n\nКОНТЕКСТ:\n{ctx}"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    t_total = time.perf_counter()
    timing  = {}

    data         = request.json or {}
    user_message = data.get("message","").strip()
    history      = data.get("history",[])

    if not user_message:
        return jsonify({"error": "Хоосон асуулт"}), 400

    # ── 1. Classify + Dense (нэгдсэн, нэг Pinecone query) ─
    t0     = time.perf_counter()
    clf    = classify_and_fetch(
        user_message,
        top_k=20 if any(kw in user_message.upper() for kw in BOOST_KEYWORDS) else 12,
    )
    timing["classify_ms"] = round((time.perf_counter() - t0) * 1000)

    if not clf["is_relevant"]:
        timing["total_ms"] = round((time.perf_counter() - t_total) * 1000)
        print(f"🚫 [{clf['method']}] {timing['classify_ms']}ms: {user_message[:60]}")
        return jsonify({"answer": REJECT_MESSAGE, "sources": [], "cached": False, "timing": timing})

    # ── 2. Cache ──────────────────────────────────────────
    use_cache = len(history) == 0
    if use_cache:
        cached = get_cached(user_message)
        if cached:
            timing["total_ms"] = round((time.perf_counter() - t_total) * 1000)
            return jsonify({**cached, "cached": True, "timing": timing})

    try:
        # ── 3. Expand + BM25 + Rerank ────────────────────
        t0      = time.perf_counter()
        matches = enrich_and_rerank(user_message, clf["matches"], top_k=6)
        timing["retrieval_ms"] = round((time.perf_counter() - t0) * 1000)

        context_text = build_context_block(matches)

        # ── 4. Conv. Memory ───────────────────────────────
        messages = []
        for m in history[-5:]:
            r, c = m.get("role",""), m.get("content","")
            if r in ("user","assistant") and c:
                messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": user_message})

        # ── 5. Claude ─────────────────────────────────────
        t0 = time.perf_counter()
        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            temperature=0,
            system=build_system_prompt(context_text),
            messages=messages,
        )
        timing["llm_ms"] = round((time.perf_counter() - t0) * 1000)

        raw_text = response.content[0].text
        passed   = faithfulness_check_fast(raw_text, context_text)

        if not passed:
            raw_text += (
                "\n\n*⚠️ Зарим мэдээлэл баримтад байхгүй байж болзошгүй. "
                "МУИС-ийн холбогдох нэгжээс лавлана уу.*"
            )

        html_text = markdown.markdown(raw_text, extensions=["tables","nl2br","fenced_code"])
        sources   = list({m["source"] for m in matches})

        if use_cache and passed:
            set_cached(user_message, html_text, sources)

        timing["total_ms"] = round((time.perf_counter() - t_total) * 1000)
        print(
            f"⏱️  [{clf['method']}] "
            f"classify:{timing['classify_ms']}ms "
            f"retrieval:{timing['retrieval_ms']}ms "
            f"llm:{timing['llm_ms']}ms "
            f"total:{timing['total_ms']}ms"
        )

        return jsonify({
            "answer": html_text, "sources": sources,
            "cached": False, "faithful": passed, "timing": timing,
        })

    except anthropic.APIStatusError as e:
        msgs = {429:"⏳ Хэт олон хүсэлт.",401:"❌ API key буруу.",529:"❌ Сервер ачаалалтай."}
        return jsonify({"error": msgs.get(e.status_code, f"❌ Claude алдаа ({e.status_code})")}), 500
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({"error": f"Алдаа гарлаа: {str(e)}"}), 500


@app.route("/health")
def health():
    try:
        stats = index.describe_index_stats()
        return jsonify({"status":"ok","vector_count":stats.total_vector_count,"cache":cache_stats()})
    except Exception as e:
        return jsonify({"status":"error","detail":str(e)}), 500


@app.route("/admin/cache/flush", methods=["POST"])
def flush_cache():
    from cache import flush_all
    return jsonify({"flushed": flush_all()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)