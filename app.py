"""
app.py — NUM chatbot Flask backend (v5)

PURPOSE:
  This is the main web server. It receives user messages from the browser,
  runs the full RAG pipeline (classify → cache → retrieve → generate), and
  returns an answer. It exposes two chat endpoints:
    POST /chat        — original endpoint, returns full JSON response when done
    GET  /chat/stream — streaming endpoint, sends tokens via Server-Sent Events

FULL PIPELINE PER REQUEST:
  1. Classify query    — is it NUM-related? (keyword check + Pinecone embedding)
  2. Cache lookup      — return cached answer immediately if available
  3. Query expansion   — add synonyms via local MN_SYNONYMS dict (<1ms, no API call)
  4. Hybrid search     — BM25 + Pinecone dense → RRF merge → cross-encoder rerank
  5. Build context     — format the top chunks into a readable context block
  6. Claude generation — send context + query to Claude Haiku, get answer
  7. Faithfulness check— verify answer doesn't hallucinate numbers not in context
  8. Cache store       — save the answer to Redis for future identical queries
  9. Return response   — send HTML-rendered answer + sources to the browser

CHANGE LOG:
  - [FIX]    EMBED_MODEL fixed to "all-mpnet-base-v2" — must match ingest.py
  - [PERF]   query_rewriter.py deleted entirely — synonym expansion is now handled
             by the local expand_query() function using MN_SYNONYMS dictionary.
             Zero extra API calls, <1ms instead of ~400ms, same retrieval quality.
  - [PERF]   max_tokens reduced 1500 → 800 — saves ~40% LLM latency
  - [PERF]   New /chat/stream endpoint — streams tokens via SSE
  - [QUAL]   faithfulness_check_fast now also catches hallucinated numbers/dates
"""

import os          # Read environment variables (API keys)
import time        # Measure how long each pipeline step takes (for timing logs)
import re          # Regular expressions — used in faithfulness number check
import json        # Parse/serialize JSON — used in SSE streaming route
import markdown    # Convert Claude's Markdown output to HTML for the browser
import anthropic   # Anthropic Python SDK — used to call Claude

# Flask imports for building the web server and streaming responses
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

from pinecone import Pinecone                          # Pinecone client for vector database queries
from sentence_transformers import SentenceTransformer  # Embeds user queries into float vectors
from dotenv import load_dotenv                         # Loads .env file into os.environ

# Our custom modules — query_rewriter is intentionally NOT imported (file deleted)
from cache import get_cached, set_cached, stats as cache_stats  # Redis cache helpers
from retrieval import bm25_index, hybrid_search                  # BM25 index + hybrid search pipeline

load_dotenv()           # Read .env file so ANTHROPIC_API_KEY, PINECONE_API_KEY etc. are available
app = Flask(__name__)   # Create the Flask web application instance

# ── Configuration ─────────────────────────────────────────

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")   # Key for authenticating with the Anthropic API
PINECONE_KEY  = os.getenv("PINECONE_API_KEY")    # Key for authenticating with Pinecone
INDEX_NAME    = "muis-chatbot"                   # Name of the Pinecone index holding our document vectors

# MUST match the EMBED_MODEL used in ingest.py.
# If they differ, query vectors and document vectors live in different embedding spaces
# and retrieval silently breaks — Pinecone returns wrong results with no error message.
EMBED_MODEL = "all-mpnet-base-v2"

RELEVANCE_THRESHOLD = 0.62   # Minimum Pinecone cosine similarity to consider a query on-topic

# Source files that count as official NUM data — used in embedding-based classification
# A query is accepted only if at least one of these sources scores above RELEVANCE_THRESHOLD
TRUSTED_SOURCES = {
    "chuluu.json", "teachers.json",
    "courses.json", "grading.json", "level.json",
    "schedule.json", "tuition.json",
}

# Shown to the user when their query fails all relevance checks
REJECT_MESSAGE = (
    "Sorry, I can only answer questions related to NUM regulations, rules, "
    "and academic policies. Your question does not fall within this scope. 🎓"
)

# If any of these strings appear in the lowercased query, reject immediately
# without even calling the embedder — saves one Pinecone round-trip (~300ms)
REJECT_TOPICS = [
    "recipe", "buuz", "cooking", "baking", "boiling", "frying", "how to make",
    "flour", "oil", "salt", "sweets", "recipe",
    "football", "basketball", "game", "chess",
    "weather", "rain", "snow", "temperature",
    "movie", "song", "music", "netflix", "youtube",
    "president", "government", "doctor", "hospital",
    "love", "husband", "girlfriend",
]

# If any of these strings appear in the lowercased query, accept immediately
# without calling the embedder — saves one Pinecone round-trip (~300ms)
NUM_KEYWORDS = [
    "regulations", "rules", "policy", "student", "learner",
    "admission", "graduation", "exam", "diploma",
    "department", "school", "num",
    "course", "registration", "register", "credit", "credit hour", "credits",
    "level", "year",
    "gpa", "average", "grade", "evaluation", "score", "notation",
    "stipend", "award", "scholarship",
    " w ", "wf", " f ", " i ", " r ", "cr", "ca", "nr", "na", "rc", "grading",
    "payment", "fee", "account", "tuition fee",
    "leave of absence", "dismissal", "drop", "removal",
    "internship", "practice", "defense",
    "teacher", "professor", "email", "phone",
    "dormitory", "housing",
]

# Synonym map used by expand_query() — replaces query_rewriter.py entirely.
# If a key appears in the query, its value (extra terms) is appended before searching.
# WHY: students say "drop a class" but regulations say "course withdrawal" — appending
# both forms means BM25 and Pinecone can match either vocabulary.
# COST: zero API calls, runs in <1ms. Previously this required one Claude Haiku call (~400ms).
MN_SYNONYMS = {
    "credit":       "credit hour credits",                          # "credit" → also search for "credit hour"
    "year":         "level course year number",                     # "year" → also search for "level"
    "which":        "number level",                                 # "which year/level" → add level number
    "gpa":          "average grade score GPA",                     # "gpa" → also search cumulative average
    "grade":        "evaluation grade score notation",             # "grade" → also search evaluation/notation
    "leave":        "take a leave student regulation",             # "leave" → also search leave of absence rules
    "drop":         "reason for dismissal from school",            # "drop" → also search dismissal regulations
    "payment":      "tuition fee fees account",                    # "payment" → also search tuition/fees
    "exam":         "exam midterm final",                          # "exam" → also search midterm/final
    "registration": "course registration selection",               # "registration" → also search course selection
    "scholarship":  "scholarship loan financial support",          # "scholarship" → also search financial aid
    "graduation":   "graduation completion requirements diploma",  # "graduation" → also search diploma requirements
    "internship":   "internship professional master",              # "internship" → also search professional practice
    "level":        "level course year credit hours",              # "level" → also search year/credit hours
}

# When these appear in the query (checked uppercase), fetch top_k=20 from Pinecone
# instead of the default 12 — these topics often span multiple regulation chunks
BOOST_KEYWORDS = {
    "W", "WF", "NR", "NA", "CA", "CR", "RC", "GPA",
    "AVERAGE", "EVALUATION", "NOTATION", "CREDIT", "CREDITS",
    "LEVEL", "COURSE", "DISMISSAL", "LEAVE", "GRADUATION",
}

# ── Load models at startup ────────────────────────────────

print(f"🔧 Loading embedder: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)          # Load the embedding model into memory once at startup
print("✅ Embedder ready.")                          # Confirm model loaded without error

claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)  # Initialize the Anthropic API client
pc     = Pinecone(api_key=PINECONE_KEY)              # Initialize the Pinecone client
index  = pc.Index(INDEX_NAME)                        # Connect to our specific Pinecone index


def _build_bm25():
    """
    Load the BM25 keyword index from disk at startup.
    If the file doesn't exist, warn the user to run ingest.py first.
    """
    if not bm25_index.load():                          # Returns False if bm25_index.pkl not found
        print("⚠️  BM25 is not working — run ingest.py first")

_build_bm25()   # Run immediately at module load so BM25 is ready before any HTTP requests arrive


# ── Helper functions ──────────────────────────────────────

def expand_query(query: str) -> str:
    """
    Append domain-specific synonyms to the query using the local MN_SYNONYMS dictionary.

    This fully replaces what query_rewriter.py's expand() function used to do, but:
      - costs zero API calls  (was one Claude Haiku call per request)
      - runs in <1ms          (was ~300-500ms network round-trip)
      - same or better quality for this narrow, predictable domain

    EXAMPLE:
      Input:  "how do I drop a course"
      Output: "how do I drop a course reason for dismissal from school"
      BM25 can now match both "drop" and "dismissal" in the source documents.
    """
    q_lower = query.lower()                                              # Lowercase once for all checks
    extras  = [exp for mn, exp in MN_SYNONYMS.items() if mn in q_lower] # Collect all matching synonym groups
    return f"{query} {' '.join(extras)}" if extras else query            # Append synonyms or return unchanged


def _dense_search(query_text: str, top_k: int) -> list[dict]:
    """
    Encode the query as a float vector and search Pinecone for the top_k nearest chunks.
    Filters out results with cosine similarity < 0.10 (near-zero scores are irrelevant noise).
    Returns a list of dicts: {text, source, score, method}.
    """
    try:
        vector = embedder.encode(query_text).tolist()                              # Convert query text to a vector
        res    = index.query(vector=vector, top_k=top_k, include_metadata=True)   # Send query to Pinecone
        result = []
        for m in res.matches:
            if m.score < 0.10:                    # Skip near-zero similarity — these are noise, not matches
                continue
            result.append({
                "text":   m.metadata.get("text", ""),                # The chunk text stored in Pinecone metadata
                "source": m.metadata.get("source", "NUM document"),  # Source filename (e.g. "grading.json")
                "score":  round(m.score, 3),                         # Cosine similarity score, 3 decimal places
                "method": "dense",                                    # Tag so downstream code knows retrieval type
            })
        return result
    except Exception as e:
        print(f"⚠️ Dense search error: {e}")   # Log the error without crashing the whole request
        return []                              # Return empty list — pipeline handles this gracefully


def classify_and_fetch(query: str, top_k: int = 12) -> dict:
    """
    Decide if the query is NUM-related AND fetch dense results — in ONE Pinecone call.

    WHY COMBINED: Previously classification and retrieval were two separate Pinecone queries
    (~3s each = 6s total). Now one query does both jobs, saving ~3s per request.

    THREE PATHS ordered fastest → slowest:
      Path 1 — Blacklist:  substring check in REJECT_TOPICS → instant reject (<1ms)
      Path 2 — Keyword:    substring check in NUM_KEYWORDS  → instant accept, then fetch Pinecone
      Path 3 — Embedding:  encode query, query Pinecone, check best trusted-source score

    RETURNS:
      {
        "is_relevant": bool,   — whether to continue the pipeline
        "method":      str,    — which path was taken (for logging/debugging)
        "matches":     list,   — dense search results (empty if rejected)
      }
    """
    query   = query.strip()    # Remove leading/trailing whitespace from user input
    q_lower = query.lower()    # Lowercase once — reused in all checks below

    if len(query) < 3:         # Queries shorter than 3 chars can't be meaningful
        return {"is_relevant": False, "method": "too_short", "matches": []}

    # Path 1: instant reject — no embedder, no Pinecone, just string matching
    if any(tok in q_lower for tok in REJECT_TOPICS):
        print(f"🚫 Blacklist reject: {query[:60]}")
        return {"is_relevant": False, "method": "blacklist", "matches": []}

    # Path 2: instant accept — keyword matched, then run Pinecone to get results
    if any(kw in q_lower for kw in NUM_KEYWORDS):
        print(f"✅ Keyword pass: {query[:60]}")
        matches = _dense_search(expand_query(query), top_k)   # expand_query adds synonyms before searching
        return {"is_relevant": True, "method": "keyword_pass", "matches": matches}

    # Path 3: no keyword match — embed the query and check Pinecone's scores
    try:
        vector = embedder.encode(expand_query(query)).tolist()                     # Embed the expanded query
        res    = index.query(vector=vector, top_k=top_k, include_metadata=True)   # One Pinecone call for both classify + fetch

        if not res["matches"]:   # Pinecone returned nothing — query is too far from any document
            return {"is_relevant": False, "method": "no_matches", "matches": []}

        best_trusted  = 0.0    # Tracks the best cosine score from a trusted NUM source
        dense_matches = []     # Accumulates all usable matches from this Pinecone call

        for m in res.matches:
            src   = m.get("metadata", {}).get("source", "")   # Which source file this chunk came from
            score = m.score                                     # Cosine similarity (0.0 – 1.0)
            if score < 0.10:                                    # Skip near-zero matches — they're noise
                continue
            dense_matches.append({
                "text":   m.metadata.get("text", ""),
                "source": src,
                "score":  round(score, 3),
                "method": "dense",
            })
            # Keep track of the highest score from any trusted source file
            if src in TRUSTED_SOURCES and score > best_trusted:
                best_trusted = score

        # Accept if the best trusted-source match clears the relevance threshold
        if best_trusted >= RELEVANCE_THRESHOLD:
            print(f"✅ Embedding pass ({best_trusted:.3f}): {query[:60]}")
            return {"is_relevant": True, "method": f"emb({best_trusted:.3f})", "matches": dense_matches}

        # Score too low — query is probably not about NUM regulations
        print(f"🚫 Low score ({best_trusted:.3f}): {query[:60]}")
        return {"is_relevant": False, "method": f"low({best_trusted:.3f})", "matches": []}

    except Exception as e:
        print(f"⚠️ Classify error: {e}")
        return {"is_relevant": False, "method": "error", "matches": []}


def enrich_and_rerank(query: str, dense_matches: list[dict], top_k: int = 6) -> list[dict]:
    """
    Expand the query locally, run BM25 keyword search, merge with dense results via RRF,
    then rerank using the cross-encoder.

    expand_query() is used here instead of calling query_rewriter.rewrite() — same synonym
    expansion, zero API calls, and it was already called in classify_and_fetch() so the
    BM25 search now gets the same enriched query that Pinecone already used.
    """
    try:
        expanded = expand_query(query)                                   # Append synonyms locally (<1ms)
        final    = hybrid_search(expanded, dense_matches, top_k=top_k)  # BM25 + RRF + cross-encoder rerank
        return final if final else dense_matches[:top_k]                 # Fall back to top dense results if hybrid fails
    except Exception as e:
        print(f"⚠️ Enrich error: {e}")
        return dense_matches[:top_k]   # Return top dense results as a safe fallback


def build_context_block(matches: list[dict]) -> str:
    """
    Format the top retrieved chunks into a labeled, readable context string for Claude.
    Each chunk is numbered and tagged with its source filename and relevance score.
    Chunks are separated by horizontal rules so Claude can distinguish between sources.
    """
    if not matches:              # If no chunks were retrieved, return empty string
        return ""
    parts = []
    for i, m in enumerate(matches, 1):                              # Number sources starting from 1
        src   = m.get("source", "NUM document")                     # Source filename
        score = m.get("rerank_score", m.get("score", 0))           # Use rerank score if available, else dense score
        parts.append(f"[Source {i}: {src} | score={score:.3f}]\n{m['text']}")   # Format one chunk
    return "\n\n---\n\n".join(parts)   # Separate chunks with horizontal rules for readability


def faithfulness_check_fast(answer: str, context: str) -> bool:
    """
    Quick two-part heuristic check that the answer is grounded in the retrieved context.

    PART 1 — Word overlap:
      If fewer than 12% of the answer's words appear in the context, the answer is likely
      hallucinated. Stopwords (len <= 2) are excluded so "the", "is", "a" don't inflate the ratio.

    PART 2 — Number/date check (added in v4):
      Every number in the answer must also appear in the context.
      This catches hallucinated credit counts, fees, percentages, clause numbers, etc.
      Example: Claude says "15 credit hours" but context says "12 credit hours" → caught.

    WHY FAST: Runs in <1ms with no API calls. Not perfect, but catches the most common
    failure modes without adding latency or cost. A full LLM-based faithfulness check
    would be more accurate but costs ~500ms and one extra API call per response.
    """
    if not context or not answer:   # Can't check without both sides — assume faithful
        return True

    # Part 1: word overlap ratio
    aw    = set(w for w in answer.lower().split() if len(w) > 2)    # Meaningful words from the answer
    cw    = set(w for w in context.lower().split() if len(w) > 2)   # Meaningful words from the context
    if not aw:                                                        # Answer has no meaningful words — skip check
        return True
    ratio = len(aw & cw) / len(aw)                                  # Fraction of answer words found in context
    if ratio < 0.12:                                                  # Less than 12% overlap = likely hallucination
        print(f"⚠️  Faithfulness suspect (word overlap {ratio:.2%})")
        return False

    # Part 2: number hallucination check
    answer_nums  = set(re.findall(r'\b\d+\.?\d*\b', answer))        # All numbers/decimals in the answer
    context_nums = set(re.findall(r'\b\d+\.?\d*\b', context))       # All numbers/decimals in the context
    if answer_nums:                                                   # Only check if answer actually contains numbers
        hallucinated = answer_nums - context_nums                    # Numbers present in answer but NOT in context
        if hallucinated:                                             # Any hallucinated number → fail
            print(f"⚠️  Faithfulness suspect (hallucinated numbers: {hallucinated})")
            return False

    return True   # Both checks passed — answer appears grounded in the context


# ── System prompt ─────────────────────────────────────────

# Static instruction block that Claude always receives, regardless of the query.
# The retrieved context chunks are appended below this in build_system_prompt().
SYSTEM_BASE = """You are the "NUM Assistant" chatbot for the National University of Mongolia.

LANGUAGE INSTRUCTIONS:
- "credit" = "credit hour", "course" = "level", "which year" = "level number"
- "GPA" = "grade point average", "grade" = "evaluation", "drop" = "dismissed from school"
- Understand any phrasing and respond based on its meaning.

OPERATING RULES:
1. Use ONLY the information provided in the CONTEXT below.
2. DO NOT add information from your own knowledge that is not in the context.
3. If the context contains an answer, respond clearly and completely.
4. If the context contains no relevant answer, respond with:
   "Sorry, I don't have information about this in my knowledge base."
5. Pass numeric information (credit hours, levels, percentages, scores, fees) exactly as given.
6. If asked about grade notations (W, WF, I, R, F, CA, NR), explain each one.
7. If asked about a teacher, clearly state their email, phone, and department.
8. Respond in English, concisely and clearly.

FORMATTING INSTRUCTIONS:
- ALWAYS format course lists as a Markdown table with columns: Course, Code, Level, Semester, Day, Time, Room, Type, Credits
- ALWAYS format teacher contact info as plain labeled lines (not a table)
- If a clause number is available → mention it in the response
- Do NOT use numbered lists for course information — always use a table
- Do NOT give generic advice such as "see your professor" or "contact the administration" """


def build_system_prompt(context: str) -> str:
    """
    Combine the static system instructions with the retrieved context chunks.
    If no context was found, tell Claude explicitly so it doesn't try to answer from memory.
    """
    ctx = context if context else "No relevant documents found in the knowledge base."
    return f"{SYSTEM_BASE}\n\nCONTEXT:\n{ctx}"   # Context appended at the end of the system prompt


# ── Routes ────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the main chat HTML page."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Full RAG pipeline — waits for the complete Claude response before returning JSON.
    Kept for backward compatibility. Prefer /chat/stream for a better user experience
    since it shows tokens to the user as they are generated instead of waiting 2-4s.
    """
    t_total = time.perf_counter()   # Start the total request timer
    timing  = {}                    # Dict to collect per-step millisecond timings

    data         = request.json or {}              # Parse the incoming JSON request body
    user_message = data.get("message", "").strip() # Extract and clean the user's question
    history      = data.get("history", [])         # Previous conversation turns for context

    if not user_message:                           # Reject empty messages immediately
        return jsonify({"error": "Empty question"}), 400

    # ── Step 1: Classify query + fetch dense results ──────
    t0  = time.perf_counter()
    clf = classify_and_fetch(
        user_message,
        # Fetch more results for queries about specific grading/credit topics
        top_k=20 if any(kw in user_message.upper() for kw in BOOST_KEYWORDS) else 12,
    )
    timing["classify_ms"] = round((time.perf_counter() - t0) * 1000)   # Record classify duration

    if not clf["is_relevant"]:   # Query failed all relevance checks — return rejection immediately
        timing["total_ms"] = round((time.perf_counter() - t_total) * 1000)
        print(f"🚫 [{clf['method']}] {timing['classify_ms']}ms: {user_message[:60]}")
        return jsonify({"answer": REJECT_MESSAGE, "sources": [], "cached": False, "timing": timing})

    # ── Step 2: Check Redis cache ─────────────────────────
    use_cache = len(history) == 0   # Only cache single-turn questions (no conversation history)
    if use_cache:
        cached = get_cached(user_message)   # Look up by normalized SHA256 hash of the query
        if cached:                          # Cache hit — skip the entire pipeline and return instantly
            timing["total_ms"] = round((time.perf_counter() - t_total) * 1000)
            return jsonify({**cached, "cached": True, "timing": timing})

    try:
        # ── Step 3: Local expand + BM25 + rerank ─────────
        t0      = time.perf_counter()
        matches = enrich_and_rerank(user_message, clf["matches"], top_k=6)   # Get final top 6 chunks
        timing["retrieval_ms"] = round((time.perf_counter() - t0) * 1000)   # Record retrieval duration

        context_text = build_context_block(matches)   # Format the 6 chunks into a labeled context string

        # ── Step 4: Build conversation history ───────────
        messages = []
        for m in history[-5:]:                               # Use only last 5 turns to stay within token budget
            r, c = m.get("role", ""), m.get("content", "")  # role = "user" or "assistant"
            if r in ("user", "assistant") and c:             # Skip any malformed history entries
                messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": user_message})   # Append the current question last

        # ── Step 5: Call Claude (non-streaming) ───────────
        t0 = time.perf_counter()
        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",          # Haiku: fastest and cheapest Claude model
            max_tokens=800,                             # Regulation answers are short — 800 is plenty
            temperature=0,                              # Deterministic — no randomness that could boost hallucinations
            system=build_system_prompt(context_text),   # System prompt containing context chunks + instructions
            messages=messages,                          # Conversation history + current question
        )
        timing["llm_ms"] = round((time.perf_counter() - t0) * 1000)   # Record LLM duration

        raw_text = response.content[0].text                          # Extract text from Claude's response object
        passed   = faithfulness_check_fast(raw_text, context_text)   # Run hallucination checks

        if not passed:   # Append a visible warning if the answer looks potentially hallucinated
            raw_text += (
                "\n\n*⚠️ Some information may not be present in the source documents. "
                "Please verify with the relevant NUM department.*"
            )

        # Convert Claude's Markdown formatting to HTML so the browser can render it properly
        html_text = markdown.markdown(raw_text, extensions=["tables", "nl2br", "fenced_code"])
        sources   = list({m["source"] for m in matches})   # Deduplicated set of source filenames

        if use_cache and passed:   # Only cache single-turn responses that passed faithfulness check
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
            "answer":   html_text,   # HTML-rendered answer string
            "sources":  sources,     # List of source filenames the answer was drawn from
            "cached":   False,       # This response was freshly generated (not from cache)
            "faithful": passed,      # Whether the faithfulness check passed
            "timing":   timing,      # Per-step timing breakdown in milliseconds
        })

    except anthropic.APIStatusError as e:
        # Map known HTTP error codes to user-friendly messages
        msgs = {429: "⏳ Too many requests.", 401: "❌ Invalid API key.", 529: "❌ Server overloaded."}
        return jsonify({"error": msgs.get(e.status_code, f"❌ Claude error ({e.status_code})")}), 500
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/chat/stream")
def chat_stream():
    """
    Streaming version of /chat using Server-Sent Events (SSE).

    WHY SSE: The original POST /chat makes the browser wait 2-4 seconds for Claude to finish
    generating the full response before anything appears on screen. With SSE, each token
    is sent to the browser as Claude generates it — the answer builds word by word,
    like ChatGPT's streaming interface. Users perceive this as much faster.

    HOW TO USE FROM THE FRONTEND:
      const es = new EventSource(`/chat/stream?message=...&history=...`);
      es.onmessage = (e) => {
          const data = JSON.parse(e.data);
          if (data.type === "token")   appendToken(data.text);              // Append each token to the UI
          if (data.type === "done")    finalise(data.sources, data.faithful); // Show sources, finalize UI
          if (data.type === "error")   showError(data.message);             // Display error message
          if (data.type === "reject")  showReject(data.message);            // Display off-topic message
          if (data.type === "cached")  showFull(data.answer, data.sources); // Render cached full response
      };

    SSE EVENT TYPES EMITTED:
      token  — one text chunk from Claude's stream (append to displayed text)
      done   — streaming finished; includes sources list and faithful flag
      error  — something went wrong (API error, empty message, etc.)
      reject — query was off-topic (show REJECT_MESSAGE)
      cached — a cached answer exists (show the full HTML at once, no streaming needed)
    """
    user_message = request.args.get("message", "").strip()   # User's question from URL query param
    history_raw  = request.args.get("history", "[]")          # Conversation history as a JSON string

    try:
        history = json.loads(history_raw)   # Parse the JSON string into a Python list
    except Exception:
        history = []   # If JSON is malformed, start with no history

    def generate():
        """
        Generator function that yields SSE-formatted strings one at a time.
        Flask's stream_with_context() keeps the request context alive inside this generator.
        Each yielded string must be in the format: "data: <json>\n\n"
        The double newline is required by the SSE specification.
        """
        if not user_message:   # Reject empty messages before doing any work
            yield f"data: {json.dumps({'type': 'error', 'message': 'Empty question'})}\n\n"
            return

        # ── Step 1: Classify + dense fetch ───────────────
        clf = classify_and_fetch(
            user_message,
            top_k=20 if any(kw in user_message.upper() for kw in BOOST_KEYWORDS) else 12,
        )
        if not clf["is_relevant"]:   # Off-topic — send rejection event and stop the generator
            yield f"data: {json.dumps({'type': 'reject', 'message': REJECT_MESSAGE})}\n\n"
            return

        # ── Step 2: Cache lookup ──────────────────────────
        use_cache = len(history) == 0   # Only cache single-turn questions
        if use_cache:
            cached = get_cached(user_message)
            if cached:   # Cache hit — send the full cached HTML as a single event (no streaming needed)
                yield f"data: {json.dumps({'type': 'cached', 'answer': cached['answer'], 'sources': cached['sources']})}\n\n"
                return

        # ── Step 3: Local expand + BM25 + rerank ─────────
        matches      = enrich_and_rerank(user_message, clf["matches"], top_k=6)   # Final top 6 chunks
        context_text = build_context_block(matches)                                # Format for Claude prompt
        sources      = list({m["source"] for m in matches})                       # Unique source filenames

        # ── Step 4: Conversation history ──────────────────
        messages = []
        for m in history[-5:]:                               # Last 5 turns only to stay within token budget
            r, c = m.get("role", ""), m.get("content", "")  # Each turn has a role and content
            if r in ("user", "assistant") and c:             # Skip malformed entries
                messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": user_message})   # Current question goes last

        # ── Step 5: Claude streaming ──────────────────────
        full_text = ""   # Accumulate the complete response text for faithfulness check after streaming ends
        try:
            # Use .stream() instead of .create() — this yields tokens as they are generated
            with claude.messages.stream(
                model="claude-haiku-4-5-20251001",          # Same model as non-streaming route
                max_tokens=800,                             # Regulation answers fit well within 800 tokens
                temperature=0,                              # Deterministic output
                system=build_system_prompt(context_text),   # Context chunks + instructions
                messages=messages,                          # Conversation history + current question
            ) as stream:
                for token in stream.text_stream:            # Each iteration yields one text token from Claude
                    full_text += token                      # Build up the full response for faithfulness check
                    # Send this token to the browser as an SSE event — browser appends it to the displayed text
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        except anthropic.APIStatusError as e:
            # Handle Claude API errors that occur mid-stream
            msgs = {429: "⏳ Too many requests.", 401: "❌ Invalid API key.", 529: "❌ Server overloaded."}
            yield f"data: {json.dumps({'type': 'error', 'message': msgs.get(e.status_code, str(e))})}\n\n"
            return   # Stop the generator — no "done" event since we errored out
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return   # Stop the generator

        # ── Step 6: Faithfulness check + cache ───────────
        passed = faithfulness_check_fast(full_text, context_text)   # Run hallucination checks on full text
        if use_cache and passed:                                      # Cache only faithful, single-turn responses
            html_text = markdown.markdown(full_text, extensions=["tables", "nl2br", "fenced_code"])
            set_cached(user_message, html_text, sources)             # Store the HTML version in Redis

        # Send the final "done" event — browser uses this to show sources and finalize the UI state
        yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'faithful': passed})}\n\n"

    return Response(
        stream_with_context(generate()),   # Keeps Flask's request context (g, request) alive inside the generator
        mimetype="text/event-stream",      # Required MIME type — tells browser this is an SSE stream, not regular JSON
        headers={
            "Cache-Control":     "no-cache",   # Prevent browsers and proxies from caching the event stream
            "X-Accel-Buffering": "no",         # Tell Nginx not to buffer the stream — tokens must arrive immediately
        },
    )


# ── Admin / monitoring routes ─────────────────────────────

@app.route("/health")
def health():
    """Health check endpoint — returns Pinecone vector count and Redis cache stats."""
    try:
        stats = index.describe_index_stats()   # Ask Pinecone for current index statistics
        return jsonify({
            "status":       "ok",
            "vector_count": stats.total_vector_count,   # Total vectors currently stored in Pinecone
            "cache":        cache_stats(),              # Redis cache entry count and memory usage
        })
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/admin/cache/flush", methods=["POST"])
def flush_cache():
    """
    Admin endpoint to wipe all cached answers from Redis.
    Useful after re-running ingest.py — old cached answers may no longer match the new data.
    """
    from cache import flush_all               # Import here to avoid any circular import issues
    return jsonify({"flushed": flush_all()})  # flush_all() deletes all keys with the chatbot: prefix


if __name__ == "__main__":
    app.run(debug=True, port=5000)   # Start Flask dev server; debug=True enables auto-reload on file changes