"""
query_rewriter.py — Query Rewriter module
Fixes:
  - HyDE only runs on long/complex queries (short queries are skipped)
  - Expand + HyDE are called concurrently → 2x speed improvement
  - Expand is limited to avoid adding too many keywords
  - Improved error fallback handling
"""

import os, concurrent.futures
import anthropic
from dotenv import load_dotenv

load_dotenv()
_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Short/simple queries → HyDE not needed
HYDE_MIN_LEN = 15  # character count threshold


# ── HyDE ─────────────────────────────────────────────────
HYDE_SYSTEM = """You are an expert in NUM (National University of Mongolia) academic regulations.
Given a user's question, write a VERY SHORT hypothetical answer in the style of NUM regulations.
Rules:
- Only 1-3 sentences
- Use NUM terminology and clause numbers where applicable
- Only include information related to NUM academic matters
- If unsure, make a reasonable guess based on the closest relevant regulation
- Respond in English"""


def hyde(query: str) -> str:
    """
    Hypothetical Document Embedding.
    Skips short queries (< HYDE_MIN_LEN characters) to save time.
    """
    if len(query.strip()) < HYDE_MIN_LEN:
        return query
    try:
        resp = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            temperature=0,
            system=HYDE_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        hypo = resp.content[0].text.strip()
        return f"{query}\n{hypo}"
    except Exception as e:
        print(f"⚠️ HyDE error: {e}")
        return query


# ── Query Expansion ───────────────────────────────────────
EXPAND_SYSTEM = """You are an expert in NUM (National University of Mongolia) academic regulations.
Expand the user's query to improve search retrieval.
Rules:
- Do NOT modify the original query — only ADD related synonym keywords
- Maximum 6 additional words (comma-separated)
- Use NUM academic terminology
- If the query is already specific enough, return it exactly as-is
Output format: original query + additional keywords (single line)
Example:
  Input:  what is GPA
  Output: what is GPA, grade point average, cumulative score, 4.0 scale, calculate GPA
Respond in English."""


def expand(query: str) -> str:
    """
    Keyword expansion — avoids adding too many keywords.
    """
    try:
        resp = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            temperature=0,
            system=EXPAND_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        result = resp.content[0].text.strip()
        # If the response is too long, keep only the original query
        if len(result) > len(query) * 3:
            return query
        return result
    except Exception as e:
        print(f"⚠️ Expand error: {e}")
        return query


# ── Combined rewrite (concurrent) ────────────────────────
def rewrite(query: str, use_hyde: bool = True, use_expand: bool = True) -> dict:
    """
    Calls Expand + HyDE concurrently to improve speed.

    Returns:
        {
          "original": str,
          "hyde":     str,   # used for embedding
          "expanded": str,   # used for additional search
        }
    """
    result = {"original": query, "hyde": query, "expanded": query}

    # Short queries → no rewrite needed
    if len(query.strip()) < 5:
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        if use_hyde:
            futures["hyde"]     = pool.submit(hyde, query)
        if use_expand:
            futures["expanded"] = pool.submit(expand, query)

        for key, fut in futures.items():
            try:
                result[key] = fut.result(timeout=5)
            except Exception as e:
                print(f"⚠️ Rewrite {key} error: {e}")
                # fallback: use original query

    if use_hyde:   print(f"📝 HyDE: {result['hyde'][:80]}...")
    if use_expand: print(f"🔍 Expand: {result['expanded'][:80]}")

    return result