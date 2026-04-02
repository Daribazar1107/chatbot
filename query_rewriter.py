"""
query_rewriter.py — Query Rewriter module
Fixes:
  - HyDE-г зөвхөн урт/нарийн асуулттай үед ажиллуулна (богино асуулт → алгасна)
  - Expand + HyDE-г concurrent дуудна → хурд 2x сайжирна
  - Expand хэт олон keyword нэмэхгүй байхаар хязгаарлана
  - Алдааны fallback сайжирсан
"""

import os, concurrent.futures
import anthropic
from dotenv import load_dotenv

load_dotenv()
_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Богино/энгийн асуулт → HyDE шаардлагагүй
HYDE_MIN_LEN = 15  # тэмдэгтийн тоо


# ── HyDE ─────────────────────────────────────────────────
HYDE_SYSTEM = """Та МУИС-ийн сургалтын журмын мэргэжилтэн.
Хэрэглэгчийн асуултад МУИС-ийн журмын хэлбэрээр МАШ ТОВЧ хариу бич.
Дүрэм:
- Зөвхөн 1-3 өгүүлбэр
- Журмын нэр томьёо, заалтын дугаарыг ашигла
- Зөвхөн МУИС-ийн боловсролтой холбоотой мэдээлэл бич
- Мэдэхгүй бол хамгийн ойрын журмын зарчмаар таамагла
- Монгол хэлээр"""


def hyde(query: str) -> str:
    """
    Hypothetical Document Embedding.
    Богино асуулт (< HYDE_MIN_LEN тэмдэгт) үед алгасна — хурд хэмнэнэ.
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
        print(f"⚠️ HyDE алдаа: {e}")
        return query


# ── Query Expansion ───────────────────────────────────────
EXPAND_SYSTEM = """Та МУИС-ийн сургалтын журмын мэргэжилтэн.
Хэрэглэгчийн асуултыг хайлт сайжруулахаар өргөтгө.
Дүрэм:
- Анхны асуултыг ӨӨРЧЛӨХГҮЙ — зөвхөн ОЙРОЛЦОО УТГАТАЙ нэмэлт үгс нэм
- Хамгийн ихдээ 6 нэмэлт үг (таслалаар тусгаарла)
- МУИС-ийн нэр томьёог ашигла
- Хэрэв асуулт аль хэдийн тодорхой бол яг анхны асуултыг буцаа
Гаралт: анхны асуулт + нэмэлт үгс (нэг мөр)
Жишээ:
  Орц: голч дүн гэж юу вэ
  Гаралт: голч дүн гэж юу вэ, GPA, голч оноо, 4.0 систем, оноо бодох
Монгол хэлээр."""


def expand(query: str) -> str:
    """
    Keyword өргөтгөл — хэт олон keyword нэмэхгүй.
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
        # Хэрэв хариу хэт урт буцаасан бол анхны асуулт хадгалж, 
        # зөвхөн эхний хэсгийг авна
        if len(result) > len(query) * 3:
            return query
        return result
    except Exception as e:
        print(f"⚠️ Expand алдаа: {e}")
        return query


# ── Нэгдсэн rewrite (concurrent) ─────────────────────────
def rewrite(query: str, use_hyde: bool = True, use_expand: bool = True) -> dict:
    """
    Expand + HyDE-г зэрэг дуудаж хурдыг сайжруулна.

    Returns:
        {
          "original": str,
          "hyde":     str,   # embed хийхэд ашиглана
          "expanded": str,   # нэмэлт хайлтад ашиглана
        }
    """
    result = {"original": query, "hyde": query, "expanded": query}

    # Богино асуулт → rewrite шаардлагагүй
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
                print(f"⚠️ Rewrite {key} алдаа: {e}")
                # fallback: анхны асуулт

    if use_hyde:    print(f"📝 HyDE: {result['hyde'][:80]}...")
    if use_expand:  print(f"🔍 Expand: {result['expanded'][:80]}")

    return result