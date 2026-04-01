import os, anthropic
from dotenv import load_dotenv
load_dotenv()
_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── HyDE ─────────────────────────────────────────────────
HYDE_SYSTEM = """Та МУИС-ийн сургалтын журам, дүрмийн мэргэжилтэн.
Хэрэглэгчийн асуултад МУИС-ийн албан журмын хэлбэрээр ТОВЧ хариу бич.
- Зөвхөн 2-4 өгүүлбэр
- Журмын нэр томьёо, заалтын дугаар ашигла
- Мэдэхгүй бол "Журмын дагуу..." гэж эхлэн таамаглал бич
- Монгол хэлээр"""


def hyde(query: str) -> str:
    """
    Hypothetical Document Embedding:
    Асуултаас хиймэл журмын хариу үүсгэнэ.
    Энэ хариуг embed хийж Pinecone-д хайна → жинхэнэ chunk-тай илүү сайн таарна.
    """
    try:
        resp = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0,
            system=HYDE_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        hypo = resp.content[0].text.strip()
        # Хиймэл хариуг анхны асуулттай нэгтгэнэ
        return f"{query}\n{hypo}"
    except Exception as e:
        print(f"⚠️ HyDE алдаа: {e}")
        return query  # fallback — анхны асуулт


# ── Query Expansion ───────────────────────────────────────
EXPAND_SYSTEM = """Та МУИС-ийн сургалтын журмын мэргэжилтэн.
Хэрэглэгчийн асуултыг хайлт сайжруулах үүднээс өргөтгө.
Зөвхөн нэмэлт түлхүүр үгсийг НЭМЭХ — анхны асуултыг өөрчлөхгүй.
Гаралт: анхны асуулт + нэмэлт үгс (нэг мөрт, таслалаар тусгаарлан)
Жишээ:
  Орц: голч дүн гэж юу вэ
  Гаралт: голч дүн гэж юу вэ, GPA, голч оноо, сурлагын чанар, 4.0 систем
Монгол хэлээр, 10-аас илүүгүй үг нэм."""


def expand(query: str) -> str:
    """
    Claude-аар асуултыг МУИС нэр томьёонд тохируулан өргөтгөнө.
    """
    try:
        resp = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            temperature=0,
            system=EXPAND_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"⚠️ Expand алдаа: {e}")
        return query


# ── Нэгдсэн rewrite ───────────────────────────────────────
def rewrite(query: str, use_hyde: bool = True, use_expand: bool = True) -> dict:
    """
    Асуултыг сайжруулж буцаана.

    Returns:
        {
          "original": str,      # анхны асуулт
          "hyde":     str,      # HyDE өргөтгөсөн (embed хийхэд ашиглана)
          "expanded": str,      # keyword өргөтгөсөн (нэмэлт хайлтад)
        }
    """
    result = {"original": query, "hyde": query, "expanded": query}

    if use_hyde:
        result["hyde"] = hyde(query)
        print(f"📝 HyDE: {result['hyde'][:80]}...")

    if use_expand:
        result["expanded"] = expand(query)
        print(f"🔍 Expand: {result['expanded'][:80]}")

    return result