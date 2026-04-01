"""
cache.py — Redis Cache module
МУИС chatbot-ын ижил асуултад хариуг кэшлэнэ.

Ажиллах зарчим:
  1. Асуултыг normalize хийж → SHA256 key үүсгэнэ
  2. Redis-ээс хайна → байвал шууд буцаана (Pinecone + Claude дуудахгүй)
  3. Байхгүй бол RAG pipeline явуулж → хариуг Redis-д хадгална
"""

import hashlib, json, os
import redis

# ── Тохиргоо ─────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB   = int(os.getenv("REDIS_DB", 0))
CACHE_TTL  = int(os.getenv("CACHE_TTL", 60 * 60 * 24))  # 24 цаг (секундээр)
CACHE_PREFIX = "muис:"

# ── Redis холболт ─────────────────────────────────────────
try:
    _redis = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=2,  # 2 секунд холбогдохгүй бол алгасна
    )
    _redis.ping()
    CACHE_ENABLED = True
    print(f"✅ Redis cache бэлэн: {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    _redis = None
    CACHE_ENABLED = False
    print(f"⚠️  Redis холбогдохгүй байна ({e}) — cache алгасна")


# ── Гол функцүүд ──────────────────────────────────────────

def _make_key(query: str) -> str:
    """
    Асуултаас тогтвортой cache key үүсгэнэ.
    Normalize: lowercase, extra whitespace арилгах.
    """
    normalized = " ".join(query.lower().split())
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return f"{CACHE_PREFIX}{digest}"


def get_cached(query: str) -> dict | None:
    """
    Cache-ээс хариу хайна.
    Returns: {"answer": ..., "sources": [...]} эсвэл None
    """
    if not CACHE_ENABLED:
        return None
    try:
        key  = _make_key(query)
        data = _redis.get(key)
        if data:
            print(f"🎯 Cache hit: {query[:50]!r}")
            return json.loads(data)
    except Exception as e:
        print(f"⚠️  Cache get алдаа: {e}")
    return None


def set_cached(query: str, answer: str, sources: list[str]) -> None:
    """
    Хариуг Redis-д хадгална (TTL = CACHE_TTL секунд).
    """
    if not CACHE_ENABLED:
        return
    try:
        key  = _make_key(query)
        data = json.dumps({"answer": answer, "sources": sources}, ensure_ascii=False)
        _redis.setex(key, CACHE_TTL, data)
        print(f"💾 Cache set: {query[:50]!r}")
    except Exception as e:
        print(f"⚠️  Cache set алдаа: {e}")


def invalidate(query: str) -> bool:
    """Тодорхой асуултын cache-ийг устгана."""
    if not CACHE_ENABLED:
        return False
    try:
        return bool(_redis.delete(_make_key(query)))
    except Exception:
        return False


def flush_all() -> bool:
    """МУИС chatbot-ын бүх cache-ийг цэвэрлэнэ (debug/admin)."""
    if not CACHE_ENABLED:
        return False
    try:
        keys = _redis.keys(f"{CACHE_PREFIX}*")
        if keys:
            _redis.delete(*keys)
        print(f"🗑️  {len(keys)} cache entry устгагдлаа")
        return True
    except Exception as e:
        print(f"⚠️  Flush алдаа: {e}")
        return False


def stats() -> dict:
    """Cache-ийн статистик (health endpoint-д ашиглана)."""
    if not CACHE_ENABLED:
        return {"enabled": False}
    try:
        keys  = _redis.keys(f"{CACHE_PREFIX}*")
        info  = _redis.info("memory")
        return {
            "enabled":     True,
            "entries":     len(keys),
            "ttl_hours":   CACHE_TTL // 3600,
            "memory_used": info.get("used_memory_human", "?"),
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}