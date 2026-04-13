"""
cache.py — Redis Cache module
Caches responses for repeated questions in the chatbot.

How it works:
  1. Normalize the query → generate a SHA256 key
  2. Look up in Redis → if found, return immediately (skips Pinecone + Claude)
  3. If not found, run the RAG pipeline → store the response in Redis
"""

import hashlib
import json
import os

import redis

# ── Configuration ─────────────────────────────────────────
REDIS_HOST   = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT   = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB     = int(os.getenv("REDIS_DB", 0))
CACHE_TTL    = int(os.getenv("CACHE_TTL", 60 * 60 * 24))  # 24 hours (in seconds)
CACHE_PREFIX = "chatbot:"

# ── Redis connection ───────────────────────────────────────
try:
    _redis = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=2,  # skip if not connected within 2 seconds
    )
    _redis.ping()
    CACHE_ENABLED = True
    print(f"✅ Redis cache ready: {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    _redis = None
    CACHE_ENABLED = False
    print(f"⚠️  Redis not available ({e}) — cache disabled")


# ── Core functions ────────────────────────────────────────

def _make_key(query: str) -> str:
    """
    Generates a stable cache key from a query.
    Normalize: lowercase, strip extra whitespace.
    """
    normalized = " ".join(query.lower().split())
    digest     = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return f"{CACHE_PREFIX}{digest}"


def get_cached(query: str) -> dict | None:
    """
    Looks up a cached response.
    Returns: {"answer": ..., "sources": [...]} or None
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
        print(f"⚠️  Cache get error: {e}")
    return None


def set_cached(query: str, answer: str, sources: list[str]) -> None:
    """
    Stores a response in Redis (TTL = CACHE_TTL seconds).
    """
    if not CACHE_ENABLED:
        return
    try:
        key  = _make_key(query)
        data = json.dumps({"answer": answer, "sources": sources}, ensure_ascii=False)
        _redis.setex(key, CACHE_TTL, data)
        print(f"💾 Cache set: {query[:50]!r}")
    except Exception as e:
        print(f"⚠️  Cache set error: {e}")


def invalidate(query: str) -> bool:
    """Deletes the cached entry for a specific query."""
    if not CACHE_ENABLED:
        return False
    try:
        return bool(_redis.delete(_make_key(query)))
    except Exception:
        return False


def flush_all() -> bool:
    """Clears all cached entries for this chatbot (debug/admin use)."""
    if not CACHE_ENABLED:
        return False
    try:
        keys = _redis.keys(f"{CACHE_PREFIX}*")
        if keys:
            _redis.delete(*keys)
        print(f"🗑️  {len(keys)} cache entries deleted")
        return True
    except Exception as e:
        print(f"⚠️  Flush error: {e}")
        return False


def stats() -> dict:
    """Returns cache statistics (used by health endpoints)."""
    if not CACHE_ENABLED:
        return {"enabled": False}
    try:
        keys = _redis.keys(f"{CACHE_PREFIX}*")
        info = _redis.info("memory")
        return {
            "enabled":     True,
            "entries":     len(keys),
            "ttl_hours":   CACHE_TTL // 3600,
            "memory_used": info.get("used_memory_human", "?"),
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}