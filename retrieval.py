"""
retrieval.py — Hybrid Search + Reranker module
Dense (Pinecone) + Sparse (BM25) нэгтгэж, Reranker-ээр шүүнэ.

Ажиллах зарчим:
  1. Dense  — Pinecone embedding similarity (semantic)
  2. Sparse — BM25 keyword exact match (lexical)
  3. RRF    — Reciprocal Rank Fusion хоёрыг нэгтгэнэ
  4. Rerank — Cross-encoder-ээр top-k дахин эрэмбэлнэ
"""

import os
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ── Reranker загвар ───────────────────────────────────────
# Олон хэлт, хурдан, 512 token хүртэл
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print(f"🔧 Reranker ачааллаж байна: {RERANKER_MODEL}")
_reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
print("✅ Reranker бэлэн.")


import pickle, re

BM25_CACHE_FILE = "bm25_index.pkl"   # disk-д хадгалах файл


# ── BM25 Index ────────────────────────────────────────────
class BM25Index:
    """
    Pinecone chunk-уудаас BM25 index үүсгэж disk-д хадгална.
    Дараагийн boot-д disk-ээс татаж ачаална — Pinecone дуудахгүй.
    """

    def __init__(self):
        self.docs:   list[dict] = []
        self.index:  BM25Okapi | None = None
        self._built = False

    # ── Үүсгэх ───────────────────────────────────────────
    def build(self, chunks: list[dict]) -> None:
        self.docs  = chunks
        tokenized  = [self._tokenize(c["text"]) for c in chunks]
        self.index = BM25Okapi(tokenized)
        self._built = True
        print(f"✅ BM25 index үүсгэгдлээ: {len(chunks)} doc")

    # ── Disk-д хадгалах ──────────────────────────────────
    def save(self, path: str = BM25_CACHE_FILE) -> None:
        with open(path, "wb") as f:
            pickle.dump({"docs": self.docs, "index": self.index}, f)
        print(f"💾 BM25 index хадгалагдлаа: {path}")

    # ── Disk-ээс ачаалах ─────────────────────────────────
    def load(self, path: str = BM25_CACHE_FILE) -> bool:
        """
        Returns True хэрэв амжилттай ачаалсан бол.
        """
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.docs   = data["docs"]
            self.index  = data["index"]
            self._built = True
            print(f"✅ BM25 index disk-ээс ачааллаа: {len(self.docs)} doc")
            return True
        except FileNotFoundError:
            print(f"ℹ️  BM25 cache файл олдсонгүй ({path}) — шинээр үүсгэнэ")
            return False
        except Exception as e:
            print(f"⚠️  BM25 load алдаа: {e} — шинээр үүсгэнэ")
            return False

    # ── Хайлт ────────────────────────────────────────────
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self._built or not self.docs:
            return []
        tokens = self._tokenize(query)
        scores = self.index.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "text":   self.docs[i]["text"],
                "source": self.docs[i]["source"],
                "score":  round(float(scores[i]), 3),
                "method": "bm25",
            }
            for i in top_idx if scores[i] > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-zа-яөүёA-ZА-ЯӨҮ0-9]+", text.lower())
        return tokens if tokens else [""]


# Глобал instance
bm25_index = BM25Index()


# ── RRF (Reciprocal Rank Fusion) ─────────────────────────
def rrf_merge(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[dict]:
    """
    Dense + Sparse үр дүнг RRF аргаар нэгтгэнэ.

    RRF score = Σ weight / (k + rank)
    dense_weight > sparse_weight: semantic хайлт голлоно.
    """
    scores: dict[str, float] = {}
    docs:   dict[str, dict]  = {}

    for rank, item in enumerate(dense_results):
        key = item["text"][:100]
        scores[key] = scores.get(key, 0) + dense_weight / (k + rank + 1)
        docs[key]   = item

    for rank, item in enumerate(sparse_results):
        key = item["text"][:100]
        scores[key] = scores.get(key, 0) + sparse_weight / (k + rank + 1)
        if key not in docs:
            docs[key] = item

    merged = sorted(docs.values(), key=lambda x: scores[x["text"][:100]], reverse=True)
    return merged


# ── Reranker ─────────────────────────────────────────────
def rerank(query: str, candidates: list[dict], top_k: int = 6) -> list[dict]:
    """
    Cross-encoder-ээр candidate chunk-уудыг дахин эрэмбэлнэ.
    Dense + Sparse-аас ирсэн top-20-г авч, хамгийн хамааралтай top_k-г буцаана.
    """
    if not candidates:
        return []

    # Cross-encoder [query, passage] хос авна
    pairs  = [[query, c["text"]] for c in candidates]
    scores = _reranker.predict(pairs)

    # Score-тай нэгтгэж эрэмбэлнэ
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for doc, score in ranked[:top_k]:
        results.append({**doc, "rerank_score": round(float(score), 3)})

    return results


# ── Hybrid Search (гол функц) ────────────────────────────
def hybrid_search(
    query:        str,
    dense_results: list[dict],   # Pinecone-ээс ирсэн
    top_k:        int = 6,
) -> list[dict]:
    """
    Dense (Pinecone) + Sparse (BM25) → RRF → Rerank

    Args:
        query:         хэрэглэгчийн асуулт
        dense_results: Pinecone-ийн search_context() үр дүн
        top_k:         эцсийн буцаах chunk тоо

    Returns:
        Rerank хийсэн top_k chunk
    """
    # 1. BM25 хайлт
    sparse_results = bm25_index.search(query, top_k=10)
    print(f"🔍 BM25: {len(sparse_results)} chunk олдлоо")

    # 2. RRF нэгтгэлт
    merged = rrf_merge(dense_results, sparse_results)
    print(f"🔀 RRF merged: {len(merged)} chunk")

    # 3. Rerank — top 20-г авч rerank хийнэ
    candidates = merged[:20]
    reranked   = rerank(query, candidates, top_k=top_k)
    print(f"🎯 Reranked top-{top_k}: scores={[r['rerank_score'] for r in reranked]}")

    return reranked