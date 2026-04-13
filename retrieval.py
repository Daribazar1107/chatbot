"""
retrieval.py — Hybrid Search + Reranker module
Combines Dense (Pinecone) + Sparse (BM25), then filters with a Reranker.

How it works:
  1. Dense  — Pinecone embedding similarity (semantic)
  2. Sparse — BM25 keyword exact match (lexical)
  3. RRF    — Reciprocal Rank Fusion to merge both results
  4. Rerank — Re-rank top-k with a cross-encoder
"""

import os
import pickle
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ── Reranker model ────────────────────────────────────────
# English-optimised, fast, supports up to 512 tokens
# Note: dense embeddings use sentence-transformers/all-mpnet-base-v2
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print(f"🔧 Loading reranker: {RERANKER_MODEL}")
_reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
print("✅ Reranker ready.")


BM25_CACHE_FILE = "bm25_index.pkl"   # file path for disk persistence


# ── BM25 Index ────────────────────────────────────────────
class BM25Index:
    """
    Builds a BM25 index from Pinecone chunks and persists it to disk.
    On the next boot the index is loaded from disk — no Pinecone call needed.
    """

    def __init__(self):
        self.docs:   list[dict] = []
        self.index:  BM25Okapi | None = None
        self._built = False

    # ── Build ─────────────────────────────────────────────
    def build(self, chunks: list[dict]) -> None:
        self.docs  = chunks
        tokenized  = [self._tokenize(c["text"]) for c in chunks]
        self.index = BM25Okapi(tokenized)
        self._built = True
        print(f"✅ BM25 index built: {len(chunks)} docs")

    # ── Save to disk ──────────────────────────────────────
    def save(self, path: str = BM25_CACHE_FILE) -> None:
        with open(path, "wb") as f:
            pickle.dump({"docs": self.docs, "index": self.index}, f)
        print(f"💾 BM25 index saved: {path}")

    # ── Load from disk ────────────────────────────────────
    def load(self, path: str = BM25_CACHE_FILE) -> bool:
        """
        Returns True if loaded successfully.
        """
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.docs   = data["docs"]
            self.index  = data["index"]
            self._built = True
            print(f"✅ BM25 index loaded from disk: {len(self.docs)} docs")
            return True
        except FileNotFoundError:
            print(f"ℹ️  BM25 cache file not found ({path}) — will build a new one")
            return False
        except Exception as e:
            print(f"⚠️  BM25 load error: {e} — will build a new one")
            return False

    # ── Search ────────────────────────────────────────────
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self._built or not self.docs:
            return []
        tokens  = self._tokenize(query)
        scores  = self.index.get_scores(tokens)
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
        # English-only tokenizer (alphanumeric + apostrophes for contractions)
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        return tokens if tokens else [""]


# Global instance
bm25_index = BM25Index()


# ── RRF (Reciprocal Rank Fusion) ──────────────────────────
def rrf_merge(
    dense_results:  list[dict],
    sparse_results: list[dict],
    k:              int   = 60,
    dense_weight:   float = 0.7,
    sparse_weight:  float = 0.3,
) -> list[dict]:
    """
    Merges dense and sparse results using Reciprocal Rank Fusion.

    RRF score = Σ weight / (k + rank)
    dense_weight > sparse_weight: semantic search takes priority.
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


# ── Reranker ──────────────────────────────────────────────
def rerank(query: str, candidates: list[dict], top_k: int = 6) -> list[dict]:
    """
    Re-ranks candidate chunks using a cross-encoder.
    Takes top-20 candidates from Dense + Sparse and returns the most
    relevant top_k chunks.
    """
    if not candidates:
        return []

    # Cross-encoder takes [query, passage] pairs
    pairs  = [[query, c["text"]] for c in candidates]
    scores = _reranker.predict(pairs)

    # Combine with scores and sort
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for doc, score in ranked[:top_k]:
        results.append({**doc, "rerank_score": round(float(score), 3)})

    return results


# ── Hybrid Search (main function) ─────────────────────────
def hybrid_search(
    query:         str,
    dense_results: list[dict],   # results from Pinecone (all-mpnet-base-v2 embeddings)
    top_k:         int = 6,
) -> list[dict]:
    """
    Dense (Pinecone) + Sparse (BM25) → RRF → Rerank

    Args:
        query:         user's question
        dense_results: results from Pinecone's search_context()
                       (embeddings generated with all-mpnet-base-v2)
        top_k:         number of final chunks to return

    Returns:
        top_k re-ranked chunks
    """
    # 1. BM25 search
    sparse_results = bm25_index.search(query, top_k=10)
    print(f"🔍 BM25: {len(sparse_results)} chunks found")

    # 2. RRF merge
    merged = rrf_merge(dense_results, sparse_results)
    print(f"🔀 RRF merged: {len(merged)} chunks")

    # 3. Rerank — take top 20 candidates and rerank
    candidates = merged[:20]
    reranked   = rerank(query, candidates, top_k=top_k)
    print(f"🎯 Reranked top-{top_k}: scores={[r['rerank_score'] for r in reranked]}")

    return reranked