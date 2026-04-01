# test_index.py
from app import embedder
from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("muis-chatbot")

test_queries = [
    "бууз хийх арга",          # хамааралгүй — бага score гарах ёстой
    "голч дүн хэрхэн тооцдог", # хамааралтай — өндөр score гарах ёстой
    "өнөөдрийн цаг агаар",     # хамааралгүй
    "кредит цаг дүүргэх",      # хамааралтай
]

for q in test_queries:
    vec = embedder.encode(q).tolist()
    res = index.query(vector=vec, top_k=3, include_metadata=True)
    print(f"\n{'='*50}")
    print(f"Асуулт: {q}")
    for m in res.matches:
        print(f"  score={m.score:.3f} | {m.metadata.get('source','?')} | {m.metadata.get('text','')[:80]}")