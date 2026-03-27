import os
import json
import csv
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ТОХИРГОО
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME   = "muis-chatbot"
DATA_FOLDER  = "data"

pc = Pinecone(api_key=PINECONE_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Индекс шалгах/үүсгэх
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

def create_teacher_story(t):
    """JSON-г ухаалаг текст болгох (Хайлтад зориулав)"""
    try:
        ovog = t.get("Багш_ажилтны_овог", "")
        ner = t.get("Багш_ажилтны_нэр", "")
        alba = t.get("Албан_тушаал", "")
        tenkhim = t.get("Харьяалах_бүтцийн_нэгжийн_нэр", "")
        bolovsrol = t.get("Эзэмшсэн_боловсрол", "")
        email = t.get("Имэйл_хаяг", "")
        
        if not ner: return None
        
        # Энэ текст Pinecone-д суух тул хайх түлхүүр үгсийг багтаасан
        return f"Багш {ovog} овогтой {ner}. {ner} багш нь {tenkhim}-д {alba} албан тушаалтай ажилладаг. Боловсролын зэрэг: {bolovsrol}. Холбоо барих имэйл: {email}."
    except:
        return None

def read_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
            return [{"text": create_teacher_story(i), "source": filepath} for i in items if create_teacher_story(i)]
    elif ext == ".pdf":
        reader = PdfReader(filepath)
        return [{"text": p.extract_text(), "source": filepath} for p in reader.pages if p.extract_text()]
    return []

def start_ingestion():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"'{DATA_FOLDER}' хавтаст файлаа хийнэ үү.")
        return

    print("🚀 Өгөгдөл боловсруулж байна...")
    all_chunks = []
    for filename in os.listdir(DATA_FOLDER):
        chunks = read_file(os.path.join(DATA_FOLDER, filename))
        all_chunks.extend(chunks)
    
    if not all_chunks:
        print("❌ Дата олдсонгүй.")
        return

    print(f"📦 {len(all_chunks)} хэсгийг Pinecone руу илгээж байна...")
    vectors = []
    for i, chunk in enumerate(all_chunks):
        embedding = embedder.encode(chunk["text"]).tolist()
        vectors.append({
            "id": f"id-{i}",
            "values": embedding,
            "metadata": {"text": chunk["text"], "source": chunk["source"]},
        })

    for i in range(0, len(vectors), 50):
        index.upsert(vectors=vectors[i:i+50])
    
    print("🎯 Индексжүүлэлт амжилттай дууслаа!")

if __name__ == "__main__":
    start_ingestion()