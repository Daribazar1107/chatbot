import os, re, json, csv, time
from PyPDF2 import PdfReader
from docx import Document
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME   = "muis-chatbot"
DATA_FOLDER  = "data"
CHUNK_SIZE   = 600
OVERLAP_SIZE = 100

pc       = Pinecone(api_key=PINECONE_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME, dimension=384, metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

# ── ТЕКСТ ХУВААХ ────────────────────────────────────────
def split_chunks(text: str, source: str) -> list[dict]:
    text   = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks = []
    start  = 0
    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end]
        if end < len(text):
            for sep in [".\n", ".\r", "։", ". "]:
                pos = chunk.rfind(sep)
                if pos > CHUNK_SIZE // 2:
                    end   = start + pos + len(sep)
                    chunk = text[start:end]
                    break
        chunk = chunk.strip()
        if len(chunk) > 30:
            chunks.append({"text": chunk, "source": source})
        start = end - OVERLAP_SIZE
        if start <= 0:
            break
    return chunks

# ── ФАЙЛ УНШИГЧИД ───────────────────────────────────────
def read_prechunked(filepath: str) -> list[dict]:
    source = os.path.basename(filepath).replace("_chunks.json", ".docx")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    results = []
    for item in data:
        text = item.get("text", "").strip()
        if len(text) < 20:
            continue
        chapter = item.get("chapter", "")
        results.append({
            "text":    text,
            "source":  f"{source} [{chapter}]" if chapter else source,
            "section": item.get("section", ""),
        })
    return results


def read_docx(filepath: str) -> list[dict]:
    doc = Document(filepath)
    src = os.path.basename(filepath)
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = list(dict.fromkeys(c.text.strip() for c in row.cells if c.text.strip()))
            if cells:
                lines.append(" | ".join(cells))
    return split_chunks("\n".join(lines), src)


def read_pdf(filepath: str) -> list[dict]:
    reader  = PdfReader(filepath)
    src     = os.path.basename(filepath)
    results = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            results.extend(split_chunks(text, f"{src} (хуудас {i+1})"))
    return results


def build_teacher_text(item: dict) -> str | None:
    try:
        ovog      = item.get("Багш_ажилтны_овог", "").strip()
        ner       = item.get("Багш_ажилтны_нэр", "").strip()
        alba      = item.get("Албан_тушаал", "").strip()
        tenkhim   = item.get("Харьяалах_бүтцийн_нэгжийн_нэр", "").strip()
        bolovsrol = item.get("Эзэмшсэн_боловсрол", "").strip()
        email     = item.get("Имэйл_хаяг", "").strip()
        if not ner:
            return None
        parts = [f"Багш: {ovog} {ner}."]
        if tenkhim:    parts.append(f"Тэнхим: {tenkhim}.")
        if alba:       parts.append(f"Албан тушаал: {alba}.")
        if bolovsrol:  parts.append(f"Боловсрол: {bolovsrol}.")
        if email:      parts.append(f"Имэйл: {email}.")
        return " ".join(parts)
    except Exception:
        return None


def read_json(filepath: str) -> list[dict]:
    src = os.path.basename(filepath)
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    items   = data if isinstance(data, list) else [data]
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "Багш_ажилтны_нэр" in item or "Багш_ажилтны_овог" in item:
            text = build_teacher_text(item)
            if text:
                results.append({"text": text, "source": src})
        else:
            text = json.dumps(item, ensure_ascii=False)
            if len(text) > 20:
                results.extend(split_chunks(text, src))
    return results


def read_csv(filepath: str) -> list[dict]:
    src   = os.path.basename(filepath)
    batch = []
    results = []
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line = ", ".join(f"{k}: {v.strip()}" for k, v in row.items() if v and v.strip())
            if line:
                batch.append(line)
            if len(batch) >= 5:
                results.extend(split_chunks("\n".join(batch), src))
                batch = []
    if batch:
        results.extend(split_chunks("\n".join(batch), src))
    return results


def read_file(filepath: str) -> list[dict]:
    name = os.path.basename(filepath)
    ext  = os.path.splitext(filepath)[1].lower()
    if name.endswith("_chunks.json"):
        return read_prechunked(filepath)
    try:
        if ext == ".docx": return read_docx(filepath)
        if ext == ".pdf":  return read_pdf(filepath)
        if ext == ".json": return read_json(filepath)
        if ext == ".csv":  return read_csv(filepath)
        print(f"  ⚠️  Дэмжигдэхгүй: {ext}")
        return []
    except Exception as e:
        print(f"  ❌ Алдаа ({name}): {e}")
        return []

# ── PINECONE-Д ОРУУЛАХ ──────────────────────────────────
def upsert_all(chunks: list[dict]):
    total   = len(chunks)
    vectors = []
    for i, chunk in enumerate(chunks):
        text      = chunk["text"]
        embedding = embedder.encode(text, show_progress_bar=False).tolist()
        meta = {"text": text[:1500], "source": chunk.get("source", "")}
        if "section" in chunk:
            meta["section"] = chunk["section"][:200]
        vectors.append({
            "id":       f"doc-{i}-{abs(hash(text)) % 100000}",
            "values":   embedding,
            "metadata": meta,
        })
        if (i + 1) % 100 == 0:
            print(f"  Embedding: {i+1}/{total}")

    print(f"\n📤 Pinecone-д оруулж байна ({len(vectors)} vector)...")
    for i in range(0, len(vectors), 50):
        index.upsert(vectors=vectors[i:i+50])
        time.sleep(0.2)
    print(f"✅ {len(vectors)} vector амжилттай орлоо.")


def start_ingestion():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"'{DATA_FOLDER}' хавтас үүсгэгдлээ.")
        return

    print("🗑️  Pinecone цэвэрлэж байна...")
    try:
        index.delete(delete_all=True, namespace="")
    except Exception:
        pass
    time.sleep(1)

    supported = {".docx", ".pdf", ".json", ".csv"}
    print(f"\n📂 '{DATA_FOLDER}' уншиж байна...")
    all_chunks = []

    for filename in sorted(os.listdir(DATA_FOLDER)):
        ext = os.path.splitext(filename)[1].lower()
        is_pre = filename.endswith("_chunks.json")
        if ext not in supported and not is_pre:
            continue
        filepath = os.path.join(DATA_FOLDER, filename)
        label    = " 📋 (урьдчилан бэлдсэн)" if is_pre else ""
        print(f"\n📄 {filename}{label}")
        chunks = read_file(filepath)
        all_chunks.extend(chunks)
        print(f"   → {len(chunks)} chunk")

    if not all_chunks:
        print("\n❌ Өгөгдөл олдсонгүй.")
        return

    print(f"\n📦 Нийт {len(all_chunks)} chunk")
    upsert_all(all_chunks)
    print("\n🎯 Индексжүүлэлт дууслаа!")


if __name__ == "__main__":
    start_ingestion()