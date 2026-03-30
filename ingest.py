import os, re, json, csv, time, hashlib
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── ТОХИРГОО ────────────────────────────────────────────
PINECONE_KEY  = os.getenv("PINECONE_API_KEY")
INDEX_NAME    = "muis-chatbot"
DATA_FOLDER   = "data"
CHUNK_SIZE    = 700
OVERLAP_SIZE  = 120
EMBED_MODEL   = "paraphrase-multilingual-mpnet-base-v2"
EMBED_DIM     = 768
BATCH_SIZE    = 50
MIN_CHUNK_LEN = 30

# ── КЛИЕНТҮҮД ───────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_KEY)
print(f"🔧 Embedder ачааллаж байна: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedder бэлэн.")

# ── PINECONE INDEX ───────────────────────────────────────
def get_or_create_index():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME in existing:
        desc = pc.describe_index(INDEX_NAME)
        current_dim = desc.dimension
        if current_dim != EMBED_DIM:
            print(f"⚠️  Index dimension буруу ({current_dim} ≠ {EMBED_DIM}). Устгаж дахин үүсгэнэ...")
            pc.delete_index(INDEX_NAME)
            time.sleep(3)
        else:
            print(f"✅ Index '{INDEX_NAME}' бэлэн (dim={current_dim}).")
            return pc.Index(INDEX_NAME)

    print(f"🆕 Index '{INDEX_NAME}' үүсгэж байна (dim={EMBED_DIM})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    time.sleep(5)
    print("✅ Index үүсгэгдлээ.")
    return pc.Index(INDEX_NAME)

# ── CHUNK ID ────────────────────────────────────────────
def make_id(text: str, idx: int) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"doc-{idx:05d}-{h}"

# ── ТЕКСТ ХУВААХ ────────────────────────────────────────
def split_chunks(text: str, source: str) -> list[dict]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    section_breaks = re.compile(r'(?=\n\d+\.\d+[\.\d]*[\s\.])')
    sections = section_breaks.split(text)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= CHUNK_SIZE:
            if len(section) >= MIN_CHUNK_LEN:
                chunks.append({"text": section, "source": source})
        else:
            start = 0
            while start < len(section):
                end = start + CHUNK_SIZE
                chunk = section[start:end]
                if end < len(section):
                    for sep in [".\n", ".\r", "։\n", ".\n\n", ". "]:
                        pos = chunk.rfind(sep)
                        if pos > CHUNK_SIZE // 2:
                            end = start + pos + len(sep)
                            chunk = section[start:end]
                            break
                chunk = chunk.strip()
                if len(chunk) >= MIN_CHUNK_LEN:
                    chunks.append({"text": chunk, "source": source})
                next_start = end - OVERLAP_SIZE
                if next_start <= start:
                    break
                start = next_start

    return chunks

# ── JSON УНШИГЧИД ───────────────────────────────────────
def build_rule_text(item: dict) -> str | None:
    bulag   = item.get("бүлэг", "").strip()
    zaalt   = item.get("заалт", "").strip()
    aguulga = item.get("агуулга", "").strip()
    tags    = item.get("таг", [])
    if not aguulga:
        return None
    parts = []
    if bulag:  parts.append(f"Сэдэв: {bulag}.")
    if zaalt:  parts.append(f"Заалт {zaalt}:")
    parts.append(aguulga)
    if tags:   parts.append(f"Холбоотой: {', '.join(tags)}.")
    return " ".join(parts)


def build_teacher_text(item: dict) -> str | None:
    keys = {
        "ovog":      ["Багш_ажилтны_овог", "ovog", "lastname"],
        "ner":       ["Багш_ажилтны_нэр",  "ner",  "firstname", "name"],
        "alba":      ["Албан_тушаал",       "alban_tushaal", "position"],
        "tenkhim":   ["Харьяалах_бүтцийн_нэгжийн_нэр", "salbar_surguuli", "department"],
        "bolovsrol": ["Эзэмшсэн_боловсрол", "degree"],
        "email":     ["Имэйл_хаяг", "email"],
    }

    def get(field):
        for k in keys[field]:
            v = item.get(k, "")
            if v: return str(v).strip()
        return ""

    ner = get("ner")
    if not ner:
        return None

    parts = [f"Багш: {get('ovog')} {ner}."]
    if t := get("tenkhim"):    parts.append(f"Тэнхим: {t}.")
    if a := get("alba"):       parts.append(f"Албан тушаал: {a}.")
    if b := get("bolovsrol"):  parts.append(f"Боловсрол: {b}.")
    if e := get("email"):      parts.append(f"Имэйл: {e}.")
    return " ".join(parts)


def build_course_text(item: dict) -> str | None:
    idx   = item.get("Хичээлийн_индекс", "").strip()
    ner   = item.get("Монгол_нэр", item.get("нэр", "")).strip()
    bagts = item.get("Багц_цаг", item.get("bagts_tsag", ""))
    desc  = item.get("Товч_агуулга", item.get("агуулга", "")).strip()
    if not ner:
        return None
    parts = [f"Хичээл: {ner}" + (f" ({idx})" if idx else "") + "."]
    if bagts: parts.append(f"Багц цаг: {bagts}.")
    if desc:  parts.append(f"Агуулга: {desc}")
    return " ".join(parts)


def build_term_text(item: dict) -> str | None:
    term = item.get("term", item.get("нэр", "")).strip()
    desc = item.get("description", item.get("тайлбар", "")).strip()
    if not desc:
        return None
    return f"Нэр томьёо — {term}: {desc}"


def read_json(filepath: str) -> list[dict]:
    src = os.path.basename(filepath)
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON алдаа ({src}): {e}")
        return []

    items   = data if isinstance(data, list) else [data]
    results = []

    for item in items:
        if not isinstance(item, dict):
            continue

        if "агуулга" in item and ("бүлэг" in item or "заалт" in item):
            text = build_rule_text(item)
        elif any(k in item for k in ["Багш_ажилтны_нэр", "ner", "firstname"]):
            text = build_teacher_text(item)
        elif "Хичээлийн_индекс" in item or "Монгол_нэр" in item:
            text = build_course_text(item)
        elif "term" in item or ("нэр" in item and "тайлбар" in item):
            text = build_term_text(item)
        else:
            text = "; ".join(f"{k}: {v}" for k, v in item.items() if v)

        if not text:
            continue
        if len(text) > CHUNK_SIZE:
            results.extend(split_chunks(text, src))
        else:
            results.append({"text": text, "source": src})

    return results

# ── DOCX УНШИГЧ ─────────────────────────────────────────
def read_docx(filepath: str) -> list[dict]:
    doc  = Document(filepath)
    src  = os.path.basename(filepath)
    lines = []

    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            lines.append(t)

    for table in doc.tables:
        for row in table.rows:
            cells = []
            seen  = set()
            for c in row.cells:
                ct = c.text.strip()
                if ct and ct not in seen:
                    cells.append(ct)
                    seen.add(ct)
            if cells:
                lines.append(" | ".join(cells))

    return split_chunks("\n".join(lines), src)

# ── PDF УНШИГЧ ──────────────────────────────────────────
def read_pdf(filepath: str) -> list[dict]:
    src     = os.path.basename(filepath)
    results = []
    try:
        reader = PdfReader(filepath)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                results.extend(split_chunks(text, f"{src} (p.{i+1})"))
    except Exception as e:
        print(f"  ❌ PDF алдаа ({src}): {e}")
    return results

# ── CSV УНШИГЧ ──────────────────────────────────────────
def read_csv(filepath: str) -> list[dict]:
    src   = os.path.basename(filepath)
    batch = []
    results = []
    try:
        with open(filepath, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                line = ", ".join(
                    f"{k}: {v.strip()}"
                    for k, v in row.items()
                    if v and v.strip()
                )
                if line:
                    batch.append(line)
                if len(batch) >= 5:
                    results.extend(split_chunks("\n".join(batch), src))
                    batch = []
        if batch:
            results.extend(split_chunks("\n".join(batch), src))
    except Exception as e:
        print(f"  ❌ CSV алдаа ({src}): {e}")
    return results

# ── ФАЙЛ ДИСТПАТЧ ───────────────────────────────────────
READERS = {
    ".docx": read_docx,
    ".pdf":  read_pdf,
    ".json": read_json,
    ".csv":  read_csv,
}

def read_file(filepath: str) -> list[dict]:
    ext = Path(filepath).suffix.lower()
    reader = READERS.get(ext)
    if reader is None:
        return []
    try:
        return reader(filepath)
    except Exception as e:
        print(f"  ❌ Алдаа ({os.path.basename(filepath)}): {e}")
        return []

# ── PINECONE UPSERT ──────────────────────────────────────
def upsert_all(chunks: list[dict], idx):
    total = len(chunks)
    print(f"🔄 {total} chunk embedding үүсгэж байна...")

    # Batch encode — дангаар encode хийхээс 3-5x хурдан
    texts      = [c["text"] for c in chunks]
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id":     make_id(chunk["text"], i),
            "values": emb.tolist(),
            "metadata": {
                "text":   chunk["text"][:1500],
                "source": chunk.get("source", ""),
            },
        })

    print(f"📤 Pinecone-д оруулж байна ({len(vectors)} vector)...")
    success = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        try:
            idx.upsert(vectors=batch)
            success += len(batch)
        except Exception as e:
            print(f"  ⚠️  Batch {i//BATCH_SIZE} алдаа: {e}")
        time.sleep(0.15)

    print(f"✅ {success}/{len(vectors)} vector амжилттай орлоо.")

# ── MAIN ────────────────────────────────────────────────
def start_ingestion():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"'{DATA_FOLDER}' хавтас үүсгэгдлээ. Файлуудаа хуулаарай.")
        return

    idx = get_or_create_index()

    print("🗑️  Өмнөх векторуудыг устгаж байна...")
    try:
        idx.delete(delete_all=True, namespace="")
        time.sleep(2)
    except Exception as e:
        print(f"  ⚠️  Устгах үед алдаа (хоосон байж болно): {e}")

    all_chunks = []
    supported  = set(READERS.keys())

    print(f"\n📂 '{DATA_FOLDER}' хавтасны файлуудыг уншиж байна...")
    for filename in sorted(os.listdir(DATA_FOLDER)):
        ext = Path(filename).suffix.lower()
        if ext not in supported:
            continue
        filepath = os.path.join(DATA_FOLDER, filename)
        chunks   = read_file(filepath)
        if chunks:
            all_chunks.extend(chunks)
            print(f"  📄 {filename} → {len(chunks)} chunk")
        else:
            print(f"  ⚠️  {filename} → chunk олдсонгүй")

    if not all_chunks:
        print("\n❌ Оруулах өгөгдөл байхгүй.")
        return

    # Давхардсан chunk хасна
    seen, unique = set(), []
    for c in all_chunks:
        key = c["text"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    removed = len(all_chunks) - len(unique)
    print(f"\n📦 Нийт: {len(all_chunks)} | Давхардсан: {removed} | Цэвэр: {len(unique)}")

    upsert_all(unique, idx)
    print("\n🎯 Индексжүүлэлт дууслаа!")

    try:
        stats = idx.describe_index_stats()
        print(f"📊 Pinecone нийт вектор: {stats.total_vector_count}")
    except Exception:
        pass


if __name__ == "__main__":
    start_ingestion()