"""
ingest.py — МУИС chatbot data ingestion
Fixes:
  - build_rule_text string-iteration bug (str → chars)
  - заалтууд array-тай item-үүдийг зөв задлах
  - teacher email нэмсэн
  - grading.json-ийн student_evaluation_rules зөв унших
  - tuition.json, level.json бүрэн унших
"""

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
    """
    Ганц заалтын dict → текст болгоно.
    Буцаах: str эсвэл None (хоосон бол)
    """
    bulag   = item.get("бүлэг", "").strip()
    zaalt   = item.get("заалт", "").strip()
    aguulga = item.get("агуулга", "").strip()
    tags    = item.get("таг", [])
    tulhuur = item.get("түлхүүр_үг", [])

    if not aguulga:
        return None

    parts = []
    if bulag:  parts.append(f"Сэдэв: {bulag}.")
    if zaalt:  parts.append(f"Заалт {zaalt}:")
    parts.append(aguulga)

    all_tags = list(tags) + list(tulhuur)
    if all_tags:
        parts.append(f"Холбоотой: {', '.join(str(t) for t in all_tags)}.")

    return " ".join(parts)


def build_zaalt_list(item: dict, src: str) -> list[dict]:
    """
    заалтууд массив агуулсан item-ийг задлана.
    Жишээ: chuluu.json — { бүлэг, таг, заалтууд: [{заалт, агуулга}, ...] }
    """
    results = []
    bulag = item.get("бүлэг", "").strip()
    tags  = item.get("таг", [])

    for zaalt_item in item.get("заалтууд", []):
        if not isinstance(zaalt_item, dict):
            continue
        # Эцэг бүлэг + тагийг нэгтгэж дамжуулна
        merged = {**zaalt_item, "бүлэг": bulag, "таг": tags}
        text = build_rule_text(merged)
        if text and len(text) >= MIN_CHUNK_LEN:
            results.append({"text": text, "source": src})

    return results


def build_teacher_text(item: dict) -> str | None:
    """
    Багшийн мэдээлэл → текст. Email-ийг нэмсэн.
    """
    def get(field: str) -> str:
        v = item.get(field, "")
        if v and str(v).strip().lower() not in ["байхгүй", "none", "null", ""]:
            return str(v).strip()
        return ""

    ner = get("ner")
    if not ner:
        return None

    parts = [f"Багш: {ner}."]
    if s := get("salbar_surguuli"): parts.append(f"Сургууль/тэнхим: {s}.")
    if a := get("alban_tushaal"):   parts.append(f"Албан тушаал: {a}.")
    if u := get("uruunii_dugaar"):  parts.append(f"Өрөөний дугаар: {u}.")
    if t := get("utas"):            parts.append(f"Утас: {t}.")
    if e := get("email"):           parts.append(f"И-мэйл: {e}.")

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


def build_regulation_table(item: dict) -> list[str]:
    """
    level.json гэх мэт regulation_table бүтэцтэй файл уншина.
    """
    title   = item.get("title", "").strip()
    content = item.get("content", [])
    results = []

    for prog in content:
        if not isinstance(prog, dict):
            continue
        hotolbor   = prog.get("хөтөлбөр", "").strip()
        tulhuur_ug = prog.get("түлхүүр_үг", [])
        tuvshinguud = prog.get("түвшин", [])

        for t in tuvshinguud:
            if not isinstance(t, dict):
                continue
            ner   = t.get("нэр", "").strip()
            bagts = t.get("багц_цаг", "").strip()
            text  = f"{title}. {hotolbor} хөтөлбөр: {ner} — нийт {bagts} багц цаг цуглуулсан байна."
            if tulhuur_ug:
                text += f" Холбоотой: {', '.join(tulhuur_ug)}."
            results.append(text)

    return results


def build_grading_chunks(item: dict) -> list[str]:
    """
    grading.json-ийн олон бүтэцтэй item-үүдийг задлана.
    types: table, definitions, regulation_section, standards, methodology
    """
    results = []
    title   = item.get("title", "").strip()
    typ     = item.get("type", "")
    content = item.get("content", {})

    if typ == "table":
        # Оноо → үсгэн дүн → тоон дүн хүснэгт
        rows = []
        for row in content:
            if isinstance(row, dict):
                rows.append(
                    f"Оноо {row.get('оноо','')}: үсгэн дүн {row.get('үсгэн_дүн','')}, "
                    f"тоон дүн {row.get('тоон_дүн','')}."
                )
        if rows:
            results.append(f"{title}\n" + " ".join(rows))

    elif typ == "definitions" and isinstance(content, dict):
        # Key-value definitions
        parts = [f"{title}."]
        for k, v in content.items():
            parts.append(f"{k}: {v}")
        results.append(" ".join(parts))

    elif typ == "definitions" and isinstance(content, list):
        # Special notation definitions (W, WF, I, R, F гэх мэт)
        for entry in content:
            if not isinstance(entry, dict):
                continue
            temd   = entry.get("тэмдэглэгээ", "").strip()
        #     names  = entry.get("өөр_нэршил", [])
            tailab = entry.get("тайлбар", "").strip()
            if temd and tailab:
                # өөр нэршлийг нэгтгэж embed хийхэд тусална
                names  = entry.get("өөр_нэршил", [])
                aka    = ", ".join(names) if names else ""
                text   = f"Үнэлгээний тэмдэглэгээ {temd}"
                if aka: text += f" (мөн: {aka})"
                text  += f": {tailab}"
                results.append(text)

    elif typ == "regulation_section":
        # student_evaluation_rules гэх мэт
        for zaalt_item in content:
            if not isinstance(zaalt_item, dict):
                continue
            zaalt   = zaalt_item.get("заалт", "").strip()
            aguulga = zaalt_item.get("агуулга", "").strip()
            tulhuur = zaalt_item.get("түлхүүр_үг", [])
            if aguulga:
                text = f"{title}. Заалт {zaalt}: {aguulga}"
                if tulhuur:
                    text += f" Холбоотой: {', '.join(tulhuur)}."
                results.append(text)

    elif typ == "standards":
        # distribution_standards
        parts = [f"{title}."]
        for grade, pct in content.items():
            parts.append(f"{grade}: {pct}")
        results.append(" ".join(parts))

    elif typ == "methodology":
        formula  = item.get("formula", "")
        details  = item.get("details", {})
        parts    = [f"{title}."]
        if formula: parts.append(f"Томьёо: {formula}.")
        for k, v in details.items():
            parts.append(str(v))
        results.append(" ".join(parts))

    return results


def build_tuition_chunks(item: dict) -> list[str]:
    """
    tuition.json-ийн төлбөрийн мэдээлэл → текст.
    """
    results = []

    # Дансны мэдээлэл
    if item.get("turuul") == "tulbur_zaavar":
        text = item.get("text", "").strip()
        if text:
            results.append(text)
        return results

    # Төлбөрийн тариф
    hicheel_jil = item.get("hicheeliin_jil", "")
    elseltin_ue = item.get("elseltiin_ue", "")
    surguuli    = item.get("surguuli", "")
    tulbur      = item.get("tulbur", {})
    erunkhii    = tulbur.get("erunkhii_suuri", "")
    mergejliin  = tulbur.get("mergejliin_suuri", "")

    if hicheel_jil and surguuli:
        text = (
            f"Сургалтын төлбөр {hicheel_jil} хичээлийн жилд. "
            f"Элссэн үе: {elseltin_ue}. "
            f"Сургууль: {surguuli}. "
        )
        if erunkhii:
            text += f"Ерөнхий суурь хичээлийн нэг багц цагийн төлбөр: {erunkhii:,} төгрөг. "
        if mergejliin:
            text += f"Мэргэжлийн суурь хичээлийн нэг багц цагийн төлбөр: {mergejliin:,} төгрөг."
        results.append(text)

    return results


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

        # ── regulation_table (level.json) ──
        if item.get("type") == "regulation_table":
            for t in build_regulation_table(item):
                if len(t) >= MIN_CHUNK_LEN:
                    results.append({"text": t, "source": src})
            continue

        # ── grading.json олон бүтэц ──
        if item.get("type") in ("table", "definitions", "regulation_section", "standards", "methodology"):
            for t in build_grading_chunks(item):
                if len(t) > CHUNK_SIZE:
                    results.extend(split_chunks(t, src))
                elif len(t) >= MIN_CHUNK_LEN:
                    results.append({"text": t, "source": src})
            continue

        # ── tuition.json ──
        if "turuul" in item or "hicheeliin_jil" in item:
            for t in build_tuition_chunks(item):
                if len(t) >= MIN_CHUNK_LEN:
                    results.append({"text": t, "source": src})
            continue

        # ── заалтууд массив агуулсан item (chuluu.json гэх мэт) ──
        # BUG FIX: өмнө нь build_rule_text(item) нь str буцааж байсан ч
        # "for text in str" гэж character-ээр iterate хийж байсан!
        if "заалтууд" in item:
            results.extend(build_zaalt_list(item, src))
            continue

        # ── Ганц заалт (заалт + агуулга) ──
        if "бүлэг" in item or ("агуулга" in item and "заалт" in item):
            text = build_rule_text(item)
            if text:
                if len(text) > CHUNK_SIZE:
                    results.extend(split_chunks(text, src))
                elif len(text) >= MIN_CHUNK_LEN:
                    results.append({"text": text, "source": src})
            continue

        # ── Багш/ажилтан ──
        if any(k in item for k in ["Багш_ажилтны_нэр", "ner", "firstname"]):
            text = build_teacher_text(item)

        # ── Хичээл ──
        elif "Хичээлийн_индекс" in item or "Монгол_нэр" in item:
            text = build_course_text(item)

        # ── Нэр томьёо ──
        elif "term" in item or ("нэр" in item and "тайлбар" in item):
            text = build_term_text(item)

        # ── Бусад ──
        else:
            text = "; ".join(f"{k}: {v}" for k, v in item.items() if v)

        if not text:
            continue
        if len(text) > CHUNK_SIZE:
            results.extend(split_chunks(text, src))
        elif len(text) >= MIN_CHUNK_LEN:
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