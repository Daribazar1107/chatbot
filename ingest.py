"""
ingest.py — NUM chatbot data ingestion
Fixes:
  - build_rule_text string-iteration bug (str → chars)
  - Correctly parses items with a provisions array
  - Added teacher email field
  - Correctly reads student_evaluation_rules from grading.json
  - Fully reads tuition.json and level.json
  - Updated all JSON field keys to match English-translated data files
  - Upgraded embedding model to all-mpnet-base-v2 (better English performance)
"""

import os, re, json, csv, time, hashlib
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURATION ────────────────────────────────────────
PINECONE_KEY  = os.getenv("PINECONE_API_KEY")
INDEX_NAME    = "muis-chatbot2"
DATA_FOLDER   = "data"
CHUNK_SIZE    = 700
OVERLAP_SIZE  = 120
EMBED_MODEL   = "all-mpnet-base-v2"  
EMBED_DIM     = 768
BATCH_SIZE    = 50
MIN_CHUNK_LEN = 30

# ── CLIENTS ──────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_KEY)
print(f"🔧 Loading embedder: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedder ready.")


# ── PINECONE INDEX ───────────────────────────────────────
def get_or_create_index():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME in existing:
        desc = pc.describe_index(INDEX_NAME)
        current_dim = desc.dimension
        if current_dim != EMBED_DIM:
            print(f"⚠️  Index dimension mismatch ({current_dim} ≠ {EMBED_DIM}). Deleting and recreating...")
            pc.delete_index(INDEX_NAME)
            time.sleep(3)
        else:
            print(f"✅ Index '{INDEX_NAME}' ready (dim={current_dim}).")
            return pc.Index(INDEX_NAME)

    print(f"🆕 Creating index '{INDEX_NAME}' (dim={EMBED_DIM})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    time.sleep(5)
    print("✅ Index created.")
    return pc.Index(INDEX_NAME)


# ── CHUNK ID ─────────────────────────────────────────────
def make_id(text: str, idx: int) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"doc-{idx:05d}-{h}"


# ── TEXT SPLITTER ─────────────────────────────────────────
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


# ── JSON PARSERS ─────────────────────────────────────────

def build_rule_text(item: dict) -> str | None:
    """
    Single provision dict → text string.
    Returns: str or None (if empty)
    """
    # Support both old Mongolian keys and new English keys
    section  = (item.get("section") or item.get("бүлэг", "")).strip()
    clause   = (item.get("clause")  or item.get("заалт", "")).strip()
    content  = (item.get("content") or item.get("агуулга", "")).strip()
    tags     = item.get("tags",     item.get("таг",        []))
    keywords = item.get("keywords", item.get("түлхүүр_үг", []))

    if not content:
        return None

    parts = []
    if section: parts.append(f"Section: {section}.")
    if clause:  parts.append(f"Clause {clause}:")
    parts.append(content)

    all_tags = list(tags) + list(keywords)
    if all_tags:
        parts.append(f"Related: {', '.join(str(t) for t in all_tags)}.")

    return " ".join(parts)


def build_provisions_list(item: dict, src: str) -> list[dict]:
    """
    Parses items with a provisions (or заалтууд) array.
    Example: chuluu_en.json — { section, tags, provisions: [{clause, content}, ...] }
    """
    results = []
    section = (item.get("section") or item.get("бүлэг", "")).strip()
    tags    = item.get("tags", item.get("таг", []))

    provision_key = "provisions" if "provisions" in item else "заалтууд"

    for provision in item.get(provision_key, []):
        if not isinstance(provision, dict):
            continue
        # Merge parent section + tags into each provision
        merged = {**provision, "section": section, "tags": tags}
        text = build_rule_text(merged)
        if text and len(text) >= MIN_CHUNK_LEN:
            results.append({"text": text, "source": src})

    return results


def build_teacher_text(item: dict) -> str | None:
    """
    Teacher info dict → text. Supports both English and Mongolian field names.
    """
    def get(field: str) -> str:
        v = item.get(field, "")
        if v and str(v).strip().lower() not in ["not available", "none", "null", ""]:
            return str(v).strip()
        return ""

    name = (
        get("name")
        or get("ner")
        or get("Багш_ажилтны_нэр")
        or " ".join(filter(None, [get("lastname"), get("firstname")]))
        or None
    )
    if not name:
        return None

    parts = [f"Teacher: {name}."]
    if s := get("department") or get("salbar_surguuli"): parts.append(f"School/Department: {s}.")
    if a := get("position")   or get("alban_tushaal"):   parts.append(f"Position: {a}.")
    if u := get("room")       or get("uruunii_dugaar"):   parts.append(f"Room: {u}.")
    if t := get("phone")      or get("utas"):             parts.append(f"Phone: {t}.")
    if e := get("email"):                                  parts.append(f"Email: {e}.")

    return " ".join(parts)


def build_course_text(item: dict) -> str | None:
    """
    Supports both English keys (course_index, name, credits, description)
    and original Mongolian keys.
    """
    idx   = (item.get("course_index")  or item.get("Хичээлийн_индекс", "")).strip()
    name  = (item.get("name")          or item.get("Монгол_нэр", item.get("нэр", ""))).strip()
    creds = (item.get("credits")       or item.get("Багц_цаг",   item.get("bagts_tsag", "")))
    desc  = (item.get("description")   or item.get("Товч_агуулга", item.get("агуулга", ""))).strip()

    if not name:
        return None

    parts = [f"Course: {name}" + (f" ({idx})" if idx else "") + "."]
    if creds: parts.append(f"Credits: {creds}.")
    if desc:  parts.append(f"Description: {desc}")
    return " ".join(parts)


def build_term_text(item: dict) -> str | None:
    term = (item.get("term") or item.get("нэр", "")).strip()
    desc = (item.get("description") or item.get("тайлбар", "")).strip()
    if not desc:
        return None
    return f"Term — {term}: {desc}"


def build_regulation_table(item: dict) -> list[str]:
    """
    Reads regulation_table structured files (e.g. level_en.json).
    Supports both English keys (program, levels, name, credits)
    and original Mongolian keys.
    """
    title   = item.get("title", "").strip()
    content = item.get("content", [])
    results = []

    for prog in content:
        if not isinstance(prog, dict):
            continue
        program  = (prog.get("program")  or prog.get("хөтөлбөр", "")).strip()
        keywords = prog.get("keywords",   prog.get("түлхүүр_үг", []))
        levels   = prog.get("levels",     prog.get("түвшин", []))

        for lvl in levels:
            if not isinstance(lvl, dict):
                continue
            lvl_name = (lvl.get("name")    or lvl.get("нэр", "")).strip()
            credits  = (lvl.get("credits") or lvl.get("багц_цаг", "")).strip()
            text = f"{title}. {program} program: {lvl_name} — requires {credits} total credits."
            if keywords:
                text += f" Related: {', '.join(keywords)}."
            results.append(text)

    return results


def build_grading_chunks(item: dict) -> list[str]:
    """
    Parses multi-structure items from grading_en.json.
    types: table, definitions, regulation_section, standards, methodology
    """
    results = []
    title   = item.get("title", "").strip()
    typ     = item.get("type", "")
    content = item.get("content", {})

    if typ == "table":
        # Score → letter grade → GPA table
        rows = []
        for row in content:
            if isinstance(row, dict):
                # Support both English and Mongolian keys
                score  = row.get("score",        row.get("оноо", ""))
                letter = row.get("letter_grade",  row.get("үсгэн_дүн", ""))
                gpa    = row.get("gpa",           row.get("тоон_дүн", ""))
                rows.append(f"Score {score}: letter grade {letter}, GPA {gpa}.")
        if rows:
            results.append(f"{title}\n" + " ".join(rows))

    elif typ == "definitions" and isinstance(content, dict):
        # Key-value definitions
        parts = [f"{title}."]
        for k, v in content.items():
            parts.append(f"{k}: {v}")
        results.append(" ".join(parts))

    elif typ == "definitions" and isinstance(content, list):
        # Special notation definitions (W, WF, I, R, F, etc.)
        for entry in content:
            if not isinstance(entry, dict):
                continue
            notation = (entry.get("notation")    or entry.get("тэмдэглэгээ", "")).strip()
            desc     = (entry.get("description") or entry.get("тайлбар",     "")).strip()
            if notation and desc:
                aliases = entry.get("aliases", entry.get("өөр_нэршил", []))
                aka     = ", ".join(aliases) if aliases else ""
                text    = f"Grade notation {notation}"
                if aka: text += f" (also known as: {aka})"
                text   += f": {desc}"
                results.append(text)

    elif typ == "regulation_section":
        # e.g. student_evaluation_rules
        for provision in content:
            if not isinstance(provision, dict):
                continue
            clause   = (provision.get("clause")   or provision.get("заалт",      "")).strip()
            body     = (provision.get("content")   or provision.get("агуулга",    "")).strip()
            keywords = provision.get("keywords",    provision.get("түлхүүр_үг",   []))
            if body:
                text = f"{title}. Clause {clause}: {body}"
                if keywords:
                    text += f" Related: {', '.join(keywords)}."
                results.append(text)

    elif typ == "standards":
        # distribution_standards
        parts = [f"{title}."]
        for grade, pct in content.items():
            parts.append(f"{grade}: {pct}")
        results.append(" ".join(parts))

    elif typ == "methodology":
        formula = item.get("formula", "")
        details = item.get("details", {})
        parts   = [f"{title}."]
        if formula: parts.append(f"Formula: {formula}.")
        for k, v in details.items():
            parts.append(str(v))
        results.append(" ".join(parts))

    return results


def build_tuition_chunks(item: dict) -> list[str]:
    """
    Parses tuition info from tuition_en.json.
    Supports both English keys (type, academic_year, admission_cohort, tuition_per_credit)
    and original keys.
    """
    results = []

    # Payment instructions entry
    is_instructions = (
        item.get("type") == "payment_instructions"
        or item.get("turuul") == "tulbur_zaavar"
    )
    if is_instructions:
        # Try English field first, fall back to original
        text = item.get("instructions") or item.get("text", "")
        if text:
            results.append(text.strip())
        return results

    # Tuition rate entry
    academic_year = item.get("academic_year",    item.get("hicheeliin_jil", ""))
    cohort        = item.get("admission_cohort", item.get("elseltiin_ue",   ""))
    school        = item.get("school",           item.get("surguuli",       ""))

    per_credit    = item.get("tuition_per_credit", {})
    general       = per_credit.get("general_foundation") if per_credit else item.get("tulbur", {}).get("erunkhii_suuri", "")
    major         = per_credit.get("major_foundation")   if per_credit else item.get("tulbur", {}).get("mergejliin_suuri", "")

    if academic_year and school:
        text = (
            f"Tuition fees for the {academic_year} academic year. "
            f"Admission cohort: {cohort}. "
            f"School: {school}. "
        )
        if general:
            text += f"General foundation course fee per credit: {int(general):,} MNT. "
        if major:
            text += f"Major foundation course fee per credit: {int(major):,} MNT."
        results.append(text)

    return results


def read_json(filepath: str) -> list[dict]:
    src = os.path.basename(filepath)
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON error ({src}): {e}")
        return []

    items   = data if isinstance(data, list) else [data]
    results = []

    for item in items:
        if not isinstance(item, dict):
            continue

        # ── regulation_table (level_en.json) ──
        if item.get("type") == "regulation_table":
            for t in build_regulation_table(item):
                if len(t) >= MIN_CHUNK_LEN:
                    results.append({"text": t, "source": src})
            continue

        # ── grading_en.json multi-structure ──
        if item.get("type") in ("table", "definitions", "regulation_section", "standards", "methodology"):
            for t in build_grading_chunks(item):
                if len(t) > CHUNK_SIZE:
                    results.extend(split_chunks(t, src))
                elif len(t) >= MIN_CHUNK_LEN:
                    results.append({"text": t, "source": src})
            continue

        # ── tuition_en.json ──
        if item.get("type") == "payment_instructions" or "turuul" in item or "hicheeliin_jil" in item or "academic_year" in item:
            for t in build_tuition_chunks(item):
                if len(t) >= MIN_CHUNK_LEN:
                    results.append({"text": t, "source": src})
            continue

        # ── Items with a provisions array (chuluu_en.json, regulations_en.json) ──
        if "provisions" in item or "заалтууд" in item:
            results.extend(build_provisions_list(item, src))
            continue

        # ── Single provision (clause + content) ──
        has_clause  = "clause"  in item or "заалт"   in item
        has_content = "content" in item or "агуулга"  in item
        has_section = "section" in item or "бүлэг"   in item
        if has_section or (has_content and has_clause):
            text = build_rule_text(item)
            if text:
                if len(text) > CHUNK_SIZE:
                    results.extend(split_chunks(text, src))
                elif len(text) >= MIN_CHUNK_LEN:
                    results.append({"text": text, "source": src})
            continue

        # ── Teacher ──
        if any(k in item for k in ["name", "ner", "Багш_ажилтны_нэр", "firstname"]):
            text = build_teacher_text(item)

        # ── Course ──
        elif any(k in item for k in ["course_index", "Хичээлийн_индекс", "name", "Монгол_нэр"]):
            text = build_course_text(item)

        # ── Term/Glossary ──
        elif "term" in item or ("name" in item and "description" in item):
            text = build_term_text(item)

        # ── Fallback ──
        else:
            text = "; ".join(f"{k}: {v}" for k, v in item.items() if v)

        if not text:
            continue
        if len(text) > CHUNK_SIZE:
            results.extend(split_chunks(text, src))
        elif len(text) >= MIN_CHUNK_LEN:
            results.append({"text": text, "source": src})

    return results


# ── DOCX READER ──────────────────────────────────────────
def read_docx(filepath: str) -> list[dict]:
    doc   = Document(filepath)
    src   = os.path.basename(filepath)
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


# ── PDF READER ───────────────────────────────────────────
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
        print(f"  ❌ PDF error ({src}): {e}")
    return results


# ── CSV READER ───────────────────────────────────────────
def read_csv(filepath: str) -> list[dict]:
    src     = os.path.basename(filepath)
    batch   = []
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
        print(f"  ❌ CSV error ({src}): {e}")
    return results


# ── FILE DISPATCH ─────────────────────────────────────────
READERS = {
    ".docx": read_docx,
    ".pdf":  read_pdf,
    ".json": read_json,
    ".csv":  read_csv,
}


def read_file(filepath: str) -> list[dict]:
    ext    = Path(filepath).suffix.lower()
    reader = READERS.get(ext)
    if reader is None:
        return []
    try:
        return reader(filepath)
    except Exception as e:
        print(f"  ❌ Error ({os.path.basename(filepath)}): {e}")
        return []


# ── PINECONE UPSERT ──────────────────────────────────────
def upsert_all(chunks: list[dict], idx):
    total = len(chunks)
    print(f"🔄 Generating embeddings for {total} chunks...")

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

    print(f"📤 Uploading to Pinecone ({len(vectors)} vectors)...")
    success = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        try:
            idx.upsert(vectors=batch)
            success += len(batch)
        except Exception as e:
            print(f"  ⚠️  Batch {i//BATCH_SIZE} error: {e}")
        time.sleep(0.15)

    print(f"✅ {success}/{len(vectors)} vectors uploaded successfully.")


# ── MAIN ─────────────────────────────────────────────────
def start_ingestion():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"'{DATA_FOLDER}' folder created. Please copy your data files into it.")
        return

    idx = get_or_create_index()

    print("🗑️  Deleting existing vectors...")
    try:
        idx.delete(delete_all=True, namespace="")
        time.sleep(2)
    except Exception as e:
        print(f"  ⚠️  Error during deletion (may already be empty): {e}")

    all_chunks = []
    supported  = set(READERS.keys())

    print(f"\n📂 Reading files from '{DATA_FOLDER}'...")
    for filename in sorted(os.listdir(DATA_FOLDER)):
        ext = Path(filename).suffix.lower()
        if ext not in supported:
            continue
        filepath = os.path.join(DATA_FOLDER, filename)
        chunks   = read_file(filepath)
        if chunks:
            all_chunks.extend(chunks)
            print(f"  📄 {filename} → {len(chunks)} chunks")
        else:
            print(f"  ⚠️  {filename} → no chunks found")

    if not all_chunks:
        print("\n❌ No data to ingest.")
        return

    # Deduplicate chunks
    seen, unique = set(), []
    for c in all_chunks:
        key = c["text"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    removed = len(all_chunks) - len(unique)
    print(f"\n📦 Total: {len(all_chunks)} | Duplicates removed: {removed} | Clean: {len(unique)}")

    upsert_all(unique, idx)
    print("\n🎯 Ingestion complete!")

    try:
        stats = idx.describe_index_stats()
        print(f"📊 Pinecone total vectors: {stats.total_vector_count}")
    except Exception:
        pass


if __name__ == "__main__":
    start_ingestion()