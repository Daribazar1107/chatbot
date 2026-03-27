"""
fetch_courses.py — Мэдээлэл, компьютерийн ухааны тэнхимийн хичээлүүд
"""
import requests, json, os, time

GRAPHQL_URL = "http://digital.num.edu.mn:4000/graphql"
OUTPUT_FILE = "data/courses.json"
DEPT        = "Мэдээлэл, компьютерийн ухааны тэнхим"
BATCH_SIZE  = 20

def gql(query):
    r = requests.post(GRAPHQL_URL, json={"query": query},
                      headers={"content-type": "application/json"}, timeout=60)
    data = r.json()
    if "errors" in data:
        print("❌ Алдаа:", data["errors"][0]["message"])
        return None
    return data

r = gql(f'{{ coursesCount(where: {{ has_department: {{ label: "{DEPT}" }} }}) }}')
total = r["data"]["coursesCount"] if r else 0
print(f"📊 Нийт хичээл: {total}")

all_raw = []
offset = 0
while offset < total:
    print(f"  📥 {offset+1}–{min(offset+BATCH_SIZE, total)} татаж байна...")
    r = gql(f"""
    {{
      courses(
        where: {{ has_department: {{ label: "{DEPT}" }} }}
        options: {{ limit: {BATCH_SIZE}, offset: {offset} }}
      ) {{
        Course_index
        Course_credits
        Degree_program
        Abstract
        Description
        has_name {{
          Entity_name
        }}
      }}
    }}
    """)
    if not r:
        break
    all_raw.extend(r["data"]["courses"])
    offset += BATCH_SIZE
    time.sleep(0.2)

courses = []
for c in all_raw:
    idx        = c.get("Course_index", "")
    names      = c.get("has_name") or []
    # ✅ Entity_name-г Монгол_нэр болгож оруулна
    mongol_ner = names[0].get("Entity_name", "") if names else ""
    credit     = c.get("Course_credits", "")
    degree     = c.get("Degree_program", "")
    desc       = c.get("Abstract") or c.get("Description") or ""

    courses.append({
        "Хичээлийн_индекс": idx,
        "Монгол_нэр":        mongol_ner if mongol_ner else idx,
        "Багц_цаг":          credit,
        "Зэрэг":             degree,
        "Товч_агуулга":      desc.strip()[:500] if desc else "",
        "Тэнхим":            DEPT,
    })

os.makedirs("data", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(courses, f, ensure_ascii=False, indent=2)

print(f"\n💾 Хадгалагдлаа: {OUTPUT_FILE}")
print(f"✅ Нийт: {len(courses)} хичээл")
print("\nЖишээ:")
for c in courses[:3]:
    print(f"  {c['Хичээлийн_индекс']} — {c['Монгол_нэр']} ({c['Багц_цаг']} багц цаг)")
print("\n▶️  Дараа нь: python ingest.py")