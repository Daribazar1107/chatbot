"""
fetch_teachers.py — Мэдээлэл, компьютерийн ухааны тэнхимийн багш нар
"""
import requests, json, os

GRAPHQL_URL = "http://digital.num.edu.mn:4000/graphql"
OUTPUT_FILE = "data/teachers.json"

def gql(query):
    r = requests.post(GRAPHQL_URL, json={"query": query},
                      headers={"content-type": "application/json"}, timeout=60)
    data = r.json()
    if "errors" in data:
        print("❌ Алдаа:", data["errors"][0]["message"])
        return None
    return data

print("📥 Мэдээлэл, компьютерийн ухааны тэнхимийн багш нарыг татаж байна...")

r = gql("""
{
  organizations(where: { label: "Мэдээлэл, компьютерийн ухааны тэнхим" }) {
    label
    staffs {
      label
      Email
      room {
        label
      }
    }
  }
}
""")

if not r or not r["data"]["organizations"]:
    print("❌ Тэнхим олдсонгүй")
    exit()

org = r["data"]["organizations"][0]
staffs = org.get("staffs", [])
print(f"✅ {len(staffs)} багш олдлоо")

teachers = []
for s in staffs:
    room_obj = s.get("room")
    oroo = ""
    if isinstance(room_obj, list) and room_obj:
        oroo = room_obj[0].get("label", "")
    elif isinstance(room_obj, dict):
        oroo = room_obj.get("label", "")

    teachers.append({
        "ner":             s.get("label", ""),
        "salbar_surguuli": "Мэдээлэл, компьютерийн ухааны тэнхим",
        "uruunii_dugaar":  oroo,
        "utas":            "",
        "email":           s.get("Email", "") or "",
    })

os.makedirs("data", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(teachers, f, ensure_ascii=False, indent=2)

print(f"💾 Хадгалагдлаа: {OUTPUT_FILE}")
print(f"📊 Нийт: {len(teachers)} багш")
print("\n▶️  Дараа нь: python ingest.py")