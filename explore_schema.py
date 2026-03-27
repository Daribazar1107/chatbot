"""
explore_schema.py — МУИС GraphQL schema бүрэн судлах
"""
import requests, json

GRAPHQL_URL = "http://digital.num.edu.mn:4000/graphql"

def gql(query):
    r = requests.post(GRAPHQL_URL, json={"query": query},
                      headers={"content-type": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()

# 1. Query root-ийн бүх боломжит query-нүүд
print("=" * 60)
print("1️⃣  Query root fields (татаж авч болох бүх зүйл):")
print("=" * 60)
result = gql("""
{
  __type(name: "Query") {
    fields {
      name
      args { name type { name kind ofType { name } } }
      type { name kind ofType { name kind ofType { name } } }
    }
  }
}
""")
fields = result["data"]["__type"]["fields"] or []
for f in fields:
    print(f"  query: {f['name']}")

# 2. Person type-ийн field-үүд
print("\n" + "=" * 60)
print("2️⃣  Person type fields:")
print("=" * 60)
for type_name in ["Person", "Staff", "Employee", "Teacher", "Bagsh"]:
    r2 = gql(f"""
    {{
      __type(name: "{type_name}") {{
        name
        fields {{
          name
          type {{ name kind ofType {{ name }} }}
        }}
      }}
    }}
    """)
    t = r2["data"]["__type"]
    if t:
        print(f"\n  ✅ Type '{type_name}' олдлоо:")
        for f in (t["fields"] or []):
            print(f"    {f['name']}")
        break
else:
    print("  ⚠️  Стандарт нэр олдсонгүй")