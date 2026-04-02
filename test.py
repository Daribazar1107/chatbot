import json
with open('data/level.json', encoding='utf-8') as f:
    data = json.load(f)
items = data if isinstance(data, list) else [data]
print(f'Нийт: {len(items)} item')
for item in items[:3]:
    print(json.dumps(item, ensure_ascii=False, indent=2)[:300])
    print('---')