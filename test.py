from app import classify_query

tests = [
    ('бууз хийх арга', False),
    ('монгол хоол хийх арга', False),
    ('өнөөдрийн цаг агаар', False),
    ('сургуулиас хасах шалтгаан', True),
    ('багшийн утасны дугаар', True),
    ('голч дүн хэрхэн тооцдог', True),
    ('чөлөө авах журам', True),
]

for q, expected in tests:
    r = classify_query(q)
    got = r["is_relevant"]
    ok = '✅' if got == expected else '❌'
    print(f"{ok} [{r['method']:15}] score={r['score']} | {q}")