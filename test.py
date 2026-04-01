from app import classify_query

tests = [
    ('бууз хийх арга', False),
    ('өнөөдрийн цаг агаар', False),
    ('монгол хоол хийх арга', False),
    ('голч дүн хэрхэн тооцдог', True),
    ('кредит цаг дүүргэх', True),
    ('шалгалтаас чөлөөлөгдөх', True),
    ('сургуулиас хасах шалтгаан', True),
    ('чөлөө авах журам', True),
    ('багшийн утасны дугаар', True),
]

for q, expected in tests:
    r = classify_query(q)
    got = r["is_relevant"]
    ok = '✅' if got == expected else '❌'
    print(f"{ok} [{r['method']:15}] score={r['score']} | {q}")