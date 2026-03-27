from flask import Flask, render_template, request, jsonify, session
import anthropic
import os

app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Session-д ашиглана

# Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# =============================================
# ИХ СУРГУУЛИЙН МЭДЭЭЛЭЛ - энд өөрчлөөрэй
# =============================================
UNIVERSITY_SYSTEM_PROMPT = """
Та Монголын Улсын Их Сургуулийн ухаалаг туслах chatbot юм.
Монгол болон Англи хэлээр хариулж чадна. Хэрэглэгч ямар хэлээр асуувал тэр хэлээр хариулна.
Хариултаа товч, тодорхой, найрсаг байлга.

## СУРГУУЛИЙН МЭДЭЭЛЭЛ:

### Элсэлт
- Элсэлтийн шалгалт: Жил бүр 6-р сарын 1-30
- Шаардлага: ЭЕШ-ын 500+ оноо
- Бүртгэл: www.elshilt.mn сайтаар
- Холбоо барих: elshilt@university.mn | 7700-1234

### Сургалтын төлбөр
- Бакалавр: 2,500,000₮ / жилд
- Магистр: 3,200,000₮ / жилд
- Доктор: 4,000,000₮ / жилд
- Тэтгэлэг: Шилдэг 10% оюутанд 50% хөнгөлөлт

### Сургуулийн цаг
- Даваа-Баасан: 08:00 - 20:00
- Бямба: 09:00 - 15:00
- Ням: амарна

### Факультетүүд
- Бизнес ба Эдийн засгийн факультет
- Мэдээллийн технологийн факультет
- Хууль зүйн факультет
- Анагаах ухааны факультет
- Инженерийн факультет

### Байршил
- Хаяг: Улаанбаатар хот, Сүхбаатар дүүрэг, Их Сургуулийн гудамж 1
- Утас: 7700-0000
- Имэйл: info@university.mn

### Номын сан
- Цагийн хуваарь: Да-Ба 08:00-21:00, Бя 09:00-17:00
- Онлайн эх сурвалж: library.university.mn

Хэрэв мэдэхгүй зүйл асуувал: "Энэ талаар дэлгэрэнгүй мэдээллийг 7700-0000 утсаар авна уу" гэж хэлнэ.
"""

@app.route("/")
def index():
    # Шинэ хэрэглэгч ирвэл түүхийг цэвэрлэнэ
    session["history"] = []
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Хоосон мессеж"}), 400

    # Харилцааны түүхийг session-д хадгална
    if "history" not in session:
        session["history"] = []

    # Хэрэглэгчийн мессежийг нэмнэ
    session["history"].append({
        "role": "user",
        "content": user_message
    })

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=UNIVERSITY_SYSTEM_PROMPT,
            messages=session["history"]
        )

        assistant_reply = response.content[0].text

        # Chatbot хариуг түүхэнд нэмнэ
        session["history"].append({
            "role": "assistant",
            "content": assistant_reply
        })

        # Session-г шинэчлэх
        session.modified = True

        return jsonify({"reply": assistant_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    session["history"] = []
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)