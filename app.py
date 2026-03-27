import os
import markdown
import anthropic
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader

load_dotenv()
app = Flask(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

pdf_context = ""

SYSTEM_PROMPT = """Та МУИС (Монгол Улсын Их Сургууль)-ийн оюутнуудад зориулсан дижитал туслах юм.
Монгол хэлээр товч, тодорхой, найрсаг байдлаар хариулна уу.
Хэрэв PDF мэдээлэл өгөгдсөн бол түүнийг үндэслэн хариулна уу.
Мэдэхгүй зүйлийг таамаглахгүй, шударгаар хэлнэ үү."""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global pdf_context
    files = request.files.getlist("file")
    if not files:
        return jsonify({"success": False, "error": "Файл олдсонгүй"}), 400

    extracted_text = ""
    chunk_count = 0

    for f in files:
        try:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
                    chunk_count += 1
        except Exception as e:
            return jsonify({"success": False, "error": f"Файл уншихад алдаа: {str(e)}"}), 500

    pdf_context = extracted_text
    return jsonify({"success": True, "chunks": chunk_count})

@app.route("/chat", methods=["POST"])
def chat():
    global pdf_context
    data = request.json
    user_msg = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_msg:
        return jsonify({"error": "Хоосон мессеж"}), 400

    # Build system prompt — append PDF context only if available
    system = SYSTEM_PROMPT
    if pdf_context:
        system += f"\n\nНэмэлт мэдээлэл (PDF):\n{pdf_context[:15000]}"

    # Keep last 10 turns (20 messages) to stay within context limits
    messages = []
    for h in history[-20:]:
        role = h.get("role")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_msg})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        answer_md = response.content[0].text
        answer_html = markdown.markdown(answer_md)
        return jsonify({"answer": answer_html})
    except anthropic.APIError as e:
        return jsonify({"error": f"API алдаа: {str(e)}"}), 502

if __name__ == "__main__":
    app.run(debug=True)