import os
import markdown
import anthropic
from flask import Flask, render_template, request, jsonify
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ТОХИРГОО
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_KEY  = os.getenv("PINECONE_API_KEY")
INDEX_NAME    = "muis-chatbot"

claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def search_context(query: str, top_k: int = 10) -> str:
    try:
        query_embedding = embedder.encode(query).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return "\n\n".join([m["metadata"].get("text", "") for m in results["matches"]])
    except:
        return ""

def build_system_prompt(context: str, user_query: str) -> str:
    return f"""Та бол МУИС-ийн оюутны туслах чатбот юм. 

ХЭРЭГЛЭГЧИЙН ХАЙЛТ: {user_query}

ДҮРЭМ:
1. Хэрэв баримт бичигт тухайн нэрээр ЗӨВХӨН 1 ХҮН олдвол түүний бүх мэдээллийг (Овог нэр, Тэнхим, Албан тушаал, Боловсрол, Имэйл) шууд дэлгэрэнгүй харуул.
2. Хэрэв 2 БОЛОН ТҮҮНЭЭС ДЭЭШ хүн олдвол:
   - Хэнийх нь ч дэлгэрэнгүй мэдээллийг (имейл, боловсрол г.м) БИТГИЙ харуул.
   - Зөвхөн "Таны хайлтаар [Тоо] хүн олдлоо. Та алийг нь хайж байна вэ?" гэж асуу.
   - Сонголт бүрт зөвхөн Овог, Нэр, Тэнхимийг нь дугаарлан жагсааж харуул.
3. Хэрэглэгч дугаар эсвэл тодорхой нэрийг сонгож хэлэх хүртэл дэлгэрэнгүй мэдээллийг нууцлах ёстой.
4. Мэдээлэл байхгүй бол "Уучлаарай, мэдээлэл алга." гэж хариулна.
5. Хариултыг Монгол хэлээр маш товч бөгөөд цэгцтэй бичнэ.

БАРИМТ БИЧИГ:
{context}"""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    history = data.get("history", [])

    try:
        context = search_context(user_message)
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history[-5:]]
        messages.append({"role": "user", "content": user_message})

        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system=build_system_prompt(context, user_message),
            messages=messages
        )
        
        formatted_reply = markdown.markdown(response.content[0].text)
        return jsonify({"answer": formatted_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)