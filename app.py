from flask import Flask, request, jsonify, render_template
from rag.rag_engine import get_rag_chain

app = Flask(__name__)

# 🔁 Load RAG chain factory once (lazy loading inside)
rag_chain_factory = get_rag_chain()

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    answer = ""
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            try:
                chain = rag_chain_factory(query)
                response = chain.invoke({"question": query})
                answer = response.get("answer", "কোনো উত্তর পাওয়া যায়নি।")
            except Exception as e:
                answer = f"Error: {str(e)}"
    return render_template("index.html", query=query, answer=answer)

@app.route("/api/ask", methods=["POST"])
def ask_api():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": " query provided"}), 400

        chain = rag_chain_factory(query)
        response = chain.invoke({"question": query})
        return jsonify({"answer": response.get("answer", "No answer found")})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
