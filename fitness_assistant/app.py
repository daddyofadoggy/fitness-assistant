import uuid

from flask import Flask, request, jsonify

from rag_utils import rag

app = Flask(__name__)
print(app)

@app.route("/", methods=["GET"])
def home():
    return "Hello, Flask is working!", 200


@app.route("/question", methods=["POST"])
def handle_question():
    print('Question')
    data = request.json
    question = data["question"]

    if not question:
        return jsonify({"error": "No question provided"}), 400

    conversation_id = str(uuid.uuid4())

    answer_data = rag(question)
    #answer_data = {"answer": "Test answer"}

    result = {
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer_data["answer"],
    }

    return jsonify(result)

@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.json
    conversation_id = data["conversation_id"]
    feedback = data["feedback"]

    if not conversation_id or feedback not in [1, -1]:
        return jsonify({"error": "Invalid input"}), 400

    result = {
        "message": f"Feedback received for conversation {conversation_id}: {feedback}"
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)