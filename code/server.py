# server.py
"""
Flask API for chatbot brain + lead prediction model.
Run: python code/server.py
"""

from flask import Flask, request, jsonify
import joblib
from chatbot_brain import handle

# -----------------
# Setup
# -----------------
app = Flask(__name__)

# Load model at startup (optional, but won't be passed to handle anymore)
MODEL_PATH = "models/lead_pipeline.joblib"
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# -----------------
# Routes
# -----------------

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Input JSON:
    {
      "session_id": "s1",
      "message": "predict likelihood",
      "note": "Customer asked for pricing and a demo"
    }
    Output: chatbot_brain.handle(...) result
    """
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    message = data.get("message", "")
    note = data.get("note", "")

    # Call handle without model argument
    result = handle(session_id, message, note)
    return jsonify(result)


@app.route("/predict", methods=["POST"])
def predict_shortcut():
    """
    Input JSON:
    {
      "note": "Customer liked the demo and asked about pricing."
    }
    Output: probability only
    """
    data = request.get_json() or {}
    note = data.get("note", "")
    if not note:
        return jsonify({"error": "No note provided"}), 400

    result = handle(
        data.get("session_id", "default"),
        "predict likelihood",
        note
    )
    return jsonify(result)


@app.route("/", methods=["GET"])
def health():
    return {"status": "ok", "message": "Chatbot brain API running"}


# -----------------
# Run server
# -----------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
