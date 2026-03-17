from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---- INIT APP ----
app = Flask(__name__)

# ---- NLTK SETUP ----
NLTK_DATA_PATH = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Ensure stopwords are available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DATA_PATH)

# ---- LOAD MODEL & VECTORIZER ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "NB_spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "transform.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ---- PREPROCESS ----
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Lowercase, remove non-alphanumeric, remove stopwords, apply stemming."""
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text).lower())
    words = [ps.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---- STATS ----
stats = {"total": 0, "spam": 0, "ham": 0}

# ---- ROUTES ----
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = data.get("text", "").strip()  # ✅ unified key name "text"

    if not message:
        return jsonify({"error": "Empty message"}), 400

    cleaned = clean_text(message)

    try:
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    confidence = round(max(prob) * 100, 2)
    label = "Spam" if result == 1 else "Ham"
    pretty = f"{'🚫 Spam' if result == 1 else '✅ Not Spam'} ({confidence}%)"

    # update stats
    stats["total"] += 1
    stats["spam" if result == 1 else "ham"] += 1

    return jsonify({
        "label": label,
        "pretty": pretty,
        "confidence": confidence,
        "stats": stats
    })

# ---- RUN ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)