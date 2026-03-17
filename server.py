from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---- NLTK SAFE LOAD (works on Render) ----
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir='/tmp')
    nltk.data.path.append('/tmp')

app = Flask(__name__)

# ---- LOAD MODEL SAFELY ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "NB_spam_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "transform.pkl"), "rb"))

# ---- PREPROCESS ----
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---- SESSION STATS (in-memory) ----
stats = {"total": 0, "spam": 0, "ham": 0}

@app.route("/")
def home():
    return render_template("index.html")

# ---- REAL-TIME API ----
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    if not message.strip():
        return jsonify({"error": "Empty message"}), 400

    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])

    result = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]
    confidence = round(max(prob) * 100, 2)

    label = "Spam" if result == 1 else "Ham"
    pretty = f"{'🚫 Spam' if result == 1 else '✅ Not Spam'} ({confidence}%)"

    # update stats
    stats["total"] += 1
    if result == 1:
        stats["spam"] += 1
    else:
        stats["ham"] += 1

    return jsonify({
        "label": label,
        "pretty": pretty,
        "confidence": confidence,
        "stats": stats
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)