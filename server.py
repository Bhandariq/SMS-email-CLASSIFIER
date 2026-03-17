from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# safe download (important for Render)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# -----------------------------
# LOAD MODEL SAFELY
# -----------------------------
try:
    model = pickle.load(open("NB_spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("transform.pkl", "rb"))
except Exception as e:
    print("❌ Error loading model:", e)

# -----------------------------
# PREPROCESSING SETUP
# -----------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

history = []

def clean_text(text):
    text = str(text).lower()
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# HOME ROUTE
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    global history

    if request.method == "POST":
        message = request.form.get("message", "")

        if message.strip() == "":
            return render_template("index.html", prediction="⚠️ Enter a message", history=history)

        try:
            cleaned = clean_text(message)
            vector = vectorizer.transform([cleaned])

            result = model.predict(vector)[0]
            prob = model.predict_proba(vector)[0]
            confidence = round(max(prob) * 100, 2)

            if result == 1:
                output = f"🚫 Spam ({confidence}%)"
            else:
                output = f"✅ Not Spam ({confidence}%)"

        except Exception as e:
            output = "❌ Error processing input"
            print("Prediction error:", e)

        # save history
        history.insert(0, (message, output))
        history = history[:5]

        return render_template("index.html", prediction=output, history=history)

    return render_template("index.html", prediction=None, history=history)

# -----------------------------
# API ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    try:
        message = data.get("message")
        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])

        result = model.predict(vector)[0]

        return jsonify({"prediction": "Spam" if result == 1 else "Ham"})

    except Exception as e:
        print("API error:", e)
        return jsonify({"error": "Prediction failed"}), 500

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)