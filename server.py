from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

# load model
model = pickle.load(open("NB_spam_model.pkl", "rb"))
vectorizer = pickle.load(open("transform.pkl", "rb"))

# preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# store last 5 results
history = []

def clean_text(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        message = request.form.get("message")

        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])

        result = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]
        confidence = round(max(prob) * 100, 2)

        if result == 1:
            output = f"🚫 Spam ({confidence}%)"
        else:
            output = f"✅ Not Spam ({confidence}%)"

        # save history
        history.insert(0, (message, output))
        if len(history) > 5:
            history.pop()

        return render_template("index.html", prediction=output, history=history)

    return render_template("index.html", prediction=None, history=history)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message")

    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])

    result = model.predict(vector)[0]

    if result == 1:
        return jsonify({"prediction": "Spam"})
    else:
        return jsonify({"prediction": "Ham"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)