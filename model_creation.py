import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (safe check)
nltk.download('stopwords', quiet=True)

print("Training started...")

# Load dataset
df = pd.read_csv("spam_data.csv", encoding="latin-1", on_bad_lines='skip')

# Keep only first 2 columns and rename
df = df.iloc[:, :2]
df.columns = ["type", "message"]

# Basic cleaning
df.dropna(inplace=True)
df['type'] = df['type'].str.lower()

# Keep only valid labels
df = df[df['type'].isin(['ham', 'spam'])]

# Convert labels
df['label'] = df['type'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

print("Dataset loaded:", len(df))

# Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

X = X.apply(clean_text)

print("Text cleaned")

# Vectorization
vectorizer = TfidfVectorizer(max_features=4000)
X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained")

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nReport:\n", classification_report(y_test, y_pred))

# Save files
with open("NB_spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("transform.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Done — model and vectorizer saved")