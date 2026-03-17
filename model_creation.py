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

# download stopwords (only once)
nltk.download('stopwords')

print("Training started...")

# load dataset
df = pd.read_csv("spam_data.csv", encoding="latin-1", on_bad_lines='skip')

# keep only first 2 columns and rename
df = df.iloc[:, :2]
df.columns = ["type", "message"]

# basic cleaning
df.dropna(inplace=True)
df['type'] = df['type'].str.lower()

# keep only valid labels
df = df[df['type'].isin(['ham', 'spam'])]

# convert labels
df['label'] = df['type'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

print("Dataset loaded:", len(df))

# preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

X = X.apply(clean_text)

print("Text cleaned")

# vectorization
vectorizer = TfidfVectorizer(max_features=4000)
X = vectorizer.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained")

# evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# save files
pickle.dump(model, open("NB_spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("transform.pkl", "wb"))

print("Done 👍")