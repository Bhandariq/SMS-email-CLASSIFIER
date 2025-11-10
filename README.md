# 📧 SMS & Email Classifier

A machine learning project that classifies SMS messages and emails as **Spam** or **Ham (legitimate)**.  
It combines **data preprocessing, model training, and deployment via Flask** into a complete end-to-end solution.

---

## ✨ Features
- 🔍 **Spam Detection** using **Naive Bayes**  
- 🧹 **Text Preprocessing**: cleaning, tokenization, stopword removal, vectorization  
- 💾 **Model Persistence**: trained model (`NB_spam_model.pkl`) and transformer (`transform.pkl`) saved for reuse  
- 🌐 **Web Interface**: Flask app with simple UI for testing messages  
- 📊 **Exploratory Data Analysis (EDA)** included in Jupyter Notebook  

---

## 📂 Project Structure
```plaintext
SMS-email-CLASSIFIER/
│
├── sms-detector-EDA.ipynb    # Dataset exploration & visualization
├── model_creation.py         # Training script for Naive Bayes model
├── server.py                 # Flask server for deployment
├── spam_data.csv             # Dataset (SMS/email samples)
├── NB_spam_model.pkl         # Trained model
├── transform.pkl             # Text transformer (vectorizer)
├── requirements.txt          # Dependencies
│
├── templates/                # HTML templates for frontend
│   └── index.html
│
└── static/                   # CSS/JS files for styling
```

---

## ⚙️ Installation & Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/Bhandariq/SMS-email-CLASSIFIER.git
   cd SMS-email-CLASSIFIER
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask server**
   ```bash
   python server.py
   ```

4. **Open in browser**  
   Visit: `http://127.0.0.1:5000/`

---

## 🖥️ Usage Example
- Enter an SMS or email text in the input box.  
- Click **Classify**.  
- The app will return either:
  - ✅ **Ham** → Legitimate message  
  - 🚫 **Spam** → Unwanted/advertisement message  

---

## 📊 Workflow
1. **Data Preprocessing**
   - Lowercasing, punctuation removal, stopword filtering  
   - Vectorization using `CountVectorizer`  

2. **Model Training**
   - Naive Bayes classifier trained on `spam_data.csv`  

3. **Model Saving**
   - Model and transformer saved as `.pkl` files  

4. **Deployment**
   - Flask app loads saved model and transformer  
   - User inputs are classified in real-time  

---

## 📸 Screenshots  
Example:
- **Homepage UI**
- <img width="1895" height="926" alt="image" src="https://github.com/user-attachments/assets/7f6455fc-d800-4cb8-acff-1fadf2fe950c" />

- **Classification Result Page**
<img width="1862" height="873" alt="image" src="https://github.com/user-attachments/assets/2431b7b6-5117-473a-87ad-721ac4e24908" />

---

## 🔮 Future Improvements
- Integrate **deep learning models** (LSTM, Transformers)  
- Add **real-time email integration** (e.g., Gmail API)  
- Improve UI with modern frameworks (React, Bootstrap)  
- Expand dataset for multilingual spam detection  

---

## 👨‍💻 Author
Developed by **[Bhandariq](https://github.com/Bhandariq)**  

---

👉 This version is **professional, detailed, and beginner-friendly**, with clear steps and placeholders for screenshots.  

