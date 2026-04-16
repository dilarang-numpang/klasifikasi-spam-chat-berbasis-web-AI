import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset yang SUDAH BENAR
data = pd.read_csv("sms_spam_indo.csv")

X = data["pesan"]   # teks
y = data["label"]   # spam / ham

# Vectorisasi teks
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Model klasifikasi
model = MultinomialNB()
model.fit(X_vec, y)

def predict_text(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]
