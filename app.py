HEAD
# app.py

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# 🔒 Cache the model & vectorizer so they don't re-run
@st.cache_resource
def train_model():
    df = pd.read_csv("IMDB Dataset.csv")
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    model = LogisticRegression()
    model.fit(X, y)
    return model, vectorizer

# UI
st.set_page_config(page_title="Fast Sentiment Analyzer", layout="wide")
st.title("⚡ Sentiment Analysis")

# Load model/vectorizer once
model, vectorizer = train_model()

# Prediction UI
st.subheader("✍️ Enter Your Movie Review")
user_input = st.text_area("Write your review:")

if st.button("Analyze Sentiment"):
    cleaned_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vector)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    st.markdown(f"## ✅ Sentiment: **{sentiment}**")
# app.py

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# 🔒 Cache the model & vectorizer so they don't re-run
@st.cache_resource
def train_model():
    df = pd.read_csv("IMDB Dataset.csv")
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    model = LogisticRegression()
    model.fit(X, y)
    return model, vectorizer

# UI
st.set_page_config(page_title="Fast Sentiment Analyzer", layout="wide")
st.title("⚡ Sentiment Analysis")

# Load model/vectorizer once
model, vectorizer = train_model()

# Prediction UI
st.subheader("✍️ Enter Your Movie Review")
user_input = st.text_area("Write your review:")

if st.button("Analyze Sentiment"):
    cleaned_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vector)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    st.markdown(f"## ✅ Sentiment: **{sentiment}**")
 de4064f (Initial commit with app, dataset, and gitignore)
