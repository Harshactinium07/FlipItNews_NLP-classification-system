
import re
import nltk
import streamlit as st
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

@st.cache_resource
def load_artifacts():
    model = load_model("news_model.keras")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, encoder

model, vectorizer, encoder = load_artifacts()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens
        if word not in stop_words and word.strip()
    ]
    return " ".join(tokens)

st.set_page_config(
    page_title="FlipItNews",
    page_icon="📰",
    layout="centered"
)

st.title("📰 FlipItNews")
st.subheader("AI-Powered News Classification")
st.write(
    "Enter a news article below to predict its category."
)

article = st.text_area(
    "News article",
    height=250,
    placeholder="Paste your news article here..."
)

if st.button("Predict Category", type="primary"):
    if not article.strip():
        st.warning("Please enter a news article first.")
    else:
        cleaned = preprocess_text(article)
        features = vectorizer.transform([cleaned])

        prediction = model.predict(
            features.toarray(),
            verbose=0
        )

        class_index = prediction.argmax(axis=1)[0]
        category = encoder.inverse_transform([class_index])[0]
        confidence = float(prediction[0][class_index]) * 100

        st.success(f"Predicted Category: {category.title()}")
        st.metric("Model Confidence", f"{confidence:.2f}%")

        st.caption(
            "Confidence is the model's predicted probability, "
            "not a guarantee of correctness."
        )
