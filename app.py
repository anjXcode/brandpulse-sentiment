import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import pickle
import nltk
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Configuration
st.set_page_config(
    page_title="BrandPulse AI | Sentiment Intelligence",
    page_icon="✈️",
    layout="wide"
)

# Download required NLTK resources silently
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

setup_nltk()

# Load Models & Vectorizers (Cached for fast performance)
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    lr_model = joblib.load("models/logistic_model.pkl")
    with open("models/lstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    lstm_model = load_model("models/lstm_sentiment_model.keras")
    return tfidf, lr_model, tokenizer, lstm_model

tfidf, lr_model, tokenizer, lstm_model = load_artifacts()

# Preprocessing Function (Matches training pipeline exactly)
lemmatizer = WordNetLemmatizer()
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'cannot', "don't", "aren't", "wasn't"}
stop_words = set(stopwords.words('english')) - negation_words
MAX_LEN = 50
LABELS = ['negative', 'neutral', 'positive']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join(words)

def predict_sentiment(text, model_choice):
    cleaned = clean_text(text)
    if model_choice == "Deep Learning (LSTM)":
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        probs = lstm_model.predict(padded, verbose=0)[0]
    else:
        vec = tfidf.transform([cleaned])
        probs = lr_model.predict_proba(vec)[0]
        # Ensure class probability order matches ['negative', 'neutral', 'positive']
        class_order = list(lr_model.classes_)
        probs = np.array([probs[class_order.index(lbl)] for lbl in LABELS])
    
    pred_label = LABELS[np.argmax(probs)]
    return pred_label, probs, cleaned

# --- SIDEBAR ---
st.sidebar.title("⚙️ Model Settings")
model_choice = st.sidebar.radio(
    "Select Classification Engine:",
    ["Deep Learning (LSTM)", "Classical NLP (Logistic Regression)"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**BrandPulse AI** analyzes customer tweets to detect brand sentiment in real time.\n\n"
    "- **LSTM:** Captures word order, context, and negations.\n"
    "- **Logistic Regression:** Fast statistical TF-IDF baseline."
)

# --- MAIN HEADER ---
st.title("✈️ BrandPulse AI — Customer Sentiment Dashboard")
st.markdown("Real-time social media sentiment monitoring powered by **Classical NLP** and **Deep Learning (LSTM)**.")

tab1, tab2 = st.tabs(["🔍 Live Tweet Analyzer", "📊 Batch CSV Insights"])

# --- TAB 1: SINGLE TWEET ANALYZER ---
with tab1:
    st.subheader("Analyze Single Customer Tweet or Feedback")
    user_input = st.text_area(
        "Enter tweet or review text:",
        placeholder="e.g., The flight was delayed by 3 hours with zero updates from the staff!",
        height=110
    )

    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            label, probs, cleaned_out = predict_sentiment(user_input, model_choice)
            
            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.markdown("### Predicted Sentiment")
                if label == "positive":
                    st.success(f"**POSITIVE** ({probs[2]*100:.1f}% confidence)")
                elif label == "neutral":
                    st.warning(f"**NEUTRAL** ({probs[1]*100:.1f}% confidence)")
                else:
                    st.error(f"**NEGATIVE** ({probs[0]*100:.1f}% confidence)")
                
                with st.expander("View Preprocessed Tokens"):
                    st.code(cleaned_out if cleaned_out else "[No valid tokens remaining]")

            with col2:
                prob_df = pd.DataFrame({
                    "Sentiment": ["Negative", "Neutral", "Positive"],
                    "Probability": probs
                })
                fig = px.bar(
                    prob_df, x="Sentiment", y="Probability", color="Sentiment",
                    color_discrete_map={"Negative": "#e74c3c", "Neutral": "#f1c40f", "Positive": "#2ecc71"},
                    range_y=[0, 1], title=f"Confidence Breakdown ({model_choice})"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter some text before clicking Analyze.")

# --- TAB 2: BATCH CSV UPLOAD ---
with tab2:
    st.subheader("Batch Sentiment Analysis on Customer Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file containing a 'text' column", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if 'text' not in batch_df.columns:
            st.error("The uploaded CSV must contain a column named `'text'`.")
        else:
            sample_df = batch_df.head(200).copy()  # Cap at 200 rows for instant UI responsiveness
            st.info(f"Analyzing first {len(sample_df)} rows using **{model_choice}**...")
            
            preds = [predict_sentiment(t, model_choice)[0] for t in sample_df['text']]
            sample_df['predicted_sentiment'] = preds

            c1, c2 = st.columns(2)
            with c1:
                counts = sample_df['predicted_sentiment'].value_counts().reset_index()
                counts.columns = ['Sentiment', 'Count']
                pie_fig = px.pie(
                    counts, names='Sentiment', values='Count', hole=0.4,
                    color='Sentiment',
                    color_discrete_map={'negative': '#e74c3c', 'neutral': '#f1c40f', 'positive': '#2ecc71'},
                    title="Overall Sentiment Distribution"
                )
                st.plotly_chart(pie_fig, use_container_width=True)

            with c2:
                st.dataframe(sample_df[['text', 'predicted_sentiment']], height=350, use_container_width=True)

            csv_export = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=csv_export,
                file_name="brandpulse_predictions.csv",
                mime="text/csv"
            )
            