# ✈️ BrandPulse AI — Customer Sentiment Intelligence

An end-to-end Natural Language Processing (NLP) and Deep Learning pipeline that classifies customer tweets into **Positive**, **Neutral**, and **Negative** sentiment, comparing a classical **TF-IDF + Logistic Regression** baseline against a sequence-aware **LSTM Neural Network**, deployed via an interactive **Streamlit** dashboard.

---

## 📁 Project Structure

```text
brandpulse-sentiment/
├── data/
│   ├── raw/                       # Original Twitter US Airline Sentiment dataset
│   └── processed/                 # Cleaned & lemmatized tweets (cleaned_tweets.csv)
├── models/
│   ├── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer
│   ├── logistic_model.pkl         # Trained Logistic Regression baseline
│   ├── lstm_tokenizer.pkl         # Fitted Keras word tokenizer
│   └── lstm_sentiment_model.keras # Trained LSTM neural network
├── notebooks/
│   ├── 01Preprocessingand_Classical.ipynb  # Regex cleaning, EDA, TF-IDF & Classical ML
│   └── 02DeepLearning_LSTM.ipynb           # Tokenization, padding, LSTM training & comparison
├── app.py                         # Interactive Streamlit web dashboard
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
