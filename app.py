import string
import re
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
import streamlit as st

# Load the fine-tuned DistilBERT model
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = TFDistilBertForSequenceClassification.from_pretrained('./models/fine_tuned_distilbert', num_labels=2)

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='tf')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = np.argmax(logits).item()
    sentiment = 'Positive' if predicted_label == 1 else 'Negative'
    return sentiment

st.title("Sentiment Analysis with DistilBERT")

review_text = st.text_area("Enter a review:")

if st.button("Submit"):
    if review_text:
        sentiment = perform_sentiment_analysis(review_text)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review.")
