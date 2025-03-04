from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import re
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import os
from transformers import DistilBertModel, DistilBertTokenizer

# Download stopwords if not available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load DistilBERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load the FAQ dataset
faq_file = r"C:\Users\aniru\Downloads\demoFAQ's (1).csv"
faq_df = pd.read_csv(faq_file)
faq_df.columns = ["Question", "Answer"]

# Function to clean and preprocess text
def extract_keywords(text):
    if not isinstance(text, str) or not text.strip():
        return []
    words = re.findall(r'\b\w+\b', text.lower())  # Tokenize
    keywords = [word for word in words if word not in stop_words]
    return keywords

# Function to encode text using DistilBERT
def encode_text(text):
    if not text.strip():
        return np.zeros(768)  # Return zero vector for empty input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model1(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Precompute FAQ embeddings and keyword lists
faq_df["Keywords"] = faq_df["Question"].apply(extract_keywords)
faq_encodings = np.array([encode_text(text) for text in faq_df["Question"]])

# Function to get FAQ response with keyword-based priority
def get_faq_response(user_query):
    user_keywords = extract_keywords(user_query)
    user_query_encoding = encode_text(user_query)

    # Step 1: Check for exact keyword matches
    for idx, faq_keywords in enumerate(faq_df["Keywords"]):
        if any(keyword in faq_keywords for keyword in user_keywords):
            return faq_df.iloc[idx]["Answer"]

    # Step 2: Compute similarity with precomputed FAQ embeddings
    similarity_scores = cosine_similarity([user_query_encoding], faq_encodings)[0]
    max_score_index = np.argmax(similarity_scores)
    max_score = similarity_scores[max_score_index]

    # Step 3: Apply stricter threshold to avoid irrelevant matches
    strict_threshold = 0.80  # Only accept highly similar responses
    if max_score > strict_threshold:
        print(max_score,faq_df.iloc[max_score]["Answer"])
        return faq_df.iloc[max_score]["Answer"]

    return "I'm sorry, I don't have information on that yet."

# Example usage
while True:
    user_input = input("\nAsk me a question (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    print("Answer:", get_faq_response(user_input))



app = Flask(__name__)
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('question')
    response = get_faq_response(user_input)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)