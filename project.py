#Sympsis : Your Healthcare Assistant for Symptom Diagnosis and Disease Prediction


from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import json
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import csv
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import re
from rapidfuzz import process, fuzz
import langdetect
import random

# Load the dataset
file_path = r"C:\Users\aniru\dataset.csv"
data = pd.read_csv(file_path)
CSV_FILE = "issues.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Email/Phone", "Issue"])

# Combine multiple symptom columns into a single text feature
symptom_cols = data.columns[:-1]
data["Combined_Symptoms"] = data[symptom_cols].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

# Features and labels
X = data["Combined_Symptoms"]
y = data["Disease"].astype(str)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Known symptoms extraction
known_symptoms = set()
symptom_to_diseases = {}


for index, row in data.iterrows():
    disease = row["Disease"]
    for symptom in row[symptom_cols]:
        if pd.notna(symptom):
            symptom = str(symptom).strip().lower()
            known_symptoms.add(symptom)
            if symptom not in symptom_to_diseases:
                symptom_to_diseases[symptom] = []
            symptom_to_diseases[symptom].append(disease)

# List of inappropriate words
inappropriate_words = {"fuck", "shit", "bullshit", "damn", "bitch"}

# Extract symptoms using fuzzy matching
# Extract symptoms using fuzzy matching
def extract_symptoms(user_input):
    words = re.findall(r"\b\w+\b", user_input.lower())
    detected_symptoms = []
    correction_suggestion = []
    
    for word in words:
        matches = process.extract(word, known_symptoms, scorer=fuzz.partial_ratio, limit=3)
        best_match = max(matches, key=lambda x: x[1], default=(None, 0))
        
        if best_match[1] == 100:
            detected_symptoms.append(best_match[0])
        elif best_match[1] >= 80:
            correction_suggestion.append(f"Do you mean {best_match[0]}?")
    
    if correction_suggestion:
        return correction_suggestion
    return detected_symptoms if detected_symptoms else []


# Prediction function
predicted_diseases = set()

# Updated prediction function
def predict_disease(user_input):
    # Check for short greetings
    if user_input.lower() in {"hi", "hello", "hey", "how are you", "what's up"}:
        return "Hello👋! How can I help you today?"
    
    # Check for inappropriate words
    if any(word in user_input.lower() for word in inappropriate_words):
        return "Sorry, but that's an inappropriate message 😑."
    
    # Check for exit phrases
    if user_input.lower() in {"bye", "exit", "quit", "goodbye"}:
        return "Take care 😊! If you feel unwell, don’t hesitate to seek medical advice. Goodbye!"
    
    # Check for long input first
    if len(user_input.split()) > 100:
        return "Please enter a short description."
    
    # Check for numeric input
    if any(char.isdigit() for char in user_input):
        return "Please enter symptoms, not numbers."
    
    # Check for invalid characters
    if re.search(r"[^a-zA-Z0-9\s]", user_input):
        return "Error: Invalid symptoms."
    
    # Detect language of input
    try:
        if len(user_input.split()) > 2:
            detected_language = langdetect.detect(user_input)
            if detected_language != "en":
                return "Please enter symptoms in English."
    except:
        return "Unable to detect language. Please enter symptoms in English."
    
    # Detect if the user input contains the word "random"
    if "random" in user_input.lower():
        # Get the list of diseases
        all_diseases = list(y.unique())  # All possible diseases from the dataset
        
        # If all diseases have been predicted before, reset the list of predicted diseases
        if len(predicted_diseases) == len(all_diseases):
            predicted_diseases.clear()  # Reset the set of predicted diseases
        
        # Pick a random disease that hasn't been predicted before
        remaining_diseases = list(set(all_diseases) - predicted_diseases)
        if remaining_diseases:
            random_disease = random.choice(remaining_diseases)
            predicted_diseases.add(random_disease)
            return f"Random disease predicted: {random_disease}"
        else:
            # If no remaining diseases, reset the set and pick any disease
            random_disease = random.choice(all_diseases)
            predicted_diseases.add(random_disease)
            return f"Random disease predicted: {random_disease}"
    
    # Extract symptoms from the input
    symptoms = extract_symptoms(user_input)
    
    if isinstance(symptoms, list) and symptoms and "Do you mean" in symptoms[0]:
        return symptoms[0]
    
    if not symptoms:
        return "Sorry, I couldn't understand ☹️. Please enter valid symptoms."
    
    # Vectorize the symptoms for prediction
    symptoms_vectorized = vectorizer.transform([' '.join(symptoms)])
    probabilities = model.predict_proba(symptoms_vectorized)[0]
    
    disease_probabilities = {model.classes_[i]: probabilities[i] for i in range(len(model.classes_))}

    # Adjust disease probabilities based on known symptoms
    for disease in disease_probabilities:
        disease_probabilities[disease] *= sum(symptom in symptom_to_diseases and disease in symptom_to_diseases[symptom] for symptom in symptoms)
    
    # Sort diseases by probability and select the top disease
    sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)
    max_confidence = sorted_diseases[0][1]
    
    # Apply a higher confidence threshold
    if max_confidence < 0.10:  # This is a higher threshold (can be adjusted)
        return "I'm not confident about any specific disease. Please consult a doctor."
    
    top_diseases = [disease for disease, prob in sorted_diseases if prob == max_confidence]
    return f"Predicted disease: {', '.join(top_diseases)}"
app = Flask(__name__)
app.secret_key = "your_secret_key" 
# Required for session management

# Store user credentials securely
USER_CREDENTIALS = {
    "user@example.com": generate_password_hash("123")
}

HISTORY_FILE = "chat_history.json"

# Function to save chat in a file
def save_chat_history(user, user_message, bot_response):
    try:
        with open(HISTORY_FILE, "r") as f:
            chat_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        chat_history = {}

    if user not in chat_history:
        chat_history[user] = []

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format the timestamp

    # Append message with timestamp
    chat_history[user].append({
        "user_message": user_message,
        "bot_response": bot_response,
        "timestamp": timestamp  # Add timestamp here
    })

    with open(HISTORY_FILE, "w") as f:
        json.dump(chat_history, f)

# Function to get chat history
def get_chat_history(user):
    try:
        with open(HISTORY_FILE, "r") as f:
            chat_history = json.load(f)
        return chat_history.get(user, [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

USER_CREDENTIALS_FILE = "user_credentials.json"
@app.route("/")
def home():
    if "messages" not in session:
        session["messages"] = []
    return render_template("index.html")

def save_user(email, password):
    hashed_password = generate_password_hash(password)
    with open(USER_CREDENTIALS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([email, hashed_password])

# Function to check if user exists
def user_exists(email):
    try:
        with open(USER_CREDENTIALS_FILE, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == email:
                    return row[1]  # Return hashed password
    except FileNotFoundError:
        return None
    return None

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if user_exists(email):
            return render_template("register.html", error="Email already registered!")

        save_user(email, password)
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        hashed_password = user_exists(email)
        if hashed_password and check_password_hash(hashed_password, password):
            session["user"] = email
            return redirect(url_for("chat"))

        return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")

@app.route("/chat")
def chat():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    if "user" not in session:
        return jsonify({"response": "Please log in to use MediQ."})

    user = session.get("user", "guest")
    user_message = request.json.get("message", "").strip().lower()


    

    # Use the predict_disease function from model.py
    response = predict_disease(user_message)

    # Save chat history
    save_chat_history(user, user_message, response)

    return jsonify({"response": response})

@app.route("/submit_issue", methods=["POST"])
def submit_issue():
    try:
        data = request.get_json()  # Get JSON data from frontend

        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        issue = data.get("issue", "").strip()

        if not name or not email or not issue:
            return jsonify({"success": False, "message": "All fields are required!"}), 400

        # Store data in CSV
        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([name, email, issue])

        return jsonify({"success": True, "message": "Issue submitted successfully!"})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
@app.route("/help", methods=["GET"])
def help_page():
    return render_template("help.html")















@app.route('/faqs')
def faqs():
    return render_template('faqs.html')

@app.route("/history")
def history():
    if "user" not in session:
        flash("Please log in to view history!", "error")
        return redirect(url_for("login"))

    user = session["user"]
    chat_history = get_chat_history(user)

    return render_template("history.html", chat_history=chat_history)

@app.route("/delete_message", methods=["POST"])
def delete_message():
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    user = session["user"]
    data = request.get_json()

    # Check if the 'index' is provided in the request
    if "index" not in data or not isinstance(data["index"], int):
        return jsonify({"success": False, "error": "Invalid index format"}), 400

    index = data.get("index")
    print(f"Attempting to delete message at index: {index}")

    try:
        with open(HISTORY_FILE, "r") as f:
            chat_history = json.load(f)
        print(f"Chat history loaded: {chat_history}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading chat history: {str(e)}")
        chat_history = {}

    # Check if the user exists in the chat history
    if user in chat_history:
        # Check if the index is valid for the given user
        if 0 <= index < len(chat_history[user]):
            print(f"Deleting message at index {index}")
            del chat_history[user][index]
            
            # Save the updated chat history back to the file
            with open(HISTORY_FILE, "w") as f:
                json.dump(chat_history, f)

            return jsonify({"success": True})  # Success if message was deleted

    print(f"Failed to delete message: Invalid index or user not found")
    return jsonify({"success": False, "error": "Invalid index or no messages found"}), 400



@app.route("/get_details", methods=["POST"])
def get_details():
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    user = session["user"]
    data = request.get_json()
    index = data.get("index")

    try:
        with open(HISTORY_FILE, "r") as f:
            chat_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        chat_history = {}

    if user in chat_history and 0 <= index < len(chat_history[user]):
        chat = chat_history[user][index]
        return jsonify({
            "success": True,
            "user_message": chat["user_message"],
            "bot_response": chat["bot_response"]
        })

    return jsonify({"success": False, "error": "Invalid index"}), 400

@app.route('/google-login')
def google_login():
    return redirect("https://accounts.google.com/signin")  # Temporary placeholder

@app.route('/logout')
def logout():
    session.clear()  # Clears user session
    return redirect(url_for('login'))  # Redirects to login page

from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords if not available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load DistilBERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load the FAQ dataset
faq_file = r"C:\Users\aniru\Downloads\demoFAQ's (1).csv"
faq_df = pd.read_csv(faq_file)
faq_df.columns = ["Question", "Answer"]

# Extract keywords efficiently
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

# Function to get FAQ response
def get_faq_response(user_query):
    user_keywords = extract_keywords(user_query)
   
    user_query_encoding = encode_text(user_query)
    print(user_keywords,user_keywords)
    # Step 1: Check for exact keyword matches
    for idx, faq_keywords in enumerate(faq_df["Keywords"]):
        if all(keyword in faq_keywords for keyword in user_keywords):
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
    


@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('question')
    response = get_faq_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True,port=4879)