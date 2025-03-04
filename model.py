import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import re
from rapidfuzz import process
import time
import os

# Load the dataset
file_path = r"C:\Users\aniru\dataset.csv"
data = pd.read_csv(file_path)

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
def extract_symptoms(user_input):
    words = re.findall(r"\b\w+\b", user_input.lower())
    detected_symptoms = []
    for word in words:
        match = process.extractOne(word, known_symptoms, score_cutoff=80)
        if match:
            detected_symptoms.append(match[0])
    return detected_symptoms

# Prediction function
def predict_disease(user_input):
    symptoms = extract_symptoms(user_input)
    if not symptoms:
        return "Sorry, I couldn't understand ☹️. Please enter valid symptoms."
    
    symptoms_vectorized = vectorizer.transform([' '.join(symptoms)])
    probabilities = model.predict_proba(symptoms_vectorized)[0]
    
    disease_probabilities = {model.classes_[i]: probabilities[i] for i in range(len(model.classes_))}
    
    for disease in disease_probabilities:
        disease_probabilities[disease] *= sum(symptom in symptom_to_diseases and disease in symptom_to_diseases[symptom] for symptom in symptoms)
    
    sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)
    max_confidence = sorted_diseases[0][1]
    
    if max_confidence < 0.05:
        return "I'm not confident about any specific disease. Please consult a doctor."
    
    top_diseases = [disease for disease, prob in sorted_diseases if prob == max_confidence]
    return f"Predicted disease : {', '.join(top_diseases)}"

# Clear screen function
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Chatbot interaction
def chatbot():
    while True:
        clear_screen()
        print("Chatbot : Hey👋, I'm MediQ. How can I assist you today?")
        inappropriate_count = 0
        
        while True:
            user_input = input("User : ").strip()
            
            if not user_input:
                print("Chatbot : Please enter some symptoms to proceed.")
                continue
            
            # Check for inappropriate messages
            if any(word in user_input.lower() for word in inappropriate_words):
                inappropriate_count += 1
                print("Chatbot : Sorry, but that's an inappropriate message 😑.")
                if inappropriate_count >= 3:
                    print("Chatbot : You've sent too many inappropriate messages. Chat ended. Restarting in 5 seconds... ⭕.")
                    time.sleep(5)
                    clear_screen()
                    break
                continue
            
            # Handle restart
            if user_input.lower() == "restart":
                print("Chatbot : Restarting chat...")
                time.sleep(2)
                clear_screen()
                break
            
            # Handle greetings
            if user_input.lower() in {"hi", "hello", "hey", "how are you", "what's up"}:
                print("Chatbot : Hello👋! How can I help you today?")
                continue
            
            # Handle exit messages
            if user_input.lower() in {"bye", "exit", "quit", "goodbye"}:
                print("Chatbot : Take care 😊! If you feel unwell, don’t hesitate to seek medical advice. Goodbye!")
                return
            
            # Predict disease
            response = predict_disease(user_input)
            print(f"Chatbot : {response}")

# Run chatbot
chatbot()