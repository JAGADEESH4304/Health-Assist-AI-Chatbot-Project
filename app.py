#MediQ : AI-Driven Healthcare Assistant for Symptom Diagnosis and Disease Prediction




from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import json
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)  # Corrected Flask app initialization
app.secret_key = "your_secret_key"  # Required for session management

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

@app.route("/")
def home():
    if "messages" not in session:
        session["messages"] = []
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if email in USER_CREDENTIALS and check_password_hash(USER_CREDENTIALS[email], password):
            session["user"] = email  # Store logged-in user session
            return redirect(url_for("chat"))  # Redirect to chat after login
        else:
            return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if email in USER_CREDENTIALS:
            return render_template("register.html", error="Email already registered!")

        # Store hashed password
        USER_CREDENTIALS[email] = generate_password_hash(password)
        return redirect(url_for("login"))

    return render_template("register.html")

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

    chatbot_responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! How can I help?",
        "bye": "Goodbye! Stay safe and healthy.",
        "thank you": "You're welcome! Take care.",
        "headache": "If you have a headache, try resting and drinking plenty of water. If symptoms persist, consult a doctor.",
        "fever": "You may have a fever. Please rest, stay hydrated, and consult a doctor if symptoms persist.",
        "cold": "A common cold can be treated with rest, hydration, and over-the-counter medications.",
        "covid": "If you have COVID-19 symptoms, isolate yourself and seek medical help."
    }

    # Check for exact match or keyword match
    response = chatbot_responses.get(user_message, "I'm sorry, I didn't understand that. Please try again with more details.")
    for keyword, reply in chatbot_responses.items():
        if keyword in user_message:
            response = reply
            break

    # Save chat history
    save_chat_history(user, user_message, response)

    return jsonify({"response": response})

@app.route("/help", methods=["GET", "POST"])
def help_page():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        issue = request.form.get("issue")

        if not name or not email or not issue:
            flash("All fields are required!", "error")
            return redirect(url_for("help_page"))

        # Here, you can process/store the data (e.g., save to a database)
        print(f"Help request received: {name}, {email}, {issue}")

        flash("Your issue has been submitted. We will contact you soon!", "success")
        return redirect(url_for("help_page"))

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

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":  # Corrected conditional check
    app.run(debug=True, port=5056)
