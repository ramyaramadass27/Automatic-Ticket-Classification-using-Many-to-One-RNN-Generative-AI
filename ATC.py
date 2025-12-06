import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle, os, re, requests, numpy as np
from time import sleep


import os
import gdown
import streamlit as st
from keras.models import load_model
import gdown
import os
import streamlit as st

# Google Drive file ID
GDRIVE_FILE_ID = "1gd9Tp-0J6_COyWppl-tiwlStYxTswQ4u"

# Local file name to save
MODEL_FILE = "bilstm.h5"  # fix typo: "bet_lstm.h5" → "best_lstm.h5"

# URL for gdown
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

def download_model(url, output_path):
    if not os.path.exists(output_path):
        with st.spinner("Downloading model (please wait)..."):
            gdown.download(url, output_path, quiet=False)
    return output_path

@st.cache_resource
def load_lstm_model():
    model_path = download_model(GDRIVE_URL, MODEL_FILE)
    model = load_model(model_path, compile=False)
    return model

# Load model
model = load_lstm_model()
from tensorflow.keras.models import load_model

model = load_model("bet_lstm.h5")

# ----------------------- Streamlit Config -----------------------
st.set_page_config(page_title="AI Support Chatbot", page_icon="🤖")

# ----------------------- Sidebar -----------------------

# Sidebar heading
import streamlit as st

# Big heading in sidebar using Markdown
import streamlit as st

# Big black heading in sidebar
st.sidebar.markdown("<h1 style='text-align: center; color: black;'>AI Support Chat Bot</h1>", unsafe_allow_html=True)

# or
# st.sidebar.title("Chat Bot")

st.sidebar.subheader("📊 Model Performance Summary")

# Your real metrics
TRAIN_ACCURACY = 0.9595      # from your training log
VAL_ACCURACY = 0.70     # best validation accuracy
TEST_ACCURACY = 0.70         # from classification report accuracy

st.sidebar.subheader("Training Accuracy")
st.sidebar.write(f"✔️ {TRAIN_ACCURACY * 100:.2f}%")

st.sidebar.subheader("Validation Accuracy")
st.sidebar.write(f"📌 {VAL_ACCURACY * 100:.2f}%")

st.sidebar.subheader("Test Accuracy")
st.sidebar.write(f"🎯 {TEST_ACCURACY * 100:.2f}%")

st.sidebar.markdown("---")

st.sidebar.subheader("🧠 Model Notes")
st.sidebar.write("""
- Model: BiLSTM (Many-to-One)
- Trained for 30 epochs
- Early stopping restored epoch 29 weights
- Confusion Matrix: 52x52 (multi-class)
""")

st.sidebar.markdown("---")
st.markdown("""
<style>

:root {
    --primary-color: #5A5A5A;
    --background-color: #E5E5E5;
    --secondary-background-color: #F2F2F2;
    --text-color: #000000;
}

/* Main background */
.main {
    background-color: #F2F2F2;
}

/* Chat input box */
.stTextInput>div>div>input {
    border-radius: 20px;
    border: 1px solid #5A5A5A;
    padding: 0.6em 1em;
    background-color: white;
}

/* Prevent input box from turning blue on click */
.stTextInput>div>div>input:focus {
    border: 1px solid #5A5A5A !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Buttons */
.stButton>button {
    background-color: #5A5A5A;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5em 1em;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #7A7A7A;
}

/* User bubble */
.chat-bubble {
    animation: fadeInUp 0.4s ease-in-out;
}

.user-bubble {
    background-color: #D3D3D3;
    color: black;
    padding: 10px;
    border-radius: 12px;
    text-align: right;
    margin: 5px 50px 5px 0;
}

</style>
""", unsafe_allow_html=True)



# ----------------------- Model & Assets -----------------------
from keras.models import load_model
import pickle

# ----------------------- Model & Assets -----------------------

# Load model from repo (not local path)
model = load_model("bet_lstm.h5", compile=False)

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


max_len = 300  # match your training setup

# ----------------------- Gemini API Setup -----------------------
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
# Load Gemini API key from Streamlit Secrets
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# ----------------------- Helper Functions -----------------------
def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s.,!?@#%&()-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def predict_queue(text):
    s = clean_text(text)
    seq = tokenizer.texts_to_sequences([s])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(pad, verbose=0)[0]
    idx = np.argmax(probs)
    queue = le.inverse_transform([idx])[0]
    return {"queue": queue, "confidence": float(probs[idx])}

def generate_reply_with_gemini(ticket_body, predicted_queue):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        return "(Gemini API key not set.)"

    prompt = f"""
You are a professional and empathetic customer support representative.
The customer's issue belongs to: {predicted_queue}.
Write a short, helpful customer support reply (2 short paragraphs + closing line).

Customer message: \"\"\"{ticket_body}\"\"\".
If message is in German, reply in German and include English translation below.
If message is in English, reply only in English.
"""

    payload = {"model": "gemini-2.0-flash", "contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"(Failed to get Gemini reply: {e})"

# ----------------------- UI -----------------------
st.title("🤖 AI Customer Support Chatbot")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

examples = [
    "I ordered a novel last week but received the wrong edition. Can I exchange it for the correct one?",
    "My software license key is not working after installation.",
    "My payment failed but the amount was deducted from my account.",
    "I cannot post new threads on the forum; it keeps showing a permission error.",
    "The lack of timely communication from the sales team has adversely impacted our client engagements.",
    "Can I get a refund for an accidental duplicate order?",
    "I bought a skincare cream last week, but it caused irritation. Can I return it and get a replacement?"

]

# Display chat history (user only — bot plain
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='background-color:#F2F2F2;
                        padding:10px;
                        margin:5px 50px 5px 0;
                        border-radius:12px;
                        text-align:right;'>
                <b>You:</b><br>{msg['content']}
            </div>
            """,
            unsafe_allow_html=True)
    elif msg["role"] == "bot":
        st.write(f"**Predicted Queue:** {msg['queue']}")
        st.write(f"**Confidence:** {msg['confidence']:.4f}")
        st.write(f"**Bot Reply:**\n\n{msg['reply']}")

st.markdown("---")

# Input area
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("Type your message here:", key="text_box", placeholder="Ask your question...")
with col2:
    send_button = st.button("➤")

# Suggested examples
if not user_input.strip():
    st.markdown("💡 **Suggested queries:**")
    for q in examples:
        if st.button(q):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.spinner("Bot is typing..."):
                pred = predict_queue(q)
                reply = generate_reply_with_gemini(q, pred["queue"])
            st.session_state.messages.append({"role": "bot", "queue": pred["queue"], "confidence": pred["confidence"], "reply": reply})
            st.rerun()

# When user sends a new message
if send_button and user_input.strip():
    text = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": text})

    with st.spinner("Bot is typing..."):
        sleep(0.5)
        pred = predict_queue(text)
        reply = generate_reply_with_gemini(text, pred["queue"])

    st.session_state.messages.append({
        "role": "bot",
        "queue": pred["queue"],
        "confidence": pred["confidence"],
        "reply": reply
    })

    st.rerun()

# Footer
st.markdown("---")
st.caption("Built with 🤖 and empathy | © 2025 AI Customer Support")


