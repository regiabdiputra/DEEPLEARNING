import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- KONFIGURASI ---
MAXLEN = 100
PADDING_TYPE = 'post'
TRUNCATING_TYPE = 'post'

def load_resources():
    tokenizer = None
    model = None

    # 1. Load Tokenizer (Fix: Menggunakan f.read())
    try:
        with open('tokenizer.json', 'r') as f:
            data = f.read()
            tokenizer = tokenizer_from_json(data)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")

    # 2. Load Model
    try:
        model = tf.keras.models.load_model('spamnet_hybrid_attention.keras')
    except Exception as e:
        print(f"Error loading model: {e}")

    return tokenizer, model

tokenizer, model = load_resources()

def predict_email(text, domain_input):
    if tokenizer is None or model is None:
        return "Error: Resources not loaded", 0.0

    # --- A. Preprocessing Teks ---
    sequences = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(sequences, maxlen=MAXLEN, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    # --- B. Preprocessing Domain ---
    domain_features = np.zeros((1, 7)) 

    # --- C. Prediksi ---
    try:
        prediction = model.predict([padded_text, domain_features], verbose=0)
        prob = prediction[0][0]
        label = "SPAM" if prob > 0.5 else "HAM (AMAN)"
        return label, prob
    except Exception as e:
        return f"Error: {str(e)}", 0.0
