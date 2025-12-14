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
    try:
        with open('tokenizer.json', 'r') as f:
            data = f.read()
            tokenizer = tokenizer_from_json(data)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")

    try:
        model = tf.keras.models.load_model('spamnet_hybrid_attention.keras')
    except Exception as e:
        print(f"Error loading model: {e}")

    return tokenizer, model

tokenizer, model = load_resources()

# Perubahan di sini: parameter kedua sekarang menerima 'features_list'
def predict_email(text, features_list):
    if tokenizer is None or model is None:
        return "Error System", 0.0

    # 1. Preprocessing Teks
    sequences = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(sequences, maxlen=MAXLEN, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    # 2. Preprocessing Fitur Tambahan
    # Jika input berupa list (dari UI baru), ubah jadi numpy array
    if isinstance(features_list, list):
        # Pastikan panjangnya 7 (sesuai input model kamu)
        # Jika user input kurang dari 7, kita pad dengan 0
        while len(features_list) < 7:
            features_list.append(0.0)
        # Jika lebih, potong
        features_list = features_list[:7]
        
        domain_features = np.array([features_list]) # Bentuk jadi (1, 7)
    else:
        # Fallback jika input lama
        domain_features = np.zeros((1, 7))

    # 3. Prediksi
    try:
        prediction = model.predict([padded_text, domain_features], verbose=0)
        prob = prediction[0][0]
        label = "SPAM 🚨" if prob > 0.5 else "HAM (AMAN) ✅"
        return label, prob
    except Exception as e:
        return f"Error: {str(e)}", 0.0
