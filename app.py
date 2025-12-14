import streamlit as st
import time
from predict_pipeline import predict_email

# --- 1. Konfigurasi Halaman (Harus di paling atas) ---
st.set_page_config(
    page_title="Spam Detective AI",
    page_icon="🕵️‍♂️",
    layout="wide", # Menggunakan lebar penuh layar
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS untuk Tampilan Lebih Keren ---
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #ff4b4b; color: white; }
    .stTextArea>div>div>textarea { background-color: #f0f2f6; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# --- 3. Header & Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620576.png", width=100)
    st.title("Tentang Aplikasi")
    st.info(
        """
        Aplikasi ini menggunakan **Hybrid LSTM + Attention Mechanism** untuk mendeteksi email spam berdasarkan konten teks dan metadata.
        """
    )
    st.write("---")
    st.write("**Panduan:**")
    st.caption("1. Masukkan Subjek & Isi Email.")
    st.caption("2. Lengkapi data domain & fitur lainnya.")
    st.caption("3. Klik tombol Prediksi.")

st.title("🕵️‍♂️ Spam Email Classifier")
st.markdown("### Analisis Keamanan Email Cerdas")
st.divider()

# --- 4. Layout Input (Menggunakan Kolom) ---
# Kita bagi layar jadi 2 kolom: Kiri (Teks) & Kanan (Metadata)
col_text, col_meta = st.columns([2, 1], gap="large")

with col_text:
    st.subheader("📝 Konten Email")
    email_subject = st.text_input("Subjek Email", placeholder="Contoh: Congratulations! You won a lottery...")
    email_body = st.text_area("Isi Body Email", height=300, placeholder="Paste seluruh isi email di sini...")

with col_meta:
    st.subheader("⚙️ Metadata Email")
    
    # Input Domain
    from_domain = st.text_input("Domain Pengirim", placeholder="contoh: classroom.google.com")
    
    # Input Angka (Jumlah Link)
    num_links = st.number_input("Jumlah Link dalam Email", min_value=0, value=0, help="Hitung berapa banyak link http/https yang ada.")
    
    # Input Pilihan (Ada Lampiran?)
    has_attachment = st.radio("Ada Lampiran (Attachment)?", ["Tidak (0)", "Ya (1)"], horizontal=True)
    
    # Konversi input radio ke angka
    attachment_val = 1 if "Ya" in has_attachment else 0
    
    st.write("---")
    predict_btn = st.button("🔍 Analisis Sekarang")

# --- 5. Logika Prediksi ---
if predict_btn:
    if not email_body:
        st.warning("⚠️ Mohon isi bagian 'Body Email' terlebih dahulu.")
    else:
        # Tampilkan animasi loading biar keren
        with st.spinner('🤖 Sedang memindai pola bahasa & metadata...'):
            time.sleep(1) # Efek dramatis (opsional)
            
            # --- PENTING: Menggabungkan Subject + Body (Opsional) ---
            # Agar model mendapat konteks lebih banyak, kita bisa gabung subject + body
            full_text = f"{email_subject} {email_body}"
            
            # --- PENTING: Menyiapkan Data Numerik ---
            # Model kamu butuh vector input kedua. Kita masukkan data dari UI ke sini.
            # Urutan list ini harus sesuai dengan urutan fitur saat training dulu.
            # Di sini saya masukkan link & attachment, sisanya 0 (karena kita tidak tahu 5 fitur lainnya).
            custom_features = [num_links, attachment_val, 0, 0, 0, 0, 0] 

            # Panggil fungsi prediksi
            label, prob = predict_email(full_text, custom_features)

        # --- 6. Menampilkan Hasil ---
        st.divider()
        st.subheader("📊 Hasil Analisis")

        # Layout hasil menggunakan kolom metric
        res_col1, res_col2, res_col3 = st.columns(3)

        is_spam = "SPAM" in label.upper()
        
        with res_col1:
            color = "red" if is_spam else "green"
            st.markdown(f"<h2 style='color: {color};'>{label}</h2>", unsafe_allow_html=True)
        
        with res_col2:
            st.metric("Tingkat Keyakinan", f"{prob*100:.2f}%")
        
        with res_col3:
            risk_level = "Sangat Tinggi" if prob > 0.8 else ("Sedang" if prob > 0.5 else "Rendah")
            st.metric("Level Risiko", risk_level)

        # Progress bar visual
        st.write("Visualisasi Probabilitas Spam:")
        st.progress(float(prob))

        if is_spam:
            st.error("🚨 **Peringatan:** Email ini mengandung indikasi kuat sebagai SPAM. Jangan klik link apapun!")
        else:
            st.success("✅ **Aman:** Email ini terlihat normal dan valid.")
            
        # Tampilkan JSON raw data (opsional, untuk debugging user)
        with st.expander("Lihat Detail Teknis"):
            st.json({
                "input_length": len(full_text),
                "domain": from_domain,
                "num_links": num_links,
                "has_attachment": bool(attachment_val),
                "raw_probability": float(prob)
            })
