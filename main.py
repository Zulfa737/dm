import streamlit as st
import joblib
import re

# Konfigurasi halaman harus menjadi perintah Streamlit pertama
st.set_page_config(
    page_title="Analisis Sentimen Film",
    page_icon="🎬",
    layout="centered"
)

# --- FUNGSI UTAMA ---

@st.cache_resource
def load_model_objects():
    """
    Memuat model dan tools preprocessing dari file pickle.
    Pastikan file .pkl berada di direktori yang sama.
    """
    try:
        model_bnb = joblib.load("model_bernoulli_nb.pkl")
        model_svm = joblib.load("model_linear_svm.pkl")
        model_ensemble = joblib.load("model_ensemble_voting.pkl")
        vectorizer = joblib.load("vectorizer_tfidf.pkl")
        tools = joblib.load("preprocessing_tools.pkl")
        return model_bnb, model_svm, model_ensemble, vectorizer, tools
    except FileNotFoundError as e:
        st.error(f"⚠️ Error: File model tidak ditemukan ({e}). Pastikan file .pkl sudah diupload.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error saat memuat model: {e}")
        return None, None, None, None, None

def preprocess_text(text, stopword_remover, stemmer):
    """
    Membersihkan teks input: lowercase, hapus karakter non-huruf, stopword removal, stemming.
    """
    # Lowercase & Hapus karakter selain huruf
    text = re.sub('[^A-Za-z]+', ' ', text).lower().strip()
    # Hapus spasi berlebih
    text = re.sub('\s+', ' ', text)
    
    # Stopword Removal & Stemming (jika tools tersedia)
    if stopword_remover:
        text = stopword_remover.remove(text)
    if stemmer:
        text = stemmer.stem(text)
        
    return text

def get_confidence_badge(prob):
    """
    Memberikan label visual berdasarkan tingkat kepercayaan prediksi.
    """
    if prob > 80:
        return "🟢 Tinggi", "success"
    elif prob > 60:
        return "🟡 Sedang", "warning"
    else:
        return "🔴 Rendah", "error"

# --- LOAD RESOURCES ---
model_bnb, model_svm, model_ensemble, vectorizer, tools = load_model_objects()

# --- USER INTERFACE (UI) ---

st.title("🎬 Analisis Sentimen Film")
st.markdown("### Ensemble Model (BernoulliNB + SVM)")
st.markdown("Aplikasi ini menganalisis ulasan film dan memprediksi apakah sentimennya **Positif** atau **Negatif**.")

# Cek apakah semua model berhasil dimuat
models_loaded = all([model_bnb, model_svm, model_ensemble, vectorizer, tools])

if not models_loaded:
    st.warning("⚠️ Aplikasi tidak dapat berjalan karena file model belum lengkap.")
else:
    st.divider()
    st.subheader("✍️ Masukkan Ulasan Film")

    # Pilihan contoh input untuk memudahkan user mencoba
    example_texts = [
        "Filmnya bagus banget, alurnya tidak ketebak!",
        "Film jelek, buang waktu saja nonton ini.",
        "Keren, aktingnya mantap sekali dan visualnya memukau.",
        "Goblok banget filmnya tidak bermutu sama sekali.",
        "Biasa aja sih, tidak terlalu bagus tapi lumayan menghibur.",
        "Luar biasa, sangat recommended untuk ditonton bersama keluarga!"
    ]

    selected_example = st.selectbox(
        "Pilih contoh ulasan (opsional):",
        ["-- Ketik manual --"] + example_texts
    )

    # Set nilai awal text area berdasarkan pilihan user
    default_text = "" if selected_example == "-- Ketik manual --" else selected_example

    input_text = st.text_area(
        "Tulis ulasan Anda di sini:", 
        value=default_text, 
        height=150,
        placeholder="Contoh: Film ini sangat menyentuh hati..."
    )

    # Tombol & Opsi
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        predict_btn = st.button("🔍 Analisis", type="primary", use_container_width=True)
    with col2:
        show_comparison = st.checkbox("Bandingkan model individu", value=True)
    with col3:
        show_details = st.checkbox("Lihat proses preprocessing", value=False)

    # --- LOGIKA PREDIKSI ---
    if predict_btn:
        if input_text.strip() == "":
            st.warning("⚠️ Harap masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner('Sedang menganalisis sentimen...'):
                try:
                    # 1. Preprocessing
                    stopword_remover = tools['stopword']
                    stemmer = tools['stemmer']
                    processed_text = preprocess_text(input_text, stopword_remover, stemmer)
                    
                    # 2. Vectorization
                    vec = vectorizer.transform([processed_text])

                    # 3. Prediksi (Ensemble)
                    pred_ensemble = model_ensemble.predict(vec)[0]
                    prob_ensemble = model_ensemble.predict_proba(vec)[0]

                    # --- TAMPILAN HASIL ---
                    st.divider()
                    st.subheader("🎯 Hasil Analisis (Ensemble)")

                    # Tentukan probabilitas kelas terpilih
                    # Asumsi: index 0 = negatif, index 1 = positif (sesuaikan dengan model Anda)
                    # Kita ambil probabilitas tertinggi sebagai confidence score
                    max_prob = max(prob_ensemble) * 100
                    conf_text, conf_type = get_confidence_badge(max_prob)

                    # Tampilkan Alert Hasil Utama
                    if pred_ensemble == "positive": # Sesuaikan label dengan output model Anda (misal: 1 atau 'positive')
                        st.success(f"### ✅ Sentimen: POSITIF")
                    else:
                        st.error(f"### ❌ Sentimen: NEGATIF")

                    # Tampilkan Detail Probabilitas
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.info(f"**Tingkat Keyakinan:**\n\n{conf_text} ({max_prob:.1f}%)")
                    
                    with col_res2:
                        st.write("**📊 Probabilitas Detail:**")
                        # Asumsi prob_ensemble[0] = Negatif, [1] = Positif
                        st.progress(prob_ensemble[1], text=f"Positif: {prob_ensemble[1]*100:.1f}%")
                        st.progress(prob_ensemble[0], text=f"Negatif: {prob_ensemble[0]*100:.1f}%")

                    # --- OPSI: BANDINGKAN MODEL ---
                    if show_comparison:
                        st.markdown("---")
                        st.markdown("#### 🤖 Perbandingan Model Individu")
                        
                        # Prediksi model individu
                        pred_bnb = model_bnb.predict(vec)[0]
                        pred_svm = model_svm.predict(vec)[0]
                        
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            st.caption("Bernoulli Naive Bayes")
                            if pred_bnb == "positive":
                                st.success("Positif")
                            else:
                                st.error("Negatif")
                        
                        with comp_col2:
                            st.caption("Linear SVM")
                            if pred_svm == "positive":
                                st.success("Positif")
                            else:
                                st.error("Negatif")

                    # --- OPSI: LIHAT PREPROCESSING ---
                    if show_details:
                        st.markdown("---")
                        st.markdown("#### 🔬 Detail Preprocessing")
                        st.text_area("Teks Asli:", value=input_text, height=70, disabled=True)
                        st.text_area("Hasil Preprocessing (Stemmed & Cleaned):", value=processed_text, height=70, disabled=True)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat proses prediksi: {e}")
