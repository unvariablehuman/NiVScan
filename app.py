import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NiVScan NER", page_icon="🦠", layout="wide")

# --- CUSTOM CSS (FORCE LIGHT MODE & SIDEBAR) ---
st.markdown("""
<style>
    /* Global White Theme */
    .stApp {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    /* Sidebar Styling - Gradient Merah Putih */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #c10a0a 0%, #ffffff 100%) !important;
    }
    
    /* Sidebar Header Text */
    section[data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        background-color: transparent !important;
    }

    /* Sidebar Divider */
    section[data-testid="stSidebar"] hr {
        border-top: 1px solid rgba(255,255,255,0.3) !important;
    }

    /* Force Sidebar fonts to White & Remove Background Highlights */
    section[data-testid="stSidebar"] .stText, 
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        background-color: transparent !important;
        font-weight: 500;
        text-shadow: 0px 0px 2px rgba(0,0,0,0.2); /* Sedikit bayangan agar terbaca di area putih */
    }

    /* Target specific Streamlit radio selection highlight to remove it */
    div[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: transparent !important;
    }

    /* Menyesuaikan warna bullet radio button menjadi putih agar serasi */
    input[type="radio"]:checked + div {
        background-color: #ffffff !important;
        border-color: #ffffff !important;
    }
    input[type="radio"] + div {
        border-color: #ffffff !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, p, span, label, div {
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Content Card */
    .content-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }

    /* Extract Button Styling */
    div.stButton > button:first-child {
        background-color: #c10a0a !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    
    /* Text Area Input Styling */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
    }

    /* Entity Highlight Styling */
    .entity-box {
        line-height: 2.5; 
        font-size: 17px; 
        color: #1a1a1a;
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #eeeeee;
    }
    .tag-disease {
        background: #ffebee;
        color: #c10a0a;
        border: 1px solid #ffcdd2;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    .tag-loc {
        background: #e0f2f1;
        color: #00796b;
        border: 1px solid #b2dfdb;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    .tag-sub {
        font-size: 0.65em;
        text-transform: uppercase;
        opacity: 0.7;
        margin-left: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---
@st.cache_resource
def load_model():
    model_path = "Percobaan/Baru Banget/Model B/model_b_best"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def predict_and_patch(text, tokenizer, model):
    words = text.split()
    if not words: return [], 0, 0
    inputs = tokenizer(words, return_tensors="pt", truncation=True, max_length=128, is_split_into_words=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    word_ids = inputs.word_ids(batch_index=0)
    raw_labels = []
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None: continue
        if word_idx != previous_word_idx:
            raw_labels.append(model.config.id2label[predictions[idx]])
        previous_word_idx = word_idx
    final_labels = []
    prev_label = 'O'
    disease_count, loc_count = 0, 0
    blacklist = {'fever', 'headache', 'myalgia', 'vomiting', 'cough', 'chills', 'fatigue', 'deaths', 'severe', 'brain', 'swelling', 'high', 'fatality', 'is', 'was', 'has'}
    for i, label in enumerate(raw_labels):
        word_clean = words[i].lower().strip('.,;:()[]"\'!?')
        if word_clean in blacklist and 'DISEASE' in label: label = 'O'
        if label.startswith('I-'):
            ent = label[2:]
            if prev_label not in [f'B-{ent}', f'I-{ent}']: label = f'B-{ent}'
        if "DISEASE" in label: disease_count += 1
        if "LOCATION" in label: loc_count += 1
        final_labels.append(label)
        prev_label = label
    return list(zip(words, final_labels)), disease_count, loc_count

def render_entities(entities):
    html_text = ""
    for word, label in entities:
        if "DISEASE" in label:
            html_text += f"<span class='tag-disease'>{word} <span class='tag-sub'>DISEASE</span></span> "
        elif "LOCATION" in label:
            html_text += f"<span class='tag-loc'>{word} <span class='tag-sub'>LOC</span></span> "
        else:
            html_text += f"{word} "
    return html_text

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🦠 NiVScan</h2>", unsafe_allow_html=True)
    st.markdown("---")
    selection = st.radio("Pilih Menu:", ["Deskripsi", "Demo Analisis"])
    st.markdown("---")
    st.markdown("**Status Model:** ✅ Ready")
    st.markdown("**Version:** 2.1 (Model B)")

# --- MAIN CONTENT ---
if selection == "Deskripsi":
    st.markdown("<h1 style='color: #c10a0a !important;'>Tentang Project NiVScan</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class='content-card'>
            <h3>Smart Nipah Virus NER</h3>
            <p>NiVScan (Nipah Virus Scan) adalah platform berbasis AI untuk ekstraksi informasi otomatis dari teks medis. 
            Sistem ini dikembangkan khusus untuk mengidentifikasi <b>Penyakit (Disease)</b> dan <b>Lokasi (Location)</b> 
            dalam narasi berita atau jurnal ilmiah terkait Virus Nipah.</p>
            <p>Dibangun menggunakan arsitektur <b>Transformer XLM-RoBERTa</b> untuk akurasi maksimal dalam konteks multibahasa.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='content-card' style='background-color: #fdf2f2;'>
            <h4>Dikembangkan Oleh:</h4>
            <p><b>Group 11</b><br>NLP Project - BINUS</p>
            <hr>
            <p><small>Powered by Streamlit Cloud</small></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Statistik Project")
    st.image("https://raw.githubusercontent.com/unvariablehuman/NiVScan/main/Percobaan/Baru%20Banget/Model%20A/eda_label_distribution.png", caption="Distribusi Label dalam Dataset Pelatihan")

elif selection == "Demo Analisis":
    st.markdown("<h1 style='color: #c10a0a !important;'>Uji Coba Ekstraksi</h1>", unsafe_allow_html=True)
    
    with st.spinner("Memuat Model XLM-RoBERTa..."):
        tokenizer, model = load_model()

    samples = {
        "Custom": "",
        "Scientific Case": "Using NiV as an important paramyxoviral model, we identified two novel regions in F that modulate the membrane fusion cascade.",
        "Kerala Outbreak": "The Nipah virus outbreak in Kerala has caused several deaths. Local authorities in Kozhikode are monitoring people who had contact with infected patients.",
        "Malaysia History": "Nipah virus was first identified in 1999 during an outbreak among pig farmers in Malaysia."
    }

    selected_sample = st.selectbox("Pilih Contoh Kalimat:", list(samples.keys()))
    default_input = samples[selected_sample] if selected_sample != "Custom" else ""

    user_input = st.text_area("Masukkan Kalimat Medis:", value=default_input, height=150)

    if st.button("Analisis Teks"):
        if user_input.strip():
            entities, disease_count, loc_count = predict_and_patch(user_input, tokenizer, model)
            
            # Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1: st.metric("Total Kata", len(user_input.split()))
            with m_col2: st.metric("Disease", disease_count)
            with m_col3: st.metric("Location", loc_count)

            st.markdown("---")
            st.markdown("### Hasil Ekstraksi")
            html_output = render_entities(entities)
            st.markdown(f"<div class='entity-box'>{html_output}</div>", unsafe_allow_html=True)
            
            with st.expander("Lihat Detail Token"):
                filtered = [(w, l) for w, l in entities if l != 'O']
                if filtered: st.table(filtered)
                else: st.info("Tidak ada entitas yang terdeteksi.")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")
