import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NiVScan NER", page_icon="🦠", layout="wide")

# --- CUSTOM CSS (CLEAN THEME & GOOGLE FONTS) ---
st.markdown("""
<style>
    /* 1. Import Font Modern dari Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* 2. Aplikasikan Font HANYA ke elemen teks (menghindari kerusakan icon Streamlit) */
    html, body, p, h1, h2, h3, h4, h5, h6, label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Global White Theme */
    .stApp {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    /* Sidebar Styling - Elegant Deep Red */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #a10808 0%, #7d0606 100%) !important;
        border-right: none !important;
    }

    /* Menghilangkan bug teks "keyboard_double..." dan icon collapse yang berantakan */
    button[title="Collapse sidebar"], 
    [data-testid="stSidebarCollapseIcon"],
    section[data-testid="stSidebar"] span[data-testid="stMarkdownContainer"] p:empty {
        display: none !important;
    }

    /* Memastikan semua teks di sidebar berwarna PUTIH agar kontras */
    section[data-testid="stSidebar"] .stText, 
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        background-color: transparent !important;
        font-weight: 500;
    }

    /* Radio Button Styling - Clean White and Larger text without selection highlight */
    div[data-testid="stSidebar"] div[role="radiogroup"] label,
    div[data-testid="stSidebar"] div[role="radiogroup"] label p,
    div[data-testid="stSidebar"] div[role="radiogroup"] label span {
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 500 !important;
    }

    /* Force all elements inside the radio group to have transparent background/shadow (removes selection/hover highlights of all shapes/versions) */
    div[data-testid="stSidebar"] div[role="radiogroup"] *,
    div[data-testid="stSidebar"] div[role="radiogroup"] *::before,
    div[data-testid="stSidebar"] div[role="radiogroup"] *::after {
        background-color: transparent !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    /* Specifically target radio option wrappers and labels to remove borders */
    div[data-testid="stSidebar"] div[role="radiogroup"] > div,
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        border: none !important;
    }
    
    /* Menyesuaikan warna bullet radio button menjadi putih (untuk div/span baru dan lama) */
    div[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"]:checked + div,
    div[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"]:checked + span {
        background-color: #ffffff !important;
        border-color: #ffffff !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] + div,
    div[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] + span {
        border: 2px solid rgba(255, 255, 255, 0.6) !important;
    }

    /* Menghilangkan scrollbar di sidebar */
    section[data-testid="stSidebar"]::-webkit-scrollbar { display: none; }
    section[data-testid="stSidebar"] { -ms-overflow-style: none; scrollbar-width: none; }
    
    /* Menghilangkan blok warna abu-abu bawaan Streamlit pada menu pilihan */
    div[data-testid="stSidebar"] .stRadio > div {
        background-color: transparent !important;
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

    /* Section Title with Left Accent Bar */
    .section-title {
        border-left: 5px solid #c10a0a;
        padding-left: 12px;
        color: #1a1a1a !important;
        font-weight: 700;
    }

    /* Dataset Table Styling */
    .dataset-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        margin-bottom: 5px;
        font-size: 15px;
    }
    .dataset-table th {
        border-bottom: 2px solid #c10a0a;
        color: #c10a0a !important;
        font-weight: 600;
        padding: 10px 8px;
    }
    .dataset-table td {
        border-bottom: 1px solid #eeeeee;
        padding: 10px 8px;
        color: #333333;
    }
    .dataset-table tr:hover {
        background-color: #fff8f8;
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
        
        # Auto-correct/force "virus" to be part of the DISEASE entity if it follows one
        if word_clean == 'virus' and ('DISEASE' in prev_label) and ('DISEASE' not in label):
            label = 'I-DISEASE'
            
        # Limit DISEASE to only words containing 'nipah', 'niv', 'henipavirus', or "virus" following a DISEASE
        if 'DISEASE' in label:
            is_valid_disease = False
            if 'nipah' in word_clean or 'niv' in word_clean or 'henipavirus' in word_clean:
                is_valid_disease = True
            elif word_clean == 'virus' and ('DISEASE' in prev_label):
                is_valid_disease = True
                
            if not is_valid_disease or word_clean in blacklist:
                label = 'O'
                
        if label.startswith('I-'):
            ent = label[2:]
            if prev_label not in [f'B-{ent}', f'I-{ent}']: label = f'B-{ent}'
            
        if label == "B-DISEASE": disease_count += 1
        if label == "B-LOCATION": loc_count += 1
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
    st.markdown("""
    <div style='text-align: center; margin-bottom: 15px;'>
        <h1 style='font-size: 34px !important; font-weight: 700 !important; color: #ffffff !important; margin: 0; padding: 0;'>🦠 NiVScan</h1>
        <p style='font-size: 13px !important; font-weight: 500 !important; color: rgba(255, 255, 255, 0.8) !important; margin: 8px 0 0 0; padding: 0; letter-spacing: 0.3px;'>
            - Nipah Virus Entity Extractor -
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 1px solid rgba(255, 255, 255, 0.2); margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    selection = st.radio("Pilih Menu:", ["Deskripsi", "Demo Analisis"])
    
# --- MAIN CONTENT ---
if selection == "Deskripsi":
    st.markdown("<h1 style='color: #1a1a1a !important;'>Tentang Project NiVScan</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class='content-card'>
            <h3 style='margin-top: 0;'>Smart Nipah Virus NER</h3>
            <p>NiVScan (Nipah Virus Scan) adalah platform berbasis AI untuk ekstraksi informasi otomatis dari teks medis. 
            Sistem ini dikembangkan khusus untuk mengidentifikasi <b>Penyakit (Disease)</b> dan <b>Lokasi (Location)</b> 
            dalam narasi berita atau jurnal ilmiah terkait Virus Nipah.</p>
            <p>Dibangun menggunakan arsitektur <b>Transformer XLM-RoBERTa</b> untuk akurasi maksimal dalam konteks multibahasa.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='content-card' style='background-color: #fcfcfc;'>
            <h4 style='margin-top: 0;'>Dikembangkan Oleh:</h4>
            <p><b>Group 11</b><br>NLP Project - BINUS</p>
            <hr style='border-top: 1px solid #eaeaea;'>
            <p><small style='color: #666;'>Powered by Streamlit Cloud</small></p>
        </div>
        """, unsafe_allow_html=True)

    # Dataset Information Section
    st.markdown("""
    <div class='content-card'>
        <h3 class='section-title' style='margin-top: 0;'>Dataset</h3>
        <table class='dataset-table'>
            <thead>
                <tr>
                    <th style='text-align: left;'>Sumber</th>
                    <th>Bahasa</th>
                    <th>Jumlah Kalimat</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style='text-align: left;'>Dataset Manual Search</td>
                    <td>ID/EN</td>
                    <td>~300</td>
                </tr>
                <tr>
                    <td style='text-align: left;'>Google News Scraping</td>
                    <td>ID/EN</td>
                    <td>~302</td>
                </tr>
                <tr>
                    <td style='text-align: left;'>Kaggle Disease Symptom</td>
                    <td>EN</td>
                    <td>~105</td>
                </tr>
                <tr>
                    <td style='text-align: left;'>Wikipedia (ID+EN)</td>
                    <td>ID/EN</td>
                    <td>~252</td>
                </tr>
                <tr>
                    <td style='text-align: left;'>WHO + CDC</td>
                    <td>EN</td>
                    <td>~88</td>
                </tr>
                <tr>
                    <td style='text-align: left;'>PubMed NCBI</td>
                    <td>EN</td>
                    <td>~4292</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


elif selection == "Demo Analisis":
    st.markdown("<h1 style='color: #1a1a1a !important;'>Uji Coba Ekstraksi</h1>", unsafe_allow_html=True)
    
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

    # Logika tombol tetap mematuhi instruksimu: hasil keluar hanya setelah tombol diklik
    if st.button("Analisis Teks", use_container_width=True):
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