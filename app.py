import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NiVScan NER", page_icon="🦠", layout="centered")

# --- CUSTOM CSS INJECTION ---
st.markdown("""
<style>
    /* Typography & Headers */
    h1, h2, h3 {
        color: #f0f0f0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Extract Button Styling */
    div.stButton > button:first-child {
        background-color: #c10a0a;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #e61919;
        box-shadow: 0 4px 15px rgba(193, 10, 10, 0.4);
        transform: translateY(-2px);
    }
    
    /* Text Area Input Styling */
    .stTextArea textarea {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333 !important;
        border-radius: 8px;
        padding: 15px;
        font-size: 15px;
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #c10a0a !important;
        box-shadow: 0 0 0 1px #c10a0a !important;
    }
    /* Entity Highlight Styling */
    .entity-box {
        line-height: 2.8; 
        font-size: 18px; 
        color: #d1d1d1;
        background-color: #141414;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2a2a2a;
    }
    .tag-disease {
        background: rgba(193, 10, 10, 0.15);
        color: #ff6b6b;
        border: 1px solid #c10a0a;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 0 3px;
        font-weight: 600;
        display: inline-block;
    }
    .tag-loc {
        background: rgba(10, 193, 166, 0.15);
        color: #2ee6c4;
        border: 1px solid #0ac1a6;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 0 3px;
        font-weight: 600;
        display: inline-block;
    }
    .tag-sub {
        font-size: 0.65em;
        text-transform: uppercase;
        opacity: 0.8;
        margin-left: 5px;
        letter-spacing: 0.5px;
    }
    
    /* Footer Styling */
    .footer-text {
        text-align: center;
        font-size: 12px;
        color: #666;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHE FUNCTION ---
@st.cache_resource
def load_model():
    # Folder path relative to repository structure
    model_path = "Percobaan/Baru Banget/Model B/model_b_best"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# --- PREDICTION & POST-PROCESSING FUNCTION ---
def predict_and_patch(text, tokenizer, model):
    words = text.split()
    if not words:
        return [], 0, 0

    inputs = tokenizer(words, return_tensors="pt", truncation=True, max_length=128, is_split_into_words=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    word_ids = inputs.word_ids(batch_index=0)
    
    raw_labels = []
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            raw_labels.append(model.config.id2label[predictions[idx]])
        previous_word_idx = word_idx
        
    final_labels = []
    prev_label = 'O'
    disease_count = 0
    loc_count = 0
    
    # Blacklist to prevent False Positives (Symptoms & Stopwords)
    blacklist = {
        'fever', 'headache', 'myalgia', 'vomiting', 'cough', 'chills', 'fatigue', 
        'nausea', 'diarrhea', 'dizziness', 'weakness', 'pain', 'ache', 'signs', 
        'symptoms', 'illness', 'and', 'the', 'in', 'of', 'to', 'a', 'an', 'or'
    }
    
    for i, label in enumerate(raw_labels):
        word_clean = words[i].lower().strip('.,;:()[]"\'!?')
        
        # Apply blacklist
        if word_clean in blacklist and 'DISEASE' in label:
            label = 'O'
            
        # Fix IOB2 transitions (I- without B-)
        if label.startswith('I-'):
            ent = label[2:]
            if prev_label not in [f'B-{ent}', f'I-{ent}']:
                label = f'B-{ent}'

        if "DISEASE" in label: disease_count += 1
        if "LOCATION" in label: loc_count += 1
                
        final_labels.append(label)
        prev_label = label
        
    return list(zip(words, final_labels)), disease_count, loc_count

# --- VISUALIZATION FUNCTION ---
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

# --- STREAMLIT UI ---
# Header Section
st.markdown("<h1 style='text-align: center; color: #c10a0a !important;'>🦠 NiVScan NER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0a0a0; margin-bottom: 30px;'>Nipah Virus Entity Extractor powered by XLM-RoBERTa</p>", unsafe_allow_html=True)

# Load Model
with st.spinner("Loading XLM-RoBERTa Model..."):
    tokenizer, model = load_model()

# Input Section
samples = {
    "Custom": "",
    "Kerala Outbreak": "The Nipah virus outbreak in Kerala has caused several deaths. Local authorities in Kozhikode are monitoring people who had contact with infected patients showing symptoms of high fever.",
    "Malaysia History": "Nipah virus was first identified in 1999 during an outbreak among pig farmers in Malaysia. The disease causes severe brain swelling and has a high fatality rate in Southeast Asia.",
    "Bangladesh Case": "Symptoms of Nipah virus include fever, headache, and cough. It can be transmitted from bats to humans or through contaminated food like raw date palm sap often found in Bangladesh."
}

selected_sample = st.selectbox("Quick Sample Select:", list(samples.keys()))
default_input = samples[selected_sample] if selected_sample != "Custom" else "Nipah virus (NiV) is a zoonotic pathogen causing outbreaks in Kerala, India, and Bangladesh with high mortality rates."

user_input = st.text_area("Enter Text / Medical News Sentence:", value=default_input, height=150)

# Analysis Logic
if st.button("Analyze Text"):
    if user_input.strip():
        entities, disease_count, loc_count = predict_and_patch(user_input, tokenizer, model)
        
        # --- DEEP ANALYSIS SECTION ---
        st.markdown("### 📊 Deep Analysis")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Total Tokens", len(user_input.split()))
        with m_col2:
            st.metric("Disease Entities", disease_count)
        with m_col3:
            st.metric("Location Entities", loc_count)

        st.markdown("---")
        st.markdown("### 🏷️ Extraction Results")
        html_output = render_entities(entities)
        
        # Wrap result in an elegant box
        st.markdown(f"<div class='entity-box'>{html_output}</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detail Table
        with st.expander("View Token Details", expanded=False):
            filtered_entities = [(w, l) for w, l in entities if l != 'O']
            if filtered_entities:
                st.table(filtered_entities)
            else:
                st.info("No DISEASE or LOCATION entities detected.")
    else:
        st.warning("Please enter some text first.")

# Footer
st.markdown("<div class='footer-text'>Developed by Group 11 - NiVScan Project<br>Powered by HuggingFace & Streamlit</div>", unsafe_allow_html=True)
