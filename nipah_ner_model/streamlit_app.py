import streamlit as st
import spacy
from spacy.pipeline import EntityRuler
from collections import defaultdict
import re
import unicodedata

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NiVScan — Nipah NER Detector",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    .stApp {
        background-color: #0a0e1a;
    }

    /* Header */
    .nivscan-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #0a0e1a 100%);
        border-bottom: 1px solid #1e3a5f;
        padding: 2rem 0 1.5rem 0;
        margin-bottom: 2rem;
    }

    .nivscan-logo {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #38bdf8;
        letter-spacing: -1px;
    }

    .nivscan-logo span {
        color: #ef4444;
    }

    .nivscan-tagline {
        font-size: 0.85rem;
        color: #64748b;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 0.2rem;
        letter-spacing: 0.05em;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
        margin: 2px 3px;
    }

    .badge-disease {
        background: #3f1212;
        color: #fca5a5;
        border: 1px solid #7f1d1d;
    }

    .badge-symptom {
        background: #3f2800;
        color: #fcd34d;
        border: 1px solid #78350f;
    }

    .badge-location {
        background: #0c2340;
        color: #7dd3fc;
        border: 1px solid #1e3a5f;
    }

    /* Highlighted text */
    .hl-disease {
        background: #7f1d1d;
        color: #fecaca;
        padding: 1px 4px;
        border-radius: 3px;
        font-weight: 500;
    }

    .hl-symptom {
        background: #78350f;
        color: #fde68a;
        padding: 1px 4px;
        border-radius: 3px;
        font-weight: 500;
    }

    .hl-location {
        background: #1e3a5f;
        color: #bae6fd;
        padding: 1px 4px;
        border-radius: 3px;
        font-weight: 500;
    }

    /* Result card */
    .result-card {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .result-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #38bdf8;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    /* Entity row */
    .entity-row {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 6px;
        background: #0f2235;
        margin-bottom: 8px;
        border: 1px solid #1a3050;
    }

    .entity-text {
        font-weight: 500;
        color: #e2e8f0;
        font-size: 0.95rem;
    }

    /* Metric card */
    .metric-card {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .metric-val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
    }

    .metric-val-disease { color: #ef4444; }
    .metric-val-symptom { color: #f59e0b; }
    .metric-val-location { color: #38bdf8; }

    .metric-lbl {
        font-size: 0.75rem;
        color: #64748b;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1b2a;
        border-bottom: 1px solid #1e3a5f;
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        padding: 0.6rem 1.2rem;
    }

    .stTabs [aria-selected="true"] {
        color: #38bdf8;
        border-bottom: 2px solid #38bdf8;
    }

    /* Textarea */
    .stTextArea textarea {
        background: #0d1b2a !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 6px !important;
        color: #e2e8f0 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.95rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    /* Button */
    .stButton button {
        background: #0369a1 !important;
        color: #e0f2fe !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        padding: 0.5rem 1.5rem !important;
        transition: background 0.15s !important;
    }

    .stButton button:hover {
        background: #0284c7 !important;
    }

    /* Divider */
    hr {
        border-color: #1e3a5f !important;
        margin: 1.5rem 0 !important;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #0d1b2a !important;
        border-right: 1px solid #1e3a5f !important;
    }

    /* Legend */
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* Highlighted result text */
    .annotated-text {
        font-size: 1rem;
        line-height: 2.2;
        color: #e2e8f0;
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #334155;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }

    /* Section label */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #38bdf8;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    /* Info box */
    .info-box {
        background: #0c2340;
        border: 1px solid #1e3a5f;
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    nlp = spacy.load("en_core_web_sm")
    nlp.remove_pipe("ner")
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True, "phrase_matcher_attr": "LOWER"})

    patterns = [
        # DISEASE
        {"label": "DISEASE", "pattern": "Nipah"},
        {"label": "DISEASE", "pattern": "nipah"},
        {"label": "DISEASE", "pattern": "virus Nipah"},
        {"label": "DISEASE", "pattern": "virus nipah"},
        {"label": "DISEASE", "pattern": "NiV"},
        {"label": "DISEASE", "pattern": "Nipah virus"},
        {"label": "DISEASE", "pattern": "ensefalitis"},
        {"label": "DISEASE", "pattern": "encephalitis"},
        {"label": "DISEASE", "pattern": [{"LOWER": "nipah"}, {"LOWER": "virus"}]},
        {"label": "DISEASE", "pattern": [{"LOWER": "encephalitis"}]},
        {"label": "DISEASE", "pattern": [{"LOWER": "ensefalitis"}]},
        {"label": "DISEASE", "pattern": [{"LOWER": "niv"}]},

        # SYMPTOM — Indonesia
        {"label": "SYMPTOM", "pattern": "demam"},
        {"label": "SYMPTOM", "pattern": "demam tinggi"},
        {"label": "SYMPTOM", "pattern": "kejang"},
        {"label": "SYMPTOM", "pattern": "kejang demam"},
        {"label": "SYMPTOM", "pattern": "batuk"},
        {"label": "SYMPTOM", "pattern": "sesak napas"},
        {"label": "SYMPTOM", "pattern": "sesak"},
        {"label": "SYMPTOM", "pattern": "mual"},
        {"label": "SYMPTOM", "pattern": "muntah"},
        {"label": "SYMPTOM", "pattern": "muntah darah"},
        {"label": "SYMPTOM", "pattern": "sakit kepala"},
        {"label": "SYMPTOM", "pattern": "sakit tenggorokan"},
        {"label": "SYMPTOM", "pattern": "sakit perut"},
        {"label": "SYMPTOM", "pattern": "pusing"},
        {"label": "SYMPTOM", "pattern": "lemas"},
        {"label": "SYMPTOM", "pattern": "lemah"},
        {"label": "SYMPTOM", "pattern": "nyeri"},
        {"label": "SYMPTOM", "pattern": "nyeri otot"},
        {"label": "SYMPTOM", "pattern": "nyeri kepala"},
        {"label": "SYMPTOM", "pattern": "koma"},
        {"label": "SYMPTOM", "pattern": "tidak sadar"},
        {"label": "SYMPTOM", "pattern": "gangguan kesadaran"},
        {"label": "SYMPTOM", "pattern": "gangguan pernapasan"},
        {"label": "SYMPTOM", "pattern": "radang otak"},
        {"label": "SYMPTOM", "pattern": "sulit bernapas"},
        {"label": "SYMPTOM", "pattern": "diare"},
        {"label": "SYMPTOM", "pattern": "disorientasi"},
        {"label": "SYMPTOM", "pattern": "kelelahan"},
        {"label": "SYMPTOM", "pattern": "mengantuk"},
        {"label": "SYMPTOM", "pattern": "tidak responsif"},
        {"label": "SYMPTOM", "pattern": "rasa sakit"},

        # SYMPTOM — English
        {"label": "SYMPTOM", "pattern": [{"LOWER": "fever"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "headache"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "cough"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "nausea"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "vomiting"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "fatigue"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "seizure"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "seizures"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "confusion"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "drowsiness"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "dizziness"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "disorientation"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "myalgia"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "difficulty"}, {"LOWER": "breathing"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "shortness"}, {"LOWER": "of"}, {"LOWER": "breath"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "chest"}, {"LOWER": "pain"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "muscle"}, {"LOWER": "pain"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "muscle"}, {"LOWER": "weakness"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "sore"}, {"LOWER": "throat"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "abdominal"}, {"LOWER": "pain"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "brain"}, {"LOWER": "swelling"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "loss"}, {"LOWER": "of"}, {"LOWER": "consciousness"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "altered"}, {"LOWER": "consciousness"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "respiratory"}, {"LOWER": "failure"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "respiratory"}, {"LOWER": "distress"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "respiratory"}, {"LOWER": "illness"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "respiratory"}, {"LOWER": "infection"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "respiratory"}, {"LOWER": "symptoms"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "acute"}, {"LOWER": "respiratory"}]},
        {"label": "SYMPTOM", "pattern": [{"LOWER": "inflammation"}, {"LOWER": "of"}, {"LOWER": "the"}, {"LOWER": "brain"}]},

        # LOCATION
        {"label": "LOCATION", "pattern": "Malaysia"},
        {"label": "LOCATION", "pattern": "India"},
        {"label": "LOCATION", "pattern": "Bangladesh"},
        {"label": "LOCATION", "pattern": "Singapore"},
        {"label": "LOCATION", "pattern": "Kerala"},
        {"label": "LOCATION", "pattern": "Indonesia"},
        {"label": "LOCATION", "pattern": "Philippines"},
        {"label": "LOCATION", "pattern": "Filipina"},
    ]

    ruler.add_patterns(patterns)
    return nlp


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def preprocess(text):
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s.,!?;:\-\'"()/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_ner(nlp, text):
    clean = preprocess(text)
    doc = nlp(clean)
    result = defaultdict(list)
    for ent in doc.ents:
        result[ent.label_].append((ent.text, ent.start_char, ent.end_char))
    return doc, dict(result), clean


def highlight_text(text, doc):
    colors = {
        "DISEASE":  ("hl-disease",  "DISEASE"),
        "SYMPTOM":  ("hl-symptom",  "SYMPTOM"),
        "LOCATION": ("hl-location", "LOCATION"),
    }
    result = ""
    last = 0
    for ent in sorted(doc.ents, key=lambda e: e.start_char):
        result += text[last:ent.start_char]
        cls, label = colors.get(ent.label_, ("", ""))
        result += f'<span class="{cls}">{ent.text} <span style="font-size:0.65rem;opacity:0.7;font-family:\'IBM Plex Mono\',monospace;">{label}</span></span>'
        last = ent.end_char
    result += text[last:]
    return result


def group_entities(entities):
    grouped = {}
    for label, items in entities.items():
        seen = []
        for text, _, _ in items:
            if text.lower() not in [s.lower() for s in seen]:
                seen.append(text)
        grouped[label] = seen
    return grouped


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
nlp = load_model()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="nivscan-header">
    <div class="nivscan-logo">NiV<span>Scan</span></div>
    <div class="nivscan-tagline">// Nipah Virus Named Entity Recognition System</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Analisis Teks", "📊  Evaluasi Model", "ℹ️  Tentang"])

# ─────────────────────────────────────────────
# TAB 1 — ANALISIS TEKS
# ─────────────────────────────────────────────
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-label">Input Teks</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Masukkan teks berita, artikel, atau kalimat tentang virus Nipah dalam Bahasa Indonesia maupun Inggris.</div>', unsafe_allow_html=True)

        samples = {
            "Kosong": "",
            "Sampel 1: Wabah di Kerala": "The Nipah virus outbreak in Kerala has caused several deaths. Local authorities in Kozhikode are monitoring people who had contact with infected patients showing symptoms of high fever and respiratory distress.",
            "Sampel 2: Deteksi di Malaysia": "Nipah virus was first identified in 1999 during an outbreak among pig farmers in Malaysia. The disease causes severe brain swelling and has a high fatality rate in Southeast Asia.",
            "Sampel 3: Gejala & Penularan": "Symptoms of Nipah virus include fever, headache, and cough. It can be transmitted from bats to humans or through contaminated food like raw date palm sap often found in Bangladesh."
        }
        
        selected_sample = st.selectbox("Pilih contoh kalimat:", list(samples.keys()))
        default_val = samples[selected_sample]

        input_text = st.text_area(
            label="",
            value=default_val,
            placeholder="Contoh: Pasien di Kerala mengalami demam tinggi dan kejang akibat virus Nipah...",
            height=220,
            label_visibility="collapsed"
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            detect_btn = st.button("🔬  Deteksi Entitas", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️  Hapus", use_container_width=True)

        if clear_btn:
            input_text = ""
            st.rerun()

        # Legend
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Legenda</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="legend-item"><span class="badge badge-disease">DISEASE</span> Nama penyakit / virus</div>
        <div class="legend-item"><span class="badge badge-symptom">SYMPTOM</span> Gejala klinis</div>
        <div class="legend-item"><span class="badge badge-location">LOCATION</span> Lokasi kejadian</div>
        """, unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="section-label">Hasil Deteksi</div>', unsafe_allow_html=True)

        if detect_btn and input_text.strip():
            doc, entities, clean_text = run_ner(nlp, input_text)
            grouped = group_entities(entities)

            # Highlighted text
            highlighted = highlight_text(clean_text, doc)
            st.markdown(f'<div class="annotated-text">{highlighted}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics
            n_disease  = len(grouped.get("DISEASE", []))
            n_symptom  = len(grouped.get("SYMPTOM", []))
            n_location = len(grouped.get("LOCATION", []))

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val metric-val-disease">{n_disease}</div>
                    <div class="metric-lbl">DISEASE</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val metric-val-symptom">{n_symptom}</div>
                    <div class="metric-lbl">SYMPTOM</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val metric-val-location">{n_location}</div>
                    <div class="metric-lbl">LOCATION</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Grouped entities
            if grouped:
                st.markdown('<div class="section-label">Entitas Terdeteksi</div>', unsafe_allow_html=True)

                label_config = {
                    "DISEASE":  ("badge-disease",  "DISEASE"),
                    "SYMPTOM":  ("badge-symptom",  "SYMPTOM"),
                    "LOCATION": ("badge-location", "LOCATION"),
                }

                for label in ["DISEASE", "SYMPTOM", "LOCATION"]:
                    items = grouped.get(label, [])
                    if items:
                        badge_cls, badge_lbl = label_config[label]
                        entities_str = ", ".join(items)
                        st.markdown(f"""
                        <div class="entity-row">
                            <span class="badge {badge_cls}">{badge_lbl}</span>
                            <span class="entity-text">{entities_str}</span>
                        </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="empty-state">
                    // tidak ada entitas terdeteksi
                </div>""", unsafe_allow_html=True)

        elif detect_btn and not input_text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            st.markdown("""
            <div class="empty-state">
                // masukkan teks dan klik deteksi
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 2 — EVALUASI MODEL
# ─────────────────────────────────────────────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Hasil Evaluasi — spaCy Rule-Based NER</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Evaluasi dilakukan pada 100 kalimat yang diambil secara acak dari dataset menggunakan random sampling (random_state=42).</div>', unsafe_allow_html=True)

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)

    eval_data = [
        ("DISEASE",  1.000, 1.000, 1.000, "metric-val-disease"),
        ("SYMPTOM",  1.000, 1.000, 1.000, "metric-val-symptom"),
        ("LOCATION", 0.957, 0.957, 0.957, "metric-val-location"),
    ]

    with col_e1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:#94a3b8">0.986</div>
            <div class="metric-lbl">MACRO F1</div>
        </div>""", unsafe_allow_html=True)

    for col, (label, p, r, f1, cls) in zip([col_e2, col_e3, col_e4], eval_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val {cls}">{f1:.3f}</div>
                <div class="metric-lbl">F1 {label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Tabel Detail</div>', unsafe_allow_html=True)

    import pandas as pd
    df_eval = pd.DataFrame({
        "Label":     ["DISEASE", "SYMPTOM", "LOCATION"],
        "Precision": [1.000, 1.000, 0.957],
        "Recall":    [1.000, 1.000, 0.957],
        "F1 Score":  [1.000, 1.000, 0.957],
        "TP": [47, 8, 22],
        "FP": [0, 0, 1],
        "FN": [0, 0, 1],
    })
    st.dataframe(df_eval, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Statistik Dataset</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        ("871", "Total kalimat", "#94a3b8"),
        ("43.0%", "DISEASE", "#ef4444"),
        ("5.7%", "SYMPTOM", "#f59e0b"),
        ("17.9%", "LOCATION", "#38bdf8"),
        ("5 sumber", "Dataset", "#94a3b8"),
    ]
    for col, (val, lbl, color) in zip([c1, c2, c3, c4, c5], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{color};font-size:1.3rem">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Catatan Limitasi:</b> Label LOCATION memiliki F1 terendah (0.957) karena model rule-based tidak dapat membedakan 
    nama negara yang muncul sebagai nama media (contoh: 'CNBC Indonesia') dengan lokasi geografis sesungguhnya. 
    Ini merupakan keterbatasan inherent dari pendekatan rule-based yang tidak mempertimbangkan konteks kalimat secara menyeluruh.
    Sebagai pengembangan ke depan, model berbasis BERT dapat meningkatkan performa untuk kasus ambiguitas konteks ini.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 3 — TENTANG
# ─────────────────────────────────────────────
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([1, 1], gap="large")

    with col_a1:
        st.markdown('<div class="section-label">Tentang NiVScan</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="result-card">
            <p style="color:#94a3b8;font-size:0.9rem;line-height:1.8">
            NiVScan adalah sistem <b style="color:#e2e8f0">Named Entity Recognition (NER)</b> berbasis rule-based 
            yang dirancang untuk mengekstrak informasi penting dari teks berita dan artikel tentang 
            <b style="color:#e2e8f0">virus Nipah (NiV)</b> secara otomatis.
            </p>
            <p style="color:#94a3b8;font-size:0.9rem;line-height:1.8">
            Sistem ini mampu mendeteksi tiga kategori entitas: nama penyakit/virus, gejala klinis, 
            dan lokasi kejadian wabah — dalam Bahasa Indonesia maupun Inggris.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Sumber Dataset</div>', unsafe_allow_html=True)
        sources = [
            ("01", "Dataset Manual", "300 kalimat dikumpulkan manual dari artikel berita"),
            ("02", "Google News RSS", "Scraping berita dengan 7 query keyword"),
            ("03", "Kaggle", "Disease Symptom Description Dataset"),
            ("04", "Wikipedia", "Artikel Virus Nipah (ID + EN)"),
            ("05", "WHO + CDC", "Halaman resmi organisasi kesehatan dunia"),
        ]
        for num, title, desc in sources:
            st.markdown(f"""
            <div class="entity-row">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#334155">{num}</span>
                <div>
                    <div style="font-weight:500;color:#e2e8f0;font-size:0.9rem">{title}</div>
                    <div style="font-size:0.78rem;color:#64748b">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_a2:
        st.markdown('<div class="section-label">Spesifikasi Teknis</div>', unsafe_allow_html=True)
        tech_items = [
            ("Language",   "Python 3.12"),
            ("NLP Library","spaCy 3.x + EntityRuler"),
            ("Approach",   "Rule-Based NER"),
            ("Patterns",   "81 patterns (DISEASE, SYMPTOM, LOCATION)"),
            ("Dataset",    "871 kalimat dari 5 sumber"),
            ("Evaluasi",   "100 kalimat random sampling"),
            ("Macro F1",   "0.986"),
            ("Deployment", "Streamlit"),
        ]
        for key, val in tech_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1e3a5f;">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#64748b">{key}</span>
                <span style="font-size:0.85rem;color:#e2e8f0;font-weight:500">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Mata Kuliah</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="result-card">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#64748b">Course</div>
            <div style="font-size:0.95rem;color:#e2e8f0;font-weight:500;margin-top:4px">COMP6885001 — Natural Language Processing</div>
            <div style="font-size:0.78rem;color:#64748b;margin-top:8px">School of Computer Science</div>
            <div style="font-size:0.78rem;color:#64748b">Academic Year 2025/2026</div>
        </div>
        """, unsafe_allow_html=True)
