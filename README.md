# NiVScan: Nipah Virus Entity Extractor 

NiVScan adalah tool berbasis Deep Learning untuk mendeteksi entitas (Named Entity Recognition) dalam teks berita atau medis terkait Virus Nipah. Proyek ini dikembangkan menggunakan **XLM-RoBERTa** dan library **spaCy** untuk memberikan ekstraksi informasi yang akurat mengenai lokasi, gejala, tanggal, dan detail medis lainnya terkait wabah Nipah.

## 🚀 Live Demo
Aplikasi ini dapat diakses langsung melalui Hugging Face Spaces:
 **[NiVScan on Hugging Face](https://huggingface.co/spaces/unvariablehuman/NiV-Scan)**

## Struktur Proyek
- **Dataset/**: Berisi data mentah, data gabungan, dan label (Silver/Gold) yang digunakan untuk pelatihan.
- **nipah_ner_model/**: Berisi model spaCy yang sudah ditraining dan aplikasi Streamlit.
- **Percobaan/**: Dokumentasi berbagai model (Model A, B, C, D) dan eksperimen selama pengembangan.
- **requirements.txt**: Daftar dependensi untuk menjalankan aplikasi.

## Instalasi & Cara Menjalankan

Jika ingin menjalankan secara lokal:

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/unvariablehuman/NiVScan.git
   cd NiVScan
   ```

2. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run nipah_ner_model/streamlit_app.py
   ```

## Model
Menggunakan arsitektur Transformer (XLM-RoBERTa) yang difine-tune khusus untuk dataset terkait Virus Nipah. Model ini mampu mengenali berbagai entitas medis dalam teks bahasa Inggris dan Indonesia.

## Pengembang
Dikembangkan oleh **Group 11 — NiVScan Project** untuk mata kuliah NLP (Semester 4 - BINUS University).

---
*Powered by Hugging Face, Streamlit, and spaCy.*
