# NiVScan: Nipah Virus Entity Extractor

NiVScan is a Deep Learning-based Named Entity Recognition (NER) system designed to extract epidemiological entities from biomedical texts related to Nipah virus outbreaks.

This project evaluates several NER architectures and deploys the optimized model using **XLM-RoBERTa** for extracting two key entity types — **DISEASE** and **LOCATION** — from unstructured epidemiological documents.

---

## 🚀 Live Demo

The deployed application is available through:

- **Streamlit Application** — [niv-scan.streamlit.app](https://niv-scan.streamlit.app/)
- **Hugging Face Space** — [huggingface.co/spaces/unvariablehuman/NiV-Scan](https://huggingface.co/spaces/unvariablehuman/NiV-Scan)

The application allows users to input biomedical text and automatically identify Nipah-related **DISEASE** and **LOCATION** entities.

---

## 📂 Project Structure

```
NiVScan/
│
├── Dataset/
│   ├── Raw dataset
│   ├── Combined dataset
│   └── Silver & Gold annotations
│
├── nipah_ner_model/
│   ├── Trained NER model
│   └── Streamlit application
│
├── Percobaan/
│   ├── Model A experiment
│   ├── Model B experiment
│   ├── Model C experiment
│   └── Model D experiment
│
├── requirements.txt
└── README.md
```

### Directory Description

| Folder/File | Description |
|---|---|
| `Dataset/` | Raw data, combined dataset, and Silver/Gold annotation files used for training and evaluation |
| `nipah_ner_model/` | Trained model and Streamlit deployment application |
| `Percobaan/` | Experiment notebooks and evaluation results from different architectures |
| `requirements.txt` | Python dependency list required to run the project |

---

## ⚙️ Installation & Local Deployment

To run NiVScan locally:

**1. Clone Repository**

```bash
git clone https://github.com/unvariablehuman/NiVScan.git
cd NiVScan
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**3. Run Streamlit Application**

```bash
streamlit run nipah_ner_model/streamlit_app.py
```

---

## 🧠 Model

NiVScan utilizes a fine-tuned **XLM-RoBERTa** Transformer architecture for biomedical Named Entity Recognition.

The final deployed model was selected based on evaluation using the manually annotated gold-standard test set, considering:

- Entity-level Macro F1-score
- Matthews Correlation Coefficient (MCC)
- Robustness under imbalanced entity distribution

The system focuses on identifying two main entity categories:

- **DISEASE** — disease-related terms
- **LOCATION** — geographical locations associated with outbreaks

---

## 📊 Experimental Models

Four architectures were evaluated during development:

| Model | Architecture |
|---|---|
| Model A | CharCNN-BiLSTM-CRF |
| Model B | XLM-RoBERTa |
| Model C | BioBERT |
| Model D | PubMedBERT |

Experimental notebooks and training processes are available through Kaggle:

| Model | Kaggle Notebook |
|---|---|
| Model A | [modela-revised](https://www.kaggle.com/code/unvariablehuman/modela-revised) |
| Model B | [model-b-niv](https://www.kaggle.com/code/unvariablehuman/model-b-niv) |
| Model C | [model-c](https://www.kaggle.com/code/unvariablehuman/model-c) |
| Model D | [model-d](https://www.kaggle.com/code/unvariablehuman/model-d) |

---

## 📚 Project Resources

| Resource | Link |
|---|---|
| Streamlit Application | [niv-scan.streamlit.app](https://niv-scan.streamlit.app/) |
| Presentation Slides (Canva) | [canva.link/m57yszmgne9r9e2](https://canva.link/m57yszmgne9r9e2) |
| Application Demo Files | [Google Drive Folder](https://drive.google.com/drive/folders/19TAjEBCbtzJj7zEesztthwtWM2APSzho?usp=sharing) |
| Complete Project Documentation | [Google Drive Folder](https://drive.google.com/drive/folders/1W8n2SnJ5i1VgkYfeIQJUmGqR1DDtg4ro?usp=sharing) |

---

## 👥 Development Team

**Group 11 — NiVScan Project**
Natural Language Processing (NLP) — Semester 4, BINUS University

| Member | Contribution |
|---|---|
| Farhan Faturachman Zidane Haristian | Presentation materials and project demonstration |
| Sabrina Arfanindia Devi | Model training, evaluation, methodology, and research paper |
| Kristian Novan | Documentation and project organization |
| Mohammad Faisal Riftiarrysid | Research supervision and model evaluation guidance |

---

## 📝 Acknowledgement

This project was developed as part of the Natural Language Processing course at Bina Nusantara University.

Powered by:

- [Hugging Face](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- [spaCy](https://spacy.io/)
- Transformer-based NLP technologies
