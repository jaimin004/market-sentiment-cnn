# 📊 Market Sentiment Analyzer  
### Multimodal Financial Sentiment Analysis System (FinBERT + MobileNetV2)

**Project Type:** Deep Learning / NLP  
**Domain:** Financial Market Analysis  
**Architecture:** Multimodal (Text + Image)  
**Framework:** PyTorch + Streamlit  

The Market Sentiment Analyzer is a deep learning–based system designed to classify financial market sentiment from both textual and visual inputs. The system predicts whether the market sentiment expressed in news content is:

- 🟢 Positive  
- 🟡 Neutral  
- 🔴 Negative  

The application supports both direct text input and financial image uploads (e.g., news screenshots), making it a multimodal sentiment analysis solution.

---

## 🧠 Project Overview

Financial markets are highly sensitive to news events, economic indicators, and global developments. Investors and traders rely heavily on sentiment-driven signals to make decisions.

This project aims to automate financial sentiment detection using:

- Financial headlines  
- News articles  
- Market summaries  
- Screenshot-based news content  

The system integrates domain-specific NLP (FinBERT) and visual feature extraction (MobileNetV2) along with OCR technology to extract and analyze financial information.

An interactive Streamlit dashboard allows users to perform real-time sentiment analysis with confidence scores and probability breakdowns.

---

## 🏗 Model Architecture

The system follows a multimodal architecture consisting of:

### 🔹 1. Textual Feature Extraction – FinBERT

**Base Model:**  
ProsusAI/finbert

FinBERT is a financial-domain adapted version of BERT trained specifically on financial text data such as earnings calls, analyst reports, and financial news.

#### Text Processing Pipeline:

1. Input text is tokenized using FinBERT tokenizer  
2. Token embeddings are generated  
3. CLS token embedding (768-dimensional) is extracted  
4. Passed through a fully connected classification layer (768 → 3)  
5. Softmax applied to compute class probabilities  

#### Output Label Mapping:

- 0 → Positive  
- 1 → Neutral  
- 2 → Negative  

---

### 🔹 2. Visual Feature Extraction – MobileNetV2

**Model Used:** MobileNetV2 Feature Extractor  

MobileNetV2 is used to extract deep visual features from financial images such as screenshots of market updates or news articles.

#### Image Processing Pipeline:

1. Image resized to 224 × 224  
2. Converted to tensor  
3. Passed through MobileNetV2 feature extractor  
4. EasyOCR extracts embedded text from the image  
5. Extracted text is passed to FinBERT classifier  
6. Sentiment is predicted based on extracted textual content  

Currently, the final sentiment decision is text-driven after OCR extraction, while visual features are extracted for future fusion expansion.

---

## ⚙ System Logic

### 📝 Text Input Logic

1. User enters financial text  
2. Text is tokenized using FinBERT tokenizer  
3. CLS embedding is extracted  
4. Embedding passed through trained classifier  
5. Softmax generates probability scores  
6. Highest probability determines sentiment  
7. Confidence score displayed  

---

### 📷 Image Input Logic

1. User uploads financial image  
2. Image processed through MobileNetV2  
3. EasyOCR extracts text from image  
4. Extracted text cleaned and tokenized  
5. FinBERT processes text  
6. Classifier predicts sentiment  
7. Sentiment and confidence score displayed  
8. Extracted text shown in expandable section  

---

## 📊 Dashboard Logic

The application is built using Streamlit 

### 🔹 Main Interface

- Mode Selection (Text / Image)  
- Sentiment Prediction Display  
- Confidence Percentage  
- Extracted Text Viewer  
- Expandable detailed output  

Session state is used to dynamically update prediction details after each analysis.

---

## 🛠 Built Using

### Programming & Frameworks
- Python  
- PyTorch  
- Streamlit  

### Deep Learning Models
- FinBERT (Financial NLP Model)  
- MobileNetV2 (Visual Feature Extractor)  

### Other Libraries
- Transformers (Hugging Face)  
- EasyOCR  
- Torchvision  
- Pandas  
- Pillow  
- TQDM  
---

## 👨‍💻 Author
Jaimin Patel\
Year: 2026  
