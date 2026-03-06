import streamlit as st
from PIL import Image
import tempfile
import os
import torch
import predict

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Market Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# -------------------- CUSTOM UI STYLE --------------------
st.markdown("""
    <style>
        .main { padding-top: 2rem; }
        .stButton>button {
            width: 100%;
            height: 3em;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Market Sentiment Analyzer")
st.subheader("FinBERT-Based Financial Sentiment Model")
st.divider()

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model_once():
    return predict.load_models()

tokenizer, finbert, classifier, visual_model, reader = load_model_once()

DEVICE = "cpu"  # Always CPU (Streamlit safe)

# -------------------- SESSION STATE --------------------
if "show_preview" not in st.session_state:
    st.session_state.show_preview = False

# -------------------- MODE SELECTION --------------------
mode = st.radio(
    "Select Input Type:",
    ["📷 Image Upload", "📝 Text Input"]
)

st.divider()

# ============================================================
# 📷 IMAGE MODE
# ============================================================
if mode == "📷 Image Upload":

    uploaded_files = st.file_uploader(
        "Upload Financial Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👁 Preview Images"):
                st.session_state.show_preview = not st.session_state.show_preview

        with col2:
            analyze_clicked = st.button("🔍 Analyze Images")

        # -------- PREVIEW --------
        if st.session_state.show_preview:
            st.subheader("🖼 Image Preview")

            cols = st.columns(min(3, len(uploaded_files)))

            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert("RGB")
                cols[idx % 3].image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

            st.divider()

        # -------- ANALYZE --------
        if analyze_clicked:

            results = []

            with st.spinner("Analyzing images..."):

                for uploaded_file in uploaded_files:

                    image = Image.open(uploaded_file).convert("RGB")

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        image.save(tmp.name)
                        temp_path = tmp.name

                    sentiment, extracted_text = predict.predict_sentiment_with_models(
                        temp_path,
                        tokenizer,
                        finbert,
                        classifier,
                        visual_model,
                        reader
                    )

                    os.remove(temp_path)

                    results.append((uploaded_file.name, sentiment, extracted_text))

            st.subheader("📊 Results")

            for name, sentiment, content in results:

                st.markdown(f"### 🖼 {name}")

                if sentiment == "Positive":
                    st.success("🟢 Positive Sentiment")
                elif sentiment == "Neutral":
                    st.warning("🟡 Neutral Sentiment")
                else:
                    st.error("🔴 Negative Sentiment")

                with st.expander("📄 Extracted Text"):
                    st.write(content)

                st.divider()

    else:
        st.info("Upload one or more financial images to analyze.")

# ============================================================
# 📝 TEXT MODE
# ============================================================
elif mode == "📝 Text Input":

    text = st.text_area("Enter Financial Text", height=200)

    if st.button("🔍 Analyze Text"):

        if text.strip() == "":
            st.warning("Please enter some text.")
        else:
            with torch.no_grad():

                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                ).to(DEVICE)

                outputs = finbert(**enc)
                cls_embed = outputs.last_hidden_state[:, 0, :]

                # TEXT-ONLY CLASSIFIER CALL
                logits = classifier(cls_embed)

                pred = torch.argmax(logits, dim=1).item()

                label_map = {
                    0: "Positive",
                    1: "Neutral",
                    2: "Negative"
                }

                sentiment = label_map[pred]

            st.subheader("📊 Result")

            if sentiment == "Positive":
                st.success("🟢 Positive Sentiment")
            elif sentiment == "Neutral":
                st.warning("🟡 Neutral Sentiment")
            else:
                st.error("🔴 Negative Sentiment")

            with st.expander("📄 Entered Text"):
                st.write(text)