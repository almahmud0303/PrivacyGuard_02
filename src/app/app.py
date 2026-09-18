from __future__ import annotations

import pandas as pd
import streamlit as st

from src.inference.model_service import analyze, available_models


def main() -> None:
    st.set_page_config(page_title="PrivacyGuard", page_icon="🔒", layout="wide")
    st.title("🔒 PrivacyGuard Model Lab")
    st.caption("Compare pattern, BiLSTM, BERT, and classical PII models without exposing your text.")

    models = available_models()
    with st.sidebar:
        st.header("Model settings")
        model_name = st.selectbox("Detection model", list(models))
        status = models[model_name]
        if status["ready"]:
            st.success("Model artifact ready")
        else:
            st.warning("Artifact is missing or outdated. Train it before model-only testing.")
        st.caption("BERT and BiLSTM use model-only NER. Classical models classify the entire text and use patterns for redaction.")
        threshold = st.slider("Minimum entity confidence", 0.0, 1.0, float(status.get("threshold", .80)), 0.05,
            help="Lower values improve recall; higher values reduce false positives.")
        fallback = st.toggle("Fallback on model failure", value=True,
            help="Only if loading/inference fails, use pattern detection instead of crashing.")

    examples = {
        "English": "My phone is 01712345678 and email is user@example.com.",
        "Bengali": "আমার ফোন নম্বর 01898765432 এবং ইমেইল test@example.com।",
        "Banglish": "amar phone 01955667788, email rahim@example.com",
    }
    example = st.selectbox("Example", ["Custom", *examples])
    initial = examples.get(example, "")
    text = st.text_area("Text to protect", value=initial, height=200)

    if not st.button("Run model", type="primary", width="stretch"):
        return
    if not text.strip():
        st.warning("Enter some text first.")
        return

    try:
        with st.spinner(f"Running {model_name}…"):
            result = analyze(text, model_name, threshold, fallback)
    except Exception as exc:
        st.error(f"{model_name} failed: {type(exc).__name__}: {exc}")
        st.info("Enable Safe fallback or train/install the selected model.")
        return

    if result.get("note"):
        st.warning(result["note"])
    if result.get("requested_model"):
        st.caption(f"Requested: {result['requested_model']} · Used: {result['model']}")

    first, second, third, fourth = st.columns(4)
    first.metric("Model used", result["model"])
    second.metric("Entities", result["entity_count"])
    third.metric("Overall risk", result["overall_risk"])
    fourth.metric("Classification", result.get("classification", "Entity detection"))

    st.subheader("Protected text")
    st.code(result["safe_text"], language=None)
    st.download_button("Download protected text", result["safe_text"], "protected_text.txt", "text/plain")

    st.subheader("Detected information")
    if result["entities"]:
        columns = ["entity", "type", "risk", "confidence", "source", "start", "end"]
        table = pd.DataFrame(result["entities"])
        st.dataframe(table[[column for column in columns if column in table]], width="stretch", hide_index=True)
    else:
        st.info("No PII span was detected at the selected confidence threshold.")


if __name__ == "__main__":
    main()
