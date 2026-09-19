from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from src.inference.model_service import analyze, available_models


APP_CSS = """
<style>
    :root {
        --pg-bg: #07111f;
        --pg-panel: rgba(15, 27, 45, 0.86);
        --pg-panel-soft: rgba(20, 37, 59, 0.72);
        --pg-line: rgba(148, 163, 184, 0.17);
        --pg-text: #f8fafc;
        --pg-muted: #9fb0c5;
        --pg-cyan: #22d3ee;
        --pg-green: #34d399;
    }

    .stApp {
        color: var(--pg-text);
        background:
            radial-gradient(circle at 88% 3%, rgba(34, 211, 238, .10), transparent 27rem),
            radial-gradient(circle at 15% 0%, rgba(59, 130, 246, .13), transparent 31rem),
            var(--pg-bg);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: rgba(6, 15, 28, .96);
        border-right: 1px solid var(--pg-line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--pg-muted);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.25rem;
        padding-bottom: 4rem;
    }

    .pg-brand {
        display: flex;
        align-items: center;
        gap: .8rem;
        margin: .1rem 0 1.8rem;
    }
    .pg-brand-mark {
        display: grid;
        place-items: center;
        width: 2.55rem;
        height: 2.55rem;
        border-radius: .85rem;
        color: #03101a;
        font-size: 1.25rem;
        background: linear-gradient(135deg, var(--pg-cyan), var(--pg-green));
        box-shadow: 0 0 28px rgba(34, 211, 238, .18);
    }
    .pg-brand-name { color: var(--pg-text); font-size: 1.08rem; font-weight: 760; }
    .pg-brand-meta { color: #71839a; font-size: .72rem; letter-spacing: .11em; text-transform: uppercase; }

    .pg-hero {
        position: relative;
        overflow: hidden;
        margin-bottom: 1.45rem;
        padding: 2.35rem 2.5rem;
        border: 1px solid var(--pg-line);
        border-radius: 1.5rem;
        background: linear-gradient(135deg, rgba(20, 40, 65, .94), rgba(10, 23, 39, .86));
        box-shadow: 0 24px 70px rgba(0, 0, 0, .22);
    }
    .pg-hero::after {
        content: "";
        position: absolute;
        width: 17rem;
        height: 17rem;
        right: -5rem;
        top: -7rem;
        border-radius: 50%;
        background: rgba(34, 211, 238, .10);
        filter: blur(4px);
    }
    .pg-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        margin-bottom: .9rem;
        padding: .35rem .72rem;
        border: 1px solid rgba(52, 211, 153, .25);
        border-radius: 999px;
        color: #86efac;
        background: rgba(16, 185, 129, .08);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
    }
    .pg-dot {
        width: .45rem;
        height: .45rem;
        border-radius: 50%;
        background: var(--pg-green);
        box-shadow: 0 0 10px var(--pg-green);
    }
    .pg-hero h1 {
        position: relative;
        z-index: 1;
        max-width: 760px;
        margin: 0 0 .65rem;
        color: var(--pg-text);
        font-size: clamp(2rem, 5vw, 3.35rem);
        line-height: 1.06;
        letter-spacing: -.045em;
    }
    .pg-gradient-text {
        color: transparent;
        background: linear-gradient(90deg, #67e8f9, #60a5fa);
        -webkit-background-clip: text;
        background-clip: text;
    }
    .pg-hero p {
        position: relative;
        z-index: 1;
        max-width: 700px;
        margin: 0;
        color: var(--pg-muted);
        font-size: 1rem;
        line-height: 1.65;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--pg-line) !important;
        border-radius: 1.15rem !important;
        background: var(--pg-panel);
        box-shadow: 0 16px 45px rgba(0, 0, 0, .13);
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div { padding: .3rem; }

    .pg-section-label {
        color: #dbeafe;
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .09em;
        text-transform: uppercase;
    }
    .pg-section-copy { margin-top: .2rem; color: var(--pg-muted); font-size: .87rem; }

    div[data-testid="stTextArea"] textarea {
        min-height: 205px;
        border: 1px solid rgba(148, 163, 184, .22);
        border-radius: 1rem;
        color: #eef6ff;
        background: rgba(4, 12, 23, .72);
        font-size: .98rem;
        line-height: 1.65;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: rgba(34, 211, 238, .72);
        box-shadow: 0 0 0 3px rgba(34, 211, 238, .10);
    }

    .stButton > button[kind="primary"] {
        min-height: 3.05rem;
        border: 0;
        border-radius: .85rem;
        color: #03131d;
        font-weight: 800;
        background: linear-gradient(90deg, var(--pg-cyan), #60a5fa);
        box-shadow: 0 12px 30px rgba(34, 211, 238, .16);
        transition: transform .16s ease, box-shadow .16s ease;
    }
    .stButton > button[kind="primary"]:hover {
        color: #03131d;
        transform: translateY(-1px);
        box-shadow: 0 16px 36px rgba(34, 211, 238, .24);
    }
    .stDownloadButton > button {
        border-color: rgba(34, 211, 238, .32);
        border-radius: .75rem;
        color: #a5f3fc;
        background: rgba(34, 211, 238, .06);
    }

    [data-testid="stMetric"] {
        min-height: 7.2rem;
        padding: 1.05rem 1.1rem;
        border: 1px solid var(--pg-line);
        border-radius: 1rem;
        background: var(--pg-panel-soft);
    }
    [data-testid="stMetricLabel"] { color: var(--pg-muted); }
    [data-testid="stMetricValue"] { color: var(--pg-text); font-size: 1.55rem; }

    .pg-result-head {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: center;
        margin: 2rem 0 .8rem;
    }
    .pg-result-head h2 { margin: 0; color: var(--pg-text); font-size: 1.45rem; }
    .pg-risk {
        padding: .38rem .7rem;
        border: 1px solid var(--risk-border);
        border-radius: 999px;
        color: var(--risk-color);
        background: var(--risk-bg);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .05em;
    }
    .pg-protected {
        margin: .35rem 0 .85rem;
        padding: 1.1rem 1.2rem;
        border: 1px solid rgba(34, 211, 238, .17);
        border-radius: .9rem;
        color: #dff9ff;
        background: rgba(3, 12, 23, .78);
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: .91rem;
        line-height: 1.7;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .pg-status {
        margin: .65rem 0 1rem;
        padding: .78rem .9rem;
        border: 1px solid var(--status-border);
        border-radius: .8rem;
        color: var(--status-color);
        background: var(--status-bg);
        font-size: .84rem;
        font-weight: 650;
    }
    .pg-flow {
        margin-top: 1.3rem;
        padding: 1rem;
        border: 1px solid var(--pg-line);
        border-radius: .9rem;
        color: #8ea1b8;
        background: rgba(15, 27, 45, .48);
        font-size: .78rem;
        line-height: 1.8;
    }
    .pg-flow b { color: #cbd5e1; }
    .pg-footer {
        margin-top: 2.4rem;
        color: #61738a;
        font-size: .76rem;
        text-align: center;
    }

    hr { border-color: var(--pg-line) !important; }
    @media (max-width: 760px) {
        .block-container { padding: 1.2rem .9rem 3rem; }
        .pg-hero { padding: 1.6rem 1.35rem; border-radius: 1.15rem; }
        .pg-hero h1 { font-size: 2.2rem; }
        .pg-result-head { align-items: flex-start; flex-direction: column; }
    }
</style>
"""


EXAMPLES = {
    "English": "My phone is 01712345678 and email is abdullah@gmail.com.",
    "বাংলা": "আমার ফোন নম্বর 01898765432 এবং ইমেইল abdullah@gmail.com",
    "Banglish": "amar phone 01955667788, email abdullah@gmail.com",
}

RISK_STYLES = {
    "CRITICAL": ("#fda4af", "rgba(244, 63, 94, .10)", "rgba(244, 63, 94, .30)"),
    "HIGH": ("#fdba74", "rgba(249, 115, 22, .10)", "rgba(249, 115, 22, .30)"),
    "MEDIUM": ("#fde047", "rgba(234, 179, 8, .09)", "rgba(234, 179, 8, .28)"),
    "LOW": ("#86efac", "rgba(34, 197, 94, .09)", "rgba(34, 197, 94, .28)"),
}


def _sidebar(models: dict[str, dict[str, str | bool | float]]) -> tuple[str, float, bool]:
    with st.sidebar:
        st.markdown(
            """
            <div class="pg-brand">
                <div class="pg-brand-mark">◇</div>
                <div><div class="pg-brand-name">PrivacyGuard</div>
                <div class="pg-brand-meta">Model Lab</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="pg-section-label">Detection settings</div>', unsafe_allow_html=True)
        model_name = st.selectbox("Model", list(models), label_visibility="collapsed")
        status = models[model_name]
        if status["ready"]:
            st.markdown(
                '<div class="pg-status" style="--status-color:#86efac;--status-bg:rgba(34,197,94,.08);'
                '--status-border:rgba(34,197,94,.24)">● Model artifact ready</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pg-status" style="--status-color:#fdba74;--status-bg:rgba(249,115,22,.08);'
                '--status-border:rgba(249,115,22,.24)">● Artifact missing or outdated</div>',
                unsafe_allow_html=True,
            )

        threshold = st.slider(
            "Minimum confidence",
            0.0,
            1.0,
            float(status.get("threshold", .80)),
            0.05,
            help="Lower values improve recall; higher values reduce false positives.",
        )
        fallback = st.toggle(
            "Safe fallback",
            value=True,
            help="Use pattern detection if the selected model cannot load or run.",
        )
        st.caption(f"Threshold: {threshold:.0%} · Pattern fallback: {'on' if fallback else 'off'}")
        st.divider()
        st.markdown('<div class="pg-section-label">How it works</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="pg-flow">
                <b>01</b> Choose a detection model<br>
                <b>02</b> Paste or select sample text<br>
                <b>03</b> Review and export protected text
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("All analysis runs locally. Text is not sent to an external AI service.")
    return model_name, threshold, fallback


def _risk_badge(level: str) -> str:
    color, background, border = RISK_STYLES.get(level, RISK_STYLES["LOW"])
    return (
        f'<span class="pg-risk" style="--risk-color:{color};--risk-bg:{background};'
        f'--risk-border:{border}">{escape(level)} RISK</span>'
    )


def _render_results(result: dict) -> None:
    risk = str(result.get("overall_risk", "LOW")).upper()
    st.markdown(
        f'<div class="pg-result-head"><h2>Analysis result</h2>{_risk_badge(risk)}</div>',
        unsafe_allow_html=True,
    )

    if result.get("note"):
        st.warning(result["note"])
    if result.get("requested_model"):
        st.caption(f"Requested: {result['requested_model']} · Used: {result['model']}")

    first, second, third, fourth = st.columns(4)
    first.metric("Model used", result["model"])
    second.metric("Entities found", result["entity_count"])
    third.metric("Overall risk", risk.title())
    fourth.metric("Classification", result.get("classification", "Entity detection"))

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="pg-section-label">Protected text</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="pg-protected">{escape(result["safe_text"])}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "↓  Download protected text",
                result["safe_text"],
                "protected_text.txt",
                "text/plain",
                width="stretch",
            )
    with right:
        with st.container(border=True):
            st.markdown('<div class="pg-section-label">Scan summary</div>', unsafe_allow_html=True)
            st.write(f"**{result['entity_count']}** sensitive span(s) detected")
            st.write(f"**{risk.title()}** overall document risk")
            st.write(f"**{result['model']}** completed the analysis")
            st.caption("Review detections before sharing or storing the output.")

    st.markdown("### Detected information")
    if result["entities"]:
        columns = ["entity", "type", "risk", "confidence", "source", "start", "end"]
        table = pd.DataFrame(result["entities"])
        visible = table[[column for column in columns if column in table]].copy()
        if "confidence" in visible:
            visible["confidence"] = pd.to_numeric(visible["confidence"], errors="coerce")
        st.dataframe(
            visible,
            width="stretch",
            hide_index=True,
            column_config={
                "entity": st.column_config.TextColumn("Detected value", width="large"),
                "type": st.column_config.TextColumn("Entity type"),
                "risk": st.column_config.TextColumn("Risk"),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.0f%%"
                ),
                "source": st.column_config.TextColumn("Source"),
                "start": st.column_config.NumberColumn("Start", format="%d"),
                "end": st.column_config.NumberColumn("End", format="%d"),
            },
        )
    else:
        st.info("No PII span was detected at the selected confidence threshold.")


def main() -> None:
    st.set_page_config(
        page_title="PrivacyGuard · PII Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

    models = available_models()
    model_name, threshold, fallback = _sidebar(models)

    st.markdown(
        """
        <section class="pg-hero">
            <div class="pg-eyebrow"><span class="pg-dot"></span> Local privacy workspace</div>
            <h1>Detect sensitive data.<br><span class="pg-gradient-text">Share text safely.</span></h1>
            <p>Compare  NLP models, identify personally identifiable information, and create a
            protected copy of your text—all from one privacy-first interface.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="pg-section-label">Text workspace</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pg-section-copy">Paste your own content or load a multilingual example.</div>',
            unsafe_allow_html=True,
        )
        example = st.radio(
            "Sample text",
            ["Custom", *EXAMPLES],
            horizontal=True,
            label_visibility="collapsed",
        )
        initial = EXAMPLES.get(example, "")
        text = st.text_area(
            "Text to protect",
            value=initial,
            height=220,
            placeholder="Paste text containing information you want to detect and protect…",
            label_visibility="collapsed",
        )
        count_col, action_col = st.columns([3, 1])
        with count_col:
            st.caption(f"{len(text):,} characters · Selected model: {model_name}")
        with action_col:
            run = st.button("Scan & protect  →", type="primary", width="stretch")

    if run:
        if not text.strip():
            st.warning("Enter some text before starting the scan.")
        else:
            try:
                with st.spinner(f"Running {model_name}…"):
                    result = analyze(text, model_name, threshold, fallback)
            except Exception as exc:
                st.error(f"{model_name} failed: {type(exc).__name__}: {exc}")
                st.info("Enable Safe fallback or train/install the selected model.")
            else:
                _render_results(result)

    st.markdown(
        '<div class="pg-footer">PrivacyGuard · Educational multilingual PII detection and redaction</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
