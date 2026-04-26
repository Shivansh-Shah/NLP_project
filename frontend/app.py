from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.pdf_utils import combine_selected_pages, extract_pdf_pages
from frontend.pipeline import (
    DEFAULT_PATHS,
    PipelineSettings,
    clear_pipeline_cache,
    run_pipeline_on_context,
    validate_settings_paths,
)


st.set_page_config(
    page_title="NLP QA Studio",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Noto+Sans+Kannada:wght@400;600;700&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: "Space Grotesk", "Noto Sans Kannada", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 15% 10%, rgba(255, 210, 120, 0.20), transparent 35%),
            radial-gradient(circle at 90% 25%, rgba(106, 208, 255, 0.18), transparent 30%),
            linear-gradient(165deg, #f7fbff 0%, #ecf4ff 42%, #f8f7ef 100%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        padding: 1.35rem 1.6rem;
        border-radius: 20px;
        border: 1px solid rgba(16, 42, 66, 0.20);
        background: linear-gradient(135deg, #0f2f4a, #1f5b7f 56%, #247e84);
        color: #f3fbff;
        margin-bottom: 1.5rem;
        box-shadow: 0 14px 34px rgba(10, 33, 57, 0.25);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        letter-spacing: 0.3px;
    }
    .hero p {
        margin: 0.45rem 0 0;
        opacity: 0.92;
        max-width: 90ch;
    }
    .panel {
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid rgba(23, 65, 92, 0.12);
        border-radius: 16px;
        padding: 0.95rem 1.05rem;
        backdrop-filter: blur(3px);
        margin-bottom: 1rem;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(23, 65, 92, 0.16);
        border-radius: 15px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 8px 24px rgba(11, 38, 59, 0.08);
    }
    .mono {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Kannada QA Studio</h1>
      <p>Upload a PDF or paste context, ask in Kannada, and run the connected KN→EN translation + English QA + EN→KN translation pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _build_settings_from_ui() -> PipelineSettings:
    family = st.session_state["ui_translation_family"]
    if family == "scratch":
        kn_en_backend = "scratch"
        en_kn_backend = "scratch"
        kn_en_checkpoint = DEFAULT_PATHS["scratch_kn_en"]
        en_kn_checkpoint = DEFAULT_PATHS["scratch_en_kn"]
    else:
        kn_en_backend = "nn_transformer"
        en_kn_backend = "nn_transformer"
        kn_en_checkpoint = DEFAULT_PATHS["legacy_kn_en"]
        en_kn_checkpoint = DEFAULT_PATHS["legacy_en_kn"]

    return PipelineSettings(
        device=st.session_state["ui_device"],
        cpu_quantize=bool(st.session_state["ui_cpu_quantize"]),
        kn_en_backend=kn_en_backend,
        en_kn_backend=en_kn_backend,
        kn_en_checkpoint=kn_en_checkpoint,
        en_kn_checkpoint=en_kn_checkpoint,
        reader_encoder_checkpoint=st.session_state["ui_reader_encoder"],
        reader_weights=st.session_state["ui_reader_weights"],
        max_gen_len=int(st.session_state["ui_max_gen_len"]),
        min_gen_len=int(st.session_state["ui_min_gen_len"]),
        repetition_penalty=float(st.session_state["ui_rep_penalty"]),
    )


if "ui_device" not in st.session_state:
    st.session_state["ui_device"] = "auto"
    st.session_state["ui_translation_family"] = "scratch"
    st.session_state["ui_cpu_quantize"] = False
    st.session_state["ui_reader_encoder"] = DEFAULT_PATHS["reader_encoder"]
    st.session_state["ui_reader_weights"] = DEFAULT_PATHS["reader_weights"]
    st.session_state["ui_max_gen_len"] = 160
    st.session_state["ui_min_gen_len"] = 6
    st.session_state["ui_rep_penalty"] = 1.2


with st.sidebar:
    st.subheader("Runtime")
    st.selectbox("Device", options=["auto", "cuda", "cpu"], key="ui_device")
    st.checkbox("CPU quantization (scratch only)", key="ui_cpu_quantize")

    st.divider()
    st.subheader("Translation Models")
    st.selectbox(
        "Model family",
        options=["scratch", "nn_transformer"],
        key="ui_translation_family",
        help="Choose only between your scratch models and nn.Transformer-trained models.",
    )

    if st.session_state["ui_translation_family"] == "scratch":
        st.caption("KN→EN: {}".format(DEFAULT_PATHS["scratch_kn_en"]))
        st.caption("EN→KN: {}".format(DEFAULT_PATHS["scratch_en_kn"]))
    else:
        st.caption("KN→EN: {}".format(DEFAULT_PATHS["legacy_kn_en"]))
        st.caption("EN→KN: {}".format(DEFAULT_PATHS["legacy_en_kn"]))

    st.divider()
    st.subheader("QA Reader")
    st.text_input("Reader encoder checkpoint", key="ui_reader_encoder")
    st.text_input("Reader weights", key="ui_reader_weights")

    st.divider()
    st.subheader("Decoding")
    st.slider("Max generation length", min_value=40, max_value=260, step=10, key="ui_max_gen_len")
    st.slider("Min generation length", min_value=1, max_value=20, key="ui_min_gen_len")
    st.slider("Repetition penalty", min_value=1.0, max_value=2.0, step=0.05, key="ui_rep_penalty")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset Reader Paths", use_container_width=True):
            st.session_state["ui_reader_encoder"] = DEFAULT_PATHS["reader_encoder"]
            st.session_state["ui_reader_weights"] = DEFAULT_PATHS["reader_weights"]
            st.rerun()
    with c2:
        if st.button("Clear Cache", use_container_width=True):
            clear_pipeline_cache()
            st.rerun()

settings = _build_settings_from_ui()
path_status = validate_settings_paths(settings)

st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.subheader("Connection Status")
for label, status in path_status.items():
    if status == "ok":
        st.success("{}: available".format(label))
    else:
        st.error("{}: missing".format(label))
st.markdown("</div>", unsafe_allow_html=True)

context_source = st.radio("Context Source", options=["PDF Upload", "Manual Context"], horizontal=True)

context_text = ""
selected_pages = []

if context_source == "PDF Upload":
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf is not None:
        pages = extract_pdf_pages(uploaded_pdf.getvalue())
        text_pages = [page_number for page_number, text in pages if text]

        if not text_pages:
            st.error("No extractable text was found in the uploaded PDF.")
        else:
            st.caption("Extracted text from {} pages.".format(len(text_pages)))
            default_pages = text_pages if len(text_pages) <= 6 else text_pages[:6]
            selected_pages = st.multiselect(
                "Select pages used as QA context",
                options=text_pages,
                default=default_pages,
            )
            context_text = combine_selected_pages(pages, selected_pages)

            with st.expander("Context Preview", expanded=False):
                st.write(context_text[:4500] if context_text else "No text selected.")
else:
    context_text = st.text_area(
        "Paste English context",
        placeholder="Paste the context passage in English...",
        height=220,
    )

question_kn = st.text_area(
    "Question in Kannada",
    placeholder="Example: ಅಪೊಲೊ 11 ಮಿಷನ್‌ನ ಕಮಾಂಡರ್ ಯಾರು?",
    height=120,
)

run_clicked = st.button("Run Connected Pipeline", type="primary", use_container_width=True)

if run_clicked:
    if context_source == "PDF Upload" and not selected_pages:
        st.error("Select at least one PDF page.")
    elif not question_kn.strip():
        st.error("Type a Kannada question.")
    elif not context_text.strip():
        st.error("No context is available.")
    elif any(status != "ok" for status in path_status.values()):
        st.error("Some required model paths are missing. Fix sidebar paths and run again.")
    else:
        with st.spinner("Running translation + reader + back-translation..."):
            try:
                result = run_pipeline_on_context(context_text, question_kn.strip(), settings=settings)
            except Exception as err:  # pragma: no cover - UI error path
                st.exception(err)
                st.stop()

        left, right = st.columns([1.3, 1.0])

        with left:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Question Translation")
            st.write("Kannada: {}".format(result["question_kn"]))
            st.write("English: {}".format(result["question_en"]))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Retrieved Evidence")
            st.write(result["highlighted_context_en"])
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Final Answer")
            st.write("English: {}".format(result["answer_en"]))
            st.write("Kannada: {}".format(result["answer_kn"]))
            st.write("Reader score: {}".format(result["answer_score"]))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Runtime")
            st.write("Device: {}".format(result["device"]))
            st.write("KN→EN backend: {}".format(result["kn_en_backend"]))
            st.write("EN→KN backend: {}".format(result["en_kn_backend"]))
            st.markdown("</div>", unsafe_allow_html=True)

        st.success("Pipeline completed.")
