# app.py
import os
import hashlib
from types import SimpleNamespace
from typing import List, Dict, Union

import gradio as gr
from langchain_community.retrievers import BM25Retriever

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow

from config import constants
from config.settings import settings
from utils.logging import logger


# =========================
# Examples (local files)
# =========================
EXAMPLES = {
    "Google 2024 Environmental Report": {
        "question": (
            "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. "
            "Also retrieve regional average CFE in Asia pacific in 2023"
        ),
        "file_paths": ["examples/google-2024-environmental-report.pdf"],
    },
    "DeepSeek-R1 Technical Report": {
        "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
        "file_paths": ["examples/DeepSeek Technical Report.pdf"],
    },
}


# =========================
# Helpers
# =========================
def _to_path(obj: Union[str, os.PathLike, object]) -> str:
    """
    Return a filesystem path for either:
      - string/Path-like (examples)
      - Gradio UploadedFile / tempfile-like objects with .name
    """
    if isinstance(obj, (str, os.PathLike)):
        return str(obj)
    name = getattr(obj, "name", None)
    if not name:
        raise ValueError("Unsupported file item; expected a path string or file-like with .name")
    return name


def _ensure_with_name(uploaded_files: List) -> List:
    """
    Ensure each item has a .name attribute for DocumentProcessor.
    If an item is a plain string path, wrap it with SimpleNamespace(name=path).
    """
    normalized = []
    for f in uploaded_files:
        if hasattr(f, "name"):
            normalized.append(f)
        else:
            normalized.append(SimpleNamespace(name=_to_path(f)))
    return normalized


def _get_file_hashes(uploaded_files: List) -> frozenset:
    """
    Generate SHA-256 hashes for uploaded files (paths or file-like).
    Uses streaming to avoid loading huge files into memory.
    """
    hashes = set()
    for f in uploaded_files:
        path = _to_path(f)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        hashes.add(h.hexdigest())
    return frozenset(hashes)


def _hash_set_to_session_id(hashes: frozenset) -> str:
    """
    Stable 12-char session ID from the set of file hashes, for per-upload Chroma isolation.
    """
    s = "".join(sorted(hashes))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _load_example(example_key: str):
    """
    For a given example key, return (files, clinician_note, question) for the Gradio UI.
    """
    if not example_key or example_key not in EXAMPLES:
        return [], "", ""

    ex = EXAMPLES[example_key]
    q = ex["question"]
    file_paths = []
    for p in ex["file_paths"]:
        if os.path.exists(p):
            file_paths.append(p)
        else:
            logger.warning(f"File not found: {p}")
    # examples are not “clinician notes”; keep that empty
    return file_paths, "", q


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


HEADERS_VERDICTS = ["Claim ID", "Claim (truncated)", "Verdict", "Top Citations"]
HEADERS_RISKS = ["Code", "Where", "Why", "Remedy"]


def _format_verdicts_table(verdicts: List[dict], claims: List[dict]) -> List[List[str]]:
    claim_text = {c.get("id"): c.get("text", "") for c in (claims or [])}
    rows: List[List[str]] = []
    for v in verdicts or []:
        cid = v.get("claim_id", "")
        rows.append([
            cid,
            _truncate(claim_text.get(cid, ""), 120),
            v.get("verdict", ""),
            ", ".join((v.get("citations") or [])[:2]) or "",
        ])
    return rows


def _format_risk_flags_table(risks: List[dict]) -> List[List[str]]:
    rows: List[List[str]] = []
    for r in risks or []:
        rows.append([
            r.get("code", ""),
            r.get("where", ""),
            _truncate(r.get("why", ""), 160),
            _truncate(r.get("remedy", ""), 160),
        ])
    return rows


# =========================
# Gradio App
# =========================
def main():
    # Friendly guard for missing API key
    if not getattr(settings, "OPENAI_API_KEY", None):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env before launching the app.")

    # Instantiate core components once
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()

    # Theming & small style tweaks
    css = """
    .title {
        font-size: 1.5em !important; 
        text-align: center !important;
        color: #FFD700; 
    }
    .subtitle {
        font-size: 1em !important; 
        text-align: center !important;
        color: #FFD700; 
    }
    .text { text-align: center; }
    """
    js = """
    function createGradioAnimation() {
        var container = document.createElement('div');
        container.id = 'gradio-animation';
        container.style.fontSize = '2em';
        container.style.fontWeight = 'bold';
        container.style.textAlign = 'center';
        container.style.marginBottom = '20px';
        container.style.color = '#eba93f';

        var text = 'Welcome to RAGChat Studio 🐥!';
        for (var i = 0; i < text.length; i++) {
            (function(i){
                setTimeout(function(){
                    var letter = document.createElement('span');
                    letter.style.opacity = '0';
                    letter.style.transition = 'opacity 0.1s';
                    letter.innerText = text[i];
                    container.appendChild(letter);
                    setTimeout(function() { letter.style.opacity = '0.9'; }, 50);
                }, i * 250);
            })(i);
        }
        var gradioContainer = document.querySelector('.gradio-container');
        gradioContainer.insertBefore(container, gradioContainer.firstChild);
        return 'Animation created';
    }
    """

    with gr.Blocks(theme=gr.themes.Citrus(), title="RAGChat Studio 🐥", css=css, js=js) as demo:
        gr.Markdown("## RAGChat Studio: powered by Docling 🐥 and LangGraph", elem_classes="subtitle")
        gr.Markdown("# How it works ✨:", elem_classes="title")
        gr.Markdown("📤 Upload your document(s), add a clinician note (or a question), then hit Submit 📝", elem_classes="text")
        gr.Markdown("Or select one of the examples, click Load Example, then Submit 📝", elem_classes="text")
        gr.Markdown("⚠️ **Note:** Accepted formats: '.pdf', '.docx', '.txt', '.md'", elem_classes="text")

        # Session state for caching retriever per file set
        session_state = gr.State({"file_hashes": frozenset(), "retriever": None, "session_id": None})

        with gr.Row():
            with gr.Column():
                # Examples
                gr.Markdown("### Example 📂")
                example_dropdown = gr.Dropdown(
                    label="Select an Example 🐥",
                    choices=list(EXAMPLES.keys()),
                    value=None,
                )
                load_example_btn = gr.Button("Load Example 🛠️")

                # Inputs
                files = gr.Files(label="📄 Upload Documents", file_types=constants.ALLOWED_TYPES)
                clinician_note = gr.Textbox(
                    label="🩺 Clinician Note",
                    lines=8,
                    placeholder="Paste clinician-written product/note here for QA & safe rewrite..."
                )
                question = gr.Textbox(
                    label="❓ Question (optional)",
                    lines=3,
                    placeholder="Optionally ask a question about your uploaded docs"
                )
                bm25_only = gr.Checkbox(label="BM25 only (skip embeddings)", value=True)
                submit_btn = gr.Button("Submit 🚀")

            with gr.Column():
                answer_output = gr.Textbox(label="📝 Rewritten/Safe Note (or Answer)", interactive=False)
                verification_output = gr.Textbox(label="✅ Verification Report", interactive=False)

                verdicts_df = gr.Dataframe(
                    label="Per-claim Verdicts",
                    headers=HEADERS_VERDICTS,
                    row_count=0,
                    col_count=len(HEADERS_VERDICTS),
                    interactive=False,
                )
                risk_flags_df = gr.Dataframe(
                    label="Risk Flags",
                    headers=HEADERS_RISKS,
                    row_count=0,
                    col_count=len(HEADERS_RISKS),
                    interactive=False,
                )

        # Load example handler (fills files + question; keeps clinician note empty)
        load_example_btn.click(
            fn=_load_example,
            inputs=[example_dropdown],
            outputs=[files, clinician_note, question],
        )

        # Main processing
        def process_inputs(note_text: str, question_text: str, uploaded_files: List, state: Dict, bm25_flag: bool):
            try:
                if not uploaded_files:
                    raise ValueError("❌ No documents uploaded")
                if not (note_text and note_text.strip()) and not (question_text and question_text.strip()):
                    raise ValueError("❌ Provide a clinician note or a question")

                current_hashes = _get_file_hashes(uploaded_files)
                new_session_id = _hash_set_to_session_id(current_hashes)

                # Build / reuse retriever for this specific file set
                if state["retriever"] is None or current_hashes != state["file_hashes"]:
                    logger.info("Processing new/changed documents...")
                    normalized_files = _ensure_with_name(uploaded_files)
                    chunks = processor.process(normalized_files)
                    if not chunks:
                        raise ValueError(
                            "❌ No readable text was extracted from your files. "
                            "If these are scanned PDFs, try uploading a text/markdown version."
                        )

                    if bm25_flag:
                        logger.info("Using BM25-only retriever")
                        retriever = BM25Retriever.from_documents(chunks)
                    else:
                        # Per-upload isolation via session_id
                        retriever = retriever_builder.build_hybrid_retriever(chunks, session_id=new_session_id)

                    state.update({"file_hashes": current_hashes, "retriever": retriever, "session_id": new_session_id})

                # Fallback prompt if only clinician note is provided
                qtext = (question_text or "").strip() or "Check and rewrite the clinician note."

                result = workflow.full_pipeline(
                    question=qtext,
                    retriever=state["retriever"],
                    clinician_note=note_text or qtext,
                )

                # Prefer rewritten_note if available; else draft_answer
                answer_text = result.get("rewritten_note") or result.get("draft_answer") or ""
                verif_text = result.get("verification_report") or ""

                # Build tables
                verdict_rows = _format_verdicts_table(result.get("verdicts", []), result.get("claims", []))
                risk_rows = _format_risk_flags_table(result.get("risk_flags", []))

                return answer_text, verif_text, verdict_rows, risk_rows, state

            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                return f"❌ Error: {str(e)}", "", [], [], state

        submit_btn.click(
            fn=process_inputs,
            inputs=[clinician_note, question, files, session_state, bm25_only],
            outputs=[answer_output, verification_output, verdicts_df, risk_flags_df, session_state],
        )

    # Single launch with a queue (works across Gradio versions)
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=5000,
        share=False,   # turn on later if needed
        debug=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
