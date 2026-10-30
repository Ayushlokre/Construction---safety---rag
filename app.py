from __future__ import annotations

import logging
import re
import streamlit as st
import chat as rag

LOGGER = logging.getLogger("osha_rag.streamlit")


# The Streamlit UI intentionally calls the same RAG pipeline as `python chat.py`.
# This avoids maintaining a second, less capable retrieval/answer implementation.


def split_answer_and_sources(answer: str):
    pattern = re.compile(
        r"\[Source:\s*([^,\]]+),\s*PDF page\s*(\d+)\]",
        flags=re.IGNORECASE,
    )
    sources = []
    seen = set()

    def collect(match):
        source_name = match.group(1).strip()
        page_number = match.group(2).strip()
        key = (source_name, page_number)
        if key not in seen:
            seen.add(key)
            sources.append(key)
        return ""

    clean_answer = pattern.sub(collect, answer)
    clean_answer = re.sub(r"[ \t]{2,}", " ", clean_answer)
    clean_answer = re.sub(r" *\n *", "\n", clean_answer)
    clean_answer = re.sub(r"\n{3,}", "\n\n", clean_answer).strip()
    return clean_answer, sources


def is_image_only_request(prompt: str) -> bool:
    """True when the user asks to see a visual without asking for an explanation."""
    q = prompt.lower()
    asks_to_show = bool(re.search(r"\b(show|display|view|see|give|provide)\b", q))
    asks_visual = bool(re.search(r"\b(image|picture|figure|diagram|chart|illustration|page|form|table)\b", q))
    asks_for_explanation = bool(re.search(
        r"\b(explain|describe|summari[sz]e|interpret|compare|what does|how does|tell me about)\b", q
    ))
    asks_for_full_page = bool(re.search(r"\b(full|entire|whole)\s+(pdf\s+)?page\b", q))
    # A request to show a page is visual-only unless the user also asks for
    # explanation/description. The renderer will bypass cropping for full pages.
    return asks_to_show and asks_visual and not asks_for_explanation


def render_answer(answer: str, image_only: bool = False):
    clean_answer, sources = split_answer_and_sources(answer)
    if image_only:
        st.caption("Displaying the original source visual below.")
    else:
        st.markdown(clean_answer or answer)
    if sources and not image_only:
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            for source_name, page_number in sources:
                st.markdown(f"- **{source_name}** — PDF page {page_number}")


def render_requested_source_pages(prompt: str, answer: str):
    """Display complete original PDF pages cited by the answer; never crop them."""
    visual_request = re.search(
        r"\b(show|display|view|see|give|provide|open)\b.{0,100}\b"
        r"(image|picture|figure|diagram|chart|table|page|form|illustration)\b"
        r"|\b(image|picture|figure|diagram|form)\b",
        prompt,
        re.I,
    )
    if not visual_request:
        return

    data_dir = rag.BASE_DIR / "data"
    citations = re.findall(
        r"\[Source:\s*([^,\]]+),\s*PDF page\s*(\d+)\]", answer, re.I
    )
    if not citations:
        st.info(
            "I couldn't identify a verified source page to display. "
            "Try naming the document or figure more specifically."
        )
        return

    try:
        import pymupdf
    except ImportError:
        st.warning("Install the project's requirements to enable PDF page images.")
        return

    seen = set()
    for source_name, page_text in citations:
        key = (source_name.strip(), int(page_text))
        if key in seen:
            continue
        seen.add(key)
        pdf_path = data_dir / key[0]
        if not pdf_path.is_file():
            st.warning(f"Source PDF not found locally: {key[0]}")
            continue

        try:
            with pymupdf.open(pdf_path) as pdf:
                page_index = key[1] - 1
                if page_index < 0 or page_index >= pdf.page_count:
                    st.warning(
                        f"PDF page {key[1]} is outside the page range for {key[0]}."
                    )
                    continue

                page = pdf[page_index]
                # Render the complete source page only. Avoid text-coordinate crops:
                # PDF text bounds can be inaccurate for figures and cause cut-off diagrams.
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(1.7, 1.7),
                    alpha=False,
                )
                with st.expander(
                    f"Complete original PDF page — PDF page {key[1]} — {key[0]}",
                    expanded=True,
                ):
                    st.image(pix.tobytes("png"), use_container_width=True)
        except Exception as exc:
            LOGGER.exception("Could not render source PDF page")
            st.warning(f"Could not render {key[0]} page {key[1]}: {exc}")


def question_cache_key(question: str) -> str:
    """Normalize exact repeat questions for per-session answer reuse."""
    return " ".join(re.findall(r"[a-z0-9]+", (question or "").lower())).strip()


@st.cache_resource
def load_collection():
    return rag.load_collection()


st.set_page_config(
    page_title="OSHA Safety Intelligence",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --osha-orange: #f28c28; --osha-blue: #3b82f6; }
    .block-container { max-width: 1420px; padding-top: 1.6rem; padding-bottom: 6rem; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(148,163,184,.18); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
    .osha-hero {
        padding: 1.45rem 1.6rem; border-radius: 18px; margin-bottom: 1rem;
        background: linear-gradient(115deg, rgba(242,140,40,.18), rgba(59,130,246,.10) 55%, rgba(15,23,42,.18));
        border: 1px solid rgba(148,163,184,.22);
    }
    .osha-kicker { color: #f2a65a; text-transform: uppercase; letter-spacing: .13em;
        font-size: .72rem; font-weight: 750; margin-bottom: .45rem; }
    .osha-title { font-size: clamp(1.65rem, 3vw, 2.45rem); line-height: 1.12;
        font-weight: 800; margin: 0; }
    .osha-subtitle { color: var(--text-color); opacity: .76; margin-top: .55rem; font-size: .96rem; }
    .osha-stat {
        padding: .8rem 1rem; border: 1px solid rgba(148,163,184,.2);
        border-radius: 13px; background: rgba(148,163,184,.055); min-height: 78px;
    }
    .osha-stat-label { font-size: .72rem; opacity: .7; text-transform: uppercase; letter-spacing: .08em; }
    .osha-stat-value { font-size: 1.03rem; font-weight: 750; margin-top: .28rem; }
    .osha-section { font-size: 1.02rem; font-weight: 750; margin: .8rem 0 .25rem; }
    div[data-testid="stChatMessage"] { border: 1px solid rgba(148,163,184,.13);
        border-radius: 14px; padding: .65rem .85rem; margin-bottom: .55rem;
        background: rgba(148,163,184,.035); }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(59,130,246,.035);
    }
    div[data-testid="stButton"] > button {
        border-radius: 11px; min-height: 3rem; text-align: left;
        border: 1px solid rgba(148,163,184,.24); transition: all .15s ease;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: rgba(242,140,40,.8); transform: translateY(-1px);
    }
    div[data-testid="stChatInput"] textarea { border-radius: 13px; }
    .osha-note { font-size: .78rem; opacity: .7; line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}

# Load the persistent collection once. This is read-only; this UI never ingests or rebuilds it.
try:
    collection = load_collection()
    chunk_count = collection.count()
    collection_error = None
except Exception as exc:
    collection = None
    chunk_count = 0
    collection_error = exc

with st.sidebar:
    st.markdown("## 🦺 OSHA Safety")
    st.caption("Local document intelligence")
    st.divider()

    st.markdown("**KNOWLEDGE BASE**")
    st.markdown(
        """
        <div class="osha-note">
        📄 OSHA 3071 · Job Hazard Analysis<br>
        📄 OSHA 3886 · Construction Safety
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**LOCAL MODELS**")
    st.code(f"Embedding  {rag.EMBED_MODEL}\nAnswer     {rag.LLM_MODEL}", language="text")

    if collection_error is None:
        st.success(f"Knowledge base ready · {chunk_count:,} chunks")
    else:
        st.error("Knowledge base unavailable")
        st.caption(str(collection_error))

    st.markdown("**RETRIEVAL PIPELINE**")
    st.markdown(
        """
        <div class="osha-note">
        ✓ Vector search + BM25<br>
        ✓ Reciprocal-rank fusion<br>
        ✓ Page-aware evidence context<br>
        ✓ Citation validation
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "For document Q&A support only. This assistant does not replace applicable "
        "OSHA standards, site-specific procedures, or a qualified safety professional."
    )
    if st.button("＋ New conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.session_state.answer_cache = {}
        st.rerun()

if collection_error is not None:
    st.error(f"Could not load the OSHA knowledge base: {collection_error}")
    st.info(
        "Check that Ollama is running and that the existing `chroma_db` folder "
        "is beside `app.py` and `chat.py`. No re-ingestion is required."
    )
    st.stop()

# Main hero
st.markdown(
    """
    <div class="osha-hero">
      <div class="osha-kicker">LOCAL · PRIVATE · SOURCE-GROUNDED</div>
      <h1 class="osha-title">OSHA Safety Intelligence Assistant</h1>
      <div class="osha-subtitle">
        Ask questions across two OSHA publications. Get concise answers grounded
        in retrieved passages, with PDF page citations and evidence-based refusals.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
stats = [
    ("DOCUMENTS", "02 indexed PDFs"),
    ("KNOWLEDGE BASE", f"{chunk_count:,} chunks"),
    ("RETRIEVAL", "Hybrid · Vector + BM25"),
    ("ANSWER MODEL", rag.LLM_MODEL),
]
for col, (label, value) in zip(metric_cols, stats):
    with col:
        st.markdown(
            f'<div class="osha-stat"><div class="osha-stat-label">{label}</div>'
            f'<div class="osha-stat-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="osha-section">Start with a suggested question</div>', unsafe_allow_html=True)
st.caption("Choose a common question or enter your own below.")
suggestions = [
    "What is a Job Hazard Analysis (JHA), and why is it important?",
    "What are the five questions to ask when conducting a JHA?",
    "What are the hazards associated with casting?",
    "What are the seven core elements of a construction safety and health program?",
    "How should chemical exposure hazards be controlled?",
    "Compare the purpose of the two OSHA documents.",
]
with st.container(border=True):
    suggestion_cols = st.columns(2, gap="medium")
    for index, suggestion in enumerate(suggestions):
        if suggestion_cols[index % 2].button(
            suggestion, key=f"suggestion_{index}", use_container_width=True
        ):
            st.session_state.pending_prompt = suggestion
            st.rerun()

if st.session_state.messages:
    st.markdown('<div class="osha-section">Conversation</div>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(message["content"], image_only=bool(message.get("image_only", False)))
                original_prompt = message.get("prompt", "")
                if original_prompt:
                    render_requested_source_pages(original_prompt, message["content"])
            else:
                st.markdown(message["content"])
else:
    st.markdown(
        """
        <div class="osha-note" style="padding:.6rem .1rem 1rem">
        <b>Good questions to ask:</b> definitions, hazard identification, JHA steps,
        documented controls, construction safety program elements, and comparisons
        supported by the two indexed publications.
        </div>
        """,
        unsafe_allow_html=True,
    )

typed_prompt = st.chat_input("Ask about a hazard, JHA step, control, or OSHA document...")
prompt = st.session_state.pending_prompt or typed_prompt
if prompt:
    st.session_state.pending_prompt = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        cache_key = question_cache_key(prompt)
        cached_answer = st.session_state.answer_cache.get(cache_key) if cache_key else None

        if cached_answer:
            answer = cached_answer
            st.caption("Reusing the previously verified answer for this exact question.")
        else:
            with st.spinner("Searching OSHA passages and checking the supporting evidence..."):
                try:
                    context = rag.retrieve_documents(prompt, collection)
                    answer = rag.generate_answer(prompt, context)
                except Exception as exc:
                    LOGGER.exception("Streamlit question handling failed")
                    answer = (
                        "I couldn't complete the request. Check that Ollama is running "
                        "and see `logs/rag.log` for details.\n\n"
                        f"Technical detail: `{exc}`"
                    )

            # Cache only completed responses, never transient runtime failures.
            if cache_key and not answer.startswith("I couldn't complete the request"):
                st.session_state.answer_cache[cache_key] = answer

        image_only = is_image_only_request(prompt)
        render_answer(answer, image_only=image_only)
        render_requested_source_pages(prompt, answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "prompt": prompt,
        "image_only": image_only,
    })
