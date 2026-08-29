"""
AI Video Assistant — Streamlit UI
Wraps `run_pipeline()` from main.py: YouTube/local file -> transcript ->
title/summary/action items/decisions/questions -> RAG chat.

Run with:
    streamlit run app.py
"""

import os

# Must be set before yt_dlp is imported anywhere in the app (this is the
# entry point, so it runs first). Disables yt-dlp's plugin discovery so a
# stray/incompatible third-party plugin cached in the environment can't
# crash extraction. See audio_processor.py for details.
os.environ.setdefault("YTDLP_NO_PLUGINS", "1")

import tempfile
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from main import run_pipeline
from core.rag_engine import ask_question
from utils.audio_processor import YouTubeDownloadBlockedError

load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# STYLING
# =============================================================================
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .main .block-container {
        padding-top: 1.6rem;
        max-width: 1180px;
    }

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 45%, #db2777 100%);
        padding: 2.1rem 2.4rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .hero h1 { margin: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.02em; }
    .hero p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 0.96rem; }

    .meta-pill {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-top: 0.7rem;
        margin-right: 0.4rem;
    }

    .stat-card {
        background: rgba(127,127,127,0.08);
        border: 1px solid rgba(127,127,127,0.22);
        border-radius: 16px;
        padding: 1.1rem 1rem;
        text-align: center;
        transition: transform 0.15s ease;
    }
    .stat-card .num { font-size: 1.7rem; font-weight: 800; color: inherit; }
    .stat-card .lbl { font-size: 0.74rem; opacity: 0.65; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.1rem; color: inherit; }

    .section-card {
        background: rgba(127,127,127,0.06);
        border: 1px solid rgba(127,127,127,0.2);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 0.9rem;
    }
    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        opacity: 0.85;
        margin-bottom: 0.6rem;
    }

    div[data-testid="stChatMessage"] { border-radius: 14px; }

    .empty-state {
        text-align: center;
        padding: 4rem 1rem;
        opacity: 0.55;
    }
    .empty-state .big-icon { font-size: 2.6rem; margin-bottom: 0.4rem; }

    .step-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-size: 0.82rem; opacity: 0.75; padding: 0.15rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
for key, val in {
    "result": None,
    "chat_history": [],
    "history": [],   # list of past {title, timestamp, result} for quick recall
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =============================================================================
# HERO
# =============================================================================
st.markdown("""
<div class="hero">
    <h1>🎬 AI Video Assistant</h1>
    <p>Drop a YouTube link or a local recording — get a transcript, summary, action items, decisions, and a chat that knows the whole meeting.</p>
    <span class="meta-pill">🎙️ Whisper transcription</span>
    <span class="meta-pill">🧠 Mistral summarization</span>
    <span class="meta-pill">💬 RAG-powered chat</span>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ New Analysis")

    input_mode = st.radio("Source type", ["YouTube URL", "Local file upload"])

    source, uploaded_file = None, None
    if input_mode == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    else:
        uploaded_file = st.file_uploader(
            "Upload audio/video",
            type=["mp4", "mp3", "wav", "m4a", "mov", "mkv", "webm"],
        )

    language = st.selectbox(
        "Language", ["english", "hinglish"],
        help="English → local Whisper. Hinglish → Sarvam AI (auto-translates to English).",
    )

    st.divider()
    run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if st.session_state.result:
        if st.button("🗑️ Clear current results", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

    if st.session_state.history:
        st.divider()
        st.caption("📂 Recent analyses")
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            if st.button(f"↺ {h['title'][:32]}", key=f"hist_{i}", use_container_width=True):
                st.session_state.result = h["result"]
                st.session_state.chat_history = []
                st.rerun()

    st.divider()
    st.caption("Pipeline: download/load → transcribe → summarize → extract → index for chat")

# =============================================================================
# RUN PIPELINE
# =============================================================================
if run_clicked:
    resolved_source = None

    if input_mode == "YouTube URL":
        if not source or not source.strip():
            st.sidebar.error("Please enter a YouTube URL.")
        else:
            resolved_source = source.strip()
    else:
        if uploaded_file is None:
            st.sidebar.error("Please upload a file.")
        else:
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            resolved_source = tmp_path

    if resolved_source:
        st.session_state.chat_history = []
        steps = [
            "📥 Loading source & extracting audio",
            "✂️ Chunking audio",
            "🎙️ Transcribing",
            "🧠 Summarizing & extracting insights",
            "🔎 Building searchable index",
        ]
        with st.status("Running the pipeline…", expanded=True) as status:
            placeholder = st.empty()
            placeholder.markdown("\n".join(f"- {s}" for s in steps))
            try:
                t0 = time.time()
                result = run_pipeline(resolved_source, language=language)
                elapsed = round(time.time() - t0, 1)
                status.update(label=f"Done in {elapsed}s ✅", state="complete", expanded=False)

                st.session_state.result = result
                st.session_state.history.append({
                    "title": result.get("title", "Untitled"),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "result": result,
                })
            except YouTubeDownloadBlockedError as e:
                status.update(label="Failed ❌", state="error", expanded=True)
                st.error(
                    "🚫 YouTube blocked this download. This is common on cloud-hosted "
                    "apps — YouTube restricts datacenter IPs.\n\n"
                    "**Try instead:** switch to **Local file upload** in the sidebar "
                    "and upload the video/audio file directly."
                )
            except Exception as e:
                status.update(label="Failed ❌", state="error", expanded=True)
                st.error(f"Pipeline error: {e}")
        if st.session_state.result:
            st.rerun()

# =============================================================================
# MAIN CONTENT
# =============================================================================
result = st.session_state.result

if result is None:
    st.markdown("""
    <div class="empty-state">
        <div class="big-icon">👋</div>
        <h3>No analysis yet</h3>
        <p>Add a YouTube URL or upload a file in the sidebar, then hit <b>Run Analysis</b>.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    transcript = result.get("transcript", "")
    summary = result.get("summary", "")
    action_items = result.get("action_items", "")
    decisions = result.get("key_decisions", "")
    questions = result.get("open_questions", "")

    st.subheader(f"📌 {result.get('title', 'Untitled')}")

    NONE_FOUND_MARKERS = (
        "no action items found",
        "no key decisions found",
        "no open questions found",
        "none found",
    )

    def _count(x):
        if isinstance(x, list):
            return len(x)
        if isinstance(x, str) and x.strip():
            text = x.strip()
            if text.lower().rstrip(".") in [m.rstrip(".") for m in NONE_FOUND_MARKERS]:
                return 0
            return len([l for l in text.split("\n") if l.strip()])
        return 0

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [
        (c1, len(transcript.split()) if transcript else 0, "Words transcribed"),
        (c2, _count(action_items), "Action items"),
        (c3, _count(decisions), "Key decisions"),
        (c4, _count(questions), "Open questions"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="num">{num}</div>
                <div class="lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Decisions", "❓ Questions", "📝 Transcript", "💬 Chat"]
    )

    def render_block(content, empty_msg="Nothing found."):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if isinstance(content, list) and content:
            for item in content:
                st.markdown(f"- {item}")
        elif isinstance(content, str) and content.strip():
            st.markdown(content)
        else:
            st.caption(empty_msg)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_summary:
        render_block(summary, "No summary generated.")
        st.download_button(
            "⬇️ Download summary (.md)", data=summary or "",
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )

    with tab_actions:
        render_block(action_items, "No action items found.")

    with tab_decisions:
        render_block(decisions, "No key decisions found.")

    with tab_questions:
        render_block(questions, "No open questions found.")

    with tab_transcript:
        st.text_area("Full transcript", transcript, height=420, label_visibility="collapsed")
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=transcript,
            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )

    with tab_chat:
        st.caption("Ask questions about this video — answered via retrieval over the transcript.")

        chat_box = st.container(height=430)
        with chat_box:
            if not st.session_state.chat_history:
                st.caption("No messages yet. Try: *\"What were the main decisions?\"*")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        user_q = st.chat_input("Ask a question about this video…")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            rag_chain = result.get("rag_chain")
            try:
                answer = ask_question(rag_chain, user_q) if rag_chain else "RAG chain unavailable for this session."
            except Exception as e:
                answer = f"⚠️ Error answering question: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()