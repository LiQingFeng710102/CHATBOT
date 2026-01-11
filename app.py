import os
import time
import json
import re
from datetime import date
from pathlib import Path

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.question_answering import load_qa_chain

# =========================
# CẤU HÌNH
# =========================
APP_TITLE = "Chat bot hỗ trợ cho sinh viên HCMUE"
APP_DIR = Path(__file__).resolve().parent
KB_JSON_PATH = APP_DIR / "chunks.json"

MODEL_NAME = "gemini-2.5-flash"
EMBED_MODEL = "models/gemini-embedding-001"

MIN_SECONDS_BETWEEN_REQUESTS = 2
MAX_REQUESTS_PER_DAY = 30

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
TOP_K = 4
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.2

# =========================
# HEADER + LOGO
# =========================
def render_header():
    # Header bây giờ dùng logo sidebar cũ
    logo_path = APP_DIR / "Logo HCMUE - Gia tri cot loi 2.png"
    if logo_path.exists():
        st.image(str(logo_path), width=120)
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
            .stApp { background-color: #f8f9fa; }

            /* Header */
            .hcmue-header {
                background-color: #ffffff;
                color: #124874;
                padding: 2rem;
                border-radius: 0 0 24px 24px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }
            .hcmue-header h1 { color: #124874; font-size: 42px; margin:0; }
            .hcmue-header p { color: #124874; opacity:0.8; font-size:18px; margin:5px 0 0 0; }

            /* Chat bubbles */
            .chat-msg-container { display: flex; width: 100%; margin-bottom: 1.5rem; }
            .justify-start { justify-content: flex-start; }
            .justify-end { justify-content: flex-end; }
            .msg-bubble { max-width: 100%; display: flex; flex-direction: column; }
            .items-start { align-items: flex-start; }
            .items-end { align-items: flex-end; }
            .msg-info { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
            .flex-row-reverse { flex-direction: row-reverse; }
            .avatar { width: 35px; height: 35px; border-radius: 12px; display:flex; align-items:center; justify-content:center; font-size:14px; font-weight:bold; }
            .bot-avatar { background-color:#124874; color:white; }
            .user-avatar { background-color:#e2e8f0; color:#475569; }
            .role-label { font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; }
            .content-bubble { width:100%; padding:12px 20px; border-radius:18px; font-size:15px; line-height:1.6; box-shadow:0 1px 3px rgba(0,0,0,0.1); }
            .bot-content { background-color:white; color:#1e293b; border-top-left-radius:2px; }
            .user-content { background-color:#124874; color:white; border-top-right-radius:2px; }
            footer { display:none !important; }
            .stButton>button { background-color:#0d3658 !important; color:#fff !important; border-radius:999px; padding:6px 12px; border:none !important; }
        </style>
        <div class="hcmue-header">
            <h1>CHATBOT HCMUE</h1>
            <p>Tư vấn quy chế đào tạo cho sinh viên Trường Đại học Sư phạm TP.HCM</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# HELPER CHAT
# =========================
def display_chat_message(role, content, thinking=False):
    is_bot = role == "assistant"
    justify = "justify-start" if is_bot else "justify-end"
    items = "items-start" if is_bot else "items-end"
    row_dir = "" if is_bot else "flex-row-reverse"
    avatar_class = "bot-avatar" if is_bot else "user-avatar"
    icon = '<i class="fas fa-robot"></i>' if is_bot else '<i class="fas fa-user-graduate"></i>'
    label = "Trợ lý HCMUE" if is_bot else "Sinh viên"
    bubble_class = "bot-content" if is_bot else "user-content"
    inner_content = '<div style="font-size:18px; color:#94a3b8; font-style:italic;">...</div>' if thinking else content
    html = f"""
    <div class="chat-msg-container {justify}">
        <div class="msg-bubble {items}">
            <div class="msg-info {row_dir}">
                <div class="avatar {avatar_class}">{icon}</div>
                <span class="role-label">{label}</span>
            </div>
            <div class="content-bubble {bubble_class}">
                {inner_content}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
def render_sidebar_content():
    # Sidebar bây giờ dùng logo header cũ
    logo_path = APP_DIR / "Logo HCMUE.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=60)
    st.sidebar.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:15px;">
            <div style="font-size:20px; font-weight:600; color:#124874;">CHATBOT HCMUE</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- API key ---
    st.sidebar.markdown('<p class="sidebar-section-title">Nhập API Key của bạn</p>', unsafe_allow_html=True)
    st.session_state.setdefault("api_key", "")
    api_key_input = st.sidebar.text_input("GOOGLE_API_KEY", type="password", value=st.session_state.api_key)
    if api_key_input:
        st.session_state.api_key = api_key_input.strip()
    if not st.session_state.api_key:
        st.sidebar.warning("Vui lòng nhập API key để sử dụng chatbot.")

    # --- Quick questions ---
    st.sidebar.markdown('<p class="sidebar-section-title">Hỏi nhanh</p>', unsafe_allow_html=True)
    quick_questions = [
        ("Điều kiện để xét học bổng", "Điều kiện để xét học bổng ?"),
        ("Cách xin giấy tạm hoãn nghĩa vụ quân sự", "Cách xin giấy tạm hoãn nghĩa vụ quân sự ?"),
        ("Điều kiện để xét tốt nghiệp", "Điều kiện để xét tốt nghiệp ?"),
        ("Điều kiện để bao lưu ?", "Điều kiện để bao lưu ?"),
    ]
    for label, query in quick_questions:
        if st.sidebar.button(label, key=f"btn_{label}", use_container_width=True):
            st.session_state.sidebar_selection = query
            st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("Làm mới cuộc hội thoại", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể hỗ trợ gì cho các bạn?"}]
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        <div style="margin-top:20px;padding:15px;background:#f8fafc;border-radius:12px;border:1px solid #f1f5f9;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px;">
                <div style="width:8px;height:8px;background:#10b981;border-radius:50%;"></div>
                <span style="font-size:13px;font-weight:700;color:#64748b;text-transform:uppercase;">Hệ thống Online</span>
            </div>
            <p style="font-size:13px;color:#94a3b8;margin:0;">Dữ liệu cập nhật dựa trên sổ tay sinh viên khóa 50.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# CHỐNG SPAM
# =========================
def allow_request():
    now = time.time()
    today = str(date.today())
    st.session_state.setdefault("last_req", 0.0)
    st.session_state.setdefault("count_today", 0)
    st.session_state.setdefault("day", today)
    if st.session_state["day"] != today:
        st.session_state["day"] = today
        st.session_state["count_today"] = 0
    if now - st.session_state["last_req"] < MIN_SECONDS_BETWEEN_REQUESTS:
        return False, "Bạn gửi hơi nhanh, vui lòng chờ một chút nhé."
    if st.session_state["count_today"] >= MAX_REQUESTS_PER_DAY:
        return False, "Bạn đã hết lượt hỏi hôm nay."
    st.session_state["last_req"] = now
    st.session_state["count_today"] += 1
    return True, ""

# =========================
# LOAD KB, VectorStore, QA Chain
# =========================
@st.cache_data
def load_kb_texts():
    if not KB_JSON_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {KB_JSON_PATH}")
    with open(KB_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["content"] for item in data if "content" in item]

@st.cache_resource(show_spinner=True)
def load_kb_vectorstore(api_key: str):
    texts = load_kb_texts()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key)
    return FAISS.from_texts(chunks, embedding=embeddings)

@st.cache_resource
def load_qa_chain_cached(api_key: str):
    prompt_template = """
Bạn là trợ lý hỗ trợ sinh viên.
Trả lời đầy đủ, rõ ràng, đúng trọng tâm.
Luôn trích dẫn NGỮ CẢNH.
Nếu không có thông tin trong NGỮ CẢNH, hãy nói "Không tìm thấy thông tin phù hợp".

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
""".strip()
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key,
                                 temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    render_header()
    render_sidebar_content()

    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.stop()

    st.session_state.setdefault("messages", [{"role": "assistant", "content": "Tôi có thể hỗ trợ gì cho các bạn?"}])
    for m in st.session_state.messages:
        display_chat_message(m["role"], m["content"])

    vs = load_kb_vectorstore(api_key)
    chain = load_qa_chain_cached(api_key)

    prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")
    if "sidebar_selection" in st.session_state and st.session_state.sidebar_selection:
        question = st.session_state.sidebar_selection
        del st.session_state.sidebar_selection
    else:
        question = prompt

    if question:
        ok, msg = allow_request()
        if not ok:
            st.warning(msg)
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            display_chat_message("user", question)

            placeholder = st.empty()
            with placeholder:
                display_chat_message("assistant", "", thinking=True)

            try:
                docs = vs.similarity_search(question, k=TOP_K)
                out = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                answer = out.get("output_text", "Xin lỗi, tôi không tìm thấy thông tin phù hợp.")
                sanitized = re.sub(r"```.*?```", "[mã đã ẩn]", answer, flags=re.S)

                words = sanitized.split(" ")
                full_display = ""
                for i in range(len(words)):
                    full_display += words[i] + " "
                    if i % 3 == 0 or i == len(words) - 1:
                        with placeholder:
                            display_chat_message("assistant", full_display.strip())
                        time.sleep(0.01)

                st.session_state.messages.append({"role": "assistant", "content": sanitized})

            except Exception as e:
                placeholder.empty()
                with placeholder:
                    display_chat_message("assistant", f"Đã xảy ra lỗi: {str(e)}")

if __name__ == "__main__":
    main()