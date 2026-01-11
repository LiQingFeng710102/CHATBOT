import os
import time
import json
import re
import base64
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.question_answering import load_qa_chain


# =========================
# CẤU HÌNH
# =========================
APP_TITLE = "Chat bot hỗ trợ cho sinh viên HCMUE"
APP_SUBTITLE = (
    "Tư vấn Quy chế cho Sinh viên hệ Chính quy "
    "Trường Đại học Sư phạm TP.HCM"
)

APP_DIR = Path(__file__).resolve().parent
KB_JSON_PATH = APP_DIR / "chunks.json"

MODEL_NAME = "gemini-2.5-flash"
EMBED_MODEL = "models/gemini-embedding-001"

MIN_SECONDS_BETWEEN_REQUESTS = 2
MAX_REQUESTS_PER_DAY = 30

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
TOP_K = 4

MAX_OUTPUT_TOKENS = 512
TEMPERATURE = 0.2

# =========================
# HEADER
# =========================
def render_header():
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
        /* ... (CSS nguyên bản của bạn giữ nguyên) ... */
        </style>
        <div class="hcmue-header">
            <h1 style="margin:0; font-size: 42px;">CHATBOT HCMUE</h1>
            <p style="margin:5px 0 0 0; opacity: 0.8; font-size: 18px;">Tư vấn quy chế đào tạo cho sinh viên Trường Đại học Sư phạm Thành phố Hồ Chí Minh</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Helper hiển thị chat
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

    if thinking:
        inner_content = '<div style="font-size:18px; color:#94a3b8; font-style:italic;">...</div>'
    else:
        inner_content = content

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
# Sidebar
# =========================
def render_sidebar_content():
    st.sidebar.markdown(
        """
        <div class="sidebar-header">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: #124874; padding: 8px; border-radius: 10px; color: white;">
                    <i class="fas fa-university" style="font-size: 20px;"></i>
                </div>
                <div>
                    <h2 style="margin:0; font-size: 28px; color: #124874;">CHATBOT HCMUE</h2>
                    <p style="margin:0; font-size: 13px; color: #64748b;">Trợ lý hỗ trợ sinh viên khóa 50</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown('<p class="sidebar-section-title">Hỏi nhanh quy chế</p>', unsafe_allow_html=True)

    quick_questions = [
        ("Điều kiện để xét học bổng", "Điều kiện để xét học bổng ?"),
        ("Cách xin giấy tạm hoãn nghĩa vụ quân sự", "Cách xin giấy tạm hoãn nghĩa vụ quân sự cho sinh viên ?"),
        ("Điều kiện để xét tốt nghiệp", "Điều kiện để xét tốt nghiệp là gì?"),
        ("Điều kiện để bao lưu ? ", " Điều kiện để bao lưu kết quả học tập ?"),
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
        <div style="margin-top: 20px; padding: 15px; background: #f8fafc; border-radius: 12px; border: 1px solid #f1f5f9;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 5px;">
                <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%;"></div>
                <span style="font-size: 13px; font-weight: 700; color: #64748b; text-transform: uppercase;">Hệ thống Online</span>
            </div>
            <p style="font-size: 13px; color: #94a3b8; margin: 0;">Dữ liệu cập nhật dựa trên sổ tay sinh viên khóa 50.</p>
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
# LOAD KB
# =========================
@st.cache_data
def load_kb_texts():
    if not KB_JSON_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {KB_JSON_PATH}")

    with open(KB_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["content"] for item in data if "content" in item]
    if not texts:
        raise ValueError("File JSON không có nội dung hợp lệ.")

    return texts

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
Trả lời ngắn gọn, rõ ràng, đúng trọng tâm của câu hỏi.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
""".strip()

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key, temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# =========================
# QUICK ANSWER
# =========================
def quick_answer(option: str) -> str:
    keyword_map = {
        "Cách xét học bổng": ["học bổng"],
        "Điều kiện xét học bổng": ["điều kiện", "học bổng"],
        "Điều kiện để tốt nghiệp": ["tốt nghiệp"],
        "Điều kiện xét học ngành thứ hai": ["ngành", "thứ hai"],
    }

    keywords = keyword_map.get(option, [])
    if not keywords:
        return "Chưa có thông tin."

    texts = load_kb_texts()
    for text in texts:
        content = text.lower()
        if all(k in content for k in keywords):
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            bullets = ["- " + " ".join(s.split()[:18]) for s in sentences[:3]]
            return "\n".join(bullets)
    return "Không tìm thấy nội dung phù hợp trong quy chế."

# =========================
# RESET CHAT
# =========================
def reset_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể hỗ trợ gì cho các bạn?"}]


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    render_header()
    render_sidebar_content()

    # --- Khởi tạo session_state cho api_key nếu chưa có ---
    st.session_state.setdefault("api_key", "")

    # --- Nhập API key từ người dùng ---
    api_key_input = st.text_input(
        "Nhập GOOGLE_API_KEY của bạn:", type="password", value=st.session_state.api_key
    )

    # Cập nhật session_state khi người dùng nhập
    if api_key_input:
        st.session_state.api_key = api_key_input.strip()

    # Lấy API key từ session_state
    api_key = st.session_state.api_key

    # Nếu key vẫn rỗng, dừng app và thông báo
    if not api_key:
        st.info("Vui lòng nhập API key để tiếp tục.")
        st.stop()

    # Khởi tạo messages nếu chưa có
    st.session_state.setdefault(
        "messages", [{"role": "assistant", "content": "Tôi có thể hỗ trợ gì cho các bạn?"}]
    )
    for m in st.session_state.messages:
        display_chat_message(m["role"], m["content"])

    # Khởi tạo Vector Store
    try:
        vs = load_kb_vectorstore(api_key)
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo VectorStore: {str(e)}")
        st.stop()

    # Khởi tạo QA chain
    try:
        chain = load_qa_chain_cached(api_key)
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo QA Chain: {str(e)}")
        st.stop()

    # Nhận input từ User
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
                answer = out.get(
                    "output_text", "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
                )
                sanitized = re.sub(r"```.*?```", "[mã đã ẩn]", answer, flags=re.S)

                # Hiển thị từng cụm từ cho hiệu ứng chat
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
                    display_chat_message(
                        "assistant", f"Đã xảy ra lỗi: {str(e)}")


if __name__ == "__main__":
    main()