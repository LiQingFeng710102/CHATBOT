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
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
            .stApp { background-color: #f8f9fa; }

            /* Header Style: khung trắng, chữ xanh */
            .hcmue-header {
                background-color: #ffffff;
                color: #f1f5f0;
                padding: 2rem;
                border-radius: 0 0 24px 24px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }
            .hcmue-header h1, .hcmue-header p { color: #124874; }

            /* Change top black bar (Streamlit banner) to HCMUE blue */
            header[role="banner"], header, .css-1v3fvcr, .css-13l3y2h, .stToolbar {
                background-color: #124874 !important;
                color: #ffffff !important;
            }
            [data-testid="stHeader"], [data-testid="stAppViewContainer"] > header {
                background-color: #124874 !important;
                color: #ffffff !important;
            }
            [data-testid="stHeader"]::before {
                content: "TRƯỜNG ĐẠI HỌC SƯ PHẠM THÀNH PHỐ HỒ CHÍ MINH"; 
                position: absolute;
                left: 60px; 
                font-size: 36px;
                font-weight: 600;
                color: #ffffff;
                z-index: 1;
                line-height: 2.8rem;
            }
            [data-testid="stChatInput"] {
                background-color: transparent !important; 
                border: none !important; /* Xóa viền ngoài */
                box-shadow: none !important; /* Xóa bóng đổ ngoài */
                padding: 10px !important;
            }
            
            [data-testid="stChatInput"] > div {
                background-color: transparent !important;
                border: none !important;
            }
            [data-testid="stChatInput"] textarea {
                background-color: #124874 !important;
                color: #124874 !important;
                -webkit-text-fill-color: #124874 !important;
                border: none !important;
            }
            [data-testid="stBottomBlockContainer"], 
            [data-testid="stBottom"], 
            .stChatInputContainer, 
            .stChatFooter {
                background-color: #124874 !important;
                background: #124874 !important;
            }
            
            footer { display: none !important; }
            [data-testid="stHeader"] { background-color: #124874 !important; }
            footer,
            .stFooter,
            [data-testid="stFooter"],
            [data-testid="stAppViewContainer"] footer,
            .stApp .main footer {
                background-color: #ffffff !important;
                color: #124874 !important;
                border-top: 1px solid #e6edf3 !important;
            }

            /* Phần tử nhập thực tế giữ nguyên để làm khung chat chính */
            [data-testid="stChatInput"] textarea,
            [data-testid="stChatInput"] input,
            [data-testid="stChatInput"] div[role="textbox"],
            .stChatInput textarea,
            .stChatInput input,
            .stChatInput [contenteditable="true"],
            .stTextInput input,
            .stTextArea textarea {
                background-color: #ffffff !important;
                color: #124874 !important;
                caret-color: #124874 !important;
                border: 1px solid #e6edf3 !important;
                border-radius: 12px !important;
                padding: 8px 12px !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important; 
                outline: none !important;
            }

            [data-testid="stChatInput"] ::placeholder,
            .stChatInput ::placeholder,
            .stTextInput ::placeholder,
            .stTextArea ::placeholder {
                color: #94a3b8 !important;
                opacity: 1 !important;
            }

            .stChatInput, .stChatInput * {
                background: transparent !important;
            }

            .stButton>button, .stButton>button[type="submit"] {
                background-color: #0d3658 !important;
                color: #ffffff !important;
                border-radius: 999px !important;
                padding: 6px 12px !important;
                border: none !important;
            }
            [data-testid="stSidebar"], .css-1lcbmhc, .css-1aumxhk {
                background-color: #ffffff !important;
                color: #124874 !important;
            }
            [data-testid="stSidebarHeader"] button {
                color: #124874 !important;
            }
            /* Message Container */
            .chat-msg-container {
                display: flex;
                width: 100%;
                margin-bottom: 1.5rem;
            }
            .justify-start { justify-content: flex-start; }
            .justify-end { justify-content: flex-end; }

            .msg-bubble {
                max-width: 100%;
                display: flex;
                flex-direction: column;
            }
            .items-start { align-items: flex-start; }
            .items-end { align-items: flex-end; }

            .msg-info {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 6px;
            }
            .flex-row-reverse { flex-direction: row-reverse; }
            
            .avatar {
                width: 35px;
                height: 35px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: bold;
            }
            .bot-avatar { bg-color: #124874; color: white; background-color: #124874; }
            .user-avatar { bg-color: #e2e8f0; color: #475569; background-color: #e2e8f0; }
            
            .role-label {
                font-size: 11px;
                font-weight: 700;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .content-bubble {
                width: 100%;
                padding: 12px 20px;
                border-radius: 18px;
                font-size: 15px;
                line-height: 1.6;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .bot-content {
                background-color: white;
                color: #1e293b;
                border: 1px solid #f1f5f9;
                border-top-left-radius: 2px;
            }
            .user-content {
                background-color: #124874;
                color: white;
                border-top-right-radius: 2px;
            }

            .dot-flashing {
                display: flex;
                gap: 4px;
                padding: 4px 0;
            }
            .dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background-color: #60a5fa;
                animation: bounce 1.5s infinite linear;
            }
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-5px); }
            }

            .stApp .main .block-container {
                max-width: 100% !important;
                padding-left: 0 !important;
                padding-right: 0 !important;
                margin: 0 auto !important;
            }
        </style>
        <div class="hcmue-header">
            <h1 style="margin:0; font-size: 42px;">CHATBOT HCMUE</h1>
            <p style="margin:5px 0 0 0; opacity: 0.8; font-size: 18px;">Tư vấn quy chế đào tạo cho sinh viên Trường Đại học Sư phạm Thành phố Hồ Chí Minh</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

#===============tạo hàm helper=======================
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
        # Hiển thị đơn giản "..." thay vì animation
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
#=============================
def render_sidebar_content():
    # Header Sidebar
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

    # Section: Hỏi nhanh
    st.sidebar.markdown('<p class="sidebar-section-title">Gợi ý một số câu hỏi</p>', unsafe_allow_html=True)

    quick_questions = [
        ("Cảnh báo học vụ", "Điều kiện bị cảnh báo học tập và buộc thôi học là gì?"),
        ("Giới hạn tín chỉ", "Giới hạn tín chỉ tối thiểu tối đa mỗi học kỳ được quy định thế nào?"),
        ("Tạm dừng học", "Sinh viên được tạm dừng học trong những trường hợp nào?"),
        ("Điều kiện tốt nghiệp ", " Điều kiện để được xét công nhận tốt nghiệp là gì?"),
    ]

    for label, query in quick_questions:
        if st.sidebar.button(label, key=f"btn_{label}", use_container_width=True):
            st.session_state.sidebar_selection = query
            st.rerun()
    st.sidebar.divider()
    # Nút làm mới hội thoại
    if st.sidebar.button("Làm mới cuộc hội thoại", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể hỗ trợ gì cho các bạn?"}]
        st.rerun()
    st.sidebar.divider()
    # Footer Sidebar
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=api_key,
    )

    return FAISS.from_texts(chunks, embedding=embeddings)


@st.cache_resource
def load_qa_chain_cached(api_key: str):
    prompt_template = """
Bạn là trợ lý hỗ trợ sinh viên.
Trả lời ngắn gọn, rõ ràng, dễ hiểu, phải dựa vào NGỮ CẢNH được cung cấp, và phải nó nó nằm ở phần nào trong tài liệu.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
""".strip()

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    return load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=prompt,
    )


# =========================
# QUICK ANSWER
# =========================
def quick_answer(option: str) -> str:
    keyword_map = {
        "Cách báo học vụ": ["cảnh báo học vụ", "buộc thôi học"],
        "Giới hạn tín chỉ": ["giới hạn", "tín chỉ"],
        "Tạm dừng học tập": ["tạm dừng", "học tập"],
        "Điều kiện tốt nghiệp": ["điều kiện", "tốt nghiệp"],
    }

    keywords = keyword_map.get(option, [])
    if not keywords:
        return "Chưa có thông tin."

    texts = load_kb_texts()

    for text in texts:
        content = text.lower()
        if all(k in content for k in keywords):
            sentences = [
                s.strip()
                for s in text.split(".")
                if s.strip()
            ]
            bullets = [
                "- " + " ".join(s.split()[:18])
                for s in sentences[:3]
            ]
            return "\n".join(bullets)

    return "Không tìm thấy nội dung phù hợp trong quy chế."


# =========================
# RESET CHAT
# =========================
def reset_chat():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Tôi có thể hỗ trợ gì cho các bạn?",
        }
    ]


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    render_header()
    render_sidebar_content()
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get(
        "GOOGLE_API_KEY"
    )

    if not api_key:
        st.error("Chưa cấu hình GOOGLE_API_KEY.")
        st.stop()

    st.session_state.setdefault(
        "messages",
        [
            {
                "role": "assistant",
                "content": "Tôi có thể hỗ trợ gì cho các bạn?",
            }
        ],
    )
    for m in st.session_state.messages:
        display_chat_message(m["role"], m["content"])

    # ... các bước xử lý vector store ...

# 1. Khởi tạo Vector Store và Chain (Làm trước khi nhận input)
    vs = load_kb_vectorstore(api_key)
    chain = load_qa_chain_cached(api_key)

    # 2. CHỈ DÙNG MỘT THANH NHẬP LIỆU DUY NHẤT
    # 2. XỬ LÝ ĐẦU VÀO (TỪ CHAT INPUT HOẶC SIDEBAR)
    prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    # Kiểm tra xem có dữ liệu từ Sidebar gửi qua không
    if "sidebar_selection" in st.session_state and st.session_state.sidebar_selection:
        question = st.session_state.sidebar_selection
        # Xóa ngay sau khi lấy để tránh lặp lại khi rerun lần sau
        del st.session_state.sidebar_selection
    else:
        question = prompt

    # 3. NẾU CÓ CÂU HỎI (TỪ BẤT KỲ NGUỒN NÀO) THÌ XỬ LÝ
    if question:
        # Kiểm tra giới hạn yêu cầu (Spam)
        ok, msg = allow_request()
        if not ok:
            st.warning(msg)
        else:
            # A. Thêm câu hỏi của User vào lịch sử và hiển thị ngay
            st.session_state.messages.append({"role": "user", "content": question})
            display_chat_message("user", question)

            # B. Tạo khung trống (placeholder) để xử lý hiệu ứng "Thinking"
            placeholder = st.empty()
            
            # Hiển thị trạng thái đang xử lý
            with placeholder:
                display_chat_message("assistant", "", thinking=True)
            
            # C. Logic xử lý AI (RAG)
            try:
                # Tìm kiếm nội dung liên quan trong file JSON
                docs = vs.similarity_search(question, k=TOP_K)
                
                # Chạy Chain để trả lời dựa trên ngữ cảnh
                out = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                answer = out.get("output_text", "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong quy chế.")
            except Exception as e:
                answer = f"Đã xảy ra lỗi khi xử lý: {str(e)}"

            # D. Xóa hiệu ứng Thinking và thay bằng hiệu ứng "suy luận" (typing)
            # Ẩn các khối mã để tránh hiển thị code thô
            try:
                sanitized = re.sub(r"```.*?```", "[mã đã ẩn]", answer, flags=re.S)
            except Exception:
                sanitized = answer

            # Hiển thị progressive typing trong cùng một placeholder
            try:
                for i in range(1, len(sanitized) + 1):
                    partial = sanitized[:i]
                    placeholder.empty()
                    with placeholder:
                        display_chat_message("assistant", partial)
                    time.sleep(0.01)
            except Exception:
                # Fallback: hiển thị toàn bộ nếu có lỗi khi animate
                placeholder.empty()
                with placeholder:
                    display_chat_message("assistant", sanitized)
            
            # E. Lưu câu trả lời của Bot vào lịch sử hội thoại
            st.session_state.messages.append({"role": "assistant", "content": sanitized})

            # F. Force Rerun để cập nhật lại giao diện hoàn chỉnh (tuỳ chọn)
            # st.rerun()
if __name__ == "__main__":
    main()