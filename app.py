import os
import time
import json
import re
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
        <div style="text-align:center; padding:30px;">
            <h1 style="color:#124874;">CHATBOT HCMUE</h1>
            <p>Tư vấn quy chế đào tạo – Trường ĐHSP TP.HCM</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# MÀN HÌNH NHẬP API KEY
# =========================
def render_api_key_screen():
    st.set_page_config(page_title=APP_TITLE, layout="centered")

    st.markdown(
        """
        <div style="max-width:480px; margin:auto; padding-top:80px;">
            <h2 style="color:#124874;">🔐 Nhập Google API Key</h2>
            <p style="color:#64748b;">
                API key chỉ được lưu tạm trong phiên làm việc.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_key = st.text_input(
        "Google API Key",
        type="password",
        placeholder="AIza...",
    )

    if st.button("Xác nhận & vào chatbot", use_container_width=True):
        if not api_key.strip():
            st.error("API key không được để trống.")
        else:
            st.session_state.api_key = api_key.strip()
            st.rerun()


# =========================
# CHAT UI HELPER
# =========================
def display_chat_message(role, content, thinking=False):
    label = "Trợ lý HCMUE" if role == "assistant" else "Sinh viên"
    if thinking:
        content = "..."

    st.markdown(
        f"""
        <div style="margin-bottom:12px;">
            <b>{label}:</b><br>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
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
        return False, "Bạn gửi hơi nhanh, vui lòng chờ một chút."

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
    with open(KB_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["content"] for item in data if "content" in item]


@st.cache_resource
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
Trả lời ngắn gọn, đúng trọng tâm.

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

    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)


# =========================
# MAIN
# =========================
def main():
    # ===== 1. BẮT BUỘC NHẬP API KEY TRƯỚC =====
    if "api_key" not in st.session_state:
        render_api_key_screen()
        st.stop()

    api_key = st.session_state.api_key

    # ===== 2. SAU KHI CÓ KEY → CHẠY APP =====
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    render_header()

    st.session_state.setdefault(
        "messages",
        [{"role": "assistant", "content": "Tôi có thể hỗ trợ gì cho bạn?"}],
    )

    for m in st.session_state.messages:
        display_chat_message(m["role"], m["content"])

    vs = load_kb_vectorstore(api_key)
    chain = load_qa_chain_cached(api_key)

    question = st.chat_input("Nhập câu hỏi của bạn...")

    if question:
        ok, msg = allow_request()
        if not ok:
            st.warning(msg)
            return

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        display_chat_message("user", question)

        placeholder = st.empty()
        with placeholder:
            display_chat_message("assistant", "", thinking=True)

        try:
            docs = vs.similarity_search(question, k=TOP_K)
            out = chain(
                {"input_documents": docs, "question": question},
                return_only_outputs=True,
            )
            answer = out.get("output_text", "Không tìm thấy thông tin phù hợp.")

            placeholder.empty()
            display_chat_message("assistant", answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as e:
            placeholder.empty()
            display_chat_message("assistant", f"Lỗi: {e}")


if __name__ == "__main__":
    main()