import os
import streamlit as st
import requests

st.set_page_config(
    page_title="NorthStar Bank",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
header [data-testid="stSidebarToggle"] {display: none;}

section[data-testid="stSidebar"] {
    background-color: #f3f6fa;
}

section[data-testid="stSidebar"] * {
    color: black;
}

.sp-box {
    background-color:#f9f9f9;
    padding:12px;
    border-radius:10px;
    border-left:4px solid #4f8bf9;
}
</style>
""", unsafe_allow_html=True)


API_BASE_URL = "http://127.0.0.1:8000/api/v1"


if "role" not in st.session_state:
    st.session_state.role = "User"

if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}
    st.session_state.current_chat = "Chat 1"


def render_chunk_content(chunk):
    content = chunk.get("content", "")
    chunk_type = chunk.get("chunk_type")
    image_path = chunk.get("image_path")

    if chunk_type == "image":
        st.markdown("🖼️ **Spend Chart / Visual Insight**")

        if content.strip():
            st.markdown(f"<div class='sp-box'>{content}</div>", unsafe_allow_html=True)

        if image_path:
            if image_path.startswith("http") or image_path.startswith("/"):
                st.image(image_path, caption="Spend Chart")
            elif os.path.exists(image_path):
                st.image(image_path, caption="Spend Chart")
            else:
                st.warning("⚠️ Image not found")

    elif chunk_type == "table":
        st.markdown("📊 **Spend Breakdown Table**")
        st.markdown(f"<div class='sp-box'>{content}</div>", unsafe_allow_html=True)


    else:
        st.markdown(f"<div class='sp-box'>{content}</div>", unsafe_allow_html=True)

with st.sidebar:

    st.session_state.role = st.radio(
        "Mode",
        ["User", "Admin"]
    )

    st.markdown("---")

    if st.button("➕ New Chat", use_container_width=True):
        new_chat = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[new_chat] = []
        st.session_state.current_chat = new_chat
        st.rerun()

    st.markdown("### Chats")
    for chat in st.session_state.chats:
        if st.button(chat, key=chat, use_container_width=True):
            st.session_state.current_chat = chat
            st.rerun()

if st.session_state.role == "Admin":
    st.title("NorthStar Bank")

    uploaded_file = st.file_uploader(
        "Upload document (PDF)",
        type=["pdf"]
    )

    if uploaded_file:
        st.success(f"Selected: {uploaded_file.name}")

        if st.button("Upload Statement"):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            with st.spinner("Uploading..."):
                r = requests.post(f"{API_BASE_URL}/upload", files=files)

            if r.status_code == 200:
                st.success("✅ Upload successful")
            else:
                st.error(r.text)


else:
    st.title("NorthStar Bank")
    st.caption("Ask questions about your credit card usage, EMI, rewards, and spend patterns")

    messages = st.session_state.chats[st.session_state.current_chat]

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    if prompt := st.chat_input("Ask about your credit card spend..."):
        messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing spends..."):
                    r = requests.post(
                        f"{API_BASE_URL}/query",
                        json={"query": prompt}
                    )

                data = r.json() if r.status_code == 200 else {}
                answer = data.get("answer", "⚠️ No insights available.")

            except Exception as e:
                answer = f"❌ Error: {e}"
                data = {}

            st.markdown(answer.replace("\n", "  \n"))

            messages.append({"role": "assistant", "content": answer})
            st.session_state.chats[st.session_state.current_chat] = messages

            retrieved_results = data.get("retrieved_results", [])

            if retrieved_results:
                with st.expander("View Retrieved Chunks", expanded=False):

                    for i, chunk in enumerate(retrieved_results[:5], 1):

                        st.markdown(f"### Chunk {i}")

                        c1, c2, c3 = st.columns(3)
                        c1.caption(f"📄 Page: {chunk.get('page')}")
                        c2.caption(f"📑 Section: {chunk.get('section')}")
                        score = chunk.get("similarity")
                        c3.caption(f"🔍 Relevance Score: {round(score,3) if score else 'N/A'}")

                        render_chunk_content(chunk)


                        st.markdown("---")