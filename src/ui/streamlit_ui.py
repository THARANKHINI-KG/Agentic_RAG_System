
import streamlit as st
import requests

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Regulatory Compliance Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS =================
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
header [data-testid="stSidebarToggle"] {display: none;}
section[data-testid="stSidebar"] {
    background-color: #f5f2e6;
}
section[data-testid="stSidebar"] * {
    color: black;
}
</style>
""", unsafe_allow_html=True)

# ================= CONFIG =================
API_BASE_URL = "http://127.0.0.1:8000/api/v1"

# ================= SESSION STATE =================
if "role" not in st.session_state:
    st.session_state.role = "User"

if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}
    st.session_state.current_chat = "Chat 1"

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## 💬 Chat Panel")

    # Role switch
    st.session_state.role = st.radio("Access Level", ["User", "Admin"])

    st.markdown("---")

    # New chat
    if st.button("➕ New Chat", use_container_width=True):
        new_chat_name = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        st.rerun()

    st.markdown("### Chats")

    # Chat list
    for chat_name in st.session_state.chats:
        if st.button(chat_name, key=chat_name, use_container_width=True):
            st.session_state.current_chat = chat_name
            st.rerun()

# ================= MAIN CONTENT =================

# ===== ADMIN VIEW =====
if st.session_state.role == "Admin":
    st.title("📂 Document Management")

    uploaded_file = st.file_uploader(
        "Upload regulatory document",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file:
        st.success(f"Selected: {uploaded_file.name}")

        if st.button("Upload"):
            try:
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
                    st.error(f"❌ Upload failed: {r.status_code}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ===== USER VIEW =====
else:
    st.title("🤖 Regulatory Compliance Intelligence")
    st.caption(f"Current chat: {st.session_state.current_chat}")

    messages = st.session_state.chats[st.session_state.current_chat]

    # Display chat history
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a regulatory compliance question"):

        # Save user message
        messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):

            try:
                with st.spinner("Thinking... ⏳"):
                    r = requests.post(
                        f"{API_BASE_URL}/query",
                        json={"query": prompt}
                    )

                if r.status_code == 200:
                    data = r.json()

                    # ✅ Extract clean answer
                    answer = data.get("answer") or "⚠️ No answer found"

                    # ✅ Extract citation info
                    document_name = data.get("document_name")
                    page_no = data.get("page_no")
                    policy_text = data.get("policy_citations")

                else:
                    answer = f"❌ API error: {r.status_code}"
                    document_name = None
                    page_no = None
                    policy_text = None

            except requests.exceptions.Timeout:
                answer = "⏳ Request timed out. Backend is taking too long."
                document_name = None
                page_no = None
                policy_text = None

            except requests.exceptions.ConnectionError:
                answer = "❌ Cannot connect to backend. Is FastAPI running?"
                document_name = None
                page_no = None
                policy_text = None

            except Exception as e:
                answer = f"❌ Error: {str(e)}"
                document_name = None
                page_no = None
                policy_text = None

            # ================= DISPLAY =================

            # ✅ Proper markdown formatting (fix bullets)
            formatted_answer = answer.replace("\n", "  \n")
            st.markdown(formatted_answer)

            # ✅ Safe citation display (no NameError)
            if document_name or page_no:
                caption_text = f"{document_name or 'Unknown Document'}"
                if page_no:
                    caption_text += f" | Page {page_no}"
                st.caption(f"📄 {caption_text}")


            # ================= SAVE =================
            messages.append({"role": "assistant", "content": answer})
            st.session_state.chats[st.session_state.current_chat] = messages

            # ================= DISPLAY =================

            # ✅ Proper markdown formatting
            formatted_answer = answer.replace("\n", "  \n")
            st.markdown(formatted_answer)

            # ✅ Citation (existing)
            if document_name or page_no:
                caption_text = f"{document_name or 'Unknown Document'}"
                if page_no:
                    caption_text += f" | Page {page_no}"
                st.caption(f"📄 {caption_text}")

            # ================= NEW: RERANKED CHUNKS DROPDOWN =================

            retrieved_results = data.get("retrieved_results", [])

            if retrieved_results:
                with st.expander("📚 View Retrieved Chunks (Reranked)", expanded=False):

                    for i, chunk in enumerate(retrieved_results[:5], 1):

                        st.markdown(f"### 🔹 Chunk {i}")

                        # Metadata row
                        col1, col2, col3 = st.columns(3)
                        col1.caption(f"📄 Page: {chunk.get('page')}")
                        col2.caption(f"📑 Section: {chunk.get('section')}")
                        col3.caption(f"📊 Score: {chunk.get('similarity')}")

                        # Content
                        st.markdown(
                            f"<div style='background-color:#f9f9f9;padding:10px;border-radius:8px'>"
                            f"{chunk.get('content')}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        # Optional image support
                        if chunk.get("chunk_type") == "image" and chunk.get("image_path"):
                            st.image(chunk.get("image_path"), caption="Extracted Image")

                        st.markdown("---")