import os

# Set offline flags BEFORE importing any HF/sentence-transformers modules
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HUGGING_FACE_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import streamlit as st
from src.admin import AdminManager
from src.chatbot import RAGChatbot
from src import config

# Custom CSS to create the bottom bar
footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: black;
    color: white;
    text-align: right;
    padding: 10px 20px;
    font-size: 14px;
    border-top: 2px solid #4CAF50; /* This adds the small border at the top of the bar */
    z-index: 999;
}
</style>

<div class="footer">
    Developed By : Sumit Gupta
</div>
"""

# Single entry app: choose Admin or User
st.set_page_config(
    page_title="S-TECH",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# basic styles
st.markdown(
    """
    <style>
    .stButton>button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Inject the HTML
st.markdown(footer_html, unsafe_allow_html=True)

st.title("📚 S-TECH")
st.write("## Welcome - Equipment Documentation AI Assistant")

# Initialize admin manager using config paths
admin = AdminManager(data_dir=config.ADMIN_DATA_DIR, vector_store_dir=config.VECTOR_STORE_DIR)

# Single status area for progress updates (one place only)
status_placeholder = st.empty()

def update_status(message: str, level: str = "info"):
    """Show a single-line status update in the shared placeholder.

    level: one of 'info', 'success', 'error', 'warning', or 'plain'
    """
    if level == "info":
        status_placeholder.info(message)
    elif level == "success":
        status_placeholder.success(message)
    elif level == "error":
        status_placeholder.error(message)
    elif level == "warning":
        status_placeholder.warning(message)
    else:
        status_placeholder.write(message)


def safe_rerun():
    """Attempt to rerun the Streamlit script using whatever API is available.

    Tries `st.experimental_rerun()`, then `st.rerun()`, then falls back to
    toggling query params which forces a rerun in older/newer Streamlit versions.
    """
    try:
        # Preferred (older/newer versions)
        if hasattr(st, "experimental_rerun"):
            return st.experimental_rerun()
        if hasattr(st, "rerun"):
            return st.rerun()
    except Exception:
        pass

    try:
        import time
        st.experimental_set_query_params(_rerun=int(time.time()))
    except Exception:
        # As a last resort stop the script (no-op but prevents crash)
        try:
            st.stop()
        except Exception:
            pass

# session state for role and admin flow
if "role" not in st.session_state:
    st.session_state.role = None
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "admin_step" not in st.session_state:
    st.session_state.admin_step = 0

# Note: Ollama is already initialized by main.py launcher

def show_user_ui():
    """Embedded user chat UI (same behavior as standalone)."""
    st.title("🤖 Technical Document AI Assistant")
    st.markdown("Ask me anything based on the loaded equipment manuals !")
    st.markdown("---")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot(persist_dir=config.VECTOR_STORE_DIR, embedding_model=config.EMBEDDING_MODEL)

    chatbot = st.session_state.chatbot

    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0
    if "selected_equipment" not in st.session_state:
        st.session_state.selected_equipment = "All Equipment"

    available_equipment = chatbot.get_available_equipment()
    equipment_options = ["All Equipment"] + available_equipment

    st.markdown("### 🔧 Equipment Filter")
    if available_equipment:
        selected_equip = st.selectbox(
            "Choose specific equipment or 'All Equipment' for auto-detection:",
            options=equipment_options,
            index=equipment_options.index(st.session_state.selected_equipment),
            key="equipment_selector",
            help=(
                "Select specific equipment to search within, or choose "
                "'All Equipment' to let the system auto-detect from your query."
            ),
        )
        st.session_state.selected_equipment = selected_equip

        if selected_equip == "All Equipment":
            st.info(
                f"🔍 **Mode:** Auto-detect | {len(available_equipment)} equipment available"
            )
        else:
            st.success(
                f"🎯 **Mode:** Locked to **{selected_equip}** | All queries search only this equipment"
            )
    else:
        st.warning(
            "⚠️ No equipment data available yet. Upload documents and build the vector store first."
        )
        st.info("📌 Admin Panel → Step 1: Add Documents → Step 2: Build Vector Store")

    st.markdown("---")

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container():
                st.markdown(f'<div class="message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f'<div class="message assistant-message"><strong>Assistant:</strong></div>', unsafe_allow_html=True)
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f'<div class="source"><strong>Source {i}:</strong> {source["text"]}</div>', unsafe_allow_html=True)
                if "follow_up" in message and message["follow_up"]:
                    st.markdown("**Suggested Follow-up Questions:**")
                    for q in message["follow_up"]:
                        st.markdown(f'<div class="followup">❓ {q}</div>', unsafe_allow_html=True)

    # Input section
    st.markdown("---")
    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_input(
            "Your question:",
            placeholder="Type your question here...",
            key=f"user_input_{st.session_state.input_counter}",
            max_chars=config.QUERY_CHAR_LIMIT,
        )
        # Display character counter
        char_count = len(user_input) if user_input else 0
        st.markdown(
            f'<div style="text-align: right; color: white; font-size: 14px; font-weight: 500; margin-top: -10px; margin-bottom: 10px;">'
            f'📝 {char_count}/{config.QUERY_CHAR_LIMIT} characters'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        send_button = st.button("Send", key="send_button")

    if send_button and user_input:
        equipment_filter = None
        auto_detected = False

        if st.session_state.selected_equipment == "All Equipment":
            detected_equipment = chatbot.detect_equipment_from_query(user_input)
            if detected_equipment:
                equipment_filter = detected_equipment
                auto_detected = True
        else:
            equipment_filter = st.session_state.selected_equipment

        if equipment_filter:
            equipment_context = (
                f" [Auto-detected: {equipment_filter}]" if auto_detected else f" [Selected: {equipment_filter}]"
            )
        else:
            equipment_context = " [All Equipment]"

        st.session_state.messages.append(
            {"role": "user", "content": user_input + equipment_context}
        )

        response = {
            "answer": "",
            "sources": [],
            "follow_up_questions": [],
        }
        live_answer = ""
        stream_placeholder = st.empty()

        with st.spinner("Thinking..."):
            for event in chatbot.stream_response(
                user_input,
                top_k=3,
                equipment_name=equipment_filter,
            ):
                if event.get("type") == "token":
                    live_answer += event.get("content", "")
                    stream_placeholder.markdown(live_answer + "▌")
                elif event.get("type") == "final":
                    response = {
                        "answer": event.get("answer", live_answer),
                        "sources": event.get("sources", []),
                        "follow_up_questions": event.get("follow_up_questions", []),
                    }

        stream_placeholder.empty()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
                "follow_up": response.get("follow_up_questions", []),
            }
        )
        st.session_state.input_counter += 1
        safe_rerun()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### 💡 About")
        st.markdown("""
        The Project is developed by **Col SP Gupta**. This is a **Retrieval-Augmented Generation (RAG)** chatbot that answers
        questions based on technical knowledge gained from the technical documents uploaded through the admin panel.
        """)
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            safe_rerun()
        if st.button("🔄 Reset"):
            st.session_state.clear()
            safe_rerun()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <small>RAG Chatbot v1.0 | Powered by Ollama & FAISS</small>
    </div>
    """, unsafe_allow_html=True)

# Role selection UI
if st.session_state.role is None:
    st.markdown("## Choose your role")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👨‍💼 Admin"):
            st.session_state.role = "admin"
            safe_rerun()
    with col2:
        if st.button("🤖 User"):
            st.session_state.role = "user"
            safe_rerun()

elif st.session_state.role == "user":
    show_user_ui()

elif st.session_state.role == "admin":
    st.header("🔐 Admin Login")
    if not st.session_state.admin_authenticated:
        uid = st.text_input("User ID", value="")
        pwd = st.text_input("Password", value="", type="password")
        if st.button("Login"):
            if uid == "sumit_admin" and pwd == "sumit@password":
                st.session_state.admin_authenticated = True
                st.success("Login successful")
                safe_rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.subheader("Admin Commands")
        st.markdown("Follow the steps below to manage documents and the vector store.")

        # Step 1: Add documents
        st.markdown("### 1️⃣ Add Documents")
        
        # Step 1 Status
        if st.session_state.admin_step == 0:
            st.info("📝 Ready to add documents. Click the button below.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            step1_disabled = st.session_state.admin_step > 1
            if st.button("Add Documents", disabled=step1_disabled, key="add_docs_btn"):
                st.session_state.admin_step = 1
                st.session_state.step1_status = "in_progress"
                safe_rerun()

        # Show if files have been uploaded in this step
        if st.session_state.admin_step >= 1:
            files = admin.get_uploaded_files()
            if files:
                st.success(f"✅ {len(files)} file(s) ready for import")
            
            st.markdown("#### Upload Files")
            st.markdown(f"**Supported Formats:** {', '.join(admin.get_supported_formats())}")
            uploaded_files = st.file_uploader("Choose files to upload", type=admin.get_supported_formats(), accept_multiple_files=True, key="file_upload")
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    update_status(f"Saving {uploaded_file.name}...")
                    file_path = os.path.join(admin.data_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    update_status(f"✅ {uploaded_file.name} uploaded successfully!", level="success")
            
            files = admin.get_uploaded_files()
            st.write(f"**Current files in system:** {len(files)}")
            if files:
                for f in files:
                    st.write(f"  • {f['name']} ({f['size']//1024} KB)")

            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("✅ Done - Proceed to Build Vector Store", key="done_docs_btn"):
                    st.session_state.admin_step = 2
                    st.session_state.step1_status = "completed"
                    update_status("Step 1 completed. Ready to build vector store.", level="success")
                    safe_rerun()
        
        # Step 1 completion status
        if hasattr(st.session_state, 'step1_status'):
            if st.session_state.step1_status == "completed":
                st.success("✅ Step 1: Documents uploaded and ready")

        st.markdown("---")

        # Step 2: Build vector store
        st.markdown("### 2️⃣ Create / Manage Vector Store")
        
        # Step 2 Status
        if st.session_state.admin_step < 2:
            st.info("📌 Complete Step 1 first - add documents")
        else:
            status = admin.check_vector_store_status()
            if status.get("exists"):
                st.success("✅ Vector store exists and is ready")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Rebuild Vector Store", key="rebuild_btn"):
                        # Create progress containers
                        progress_bar = st.progress(0, text="Starting...")
                        status_text = st.empty()
                        
                        def progress_callback(step, total, message):
                            progress = int((step / total) * 100)
                            progress_bar.progress(progress, text=f"Step {step}/{total}: {message}")
                            status_text.info(f"⏳ {message}")
                        
                        with st.spinner("🔨 Building vector store..."):
                            result = admin.build_vector_store(progress_callback=progress_callback)
                        
                        progress_bar.progress(100, text="Complete!")
                        
                        if result["success"]:
                            st.session_state.rebuild_status = "success"
                            status_text.success(f"✅ {result['message']}")
                            st.balloons()
                        else:
                            st.session_state.rebuild_status = "failed"
                            status_text.error(f"❌ {result['message']}")
                        safe_rerun()
                
                with col2:
                    if st.button("🗑️ Delete Vector Store", key="delete_btn"):
                        import shutil
                        update_status("⏳ Deleting vector store...")
                        if os.path.exists(admin.vector_store_dir):
                            shutil.rmtree(admin.vector_store_dir)
                        update_status("✅ Vector store deleted", level="success")
                        st.session_state.admin_step = 1
                        safe_rerun()
                
                # Show rebuild status if just completed
                if hasattr(st.session_state, 'rebuild_status'):
                    if st.session_state.rebuild_status == "success":
                        st.success("✅ Vector store rebuild completed successfully!")
                    elif st.session_state.rebuild_status == "failed":
                        st.error("❌ Vector store rebuild failed. Check the messages above.")
            else:
                st.warning("⚠️ No vector store found yet.")
                if st.button("🚀 Build Vector Store Now", key="build_btn"):
                    # Create progress containers
                    progress_bar = st.progress(0, text="Initializing...")
                    status_text = st.empty()
                    
                    def progress_callback(step, total, message):
                        progress = int((step / total) * 100)
                        progress_bar.progress(progress, text=f"Step {step}/{total}: {message}")
                        status_text.info(f"⏳ {message}")
                    
                    with st.spinner("🔨 Processing documents and building vector store..."):
                        result = admin.build_vector_store(progress_callback=progress_callback)
                    
                    progress_bar.progress(100, text="Complete!")
                    
                    if result["success"]:
                        st.session_state.vector_store_built = True
                        status_text.success(f"✅ {result['message']}")
                        st.balloons()
                    else:
                        status_text.error(f"❌ {result['message']}")
                    safe_rerun()
                
                if hasattr(st.session_state, 'vector_store_built'):
                    if st.session_state.vector_store_built:
                        st.success("✅ Vector store created successfully!")

        st.markdown("---")

        # Step 3: Status / Summary
        st.markdown("### 3️⃣ System Status & Summary")
        
        if st.button("📊 Show Detailed Status", key="status_btn"):
            st.session_state.admin_step = 3
            safe_rerun()
        
        if st.session_state.admin_step >= 3:
            st.subheader("System Status")
            files = admin.get_uploaded_files()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Total Files", len(files))
            
            if files:
                ftypes = {}
                for f in files:
                    ftypes[f['type']] = ftypes.get(f['type'], 0) + 1
                st.write("**File Types:**")
                for k, v in ftypes.items():
                    st.write(f"  • {k.upper()}: {v}")
            
            vs = admin.check_vector_store_status()
            col1, col2 = st.columns(2)
            with col1:
                if vs.get("exists"):
                    st.success("✅ Vector Store: Ready")
                else:
                    st.warning("⚠️ Vector Store: Not Ready")
            
            # Show equipment statistics if vector store exists
            if vs.get("exists"):
                try:
                    from src.vectorstore import FaissVectorStore
                    temp_vs = FaissVectorStore(persist_dir="vector_store")
                    temp_vs.load()
                    equipment_list = temp_vs.get_available_equipment()
                    
                    st.markdown("---")
                    st.write("**📦 Equipment in Knowledge Base:**")
                    st.metric("Total Equipment", len(equipment_list))
                    
                    if equipment_list:
                        st.write("**Equipment List:**")
                        for eq in equipment_list:
                            # Count documents for this equipment
                            doc_count = sum(1 for m in temp_vs.metadata if m.get('equipment_name') == eq)
                            st.write(f"  • **{eq}**: {doc_count} chunks")
                except Exception as e:
                    st.warning(f"Could not load equipment statistics: {str(e)}")

        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col3:
            if st.button("🚪 Logout Admin"):
                st.session_state.admin_authenticated = False
                st.session_state.role = None
                st.session_state.admin_step = 0
                update_status("✅ Logged out successfully", level="success")
                safe_rerun()

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray; padding: 1rem;'><small>RAG Admin Panel v1.0</small></div>", unsafe_allow_html=True)
