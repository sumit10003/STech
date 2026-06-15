import os

# Set offline flags BEFORE importing any HF/sentence-transformers modules
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HUGGING_FACE_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import streamlit as st
from src.chatbot import RAGChatbot
from src import config

@st.cache_resource
def get_chatbot():
    """Cache the chatbot to prevent reloading on each Streamlit rerun."""
    return RAGChatbot(
        persist_dir=config.VECTOR_STORE_DIR,
        embedding_model=config.EMBEDDING_MODEL,
    )

# Page config
st.set_page_config(
    page_title="WELCOME - Equipment Documentation Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #111111; /* default readable text color */
    }
    .user-message {
        background-color: #e3f2fd; /* light blue */
        border-left: 4px solid #1976d2;
        color: #0b2545; /* darker blue text for contrast */
    }
    .assistant-message {
        background-color: #f5f5f5; /* light gray */
        border-left: 4px solid #666;
        color: #111111;
    }
    .source {
        background-color: #008fcc; /* deep blue for sources */
        border-left: 4px solid #ff9800;
        color: #111111; /* black text on dark blue */
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .followup {
        background-color: #e8f5e9; /* light green */
        border-left: 4px solid #4caf50;
        color: #0b3d19; /* dark green text */
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Text Based Technical Document AI Assistant")
st.markdown("Ask me anything based on the loaded equipment manuals !")
st.markdown("---")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = get_chatbot()
## recent addition to clear input box
# Counter to force input clear
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

if "selected_equipment" not in st.session_state:
    st.session_state.selected_equipment = "All Equipment"

# Initialize chatbot
chatbot = st.session_state.chatbot

# Get available equipment from vector store
available_equipment = chatbot.get_available_equipment()
equipment_options = ["All Equipment"] + available_equipment

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.container():
            st.markdown(f'<div class="message user-message"><strong>You:</strong> {message["content"]}</div>', 
                       unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f'<div class="message assistant-message"><strong>Assistant:</strong></div>', 
                       unsafe_allow_html=True)
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f'<div class="source"><strong>Source {i}:</strong> {source["text"]}</div>', 
                                   unsafe_allow_html=True)
            
            if "follow_up" in message and message["follow_up"]:
                st.markdown("**Suggested Follow-up Questions:**")
                for q in message["follow_up"]:
                    st.markdown(f'<div class="followup">❓ {q}</div>', unsafe_allow_html=True)
                # Add two blank lines after follow-up questions
                st.markdown("<br><br>", unsafe_allow_html=True)
        # Clear input box
        st.session_state.user_input = ""

        # Add a horizontal line after every assistant response
        st.markdown("___")
        # Add vertical space using a non-breaking space and a break tag with custom height
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# Input section
st.markdown("---")

# Equipment selection
st.markdown("### 🔧 Equipment Filter")

if available_equipment:
    selected_equip = st.selectbox(
        "Choose specific equipment or 'All Equipment' for auto-detection:",
        options=equipment_options,
        index=equipment_options.index(st.session_state.selected_equipment),
        key="equipment_selector",
        help="Select specific equipment to search within, or choose 'All Equipment' to let the system auto-detect from your query."
    )
    st.session_state.selected_equipment = selected_equip
    
    # Show info about current selection
    if selected_equip == "All Equipment":
        st.info(f"🔍 **Mode:** Auto-detect | {len(available_equipment)} equipment available | Mention equipment in query OR search all")
    else:
        st.success(f"🎯 **Mode:** Locked to **{selected_equip}** | All queries search only this equipment")
else:
    st.warning("⚠️ No equipment data available yet. Please ensure documents have been uploaded and vector store has been built with proper naming convention: Equipment_Model_OEM_DocType.pdf")
    st.info("📌 Go to Admin Panel → Step 1: Add Documents → Step 2: Build Vector Store")

st.markdown("---")

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Your question:",
        placeholder="Type your question here...",
        key=f"user_input_{st.session_state.input_counter}",
        max_chars=config.QUERY_CHAR_LIMIT
    )
    # Display character counter with prominent styling
    char_count = len(user_input) if user_input else 0
    st.markdown(
        f'<div style="text-align: right; color: white; font-size: 14px; font-weight: 500; margin-top: -10px; margin-bottom: 10px;">'
        f'📝 {char_count}/{config.QUERY_CHAR_LIMIT} characters'
        f'</div>',
        unsafe_allow_html=True
    )

with col2:
    send_button = st.button("Send", key="send_button")

# Handle user input
if send_button and user_input:
    # Determine equipment filter
    equipment_filter = None
    auto_detected = False
    
    if st.session_state.selected_equipment == "All Equipment":
        # Try to auto-detect equipment from query
        detected_equipment = chatbot.detect_equipment_from_query(user_input)
        if detected_equipment:
            equipment_filter = detected_equipment
            auto_detected = True
    else:
        # Use explicitly selected equipment
        equipment_filter = st.session_state.selected_equipment
    
    # Add user message to history with equipment context
    if equipment_filter:
        if auto_detected:
            equipment_context = f" [Auto-detected: {equipment_filter}]"
        else:
            equipment_context = f" [Selected: {equipment_filter}]"
    else:
        equipment_context = " [All Equipment]"
    
    st.session_state.messages.append({
        "role": "user",
        "content": user_input + equipment_context
    })

    response = {
        "answer": "",
        "sources": [],
        "follow_up_questions": [],
    }
    live_answer = ""
    stream_placeholder = st.empty()

    # Stream response from chatbot with equipment filter
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
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", []),
        "follow_up": response.get("follow_up_questions", [])
    })
    
    # Increment counter to clear the input box
    st.session_state.input_counter += 1
    
    st.rerun()

# Sidebar info
with st.sidebar:
    st.markdown("### 💡 About")
    st.markdown("""
    This is a **Retrieval-Augmented Generation (RAG)** chatbot that answers 
    questions based on documents uploaded through the admin panel.
    
    **Features:**
    - 🔧 **Smart Equipment Filtering**
      - Select specific equipment from dropdown
      - OR mention equipment in your query
      - Auto-detection from query text
    - 📚 Retrieves relevant documents
    - 🤖 AI-powered responses
    - 💬 Follow-up suggestions
    - 📖 Source citations
    
    **How to use:**
    
    **Option 1: Explicit Selection**
    1. Select equipment from dropdown
    2. All queries search only that equipment
    
    **Option 2: Auto-Detection**
    1. Keep dropdown on "All Equipment"
    2. Mention equipment in query:
       - "Radar_3D_BEL display removal"
       - "How to service Laptop_Latitude7330_Dell?"
    3. System auto-detects and filters
    
    **Option 3: Search All**
    1. Keep "All Equipment" selected
    2. Ask general questions
    3. Search across all documentation
    """)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <small>RAG Chatbot v1.0 | Powered by Ollama & FAISS</small>
</div>
""", unsafe_allow_html=True)
