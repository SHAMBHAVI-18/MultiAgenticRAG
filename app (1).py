import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from backend import RAGOrchestrator, initialize_system

# 1. Page Config
st.set_page_config(page_title="Enterprise Governance Chatbot", page_icon="🔒", layout="wide")
st.title("🔒 Enterprise RAG Chatbot")

# 2. Sidebar - Authentication (Optional)
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password")
    
    st.divider()
    
    st.header("🔐 Authentication")
    # Initialize session state for auth if not exists
    if "auth_status" not in st.session_state:
        st.session_state.auth_status = False
        st.session_state.user_session = "session_user_1"

    # Show Login Form if NOT logged in
    if not st.session_state.auth_status:
        st.info("Log in to access personal data.")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if "orchestrator" in st.session_state:
                result = st.session_state.orchestrator.login(email, password, st.session_state.user_session)
                if result.verified:
                    st.session_state.auth_status = True
                    st.session_state.user_email = email
                    st.success("✅ Logged in!")
                    st.rerun()
                else:
                    st.error("❌ Invalid Credentials")
            else:
                st.warning("System initializing... try again in 5 seconds.")
    
    # Show Logout button if IS logged in
    else:
        st.success(f"👤 Logged in as: {st.session_state.user_email}")
        if st.button("Logout"):
            st.session_state.auth_status = False
            # Log out in backend too
            if "orchestrator" in st.session_state:
                st.session_state.orchestrator.logout(st.session_state.user_session)
            del st.session_state.user_email
            st.rerun()

# 3. Main Chat Interface (ALWAYS VISIBLE)
if not api_key:
    st.warning("Please enter your Google API Key in the sidebar to start chatting.")
    st.stop()

# Initialize System Once
if "orchestrator" not in st.session_state:
    with st.spinner("Initializing Knowledge Base..."):
        try:
            data_df, creds_df, vector_store = initialize_system()
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
            st.session_state.orchestrator = RAGOrchestrator(data_df, creds_df, vector_store, llm)
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I can answer general questions about company policies. For personal data, please log in."}]
        except Exception as e:
            st.error(f"Startup Error: {e}")
            st.stop()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask about policies or your data..."):
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process Query
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.orchestrator.process_query(prompt, st.session_state.user_session)
            st.markdown(response)
    
    # 3. Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
