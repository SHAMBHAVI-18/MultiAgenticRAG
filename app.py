import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from backend import RAGOrchestrator, initialize_system

# 1. Page Config
st.set_page_config(page_title="Enterprise Governance Chatbot", page_icon="🔒", layout="wide")
st.title("🔒 Enterprise RAG Chatbot")

# 2. Sidebar - Configuration & Auth
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password")
    if not api_key:
        st.warning("Please enter your Google API Key.")
        st.stop()

    st.divider()
    
    st.header("🔐 Authentication")
    if "user_session" not in st.session_state:
        st.session_state.user_session = "session_user_1"

    if "auth_status" not in st.session_state:
        st.session_state.auth_status = False
    
    if not st.session_state.auth_status:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            # Initialize if not ready
            if "orchestrator" not in st.session_state:
                 # Load system
                data_df, creds_df, vector_store = initialize_system()
                llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
                st.session_state.orchestrator = RAGOrchestrator(data_df, creds_df, vector_store, llm)
            
            # Attempt Login
            result = st.session_state.orchestrator.login(email, password, st.session_state.user_session)
            if result.verified:
                st.session_state.auth_status = True
                st.session_state.user_email = email
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid Credentials")
    else:
        st.success(f"Logged in as: {st.session_state.user_email}")
        if st.button("Logout"):
            st.session_state.auth_status = False
            del st.session_state.user_email
            st.rerun()

# 3. Main Chat Interface
if st.session_state.get("auth_status"):
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "System Online. I am ready to answer HR and Finance queries."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Enter your query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = st.session_state.orchestrator.process_query(prompt, st.session_state.user_session)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please log in using the sidebar to access the chatbot.")
