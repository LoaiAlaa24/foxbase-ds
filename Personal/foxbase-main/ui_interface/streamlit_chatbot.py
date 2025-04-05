import os
import streamlit as st
import requests

BACKEND_URL = os.getenv("BACKEND_URL")

# Set page config with light mode and custom logo
st.set_page_config(page_title="Chatbot", layout="wide", page_icon="assets/logo.png")

st.image("./assets/logo.png", width=150)
st.title("💬 AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message...")
if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Call FastAPI backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        response = requests.post(BACKEND_URL, json={"query": user_input}, stream=True)
        
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    decoded_chunk = chunk.decode("utf-8")
                    full_response += decoded_chunk
                    message_placeholder.markdown(full_response)
        else:
            full_response = "⚠️ Error: Unable to fetch response."
            message_placeholder.markdown(full_response)
    
    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
