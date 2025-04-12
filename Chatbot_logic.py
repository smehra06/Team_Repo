# chatbot_logic.py
import streamlit as st

def chatbot_page():
    st.title("💬 Chatbot")
    st.write("Ask a question, and I'll try my best to answer it!")

    # Predefined responses for simplicity
    responses = {
        "hello": "Hi there! How can I assist you today?",
        "how are you": "I'm just code, but I'm doing great! 😊",
        "bye": "See you next time! 👋",
        "what is this project about": "This app predicts delays in food hamper deliveries using machine learning!",
        "default": "Hmm... I didn't understand that. Can you rephrase?"
    }

    # User input box
    user_input = st.text_input("You:", key="user_input")

    # Show response
    if user_input:
        response = responses.get(user_input.lower(), responses["default"])
        st.text_area("Chatbot:", value=response, height=100, max_chars=None)
