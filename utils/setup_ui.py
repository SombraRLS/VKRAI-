import streamlit as st

def configure_page():
    """Настройка страницы Streamlit"""
    st.set_page_config(
        page_title="Equipment Guardian",
        page_icon="🛠️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Пользовательские стили CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
