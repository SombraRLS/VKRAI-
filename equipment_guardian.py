import streamlit as st
from pages import data_explorer, model_trainer, deployment
from utils import setup_ui

def main():
    # Настройка UI
    setup_ui.configure_page()
    
    # Навигационное меню
    st.sidebar.title("Equipment Guardian")
    
    pages = {
        "🔍 Data Explorer": data_explorer.show,
        "🤖 Model Trainer": model_trainer.show,
        "🚀 Deployment": deployment.show
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[selection]()
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Equipment Guardian** v1.0  
    Predictive maintenance solution  
    © 2023 Industrial AI Team
    """)

if __name__ == "__main__":
    main()
