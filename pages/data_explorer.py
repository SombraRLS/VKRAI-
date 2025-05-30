import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from utils.data_processing import load_and_preprocess

def show():
    st.title("🔍 Equipment Data Explorer")
    st.markdown("Explore your equipment sensor data and analyze failure patterns.")
    
    uploaded_file = st.file_uploader("Upload equipment data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = load_and_preprocess(uploaded_file)
        
        st.success("✅ Data loaded successfully!")
        st.dataframe(df.head(), use_container_width=True)
        
        # Анализ данных
        st.header("Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Failure Distribution")
            fig = px.pie(df, names='Failure Type', 
                         title='Equipment Failure Types Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Feature Correlation")
            corr = df.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        # Детальный анализ признаков
        st.subheader("Feature Analysis")
        selected_feature = st.selectbox("Select feature to analyze", 
                                      df.select_dtypes(include=np.number).columns)
        
        if selected_feature:
            fig = px.histogram(df, x=selected_feature, color='Failure Type',
                              marginal="box", nbins=50,
                              title=f"{selected_feature} Distribution by Failure Type")
            st.plotly_chart(fig, use_container_width=True)
