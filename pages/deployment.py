import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
try:
    from utils.data_processing import preprocess_input
except ImportError:
    from .utils.data_processing import preprocess_input

def show():
    st.title("🚀 Model Deployment Center")
    st.markdown("Deploy your trained model for real-time predictions.")
    
    if 'deployment_ready' not in st.session_state:
        st.session_state.deployment_ready = False
    
    if not st.session_state.deployment_ready:
        st.warning("Please train and select a model first.")
        return
    
    st.success("Your model is ready for deployment!")
    
    deployment_option = st.radio(
        "Select deployment method:",
        ["Local REST API", "Cloud Deployment", "Edge Device"]
    )
    
    st.header("Test Your Model")
    
    with st.form("prediction_form"):
        st.subheader("Equipment Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            equipment_type = st.selectbox(
                "Equipment Type", 
                ["Type A", "Type B", "Type C"]
            )
            temperature = st.number_input(
                "Temperature (K)", 
                min_value=250.0, max_value=400.0, value=298.0
            )
            vibration = st.number_input(
                "Vibration Level", 
                min_value=0.0, max_value=10.0, value=2.5
            )
        
        with col2:
            pressure = st.number_input(
                "Pressure (kPa)", 
                min_value=50.0, max_value=200.0, value=101.3
            )
            runtime = st.number_input(
                "Runtime Hours", 
                min_value=0, max_value=10000, value=500
            )
            maintenance = st.number_input(
                "Days Since Last Maintenance", 
                min_value=0, max_value=365, value=30
            )
        
        if st.form_submit_button("Predict Failure"):
            input_data = pd.DataFrame([{
                "Type": equipment_type,
                "Temperature": temperature,
                "Vibration": vibration,
                "Pressure": pressure,
                "Runtime": runtime,
                "Maintenance": maintenance
            }])
            
            processed_data = preprocess_input(input_data)
            model = st.session_state.best_model[1]
            
            prediction = model.predict(processed_data)
            proba = model.predict_proba(processed_data)
            
            failure_types = ["None", "Mechanical", "Electrical", "Thermal", "Other"]
            
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Predicted Failure Type", 
                    failure_types[prediction[0]]
                )
            
            with col2:
                st.metric(
                    "Confidence", 
                    f"{max(proba[0])*100:.1f}%"
                )
            
            # Визуализация вероятностей
            proba_df = pd.DataFrame({
                "Failure Type": failure_types,
                "Probability": proba[0]
            }).sort_values("Probability", ascending=False)
            
            fig = px.bar(
                proba_df, 
                x="Failure Type", 
                y="Probability",
                text="Probability",
                title="Failure Probability Distribution"
            )
            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    if deployment_option == "Local REST API":
        st.header("Local REST API Deployment")
        st.code("""
        # Sample Python code to deploy your model with Flask
        from flask import Flask, request, jsonify
        import joblib
        
        app = Flask(__name__)
        model = joblib.load('equipment_model.pkl')
        
        @app.route('/predict', methods=['POST'])
        def predict():
            data = request.get_json()
            prediction = model.predict([data['features']])
            return jsonify({
                'prediction': int(prediction[0]),
                'confidence': float(max(model.predict_proba([data['features']])[0]))
            })
            
        if __name__ == '__main__':
            app.run(host='0.0.0.0', port=5000)
        """)
        
        # Кнопка для скачивания модели
        model_bytes = BytesIO()
        joblib.dump(st.session_state.best_model[1], model_bytes)
        st.download_button(
            label="📥 Download Trained Model",
            data=model_bytes.getvalue(),
            file_name="equipment_model.pkl",
            mime="application/octet-stream"
        )
