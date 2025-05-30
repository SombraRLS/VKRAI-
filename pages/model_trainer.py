import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils.data_processing import load_and_preprocess
from utils.model_training import train_models, transform_test_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def show():
    st.title("🤖 Smart Model Trainer")
    st.markdown("Train and evaluate predictive maintenance models.")
    
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
        st.session_state.best_model = None
    
    uploaded_file = st.file_uploader("Upload training data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = load_and_preprocess(uploaded_file)
        if 'Failure Type' not in df.columns:
            st.error("The dataset must contain the column 'Failure Type'.")
            return
        
        X = df.drop(columns=['Failure Type'])
        y = df['Failure Type']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model_options = {
            "Gradient Boosting": "gb",
            "Random Forest": "rf",
            "Neural Network": "nn",
            "Support Vector Machine": "svm"
        }
        
        selected_models = st.multiselect(
            "Select models to train",
            list(model_options.keys()),
            default=["Gradient Boosting", "Random Forest"]
        )
        
        if st.button("🚀 Train Models"):
            if not selected_models:
                st.warning("Please select at least one model to train.")
                return
            
            with st.spinner("Training models..."):
                # Обучаем модели
                models = train_models(
                    X_train, y_train,
                    [model_options[m] for m in selected_models]
                )
                
                # Трансформируем тестовый набор
                X_test_transformed = transform_test_data(X_test)
                
                # Сохраняем для оценки и визуализации
                st.session_state.X_test = X_test_transformed
                st.session_state.y_test = y_test
                st.session_state.trained_models = models
                
                # Оценка моделей и выбор лучшей
                best_score = -1
                best_model_tuple = None
                for name, model in models.items():
                    score = evaluate_model(model, X_test_transformed, y_test)
                    if score > best_score:
                        best_score = score
                        best_model_tuple = (name, model)
                
                st.session_state.best_model = best_model_tuple
                st.success(f"Model training completed! Best model: {best_model_tuple[0]} (Accuracy: {best_score:.3f})")
                
                show_model_results()
    
    if 'best_model' in st.session_state and st.session_state.best_model:
        st.header("Model Deployment")
        model_name, _ = st.session_state.best_model
        
        st.success(f"Recommended model for deployment: **{model_name}**")
        
        if st.button("🛠️ Prepare for Deployment"):
            st.session_state.deployment_ready = True
            st.experimental_rerun()

def show_model_results():
    st.header("Training Results")
    
    if not st.session_state.trained_models:
        st.warning("No trained models available.")
        return
    
    metrics = []
    for name, model in st.session_state.trained_models.items():
        y_pred = model.predict(st.session_state.X_test)
        report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
        metrics.append({
            "Model": name,
            "Accuracy": report['accuracy'],
            "Precision": report['weighted avg']['precision'],
            "Recall": report['weighted avg']['recall'],
            "F1-Score": report['weighted avg']['f1-score']
        })
    
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-Score": "{:.3f}"
    }))
    # Визуализация
    fig = go.Figure()
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        fig.add_trace(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df[metric],
            name=metric
        ))
    fig.update_layout(
        barmode='group',
        title_text='Model Performance Comparison',
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig)
