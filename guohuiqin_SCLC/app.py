import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import warnings

# Suppress TensorFlow deprecated warnings (optional)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure matplotlib for better visualization
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 12

# Set page config with a custom icon
st.set_page_config(
    page_title="Small Cell Lung Cancer Immunotherapy Response Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {background-color: #f5f6f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stNumberInput>label {font-weight: bold; color: #2c3e50;}
    .sidebar .sidebar-content {background-color: #ecf0f1;}
    h1 {color: #2c3e50; text-align: center;}
    h2 {color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🧬 Small Cell Lung Cancer Immunotherapy Response Prediction")
st.markdown("""
    This tool uses transcriptome data to predict immunotherapy response for small cell lung cancer and provides interpretable insights 
    using SHAP (SHapley Additive exPlanations) visualizations. Adjust gene expression levels in the sidebar 
    to see how they influence the prediction.
""")

# Load and prepare background data
@st.cache_data
def load_background_data():
    # Get the absolute path to the data file
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data', 'train_5.xlsx')
    df = pd.read_excel(data_path)
    return df[['JAG2', 'PLAU', 'CXCL2', 'TNC', 'FGF6']]

# Load the pre-trained model
@st.cache_resource
def load_model():
    # Get the absolute path to the model file
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'data', 'MODEL_2025_04_15_19_41_16.h5')
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model from {model_path}. Error: {str(e)}")

# Initialize data and model
try:
    background_data = load_background_data()
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.stop()

# Default values for genes
default_values = {
    'JAG2': 24.81,
    'PLAU': 22.00,
    'CXCL2': 30.52,
    'TNC': 26.0,
    'FGF6': 19.58
}

# Sidebar configuration
st.sidebar.header("🎛️ Gene Expression Inputs")
st.sidebar.markdown("Adjust the expression levels below:")

# Reset button
if st.sidebar.button("Reset to Defaults", key="reset"):
    st.session_state.update(default_values)

# Input fields with organized layout
gene_features = list(default_values.keys())
gene_values = {}
col1, col2 = st.sidebar.columns(2)

for i, gene in enumerate(gene_features):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        gene_values[gene] = st.number_input(
            gene,
            min_value=float(background_data[gene].min()),
            max_value=float(background_data[gene].max()),
            value=default_values[gene],
            step=0.01,
            format="%.2f",
            key=gene
        )

# Prepare input data
def prepare_input_data():
    return pd.DataFrame([gene_values])

# Main content area
if st.button("🔍 Calculate SHAP Values", key="calculate"):
    input_df = prepare_input_data()
    background_processed = background_data
    
    # Prediction
    prediction = model.predict(input_df.values, verbose=0)[0][0]
    st.header("📊 Model Prediction")
    st.metric("Response Probability", f"{prediction:.4f}", 
             delta=None if prediction < 0.5 else "Likely Positive",
             delta_color="normal")
    
    # SHAP calculation
    explainer = shap.DeepExplainer(model, background_processed.values)
    shap_values = np.squeeze(np.array(explainer.shap_values(input_df.values)))
    base_value = float(explainer.expected_value[0].numpy())
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Force Plot", "Waterfall Plot", "Decision Plot", "Feature Importance"])
    
    with tab1:
        st.subheader("SHAP Force Plot")
        explanation = shap.Explanation(
            values=shap_values, base_values=base_value, data=input_df.values,
            feature_names=input_df.columns, display_data=input_df.round(2).values
        )
        shap.plots.force(explanation, matplotlib=True, show=False, figsize=(20, 3))
        st.pyplot(plt.gcf(), clear_figure=True)
    
    with tab2:
        st.subheader("SHAP Waterfall Plot")
        shap.waterfall_plot(
            shap.Explanation(shap_values, base_value, input_df.iloc[0], input_df.columns.tolist()),
            show=False
        )
        st.pyplot(plt.gcf(), clear_figure=True)
    
    with tab3:
        st.subheader("SHAP Decision Plot")
        shap.decision_plot(base_value, shap_values, input_df.values[0], 
                         feature_names=input_df.columns.tolist(), show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    
    with tab4:
        st.subheader("Feature Importance Summary")
        importance_df = pd.DataFrame({'Gene': input_df.columns, 'SHAP Value': shap_values})
        importance_df = importance_df.sort_values('SHAP Value', ascending=False)
        st.dataframe(importance_df.style.background_gradient(cmap='RdYlGn', subset=['SHAP Value']))
        st.bar_chart(importance_df.set_index('Gene'))

# Information section
with st.expander("ℹ️ About This Model and SHAP Visualizations", expanded=True):
    st.markdown("""
    ### Model Overview
    This model uses 5 key genes from transcriptome data to predict immunotherapy response for small cell lung cancer.
    
    ### SHAP Visualization Guide
    - **Force Plot**: Visualizes how each gene pushes the prediction from the base value (red = positive, blue = negative).
    - **Waterfall Plot**: Breaks down individual feature contributions step-by-step.
    - **Decision Plot**: Shows the cumulative effect of features on the prediction.
    - **Feature Importance**: Ranks genes by their impact on the prediction (positive/negative SHAP values).
    """)

# Footer
st.markdown("---")
st.markdown(f"Generated on: April 16, 2025 | Powered by xAI")