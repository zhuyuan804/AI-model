import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Suppress TensorFlow warnings by setting environment variable and using compat API
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs except critical errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configure matplotlib
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 12

# Set page config
st.set_page_config(
    page_title="Sepsis Risk Prediction Model",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Constants
GENE_LIST = ['DEFA4', 'CRISP3', 'EPHB1', 'ARG2', 'SESN3', 'IRAK3', 'CEACAM8', 'MME', 'LIN7A', 'MS4A3']

# Cached data loading functions
@st.cache_data
def load_background_data():
    try:
        df = pd.read_excel('heyuexian/data/train_top10.xlsx')
        return df[GENE_LIST]
    except FileNotFoundError:
        st.error("Background data file 'heyuexian/data/train_top10.xlsx' not found.")
        return pd.DataFrame(columns=GENE_LIST)

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('heyuexian/data/MODEL_2024_12_24_18_52_36.h5')
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Initialize data and model
background_data = load_background_data()
model = load_model()

# Set default values based on data median (or mean, if preferred)
def get_default_values(data):
    if data.empty:
        return {gene: 0.0 for gene in GENE_LIST}  # Fallback if data is missing
    return {gene: float(data[gene].median()) for gene in GENE_LIST}

DEFAULT_VALUES = get_default_values(background_data)

# Helper functions
def prepare_input_data(gene_values):
    return pd.DataFrame([gene_values], columns=GENE_LIST)

def initialize_session_state():
    if 'gene_values' not in st.session_state:
        st.session_state.gene_values = DEFAULT_VALUES.copy()

# Main UI
st.title("🩺 Sepsis Gene Expression Prediction Model")
st.markdown("""
    This tool predicts sepsis probability using transcriptome data of 10 key genes, 
    with SHAP visualizations for interpretable insights. Adjust gene expression 
    levels in the sidebar to explore their impact.
""")

# Sidebar configuration
st.sidebar.header("🎛️ Gene Expression Inputs")
st.sidebar.markdown("Adjust the expression levels below:")

# Initialize session state
initialize_session_state()

# Reset button
if st.sidebar.button("Reset to Defaults", key="reset"):
    st.session_state.gene_values = DEFAULT_VALUES.copy()

# Input fields
col1, col2 = st.sidebar.columns(2)
for i, gene in enumerate(GENE_LIST):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        min_val = float(background_data[gene].min()) if not background_data.empty else 0.0
        max_val = float(background_data[gene].max()) if not background_data.empty else 1000.0
        st.session_state.gene_values[gene] = st.number_input(
            gene,
            min_value=min_val,
            max_value=max_val,
            value=min(max(st.session_state.gene_values[gene], max_val), min_val),  # Clamp value to bounds
            step=0.01,
            format="%.2f",
            key=gene
        )

# Main content
if st.button("🔍 Calculate SHAP Values", key="calculate") and model is not None:
    input_df = prepare_input_data(st.session_state.gene_values)
    
    try:
        # Prediction
        prediction = model.predict(input_df.values, verbose=0)[0][0]
        st.header("📊 Model Prediction")
        st.metric("Sepsis Probability", f"{prediction:.4f}",
                 delta="Likely Positive" if prediction >= 0.5 else "Likely Negative",
                 delta_color="normal" if prediction < 0.5 else "inverse")
        
        # SHAP calculation
        explainer = shap.DeepExplainer(model, background_data.values)
        shap_values = np.squeeze(explainer.shap_values(input_df.values))
        base_value = float(explainer.expected_value[0].numpy())
        
        # Visualization tabs
        tabs = st.tabs(["Force Plot", "Waterfall Plot", "Decision Plot", "Feature Importance"])
        
        with tabs[0]:
            st.subheader("SHAP Force Plot")
            explanation = shap.Explanation(
                values=shap_values, base_values=base_value, data=input_df.values,
                feature_names=GENE_LIST, display_data=input_df.round(2).values
            )
            shap.plots.force(explanation, matplotlib=True, show=False, figsize=(20, 3))
            st.pyplot(plt.gcf(), clear_figure=True)
        
        with tabs[1]:
            st.subheader("SHAP Waterfall Plot")
            shap.waterfall_plot(
                shap.Explanation(shap_values, base_value, input_df.iloc[0], GENE_LIST),
                show=False
            )
            st.pyplot(plt.gcf(), clear_figure=True)
        
        with tabs[2]:
            st.subheader("SHAP Decision Plot")
            shap.decision_plot(base_value, shap_values, input_df.values[0], 
                             feature_names=GENE_LIST, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
        
        with tabs[3]:
            st.subheader("Feature Importance Summary")
            importance_df = pd.DataFrame({'Gene': GENE_LIST, 'SHAP Value': shap_values})
            importance_df = importance_df.sort_values('SHAP Value', ascending=False)
            st.dataframe(importance_df.style.background_gradient(cmap='RdYlGn', subset=['SHAP Value']))
            st.bar_chart(importance_df.set_index('Gene'))
            
    except Exception as e:
        st.error(f"An error occurred during prediction or SHAP calculation: {str(e)}")

# Information section
with st.expander("ℹ️ About This Model and SHAP Visualizations", expanded=True):
    st.markdown("""
    ### Model Overview
    This model leverages 10 key genes from transcriptome data to predict sepsis probability.
    
    ### SHAP Visualization Guide
    - **Force Plot**: Visualizes how each gene pushes the prediction from the base value (red = positive, blue = negative).
    - **Waterfall Plot**: Details individual feature contributions step-by-step.
    - **Decision Plot**: Shows the cumulative effect of features on the prediction.
    - **Feature Importance**: Ranks genes by their impact (positive/negative SHAP values).
    """)

# Footer
st.markdown("---")
st.markdown("Generated on: April 08, 2025 | Powered by xAI")
