import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="SCLC Gene Prediction SHAP Visualization", layout="wide")

# Title
st.title("Small Cell Lung Cancer Gene Prediction Model - SHAP Visualization")

# Load and prepare background data
@st.cache_data
def load_background_data():
    # Replace with your actual data file containing the 10 genes
    df = pd.read_excel('data/train_10.xlsx')  # Hypothetical file name
    return df[['CA8', 'GLS2', 'SCEL', 'PTPRB', 'SLC39A8', 'ANKRD1', 'MAOB', 'PMP22', 'ABCA8', 'SSPN']]

# Load the pre-trained model
@st.cache_resource
def load_model():
    # Replace with your actual model file
    model = tf.keras.models.load_model('data/MODEL_2024_12_18_16_12_51.h5')  # Hypothetical model file
    return model

# Initialize data and model
background_data = load_background_data()
model = load_model()

# Default values for the 10 genes (example values, adjust based on your data)
default_values = {
    'CA8': 0.5,
    'GLS2': 1.2,
    'SCEL': 0.1,
    'PTPRB': 0.8,
    'SLC39A8': 0.0,
    'ANKRD1': 1.5,
    'MAOB': 0.5,
    'PMP22': 0.7,
    'ABCA8': 0.4,
    'SSPN': 0.9
}

# Create input form in sidebar
st.sidebar.header("Gene Expression Inputs")

# Add reset button
if st.sidebar.button("Reset to Default Values"):
    st.session_state.update(default_values)

# Input fields for gene expression levels
gene_features = ['CA8', 'GLS2', 'SCEL', 'PTPRB', 'SLC39A8', 'ANKRD1', 'MAOB', 'PMP22', 'ABCA8', 'SSPN']
gene_values = {}

for gene in gene_features:
    gene_values[gene] = st.sidebar.number_input(
        gene,
        min_value=float(background_data[gene].min()),
        max_value=float(background_data[gene].max()),
        value=default_values[gene],
        step=0.01,
        format="%.2f",
        key=gene
    )

# Prepare input data for prediction
def prepare_input_data():
    input_data = gene_values
    return pd.DataFrame([input_data])

# Calculate SHAP values and display results
if st.button("Calculate SHAP Values"):
    # Prepare input data
    input_df = prepare_input_data()
    
    # Use background data directly
    background_processed = background_data
    
    # Get model prediction
    prediction = model.predict(input_df.values)
    st.header("Model Prediction")
    st.write(f"Prediction Value: {prediction[0][0]:.4f}")
    
    # Calculate SHAP values using DeepExplainer
    explainer = shap.DeepExplainer(model, background_processed.values)
    shap_values = np.squeeze(np.array(explainer.shap_values(input_df.values)))
    
    # Create SHAP force plot
    st.header("SHAP Force Plot")
    base_value = float(explainer.expected_value[0].numpy())
    
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=input_df.values,
        feature_names=input_df.columns,
        display_data=input_df.round(2).values
    )
    shap.plots.force(
        explanation,
        matplotlib=True,
        show=False,
        contribution_threshold=0.1
    )
    st.pyplot(plt.gcf())
    
    # Display feature importance summary
    st.header("Feature Importance Summary")
    importance_df = pd.DataFrame({
        'Gene': input_df.columns,
        'SHAP Value': shap_values
    })
    importance_df = importance_df.sort_values('SHAP Value', ascending=False)
    st.dataframe(importance_df)

    # Add explanation about SHAP values
st.markdown("""
### About SHAP Values
SHAP (SHapley Additive exPlanations) values indicate how much each gene contributes to the prediction:
- Positive values (red) increase the prediction score
- Negative values (blue) decrease the prediction score
- The magnitude reflects the importance of each gene
""")