import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Bladder Cancer Immunotherapy Prediction - SHAP Visualization", layout="wide")

# Title
st.title("Bladder Cancer Immunotherapy Prediction Model - SHAP Visualization")

# Load and prepare background data
@st.cache_data
def load_background_data():
    # Replace with your actual data file containing the 10 genes
    df = pd.read_excel('data/train_10.xlsx')  # Hypothetical file name
    return df[['PSMB8', 'CTSS', 'HLA-DPB1', 'NBPF1', 'REC8', 'RANGRF', 'UBE2L6', 'MMAA', 'TOX', 'PDCD1']]

# Load the pre-trained model
@st.cache_resource
def load_model():
    # Replace with your actual model file
    model = tf.keras.models.load_model('data/MODEL_2025_01_23_16_52_07.h5')  # Hypothetical model file
    return model

# Initialize data and model
background_data = load_background_data()
model = load_model()

# Default values for the 10 genes (example values, adjust based on your data)
default_values = {
    'PSMB8': 0.5,
    'CTSS': 0.8,
    'HLA-DPB1': 0.3,
    'NBPF1': 0.6,
    'REC8': 0.4,
    'RANGRF': 1.0,
    'UBE2L6': 0.7,
    'MMAA': 0.9,
    'TOX': 0.2,
    'PDCD1': 1.2
}

# Create input form in sidebar
st.sidebar.header("Gene Expression Inputs")

# Add reset button
if st.sidebar.button("Reset to Default Values"):
    st.session_state.update(default_values)

# Input fields for gene expression levels
gene_features = ['PSMB8', 'CTSS', 'HLA-DPB1', 'NBPF1', 'REC8', 'RANGRF', 'UBE2L6', 'MMAA', 'TOX', 'PDCD1']
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
    st.write(f"Immunotherapy Response Prediction Value: {prediction[0][0]:.4f}")
    
    # Calculate SHAP values using DeepExplainer
    explainer = shap.DeepExplainer(model, background_processed.values)
    shap_values = np.squeeze(np.array(explainer.shap_values(input_df.values)))
    base_value = float(explainer.expected_value[0].numpy())
    
    # Create SHAP Force Plot
    st.header("SHAP Force Plot")
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
    plt.clf()  # Clear the figure to avoid overlap

    # Create SHAP Waterfall Plot
    st.header("SHAP Waterfall Plot")
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns.tolist()
        ),
        show=False
    )
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the figure

    # Create SHAP Decision Plot
    st.header("SHAP Decision Plot")
    shap.decision_plot(
        base_value,
        shap_values,
        input_df.values[0],
        feature_names=input_df.columns.tolist(),
        show=False
    )
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the figure
    
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
### About SHAP Visualizations
- **Force Plot**: Shows how features push the prediction from the base value (red for positive, blue for negative contributions).
- **Waterfall Plot**: Details the contribution of each feature to the final prediction step-by-step.
- **Decision Plot**: Illustrates how the model's prediction evolves as each feature’s contribution is added.
- **SHAP Values**: Positive values increase the immunotherapy response score, negative values decrease it, and the magnitude reflects importance.
""")

# Footer with current date
st.markdown(f"Generated on: February 24, 2025")