import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Medical Prediction SHAP Visualization", layout="wide")

# Title
st.title("Medical Prediction Model SHAP Values Visualization")

# Load and prepare background data
@st.cache_data
def load_background_data():
    df = pd.read_excel('data/train_13.xlsx')
    return df.iloc[:,:-2]

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('data/MODEL_2024_12_31_12_43_00.h5')
    return model

# Initialize label encoders and load data
background_data = load_background_data()

# Get model
model = load_model()

# Default values
default_values = {
    'Tumor size (cm)': 4.0,
    'Adjuvant endocrine therapy': 0.6449,
    'Lymph node positive': 5,
    'EGFR': -0.27825,
    'MPHOSPH10': 1.463553,
    'ACOX2': 0.0,
    'CASP3': 0.58796,
    'ARL3': 0.139503,
    'KRT18': -0.13279,
    'FAM102A': 0.401588,
    'STEAP3': 0.0,
    'HER2 status (IHC)': 2,
    'BUB1B': 0.534714
}

# Create input form
st.sidebar.header("Input Features")

# Add reset button
if st.sidebar.button("Reset to Default Values"):
    st.session_state.update(default_values)

# Input fields
tumor_size = st.sidebar.number_input("Tumor size (cm)", 
                                   min_value=float(background_data['Tumor size (cm)'].min()),
                                   max_value=float(background_data['Tumor size (cm)'].max()),
                                   value=default_values['Tumor size (cm)'],
                                   step=0.1,
                                   key='Tumor size (cm)')

# Numeric inputs for previously categorical features
adjuvant_therapy = st.sidebar.number_input(
    "Adjuvant endocrine therapy",
    min_value=0.0,
    max_value=1.0,
    value=default_values['Adjuvant endocrine therapy'],
    step=0.01,
    key='Adjuvant endocrine therapy'
)

lymph_node = st.sidebar.number_input(
    "Lymph node positive",
    min_value=int(background_data['Lymph node positive'].min()),
    max_value=int(background_data['Lymph node positive'].max()),
    value=default_values['Lymph node positive'],
    key='Lymph node positive'
)

her2_status = st.sidebar.number_input(
    "HER2 status (IHC)",
    min_value=0,
    max_value=2,
    value=default_values['HER2 status (IHC)'],
    step=1,
    key='HER2 status (IHC)'
)

# Protein expression levels
st.sidebar.header("Protein Expression Levels")
protein_features = ['EGFR', 'MPHOSPH10', 'ACOX2', 'CASP3', 'ARL3', 'KRT18', 'FAM102A', 'STEAP3', 'BUB1B']
protein_values = {}

for protein in protein_features:
    protein_values[protein] = st.sidebar.number_input(
        protein,
        min_value=float(background_data[protein].min()),
        max_value=float(background_data[protein].max()),
        value=default_values[protein],
        step=0.00001,
        format="%.5f",
        key=protein
    )

# Create input data for prediction
def prepare_input_data():
    input_data = {
        'Tumor size (cm)': tumor_size,
        'Adjuvant endocrine therapy': adjuvant_therapy,
        'Lymph node positive': lymph_node,
        'HER2 status (IHC)': her2_status,
    }
    # Add protein values
    input_data.update(protein_values)
    return pd.DataFrame([input_data])

# Calculate SHAP values when user clicks the button
if st.button("Calculate SHAP Values"):
    # Prepare input data
    input_df = prepare_input_data()
    
    # Use background data directly since all values are numeric now
    background_processed = background_data
    
    # Get model prediction
    prediction = model.predict(input_df.values)
    st.header("Model Prediction")
    st.write(f"Prediction Value: {prediction[0][0]:.4f}")
    
    # Calculate SHAP values using GradientExplainer for deep learning models
    explainer = shap.DeepExplainer(model, background_processed.values)
    print(input_df.values)
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
    
    # Create SHAP waterfall plot
    # st.header("SHAP Waterfall Plot")
    # fig = plt.figure()
    # shap.waterfall_plot(shap.Explanation(
    #     values=shap_values[0],
    #     base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
    #     data=input_df.iloc[0],
    #     feature_names=input_df.columns
    # ), show=False)
    # st.pyplot(fig)
    
    # Display feature importance summary
    st.header("Feature Importance Summary")
    importance_df = pd.DataFrame({
        'Feature': input_df.columns,
        'SHAP Value': shap_values
    })
    importance_df = importance_df.sort_values('SHAP Value', ascending=False)
    st.dataframe(importance_df)

# Add some information about the visualization
st.markdown("""
### About SHAP Values
SHAP (SHapley Additive exPlanations) values show how much each feature contributes to the prediction:
- Positive values (red) push the prediction higher
- Negative values (blue) push the prediction lower
- The magnitude of the value shows how important that feature is
""")
