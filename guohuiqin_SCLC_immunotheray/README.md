# Medical Prediction SHAP Visualization

This Streamlit application visualizes SHAP (SHapley Additive exPlanations) values for medical predictions based on various features including tumor size, protein expressions, and other clinical indicators.

## Setup and Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

## Features

- Input various medical features through an interactive interface
- View SHAP force plots showing feature contributions
- Analyze feature importance through waterfall plots
- See a summary of feature importance values

## Input Features

- Tumor size (cm)
- Adjuvant endocrine therapy
- Lymph node positive status
- Various protein expression levels (EGFR, MPHOSPH10, ACOX2, etc.)
- HER2 status
