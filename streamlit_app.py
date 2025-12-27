# ============================================================================
# SPLEEN DIMENSION PREDICTION - STREAMLIT APP (FIXED)
# ============================================================================
"""
Streamlit-based web application for spleen dimension prediction.

IMPORTANT: Install dependencies first:
    pip install -r requirements.txt
    
Or manually:
    pip install streamlit pandas numpy scikit-learn joblib plotly

Usage:
    streamlit run streamlit_app.py

Then visit: http://localhost:8501
"""

import streamlit as st
import sys

# ============================================================================
# DEPENDENCY CHECK AND INSTALLATION
# ============================================================================

def check_dependencies():
    """Check and install missing dependencies"""
    missing = []
    
    deps = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'joblib': 'joblib',
        'sklearn': 'scikit-learn',
        'plotly': 'plotly'
    }
    
    for module_name, package_name in deps.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    return missing

# Check and warn about missing dependencies
missing_deps = check_dependencies()

if missing_deps:
    st.error("""
    ❌ **Missing Dependencies!**
    
    Please install the required packages before running this app:
    
    ```bash
    pip install -r requirements.txt
    ```
    
    Or install individually:
    ```bash
    pip install streamlit pandas numpy scikit-learn joblib plotly
    ```
    
    Missing packages: """ + ", ".join(missing_deps) + """
    
    After installation, restart Streamlit:
    ```bash
    streamlit run streamlit_app.py
    ```
    """)
    st.stop()

# ============================================================================
# NOW IMPORT ALL REQUIRED MODULES
# ============================================================================

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Spleen Dimension Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #667eea;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS (Cached for performance)
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained models from pickle file"""
    try:
        # Try to find the model file
        import os
        
        # Check current directory
        if os.path.exists('final_optimized_spleen_models.pkl'):
            filepath = 'final_optimized_spleen_models.pkl'
        # Check if running from different directory
        elif os.path.exists('../final_optimized_spleen_models.pkl'):
            filepath = '../final_optimized_spleen_models.pkl'
        else:
            st.error("""
            ❌ **Model file not found!**
            
            Please ensure `final_optimized_spleen_models.pkl` is in the same folder as `streamlit_app.py`
            
            Current directory: """ + os.getcwd() + """
            
            Files in current directory:
            """ + str(os.listdir('.')))
            st.stop()
        
        package = joblib.load(filepath)
        return package
        
    except FileNotFoundError as e:
        st.error(f"""
        ❌ **Error: Model file not found!**
        
        Please ensure `final_optimized_spleen_models.pkl` exists in the application directory.
        
        {str(e)}
        """)
        st.stop()
    except Exception as e:
        st.error(f"""
        ❌ **Error loading model!**
        
        {str(e)}
        """)
        st.stop()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_dimensions(age, weight, height, models_package):
    """
    Predict spleen dimensions.
    """
    try:
        # Calculate BSA
        bsa = np.sqrt((height * weight) / 3600)
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'weight': [weight],
            'Height': [height],
            'BSA': [bsa]
        })
        
        # Get predictions
        models = models_package['models']
        targets = models_package['targets']
        
        predictions = {}
        for target in targets:
            scaler = models[target]['scaler']
            model = models[target]['model']
            
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            predictions[target] = float(pred)
        
        return {
            'length': predictions['Length (cm)'],
            'width': predictions['Width (cm)'],
            'volume': predictions['volume (cm3)'],
            'bsa': bsa
        }
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

# ============================================================================
# MAIN APP
# ============================================================================

# Load models
models_package = load_models()

# Header
st.markdown("""
    <h1 style='text-align: center; color: #667eea;'>
    🫀 Spleen Dimension Predictor
    </h1>
    <p style='text-align: center; color: #666; font-size: 1.1em;'>
    Advanced ML-Based Ultrasound Measurement Prediction System
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar - Information
with st.sidebar:
    st.markdown("### 📊 Model Information")
    
    results = models_package['results']
    
    st.markdown("**Length Model**")
    st.info(f"Model: {results['Length (cm)']['model_name']}\nR²: {results['Length (cm)']['test_r2']:.4f}")
    
    st.markdown("**Width Model**")
    st.info(f"Model: {results['Width (cm)']['model_name']}\nR²: {results['Width (cm)']['test_r2']:.4f}")
    
    st.markdown("**Volume Model**")
    st.info(f"Model: {results['volume (cm3)']['model_name']}\nR²: {results['volume (cm3)']['test_r2']:.4f}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This application uses machine learning models trained on 500+ 
    patient ultrasound records to predict spleen dimensions based on:
    - Age
    - Weight
    - Height
    - Body Surface Area (BSA)
    """)
    
    st.warning("""
    ⚠️ **Clinical Disclaimer:**
    
    These predictions are for preliminary screening purposes only. 
    NOT a substitute for professional medical examination.
    """)

# Main content - Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 Patient Information")
    
    patient_id = st.text_input("Patient ID / Name (optional)", "Anonymous Patient")
    
    st.markdown("#### Measurements")
    
    age = st.number_input(
        "Age (years)",
        min_value=0,
        max_value=120,
        value=45,
        step=1,
        help="Patient age in years"
    )
    
    weight = st.number_input(
        "Weight (kg)",
        min_value=0.0,
        max_value=300.0,
        value=65.0,
        step=0.1,
        help="Patient weight in kilograms"
    )
    
    height = st.number_input(
        "Height (cm)",
        min_value=0.0,
        max_value=300.0,
        value=165.0,
        step=0.1,
        help="Patient height in centimeters"
    )
    
    # Calculate BSA and BMI
    bsa = np.sqrt((height * weight) / 3600)
    bmi = weight / ((height/100)**2)
    
    st.markdown("#### Calculated Values")
    col_bsa, col_bmi = st.columns(2)
    with col_bsa:
        st.metric("Body Surface Area (BSA)", f"{bsa:.3f} m²")
    with col_bmi:
        st.metric("BMI", f"{bmi:.1f}")

with col2:
    st.markdown("### 🔮 Predictions")
    
    if st.button("🔮 Predict Dimensions", use_container_width=True, type="primary"):
        # Validate inputs
        if age < 0 or age > 120:
            st.error("❌ Age must be between 0 and 120 years")
        elif weight < 0 or weight > 300:
            st.error("❌ Weight must be between 0 and 300 kg")
        elif height < 0 or height > 300:
            st.error("❌ Height must be between 0 and 300 cm")
        else:
            # Make prediction
            with st.spinner("Analyzing patient data..."):
                predictions = predict_dimensions(age, weight, height, models_package)
            
            if predictions:
                # Display results
                st.success("✅ Predictions generated successfully!")
                
                st.markdown("#### 📊 Results")
                
                col_length, col_width, col_volume = st.columns(3)
                
                with col_length:
                    st.metric("📏 Length", f"{predictions['length']:.2f} cm")
                
                with col_width:
                    st.metric("📊 Width", f"{predictions['width']:.2f} cm")
                
                with col_volume:
                    st.metric("📦 Volume", f"{predictions['volume']:.0f} cm³")

# Results Summary Section
st.markdown("---")
st.markdown("### 📈 Model Performance & Accuracy")

perf_col1, perf_col2, perf_col3 = st.columns(3)

with perf_col1:
    st.metric(
        "Length R²",
        f"{results['Length (cm)']['test_r2']:.4f}",
        "Fair accuracy"
    )
    st.metric("Length MAE", f"{results['Length (cm)']['test_mae']:.4f} cm")
    st.metric("Length MAPE", f"{results['Length (cm)']['test_mape']*100:.2f}%")

with perf_col2:
    st.metric(
        "Width R²",
        f"{results['Width (cm)']['test_r2']:.4f}",
        "High variability"
    )
    st.metric("Width MAE", f"{results['Width (cm)']['test_mae']:.4f} cm")
    st.metric("Width MAPE", f"{results['Width (cm)']['test_mape']*100:.2f}%")

with perf_col3:
    st.metric(
        "Volume R²",
        f"{results['volume (cm3)']['test_r2']:.4f}",
        "Moderate accuracy"
    )
    st.metric("Volume MAE", f"{results['volume (cm3)']['test_mae']:.2f} cm³")
    st.metric("Volume MAPE", f"{results['volume (cm3)']['test_mape']*100:.2f}%")

# Batch Prediction Section
st.markdown("---")
st.markdown("### 📋 Batch Prediction")

st.markdown("Upload a CSV file with patient data for batch predictions")

col_upload, col_example = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="CSV should have columns: age, weight, height"
    )

with col_example:
    if st.button("📥 Download Example CSV"):
        example_df = pd.DataFrame({
            'age': [30, 45, 60],
            'weight': [60, 65, 70],
            'height': [160, 165, 170]
        })
        csv = example_df.to_csv(index=False)
        st.download_button(
            label="example_patients.csv",
            data=csv,
            file_name="example_patients.csv",
            mime="text/csv"
        )

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate columns
        required_cols = ['age', 'weight', 'height']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ CSV must contain columns: {required_cols}")
        else:
            st.markdown(f"Processing {len(df)} patients...")
            
            # Make predictions
            predictions_list = []
            for idx, row in df.iterrows():
                pred = predict_dimensions(
                    row['age'],
                    row['weight'],
                    row['height'],
                    models_package
                )
                if pred:
                    predictions_list.append({
                        'patient_id': idx + 1,
                        'age': row['age'],
                        'weight': row['weight'],
                        'height': row['height'],
                        'length_cm': round(pred['length'], 2),
                        'width_cm': round(pred['width'], 2),
                        'volume_cm3': round(pred['volume'], 0)
                    })
            
            # Display results
            results_df = pd.DataFrame(predictions_list)
            st.dataframe(results_df, use_container_width=True)
            
            # Download option
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f"spleen_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #999; font-size: 0.9em;'>
    Spleen Dimension Prediction System | Powered by Advanced Machine Learning | © 2025
    </p>
""", unsafe_allow_html=True)