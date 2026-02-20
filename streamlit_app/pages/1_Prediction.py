"""
Prediction page for making fraud predictions.
"""

import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import time

st.set_page_config(page_title="Prediction", page_icon="🔍", layout="wide")

API_URL = "http://localhost:8000"

# Title
st.title("🔍 Fraud Prediction")
st.markdown("Analyze individual transactions or upload a batch for fraud detection")

# Check API
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    api_online = response.status_code == 200
except:
    api_online = False

if not api_online:
    st.error("⚠️ API is offline. Please start the API server.")
    st.stop()

# Sample transactions
SAMPLES = {
    "Low Risk Transaction": {
        "Time": 12345.0,
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
        "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
        "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
        "Amount": 25.50
    },
    "Medium Risk Transaction": {
        "Time": 54321.0,
        "V1": -2.5, "V2": 1.8, "V3": -1.2, "V4": 2.1,
        "V5": -1.5, "V6": 0.8, "V7": -0.5, "V8": 1.2,
        "V9": -0.9, "V10": 1.4, "V11": -1.1, "V12": 0.7,
        "V13": -1.8, "V14": 1.6, "V15": -0.4, "V16": 0.9,
        "V17": -1.3, "V18": 1.1, "V19": -0.6, "V20": 0.5,
        "V21": -1.4, "V22": 1.7, "V23": -0.8, "V24": 0.6,
        "V25": -1.2, "V26": 1.5, "V27": -0.7, "V28": 0.4,
        "Amount": 250.00
    },
    "High Risk Transaction": {
        "Time": 98765.0,
        "V1": -5.2, "V2": 4.1, "V3": -3.8, "V4": 4.5,
        "V5": -4.2, "V6": 3.5, "V7": -3.1, "V8": 4.8,
        "V9": -3.6, "V10": 4.2, "V11": -4.5, "V12": 3.8,
        "V13": -5.1, "V14": 4.6, "V15": -3.4, "V16": 3.9,
        "V17": -4.8, "V18": 4.3, "V19": -3.7, "V20": 3.2,
        "V21": -4.4, "V22": 4.7, "V23": -3.9, "V24": 3.6,
        "V25": -4.1, "V26": 4.4, "V27": -3.5, "V28": 3.3,
        "Amount": 5000.00
    }
}

# Tabs
tab1, tab2 = st.tabs(["📝 Single Prediction", "📊 Batch Upload"])

with tab1:
    st.markdown("### Enter Transaction Details")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("#### Quick Load")
        selected_sample = st.selectbox(
            "Load sample transaction",
            ["None", "Low Risk Transaction", "Medium Risk Transaction", "High Risk Transaction"]
        )
        
        if st.button("Load Sample", use_container_width=True):
            if selected_sample != "None":
                st.session_state.sample_data = SAMPLES[selected_sample]
                st.rerun()
    
    with col1:
        # Initialize session state
        if 'sample_data' not in st.session_state:
            st.session_state.sample_data = None
        
        # Basic fields
        col_time, col_amount = st.columns(2)
        
        with col_time:
            time_val = st.number_input(
                "Time (seconds)",
                min_value=0.0,
                value=float(st.session_state.sample_data['Time']) if st.session_state.sample_data else 12345.0,
                help="Seconds elapsed since first transaction"
            )
        
        with col_amount:
            amount_val = st.number_input(
                "Amount ($)",
                min_value=0.0,
                value=float(st.session_state.sample_data['Amount']) if st.session_state.sample_data else 100.0,
                help="Transaction amount in dollars"
            )
        
        # V features in expander
        with st.expander("🔢 PCA Features (V1-V28)", expanded=False):
            st.markdown("*These are PCA-transformed features from the original dataset*")
            
            v_features = {}
            
            # Create 4 columns for V features
            for i in range(0, 28, 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < 28:
                        v_num = i + j + 1
                        with col:
                            default_val = st.session_state.sample_data[f'V{v_num}'] if st.session_state.sample_data else 0.0
                            v_features[f'V{v_num}'] = st.number_input(
                                f"V{v_num}",
                                value=float(default_val),
                                format="%.6f",
                                key=f"v{v_num}"
                            )
        
        st.markdown("---")
        
        # Predict button
        if st.button("🎯 Check Transaction", type="primary", use_container_width=True):
            # Prepare transaction
            transaction = {
                "Time": time_val,
                "Amount": amount_val,
                **v_features
            }
            
            # Make prediction
            with st.spinner("Analyzing transaction..."):
                try:
                    start = time.time()
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=transaction,
                        timeout=5
                    )
                    latency = (time.time() - start) * 1000
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display result
                        st.markdown("### 📊 Prediction Result")
                        
                        # Status card
                        if result['is_fraud']:
                            st.error(f"🚨 **FRAUD DETECTED** - Risk Level: {result['risk_level']}")
                        else:
                            st.success(f"✅ **LEGITIMATE TRANSACTION** - Risk Level: {result['risk_level']}")
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Fraud Probability",
                                f"{result['fraud_probability']:.2%}"
                            )
                        
                        with col2:
                            st.metric(
                                "Risk Level",
                                result['risk_level']
                            )
                        
                        with col3:
                            st.metric(
                                "Response Time",
                                f"{result['response_time_ms']:.2f}ms"
                            )
                        
                        # Probability bar
                        st.markdown("#### Fraud Probability")
                        prob_pct = result['fraud_probability'] * 100
                        
                        if prob_pct < 30:
                            bar_color = "green"
                        elif prob_pct < 60:
                            bar_color = "orange"
                        else:
                            bar_color = "red"
                        
                        st.markdown(f"""
                        <div style="background-color: #f0f0f0; border-radius: 10px; padding: 5px;">
                            <div style="background-color: {bar_color}; width: {prob_pct}%; 
                                        height: 30px; border-radius: 8px; text-align: center; 
                                        line-height: 30px; color: white; font-weight: bold;">
                                {prob_pct:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional info
                        with st.expander("📋 Full Response"):
                            st.json(result)
                    
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

with tab2:
    st.markdown("### Upload Batch File")
    st.info("Upload a CSV file with transaction data. File must include Time, Amount, and V1-V28 columns.")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="CSV file with columns: Time, V1-V28, Amount"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} transactions")
            
            # Show preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Validate columns
            required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ All required columns present")
                
                # Process batch
                if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                    with st.spinner(f"Processing {len(df)} transactions..."):
                        try:
                            # Prepare batch request
                            transactions = df[required_cols].to_dict('records')
                            
                            start = time.time()
                            response = requests.post(
                                f"{API_URL}/predict_batch",
                                json={"transactions": transactions},
                                timeout=30
                            )
                            duration = time.time() - start
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                # Show results
                                st.success(f"✅ Processed {result['total_transactions']} transactions in {duration:.2f}s")
                                
                                # Metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total Processed", result['total_transactions'])
                                
                                with col2:
                                    st.metric("Frauds Detected", result['fraud_count'])
                                
                                with col3:
                                    fraud_rate = result['fraud_count'] / result['total_transactions'] * 100
                                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                                
                                # Create results dataframe
                                results_df = pd.DataFrame([
                                    {
                                        'is_fraud': p['is_fraud'],
                                        'fraud_probability': p['fraud_probability'],
                                        'risk_level': p['risk_level']
                                    }
                                    for p in result['predictions']
                                ])
                                
                                # Combine with original data
                                output_df = pd.concat([
                                    df[['Time', 'Amount']],
                                    results_df
                                ], axis=1)
                                
                                # Display results
                                st.markdown("### 📊 Results")
                                
                                # Filter options
                                col1, col2 = st.columns(2)
                                with col1:
                                    show_fraud_only = st.checkbox("Show fraud only")
                                with col2:
                                    risk_filter = st.multiselect(
                                        "Filter by risk level",
                                        ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                                        default=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                                    )
                                
                                # Apply filters
                                filtered_df = output_df.copy()
                                if show_fraud_only:
                                    filtered_df = filtered_df[filtered_df['is_fraud'] == True]
                                filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]
                                
                                st.dataframe(filtered_df, use_container_width=True, height=400)
                                
                                # Download button
                                csv = output_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            else:
                                st.error(f"Error: {response.status_code}")
                        
                        except Exception as e:
                            st.error(f"Error processing batch: {e}")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Sidebar info
with st.sidebar:
    st.markdown("### 💡 Tips")
    st.markdown("""
    **Single Prediction:**
    - Use sample transactions to test
    - V features are PCA components
    - Response time should be <100ms
    
    **Batch Upload:**
    - Upload CSV with required columns
    - Process up to 1000 transactions
    - Download results as CSV
    """)
    
    st.markdown("### 📖 API Reference")
    st.markdown(f"[View API Docs]({API_URL}/docs)")
    
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
    st.caption(f"Current: {threshold}")