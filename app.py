import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Configure page
st.set_page_config(
    page_title="Pizza Delivery Time Predictor - DEBUG",
    page_icon="🔍",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        if not os.path.exists('realistic_rf_model.pkl'):
            return None, "Model file 'realistic_rf_model.pkl' not found. Please run the realistic training script first."
        
        with open('realistic_rf_model.pkl', 'rb') as model_file:
            model_info = pickle.load(model_file)
        
        if isinstance(model_info, dict):
            model = model_info['model']
            features = model_info['features']
            performance = model_info.get('model_performance', {})
            
            return {
                'model': model,
                'features': features,
                'n_features': len(features),
                'performance': performance
            }, None
        else:
            return None, "Invalid model format. Please retrain the model."
            
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# Load model
model_data, error_message = load_model()

def main():
    st.title("🔍 Pizza Delivery Time Predictor - DEBUG MODE")
    st.markdown("---")
    
    if error_message:
        st.error(f"❌ {error_message}")
        return
    
    # Show model information
    if model_data:
        st.subheader("🔧 Model Information")
        st.write(f"**Features expected by model:** {model_data['features']}")
        st.write(f"**Number of features:** {model_data['n_features']}")
        if model_data['performance']:
            st.write(f"**Model performance:** {model_data['performance']}")
    
    st.markdown("---")
    
    # Prediction form
    with st.form("prediction_form"):
        st.subheader("🔧 Enter Order Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pizza_complexity = st.slider("Pizza Complexity", min_value=1, max_value=5, value=3)
            order_hour = st.slider("Order Hour", min_value=0, max_value=23, value=14)
            restaurant_avg_time = st.slider("Restaurant Avg Time (minutes)", min_value=10, max_value=60, value=25)
            distance = st.slider("Distance (km)", min_value=1, max_value=10, value=5)
        
        with col2:
            topping_density = st.slider("Topping Density", min_value=1, max_value=5, value=2)
            traffic_level = st.slider("Traffic Level", min_value=1, max_value=5, value=3)
            is_peak_hour = st.selectbox("Peak Hour?", options=[0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
            is_weekend = st.selectbox("Weekend?", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
        
        submitted = st.form_submit_button("🔍 DEBUG PREDICTION", type="primary")
        
        if submitted:
            st.markdown("---")
            st.subheader("🔍 COMPLETE DEBUG INFORMATION")
            
            # Original app logic (your current code)
            st.write("### 📋 ORIGINAL APP LOGIC:")
            input_values_original = [pizza_complexity, order_hour, restaurant_avg_time, 
                                   distance, topping_density, traffic_level, 
                                   is_peak_hour, is_weekend]
            input_data_original = np.array([input_values_original])
            
            st.write(f"**Input values (original):** {input_values_original}")
            st.write(f"**Input shape (original):** {input_data_original.shape}")
            st.write(f"**Features order (original):** ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 'Distance (km)', 'Topping Density', 'Traffic Level', 'Is Peak Hour', 'Is Weekend']")
            
            # Model prediction
            if model_data:
                model = model_data['model']
                predicted_duration_original = model.predict(input_data_original)[0]
                st.write(f"**Raw prediction (original):** {predicted_duration_original}")
                st.success(f"**ORIGINAL RESULT: {predicted_duration_original:.1f} minutes**")
            
            st.markdown("---")
            
            # Diagnostic script logic
            st.write("### 🔬 DIAGNOSTIC SCRIPT LOGIC:")
            
            # Calculate engineered features like diagnostic script
            is_peak_hour_calc = 1 if ((order_hour >= 11 and order_hour <= 14) or 
                                     (order_hour >= 17 and order_hour <= 20)) else 0
            
            # Default weekend calculation (since we don't have order_month)
            is_weekend_calc = 0  # Default to weekday
            
            st.write(f"**Calculated Peak Hour:** {is_peak_hour_calc} (based on hour {order_hour})")
            st.write(f"**Peak Hour Logic:** (11-14) OR (17-20) = {(order_hour >= 11 and order_hour <= 14) or (order_hour >= 17 and order_hour <= 20)}")
            st.write(f"**Calculated Weekend:** {is_weekend_calc} (default to weekday)")
            
            # Using calculated features
            input_values_calc = [pizza_complexity, order_hour, restaurant_avg_time, 
                               distance, topping_density, traffic_level, 
                               is_peak_hour_calc, is_weekend_calc]
            input_data_calc = np.array([input_values_calc])
            
            st.write(f"**Input values (calculated):** {input_values_calc}")
            st.write(f"**Input shape (calculated):** {input_data_calc.shape}")
            
            if model_data:
                predicted_duration_calc = model.predict(input_data_calc)[0]
                st.write(f"**Raw prediction (calculated):** {predicted_duration_calc}")
                st.success(f"**CALCULATED RESULT: {predicted_duration_calc:.1f} minutes**")
            
            st.markdown("---")
            
            # Model feature order check
            if model_data:
                st.write("### 🎯 MODEL FEATURE ORDER CHECK:")
                expected_features = model_data['features']
                st.write(f"**Model expects:** {expected_features}")
                
                # Create input using model's expected order
                feature_values = {
                    'Pizza Complexity': pizza_complexity,
                    'Order Hour': order_hour,
                    'Restaurant Avg Time': restaurant_avg_time,
                    'Distance (km)': distance,
                    'Topping Density': topping_density,
                    'Traffic Level': traffic_level,
                    'Is Peak Hour': is_peak_hour,
                    'Is Weekend': is_weekend
                }
                
                input_values_ordered = [feature_values[feature] for feature in expected_features]
                input_data_ordered = np.array([input_values_ordered])
                
                st.write(f"**Input values (model order):** {input_values_ordered}")
                st.write(f"**Input shape (model order):** {input_data_ordered.shape}")
                
                predicted_duration_ordered = model.predict(input_data_ordered)[0]
                st.write(f"**Raw prediction (model order):** {predicted_duration_ordered}")
                st.success(f"**MODEL ORDER RESULT: {predicted_duration_ordered:.1f} minutes**")
            
            st.markdown("---")
            
            # Comparison
            st.write("### ⚖️ COMPARISON:")
            st.write(f"**User Input Peak Hour:** {is_peak_hour}")
            st.write(f"**Calculated Peak Hour:** {is_peak_hour_calc}")
            st.write(f"**User Input Weekend:** {is_weekend}")
            st.write(f"**Calculated Weekend:** {is_weekend_calc}")
            
            if is_peak_hour != is_peak_hour_calc:
                st.warning("⚠️ Peak Hour values differ!")
            if is_weekend != is_weekend_calc:
                st.warning("⚠️ Weekend values differ!")
            
            # Show all predictions for comparison
            st.write("### 📊 ALL PREDICTIONS:")
            if model_data:
                results_df = pd.DataFrame({
                    'Method': ['Original App', 'Calculated Features', 'Model Order'],
                    'Prediction': [f"{predicted_duration_original:.1f}", 
                                 f"{predicted_duration_calc:.1f}", 
                                 f"{predicted_duration_ordered:.1f}"],
                    'Peak Hour': [is_peak_hour, is_peak_hour_calc, is_peak_hour],
                    'Weekend': [is_weekend, is_weekend_calc, is_weekend]
                })
                st.dataframe(results_df, use_container_width=True)
            
            # Test with specific values from diagnostic
            st.markdown("---")
            st.write("### 🧪 TEST WITH SPECIFIC VALUES:")
            
            # Test case 1: Peak hour should be 1 for hour 14
            test_hour = 14
            test_peak = 1 if ((test_hour >= 11 and test_hour <= 14) or 
                             (test_hour >= 17 and test_hour <= 20)) else 0
            st.write(f"**Test:** Hour {test_hour} should give Peak Hour = {test_peak}")
            
            # Test case 2: Non-peak hour should be 0 for hour 10
            test_hour2 = 10
            test_peak2 = 1 if ((test_hour2 >= 11 and test_hour2 <= 14) or 
                              (test_hour2 >= 17 and test_hour2 <= 20)) else 0
            st.write(f"**Test:** Hour {test_hour2} should give Peak Hour = {test_peak2}")

if __name__ == "__main__":
    main()
