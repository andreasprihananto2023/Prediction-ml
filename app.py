import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Configure page
st.set_page_config(
    page_title="Pizza Delivery Time Predictor",
    page_icon="🍕",
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
    st.title("🍕 Pizza Delivery Time Predictor")
    st.markdown("---")
    
    if error_message:
        st.error(f"❌ {error_message}")
        return
    
    # Show model performance if available
    if model_data and model_data['performance']:
        col1, col2, col3 = st.columns(3)
        perf = model_data['performance']
        
        with col1:
            if 'r2_score' in perf:
                st.metric("Model R² Score", f"{perf['r2_score']:.3f}")
        with col2:
            if 'mae' in perf:
                st.metric("Mean Abs Error", f"{perf['mae']:.1f} min")
        with col3:
            if 'rmse' in perf:
                st.metric("RMSE", f"{perf['rmse']:.1f} min")
    
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
        
        submitted = st.form_submit_button("🚀 Predict Delivery Time", type="primary")
        
        if submitted:
            try:
                # Prepare input data (EXACT same order as debug version)
                input_values = [pizza_complexity, order_hour, restaurant_avg_time, 
                              distance, topping_density, traffic_level, 
                              is_peak_hour, is_weekend]
                
                input_data = np.array([input_values])
                
                # Make prediction
                model = model_data['model']
                predicted_duration = model.predict(input_data)[0]
                
                # Debug info (temporary - remove after verification)
                st.write(f"DEBUG - Input values: {input_values}")
                st.write(f"DEBUG - Input shape: {input_data.shape}")
                st.write(f"DEBUG - Features order: {model_data['features']}")
                st.write(f"DEBUG - Raw prediction: {predicted_duration}")
                
                # Display result
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**Predicted Delivery Time: {predicted_duration:.1f} minutes**")
                    
                    # Show input summary
                    st.subheader("📋 Order Summary")
                    feature_df = pd.DataFrame({
                        'Feature': model_data['features'],
                        'Value': input_values
                    })
                    st.dataframe(feature_df, use_container_width=True)
                
                with col2:
                    st.subheader("📈 Delivery Factors")
                    
                    # Simple factor analysis
                    factors = []
                    if is_peak_hour:
                        factors.append("⏰ Peak hour may increase delivery time")
                    if traffic_level >= 4:
                        factors.append("🚗 Heavy traffic detected")
                    if distance >= 7:
                        factors.append("📍 Long distance delivery")
                    if pizza_complexity >= 4:
                        factors.append("🍕 Complex pizza preparation")
                    if restaurant_avg_time >= 35:
                        factors.append("🏪 Restaurant has longer prep time")
                    
                    if not factors:
                        factors.append("✅ Optimal conditions for delivery")
                    
                    for factor in factors:
                        st.write(f"• {factor}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
