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

def calculate_engineered_features(order_hour, order_month=None):
    """Calculate engineered features consistently with training"""
    
    # Peak hour calculation (same as diagnostic script)
    is_peak_hour = 1 if ((order_hour >= 11 and order_hour <= 14) or 
                         (order_hour >= 17 and order_hour <= 20)) else 0
    
    # Weekend calculation - if no month provided, use current logic
    # Note: The diagnostic script uses months 6,7,8,9 as weekend which seems incorrect
    # This might be a day-of-week encoding issue. Let's use a more realistic approach
    if order_month is None:
        # Default to weekday (0) since we don't have day-of-week info
        is_weekend = 0
    else:
        # Use the same logic as diagnostic script for consistency
        is_weekend = 1 if order_month in [6, 7, 8, 9] else 0
    
    return is_peak_hour, is_weekend

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
    
    # Show feature order from model
    if model_data:
        st.info(f"📋 Model expects features in this order: {model_data['features']}")
    
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
            order_month = st.selectbox("Order Month", options=list(range(1, 13)), index=5)
            
            # Show calculated engineered features
            calc_peak, calc_weekend = calculate_engineered_features(order_hour, order_month)
            st.info(f"📊 Calculated: Peak Hour = {calc_peak}, Weekend = {calc_weekend}")
        
        submitted = st.form_submit_button("🚀 Predict Delivery Time", type="primary")
        
        if submitted:
            try:
                # Calculate engineered features
                is_peak_hour, is_weekend = calculate_engineered_features(order_hour, order_month)
                
                # Prepare input data using the EXACT same order as the model expects
                model_features = model_data['features']
                
                # Create a dictionary mapping feature names to values
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
                
                # Create input array in the exact order expected by the model
                input_values = [feature_values[feature] for feature in model_features]
                input_data = np.array([input_values])
                
                # Make prediction
                model = model_data['model']
                predicted_duration = model.predict(input_data)[0]
                
                # Debug info
                st.write("🔍 **DEBUG INFORMATION:**")
                st.write(f"Model expects features: {model_features}")
                st.write(f"Input values provided: {input_values}")
                st.write(f"Input shape: {input_data.shape}")
                st.write(f"Raw prediction: {predicted_duration}")
                
                # Display result
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**Predicted Delivery Time: {predicted_duration:.1f} minutes**")
                    
                    # Show input summary
                    st.subheader("📋 Order Summary")
                    feature_df = pd.DataFrame({
                        'Feature': model_features,
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
                
                # Data consistency check
                st.markdown("---")
                st.subheader("🔍 Data Consistency Check")
                
                # Check if this combination might be in training data
                combo_hash = hash(tuple(input_values))
                st.write(f"Input combination hash: {combo_hash}")
                
                # Show realistic range check
                if predicted_duration < 5:
                    st.warning("⚠️ Prediction seems too low for realistic delivery time")
                elif predicted_duration > 120:
                    st.warning("⚠️ Prediction seems too high for realistic delivery time")
                else:
                    st.success("✅ Prediction is within realistic range")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.write("Please check that the model file is compatible with this application.")

if __name__ == "__main__":
    main()
