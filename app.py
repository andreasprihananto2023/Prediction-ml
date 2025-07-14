import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Pizza Delivery Time Predictor",
    page_icon="🍕",
    layout="wide"
)

# Load model with better error handling and debugging
@st.cache_resource
def load_model():
    """Load the trained model with enhanced debugging"""
    try:
        model_file = 'realistic_rf_model.pkl'
        
        # Check if file exists
        if not os.path.exists(model_file):
            return None, f"❌ Model file '{model_file}' not found. Please run realistic_training.py first."
        
        # Check file age
        file_age = datetime.now().timestamp() - os.path.getmtime(model_file)
        hours_old = file_age / 3600
        
        # Load model
        with open(model_file, 'rb') as f:
            model_info = pickle.load(f)
        
        # Validate model structure
        if not isinstance(model_info, dict):
            return None, "❌ Invalid model format. Please retrain the model with realistic_training.py."
        
        required_keys = ['model', 'features', 'model_performance']
        missing_keys = [key for key in required_keys if key not in model_info]
        if missing_keys:
            return None, f"❌ Model missing required keys: {missing_keys}. Please retrain the model."
        
        # Extract model components
        model = model_info['model']
        features = model_info['features']
        performance = model_info.get('model_performance', {})
        
        # Validate features
        expected_features = [
            'Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
            'Distance (km)', 'Topping Density', 'Traffic Level', 
            'Is Peak Hour', 'Is Weekend'
        ]
        
        if features != expected_features:
            return None, f"❌ Feature mismatch. Expected: {expected_features}, Got: {features}"
        
        return {
            'model': model,
            'features': features,
            'n_features': len(features),
            'performance': performance,
            'file_age_hours': hours_old,
            'model_type': model_info.get('model_type', 'Unknown'),
            'noise_added': model_info.get('noise_added', False)
        }, None
        
    except Exception as e:
        return None, f"❌ Error loading model: {str(e)}"

# Load the model
model_data, error_message = load_model()

# Main app
def main():
    st.title("🍕 Pizza Delivery Time Predictor")
    st.markdown("---")
    
    # Model status section
    st.subheader("📊 Model Status")
    
    if error_message:
        st.error(error_message)
        st.info("📝 **To fix this issue:**")
        st.info("1. Navigate to your project directory")
        st.info("2. Run: `python realistic_training.py`")
        st.info("3. Wait for training to complete")
        st.info("4. Restart this Streamlit app")
        
        # Show available files for debugging
        files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if files:
            st.info(f"📁 Available .pkl files: {files}")
        else:
            st.warning("📁 No .pkl files found in current directory")
        
        return
    
    # Model loaded successfully
    st.success("✅ Model loaded successfully!")
    
    # Detailed model info
    with st.expander("🔍 Model Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Type", model_data['model_type'])
            st.metric("Features Count", model_data['n_features'])
            st.metric("File Age", f"{model_data['file_age_hours']:.1f} hours")
        
        with col2:
            perf = model_data['performance']
            if perf:
                st.metric("Test R²", f"{perf.get('test_r2', 0):.3f}")
                st.metric("Test MAE", f"{perf.get('test_mae', 0):.1f} min")
                st.metric("CV MAE", f"{perf.get('cv_mae', 0):.1f} min")
        
        # Show if noise was added
        if model_data.get('noise_added', False):
            st.info("🔊 This model was trained with added noise for more realistic predictions")
        
        # Show features
        st.write("**Features Used:**")
        for i, feature in enumerate(model_data['features'], 1):
            st.write(f"{i}. {feature}")
    
    # Sidebar with quick stats
    with st.sidebar:
        st.header("📊 Quick Stats")
        perf = model_data['performance']
        if perf:
            st.metric("Accuracy (R²)", f"{perf.get('test_r2', 0):.1%}")
            st.metric("Avg Error", f"±{perf.get('test_mae', 0):.1f} min")
            
            # Performance interpretation
            test_r2 = perf.get('test_r2', 0)
            if test_r2 > 0.9:
                st.warning("⚠️ Very high accuracy - may indicate overfitting")
            elif test_r2 > 0.7:
                st.success("✅ Good model performance")
            else:
                st.info("📈 Moderate performance")
    
    # Main prediction interface
    st.markdown("""
    This app predicts pizza delivery time based on various factors. The model has been trained
    with realistic noise to provide more practical predictions.
    """)
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("🔧 Enter Order Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pizza_complexity = st.slider(
                "Pizza Complexity", 
                min_value=1, max_value=5, value=3,
                help="1 = Simple (Margherita), 5 = Very Complex (Supreme with extra toppings)"
            )
            
            order_hour = st.slider(
                "Order Hour", 
                min_value=0, max_value=23, value=18,
                help="Hour of the day (0-23). Peak hours are 11-14 and 17-20"
            )
            
            restaurant_avg_time = st.slider(
                "Restaurant Avg Time (minutes)", 
                min_value=10, max_value=60, value=25,
                help="Average preparation time for this restaurant"
            )
            
            distance = st.slider(
                "Distance (km)", 
                min_value=1, max_value=10, value=3,
                help="Delivery distance in kilometers"
            )
        
        with col2:
            topping_density = st.slider(
                "Topping Density", 
                min_value=1, max_value=5, value=2,
                help="1 = Light toppings, 5 = Heavy toppings"
            )
            
            traffic_level = st.slider(
                "Traffic Level", 
                min_value=1, max_value=5, value=2,
                help="1 = No traffic, 5 = Heavy traffic"
            )
            
            is_peak_hour = st.selectbox(
                "Peak Hour?", 
                options=[0, 1], 
                index=0 if not ((11 <= order_hour <= 14) or (17 <= order_hour <= 20)) else 1,
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Peak hours: 11-14 (lunch) and 17-20 (dinner)"
            )
            
            is_weekend = st.selectbox(
                "Weekend?", 
                options=[0, 1], 
                index=0,
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Weekend affects delivery patterns"
            )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Predict Delivery Time", type="primary")
        
        if submitted:
            try:
                # Prepare input data (EXACT same order as training)
                input_data = np.array([[
                    pizza_complexity,      # Index 0: Pizza Complexity
                    order_hour,           # Index 1: Order Hour  
                    restaurant_avg_time,  # Index 2: Restaurant Avg Time
                    distance,             # Index 3: Distance (km)
                    topping_density,      # Index 4: Topping Density
                    traffic_level,        # Index 5: Traffic Level
                    is_peak_hour,         # Index 6: Is Peak Hour
                    is_weekend            # Index 7: Is Weekend
                ]])
                
                # Debug: Show input array
                st.write("**Debug - Input Array:**")
                st.code(f"Input: {input_data[0].tolist()}")
                
                # Make prediction
                model = model_data['model']
                predicted_duration = model.predict(input_data)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                # Main result
                st.success(f"**🕐 Estimated Delivery Time: {predicted_duration:.1f} minutes**")
                
                # Add confidence interval if available
                if 'prediction_std' in model_data['performance']:
                    std = model_data['performance']['prediction_std']
                    lower = max(5, predicted_duration - std)  # Minimum 5 minutes
                    upper = predicted_duration + std
                    st.info(f"**📊 Confidence Range: {lower:.1f} - {upper:.1f} minutes**")
                
                # Time categorization
                if predicted_duration <= 20:
                    st.success("🟢 **Fast Delivery** - Excellent!")
                elif predicted_duration <= 35:
                    st.warning("🟡 **Normal Delivery** - Standard time")
                else:
                    st.error("🔴 **Slow Delivery** - Longer than usual")
                
                # Convert to hours and minutes for user-friendly display
                hours = int(predicted_duration // 60)
                minutes = int(predicted_duration % 60)
                
                if hours > 0:
                    st.info(f"📅 **Total Time**: {hours} hour(s) {minutes} minute(s)")
                else:
                    st.info(f"📅 **Total Time**: {minutes} minute(s)")
                
                # Show prediction breakdown
                with st.expander("🔍 Prediction Breakdown"):
                    # Input summary
                    input_df = pd.DataFrame({
                        'Feature': model_data['features'],
                        'Value': [pizza_complexity, order_hour, restaurant_avg_time, 
                                distance, topping_density, traffic_level, 
                                is_peak_hour, is_weekend],
                        'Description': [
                            f"Complexity level {pizza_complexity}/5",
                            f"{order_hour}:00 ({'Peak' if is_peak_hour else 'Off-peak'})",
                            f"{restaurant_avg_time} minutes prep time",
                            f"{distance} km distance",
                            f"Topping density {topping_density}/5", 
                            f"Traffic level {traffic_level}/5",
                            "Peak hour" if is_peak_hour else "Off-peak hour",
                            "Weekend" if is_weekend else "Weekday"
                        ]
                    })
                    st.dataframe(input_df, use_container_width=True)
                    
                    # Performance info
                    st.write("**Model Performance:**")
                    perf = model_data['performance']
                    st.write(f"- Test Accuracy (R²): {perf.get('test_r2', 0):.1%}")
                    st.write(f"- Average Error: ±{perf.get('test_mae', 0):.1f} minutes")
                    st.write(f"- Cross-Validation Error: ±{perf.get('cv_mae', 0):.1f} minutes")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.error("🔧 This might indicate a model compatibility issue. Try retraining the model.")
                
                # Debug info
                st.write("**Debug Information:**")
                st.write(f"- Model type: {type(model_data['model'])}")
                st.write(f"- Input shape: {input_data.shape}")
                st.write(f"- Features expected: {len(model_data['features'])}")
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ How It Works")
    
    st.markdown("""
    This predictor uses a Random Forest model trained on pizza delivery data with the following features:
    
    - **Pizza Complexity**: How difficult the pizza is to prepare (affects kitchen time)
    - **Order Hour**: Time of day (affects restaurant workload and traffic)
    - **Restaurant Avg Time**: Baseline preparation time for the restaurant
    - **Distance**: Delivery distance in kilometers
    - **Topping Density**: Amount of toppings (affects preparation time)
    - **Traffic Level**: Current traffic conditions
    - **Peak Hour**: Whether it's during busy periods (11-14, 17-20)
    - **Weekend**: Weekend vs weekday patterns
    
    The model provides realistic predictions by accounting for real-world variability in delivery times.
    """)
    
    # Test the model with a known case
    if st.button("🧪 Test Model with Sample Data"):
        # Use the same test case as in realistic_training.py
        test_data = np.array([[3, 14, 25, 5, 2, 3, 1, 0]])
        test_prediction = model_data['model'].predict(test_data)[0]
        
        st.info(f"**Test Case Result**: {test_prediction:.1f} minutes")
        st.write("Input: Complexity=3, Hour=14, RestTime=25, Distance=5, Density=2, Traffic=3, Peak=1, Weekend=0")
        
        # Compare with expected range
        expected_min, expected_max = 20, 50  # Reasonable range for this input
        if expected_min <= test_prediction <= expected_max:
            st.success("✅ Test prediction looks reasonable")
        else:
            st.warning(f"⚠️ Test prediction seems unusual (expected {expected_min}-{expected_max} minutes)")

if __name__ == "__main__":
    main()
