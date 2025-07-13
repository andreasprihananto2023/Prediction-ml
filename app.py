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

# Load the model
model_data, error_message = load_model()

# Main app
def main():
    st.title("🍕 Pizza Delivery Time Predictor")
    st.markdown("---")
    
    if error_message:
        st.error(f"❌ {error_message}")
        st.info("📝 **To fix this issue:**")
        st.info("1. Make sure you have 'Train Data.xlsx' in the same directory")
        st.info("2. Run the training script to generate the model file")
        st.info("3. Then restart this Streamlit app")
        return
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.success("✅ Model loaded successfully!")
        
        perf = model_data['performance']
        if perf:
            st.metric("R² Score", f"{perf.get('test_r2', 0):.3f}")
            st.metric("MAE (minutes)", f"{perf.get('test_mae', 0):.2f}")
            st.metric("CV MAE", f"{perf.get('cv_mae', 0):.2f}")
            
            # Show prediction uncertainty
            if 'prediction_std' in perf:
                st.metric("Prediction Std", f"±{perf.get('prediction_std', 0):.1f} min")
        
        st.subheader("Features Used:")
        for i, feature in enumerate(model_data['features'], 1):
            st.write(f"{i}. {feature}")
    
    # Main content
    st.markdown("""
    This app predicts pizza delivery time based on various factors like pizza complexity, 
    order time, distance, and traffic conditions.
    """)
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("🔧 Enter Order Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pizza_complexity = st.slider(
                "Pizza Complexity", 
                min_value=1, max_value=5, value=3,
                help="1 = Simple, 5 = Very Complex"
            )
            
            order_hour = st.slider(
                "Order Hour", 
                min_value=0, max_value=23, value=14,
                help="Hour of the day (0-23)"
            )
            
            restaurant_avg_time = st.slider(
                "Restaurant Avg Time (minutes)", 
                min_value=10, max_value=60, value=25,
                help="Average preparation time for this restaurant"
            )
            
            distance = st.slider(
                "Distance (km)", 
                min_value=1, max_value=10, value=5,
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
                min_value=1, max_value=5, value=3,
                help="1 = No traffic, 5 = Heavy traffic"
            )
            
            # Remove manual input for peak hour and weekend
            # These should be calculated automatically
            
        # Calculate engineered features automatically
        is_peak_hour = 1 if ((order_hour >= 11 and order_hour <= 14) or 
                           (order_hour >= 17 and order_hour <= 20)) else 0
        
        # For weekend, we need order month - let's add this input
        order_month = st.slider(
            "Order Month", 
            min_value=1, max_value=12, value=6,
            help="Month of the year (1-12)"
        )
        
        is_weekend = 1 if order_month in [6, 7, 8, 9] else 0
        
        # Display calculated values
        st.info(f"🕐 Peak Hour: {'Yes' if is_peak_hour else 'No'} (automatically calculated)")
        st.info(f"📅 Weekend: {'Yes' if is_weekend else 'No'} (automatically calculated)")
        
        # Submit button
        submitted = st.form_submit_button("🚀 Predict Delivery Time", type="primary")
        
        if submitted:
            try:
                # Ensure feature order matches training
                # Based on diagnostic script, the order should be:
                # ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
                #  'Distance (km)', 'Topping Density', 'Traffic Level', 
                #  'Is Peak Hour', 'Is Weekend']
                
                input_data = np.array([[
                    pizza_complexity,      # Pizza Complexity
                    order_hour,           # Order Hour
                    restaurant_avg_time,  # Restaurant Avg Time
                    distance,             # Distance (km)
                    topping_density,      # Topping Density
                    traffic_level,        # Traffic Level
                    is_peak_hour,         # Is Peak Hour (calculated)
                    is_weekend            # Is Weekend (calculated)
                ]])
                
                # Verify input shape
                expected_features = len(model_data['features'])
                if input_data.shape[1] != expected_features:
                    st.error(f"Feature mismatch: Expected {expected_features}, got {input_data.shape[1]}")
                    return
                
                # Make prediction
                model = model_data['model']
                predicted_duration = model.predict(input_data)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                # Main result
                st.success(f"**🕐 Estimated Delivery Time: {predicted_duration:.1f} minutes**")
                
                # Time categorization
                if predicted_duration <= 20:
                    st.success("🟢 **Fast Delivery** - Excellent delivery time!")
                elif predicted_duration <= 30:
                    st.warning("🟡 **Normal Delivery** - Standard delivery time")
                else:
                    st.error("🔴 **Slow Delivery** - Longer than usual")
                
                # Convert to hours and minutes
                hours = int(predicted_duration // 60)
                minutes = int(predicted_duration % 60)
                
                if hours > 0:
                    st.info(f"📅 **Total Time**: {hours} hour(s) {minutes} minute(s)")
                else:
                    st.info(f"📅 **Total Time**: {minutes} minute(s)")
                
                # Show input summary
                with st.expander("📝 Input Summary"):
                    input_df = pd.DataFrame({
                        'Feature': model_data['features'],
                        'Value': [pizza_complexity, order_hour, restaurant_avg_time, 
                                distance, topping_density, traffic_level, 
                                is_peak_hour, is_weekend]
                    })
                    st.dataframe(input_df, use_container_width=True)
                
                # Show calculated features
                with st.expander("🔧 Calculated Features"):
                    st.write(f"**Peak Hour Logic**: Order hour {order_hour} -> {'Peak' if is_peak_hour else 'Not Peak'}")
                    st.write(f"**Weekend Logic**: Order month {order_month} -> {'Weekend' if is_weekend else 'Weekday'}")
                    st.write("Peak hours: 11-14 and 17-20")
                    st.write("Weekend months: 6, 7, 8, 9")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.error("Please check your input values and try again.")
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.write(f"Expected features: {model_data['features']}")
                    st.write(f"Input shape: {input_data.shape if 'input_data' in locals() else 'Not created'}")
                    st.write(f"Model type: {type(model_data['model'])}")
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ Feature Information")
    
    feature_info = {
        "Pizza Complexity": "Difficulty level of pizza preparation (1-5)",
        "Order Hour": "Hour of the day when order is placed (0-23)",
        "Restaurant Avg Time": "Average preparation time for the restaurant",
        "Distance (km)": "Delivery distance in kilometers",
        "Topping Density": "Amount of toppings on the pizza (1-5)",
        "Traffic Level": "Current traffic conditions (1-5)",
        "Is Peak Hour": "Automatically calculated: 11-14 and 17-20",
        "Is Weekend": "Automatically calculated: months 6,7,8,9"
    }
    
    for feature, description in feature_info.items():
        st.write(f"**{feature}**: {description}")

    # Additional debugging section
    if st.checkbox("🔍 Show Debug Information"):
        st.subheader("Debug Information")
        st.write("**Model Features:**")
        st.write(model_data['features'])
        st.write("**Model Performance:**")
        st.write(model_data['performance'])

if __name__ == "__main__":
    main()
