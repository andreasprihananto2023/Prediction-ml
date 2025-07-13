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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #ff6b6b, #ffa726);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #ff6b6b;
    }
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        if not os.path.exists('realistic_rf_model.pkl'):
            return None, "Model file not found. Please ensure the model is trained first."
        
        with open('realistic_rf_model.pkl', 'rb') as model_file:
            model_info = pickle.load(model_file)
        
        if isinstance(model_info, dict):
            return {
                'model': model_info['model'],
                'features': model_info['features'],
                'performance': model_info.get('model_performance', {})
            }, None
        else:
            return None, "Invalid model format."
            
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# Load model
model_data, error_message = load_model()

def get_time_category(hour):
    """Get time category for better UX"""
    if 6 <= hour <= 10:
        return "🌅 Morning"
    elif 11 <= hour <= 14:
        return "🌞 Lunch Time"
    elif 15 <= hour <= 17:
        return "☕ Afternoon"
    elif 18 <= hour <= 22:
        return "🌆 Dinner Time"
    else:
        return "🌙 Late Night"

def get_complexity_description(complexity):
    """Get pizza complexity description"""
    descriptions = {
        1: "🍕 Simple (Margherita, Pepperoni)",
        2: "🍕 Basic (Hawaiian, Mushroom)",
        3: "🍕 Standard (Veggie, Meat Lovers)",
        4: "🍕 Complex (Gourmet, Specialty)",
        5: "🍕 Premium (Custom, Multiple Toppings)"
    }
    return descriptions.get(complexity, "🍕 Pizza")

def get_traffic_description(level):
    """Get traffic level description"""
    descriptions = {
        1: "🟢 Very Light",
        2: "🟡 Light",
        3: "🟠 Moderate",
        4: "🔴 Heavy",
        5: "🔴 Very Heavy"
    }
    return descriptions.get(level, "Traffic")

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍕 Pizza Delivery Time Predictor</h1>
        <p>Get accurate delivery time estimates for your pizza orders</p>
    </div>
    """, unsafe_allow_html=True)
    
    if error_message:
        st.error(f"❌ {error_message}")
        st.info("Please ensure the model is properly trained and the file 'realistic_rf_model.pkl' exists.")
        return
    
    # Show model performance if available
    if model_data and model_data['performance']:
        col1, col2, col3 = st.columns(3)
        perf = model_data['performance']
        
        with col1:
            st.metric("Model Accuracy", f"{perf.get('r2_score', 0):.2%}")
        with col2:
            st.metric("Average Error", f"{perf.get('mae', 0):.1f} min")
        with col3:
            st.metric("Model Type", "Random Forest")
    
    st.markdown("---")
    
    # Main prediction form
    with st.form("prediction_form"):
        st.subheader("📋 Order Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🍕 Pizza Information")
            pizza_complexity = st.select_slider(
                "Pizza Complexity",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=get_complexity_description
            )
            
            topping_density = st.select_slider(
                "Topping Density",
                options=[1, 2, 3, 4, 5],
                value=2,
                format_func=lambda x: f"Level {x} - {'Light' if x <= 2 else 'Medium' if x <= 3 else 'Heavy'}"
            )
            
            st.markdown("### 🏪 Restaurant Information")
            restaurant_avg_time = st.slider(
                "Restaurant Average Preparation Time (minutes)",
                min_value=10,
                max_value=60,
                value=25,
                step=5,
                help="Average time this restaurant takes to prepare orders"
            )
        
        with col2:
            st.markdown("### 🕐 Time & Location")
            order_hour = st.select_slider(
                "Order Hour",
                options=list(range(24)),
                value=14,
                format_func=lambda x: f"{x:02d}:00 - {get_time_category(x)}"
            )
            
            distance = st.slider(
                "Distance from Restaurant (km)",
                min_value=1,
                max_value=10,
                value=5,
                help="Distance between restaurant and delivery location"
            )
            
            st.markdown("### 🚗 Traffic & Timing")
            traffic_level = st.select_slider(
                "Current Traffic Level",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=get_traffic_description
            )
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                is_peak_hour = st.selectbox(
                    "Peak Hour?",
                    options=[0, 1],
                    index=1,
                    format_func=lambda x: "Yes (11-14 or 17-20)" if x == 1 else "No"
                )
            
            with col2_2:
                is_weekend = st.selectbox(
                    "Weekend?",
                    options=[0, 1],
                    index=0,
                    format_func=lambda x: "Yes (Summer months)" if x == 1 else "No"
                )
        
        # Prediction button
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            submitted = st.form_submit_button("🚀 Predict Delivery Time", type="primary", use_container_width=True)
        
        if submitted:
            try:
                # Prepare input data
                input_values = [pizza_complexity, order_hour, restaurant_avg_time, 
                              distance, topping_density, traffic_level, 
                              is_peak_hour, is_weekend]
                
                input_data = np.array([input_values])
                
                # Make prediction
                model = model_data['model']
                predicted_duration = model.predict(input_data)[0]
                
                # Display result
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>⏱️ Predicted Delivery Time</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{predicted_duration:.0f} minutes</h1>
                    <p style="font-size: 1.2rem;">Estimated delivery at {(datetime.now()).strftime('%H:%M')} + {predicted_duration:.0f} min</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show order summary
                st.subheader("📋 Order Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>🍕 Pizza Details</h4>
                        <p><strong>Complexity:</strong> {get_complexity_description(pizza_complexity)}</p>
                        <p><strong>Topping Density:</strong> Level {topping_density}</p>
                        <p><strong>Prep Time:</strong> {restaurant_avg_time} minutes</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>🚗 Delivery Details</h4>
                        <p><strong>Order Time:</strong> {order_hour:02d}:00 - {get_time_category(order_hour)}</p>
                        <p><strong>Distance:</strong> {distance} km</p>
                        <p><strong>Traffic:</strong> {get_traffic_description(traffic_level)}</p>
                        <p><strong>Peak Hour:</strong> {'Yes' if is_peak_hour else 'No'}</p>
                        <p><strong>Weekend:</strong> {'Yes' if is_weekend else 'No'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Factors affecting delivery time
                st.subheader("📊 Factors Affecting Delivery Time")
                
                factors = []
                if is_peak_hour:
                    factors.append("🔴 Peak hour increases delivery time")
                if traffic_level >= 4:
                    factors.append("🔴 Heavy traffic increases delivery time")
                if distance >= 7:
                    factors.append("🔴 Long distance increases delivery time")
                if pizza_complexity >= 4:
                    factors.append("🔴 Complex pizza increases preparation time")
                if restaurant_avg_time >= 35:
                    factors.append("🔴 Restaurant has longer preparation time")
                
                if not factors:
                    factors.append("🟢 Optimal conditions for fast delivery")
                
                for factor in factors:
                    st.markdown(f"• {factor}")
                
                # Quick tips
                st.markdown("""
                <div class="info-box">
                    <h4>💡 Quick Tips for Faster Delivery:</h4>
                    <ul>
                        <li>Order during off-peak hours (avoid 11-14 and 17-20)</li>
                        <li>Choose simpler pizzas for faster preparation</li>
                        <li>Consider ordering from closer restaurants</li>
                        <li>Check traffic conditions before ordering</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check your input values and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🍕 Pizza Delivery Time Predictor | Powered by Machine Learning</p>
        <p><small>Predictions are estimates based on historical data and may vary with real-world conditions</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
