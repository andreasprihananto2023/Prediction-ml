import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Configure page
st.set_page_config(
    page_title="Pizza Delivery Time Predictor - Debug Mode",
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
                'performance': performance,
                'full_info': model_info  # Keep full info for debugging
            }, None
        else:
            return None, "Invalid model format. Please retrain the model."
            
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# Load diagnostic results for comparison
@st.cache_resource
def load_diagnostic_results():
    """Load diagnostic results if available"""
    try:
        if os.path.exists('comprehensive_diagnostic_results.pkl'):
            with open('comprehensive_diagnostic_results.pkl', 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        return None

# Load original training data for comparison
@st.cache_resource
def load_training_data():
    """Load original training data"""
    try:
        if os.path.exists('Train Data.xlsx'):
            data = pd.read_excel('Train Data.xlsx')
            
            # Add engineered features (same as diagnostic)
            data['Is Peak Hour'] = np.where(((data['Order Hour'] >= 11) & (data['Order Hour'] <= 14)) |
                                           ((data['Order Hour'] >= 17) & (data['Order Hour'] <= 20)), 1, 0)
            data['Is Weekend'] = np.where(data['Order Month'].isin([6, 7, 8, 9]), 1, 0)
            
            return data
        return None
    except Exception as e:
        return None

# Load everything
model_data, error_message = load_model()
diagnostic_results = load_diagnostic_results()
training_data = load_training_data()

def main():
    st.title("🍕 Pizza Delivery Predictor - Debug Mode")
    st.markdown("---")
    
    if error_message:
        st.error(f"❌ {error_message}")
        return
    
    # Debug information
    st.subheader("🔍 Debug Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Info")
        if model_data:
            st.write(f"**Features**: {model_data['features']}")
            st.write(f"**Model Type**: {type(model_data['model'])}")
            st.write(f"**Performance**: {model_data['performance']}")
    
    with col2:
        st.subheader("📋 Diagnostic Info")
        if diagnostic_results:
            st.write(f"**Dataset Shape**: {diagnostic_results['dataset_info']['shape']}")
            st.write(f"**Target**: {diagnostic_results['dataset_info']['target_column']}")
            st.write(f"**Unique Values**: {diagnostic_results['target_analysis']['unique_values']}")
            st.write(f"**Mean**: {diagnostic_results['target_analysis']['mean']:.2f}")
            st.write(f"**Std**: {diagnostic_results['target_analysis']['std']:.2f}")
    
    # Sample data comparison
    if training_data is not None:
        st.subheader("📋 Sample Training Data")
        features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
                   'Distance (km)', 'Topping Density', 'Traffic Level', 
                   'Is Peak Hour', 'Is Weekend']
        
        # Find target column
        target_col = 'Delivery Duration (Min)'
        if target_col not in training_data.columns:
            possible_targets = ['Delivery Duration (min)', 'Duration', 'Delivery Time', 'Delivery Duration']
            for alt_target in possible_targets:
                if alt_target in training_data.columns:
                    target_col = alt_target
                    break
        
        sample_data = training_data[features + [target_col]].head(10)
        st.dataframe(sample_data, use_container_width=True)
    
    st.markdown("---")
    
    # Prediction form with debugging
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
        
        submitted = st.form_submit_button("🚀 Predict & Debug", type="primary")
        
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
                
                # Debug section
                st.markdown("---")
                st.subheader("🔍 Debug Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Model Prediction")
                    st.success(f"**Predicted Duration: {predicted_duration:.1f} minutes**")
                    
                    # Show input array
                    st.subheader("📋 Input Array")
                    st.write(f"Shape: {input_data.shape}")
                    st.write(f"Values: {input_values}")
                    
                    # Show feature mapping
                    st.subheader("🔗 Feature Mapping")
                    feature_df = pd.DataFrame({
                        'Feature': model_data['features'],
                        'Value': input_values
                    })
                    st.dataframe(feature_df, use_container_width=True)
                
                with col2:
                    st.subheader("🔍 Training Data Matches")
                    
                    if training_data is not None:
                        # Find exact matches in training data
                        features = model_data['features']
                        target_col = 'Delivery Duration (Min)'
                        if target_col not in training_data.columns:
                            possible_targets = ['Delivery Duration (min)', 'Duration', 'Delivery Time', 'Delivery Duration']
                            for alt_target in possible_targets:
                                if alt_target in training_data.columns:
                                    target_col = alt_target
                                    break
                        
                        # Create filter condition
                        condition = (
                            (training_data['Pizza Complexity'] == pizza_complexity) &
                            (training_data['Order Hour'] == order_hour) &
                            (training_data['Restaurant Avg Time'] == restaurant_avg_time) &
                            (training_data['Distance (km)'] == distance) &
                            (training_data['Topping Density'] == topping_density) &
                            (training_data['Traffic Level'] == traffic_level) &
                            (training_data['Is Peak Hour'] == is_peak_hour) &
                            (training_data['Is Weekend'] == is_weekend)
                        )
                        
                        matches = training_data[condition]
                        
                        if len(matches) > 0:
                            st.success(f"**Found {len(matches)} exact matches in training data:**")
                            actual_values = matches[target_col].values
                            st.write(f"Actual values: {actual_values}")
                            st.write(f"Predicted: {predicted_duration:.1f}")
                            
                            if len(set(actual_values)) == 1:
                                actual_val = actual_values[0]
                                diff = abs(predicted_duration - actual_val)
                                st.write(f"**Difference**: {diff:.6f}")
                                
                                if diff < 0.001:
                                    st.success("✅ Perfect match!")
                                elif diff < 0.1:
                                    st.warning("⚠️ Very close match")
                                else:
                                    st.error("❌ Significant difference!")
                            else:
                                st.info(f"Multiple actual values: {list(set(actual_values))}")
                        else:
                            st.warning("⚠️ No exact matches found in training data")
                            
                            # Find closest matches
                            st.subheader("🔍 Closest Matches")
                            # Calculate similarity score for each row
                            similarities = []
                            for idx, row in training_data.iterrows():
                                score = sum([
                                    abs(row['Pizza Complexity'] - pizza_complexity),
                                    abs(row['Order Hour'] - order_hour),
                                    abs(row['Restaurant Avg Time'] - restaurant_avg_time),
                                    abs(row['Distance (km)'] - distance),
                                    abs(row['Topping Density'] - topping_density),
                                    abs(row['Traffic Level'] - traffic_level),
                                    abs(row['Is Peak Hour'] - is_peak_hour),
                                    abs(row['Is Weekend'] - is_weekend)
                                ])
                                similarities.append((idx, score))
                            
                            # Sort by similarity (lowest score = most similar)
                            similarities.sort(key=lambda x: x[1])
                            
                            # Show top 3 most similar
                            st.write("**Top 3 most similar rows:**")
                            for i, (idx, score) in enumerate(similarities[:3]):
                                row = training_data.iloc[idx]
                                st.write(f"{i+1}. Similarity score: {score:.1f}, Target: {row[target_col]:.1f}")
                                st.write(f"   Features: {[row[f] for f in features]}")
                
                # Test with some known values from training data
                if training_data is not None:
                    st.subheader("🧪 Test with Known Training Examples")
                    test_examples = training_data.head(3)
                    
                    for i, (idx, row) in enumerate(test_examples.iterrows()):
                        test_input = np.array([[row[f] for f in features]])
                        test_prediction = model.predict(test_input)[0]
                        actual_value = row[target_col]
                        
                        st.write(f"**Example {i+1}:**")
                        st.write(f"  Input: {[row[f] for f in features]}")
                        st.write(f"  Actual: {actual_value:.1f}, Predicted: {test_prediction:.1f}")
                        st.write(f"  Difference: {abs(test_prediction - actual_value):.6f}")
                        
                        if abs(test_prediction - actual_value) < 0.001:
                            st.success("  ✅ Perfect match!")
                        else:
                            st.error("  ❌ Mismatch detected!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
