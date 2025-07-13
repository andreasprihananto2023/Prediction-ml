import pandas as pd
import pickle
import os

print("🔍 DEBUGGING TARGET COLUMN MISMATCH")
print("=" * 50)

# Check the Excel file columns
try:
    data = pd.read_excel('Train Data.xlsx')
    print("📊 EXCEL FILE COLUMNS:")
    print("-" * 30)
    for i, col in enumerate(data.columns):
        print(f"{i+1}. '{col}'")
    
    print(f"\nTotal columns: {len(data.columns)}")
    
    # Look for potential target columns
    potential_targets = []
    for col in data.columns:
        if any(keyword in col.lower() for keyword in ['duration', 'time', 'delivery', 'estimated']):
            potential_targets.append(col)
    
    print(f"\n🎯 POTENTIAL TARGET COLUMNS:")
    print("-" * 30)
    for target in potential_targets:
        print(f"- '{target}'")
        print(f"  Sample values: {data[target].head(3).tolist()}")
        print(f"  Unique values: {data[target].nunique()}")
        print(f"  Data type: {data[target].dtype}")
        print()
    
except Exception as e:
    print(f"❌ Error loading Excel file: {e}")

# Check the model file
print("\n🤖 MODEL FILE ANALYSIS:")
print("-" * 30)
try:
    if os.path.exists('realistic_rf_model.pkl'):
        with open('realistic_rf_model.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        print("✅ Model file exists")
        print(f"Model type: {type(model_info)}")
        
        if isinstance(model_info, dict):
            print("📋 Model info keys:")
            for key in model_info.keys():
                print(f"  - {key}")
            
            if 'features' in model_info:
                print(f"\n🔧 Features used in model:")
                for i, feature in enumerate(model_info['features'], 1):
                    print(f"  {i}. {feature}")
            
            if 'target_column' in model_info:
                print(f"\n🎯 Target column used: '{model_info['target_column']}'")
            
            if 'model_performance' in model_info:
                print(f"\n📈 Model performance:")
                perf = model_info['model_performance']
                for key, value in perf.items():
                    print(f"  {key}: {value}")
        
    else:
        print("❌ Model file 'realistic_rf_model.pkl' not found")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Check diagnostic results if available
print("\n🔍 DIAGNOSTIC RESULTS:")
print("-" * 30)
try:
    if os.path.exists('comprehensive_diagnostic_results.pkl'):
        with open('comprehensive_diagnostic_results.pkl', 'rb') as f:
            diagnostic_info = pickle.load(f)
        
        print("✅ Diagnostic file exists")
        
        if 'dataset_info' in diagnostic_info:
            target_col = diagnostic_info['dataset_info'].get('target_column', 'Not found')
            print(f"🎯 Target column from diagnostic: '{target_col}'")
            
            columns = diagnostic_info['dataset_info'].get('columns', [])
            print(f"\n📊 All columns from diagnostic:")
            for i, col in enumerate(columns, 1):
                print(f"  {i}. '{col}'")
    else:
        print("❌ Diagnostic file not found")
        
except Exception as e:
    print(f"❌ Error loading diagnostic results: {e}")

# Recommendations
print("\n💡 RECOMMENDATIONS:")
print("-" * 30)
print("1. Check if the target column names match between:")
print("   - Excel file")
print("   - Model training script")
print("   - Diagnostic script")
print("   - Streamlit app")
print()
print("2. Common target column names to check:")
print("   - 'Delivery Duration (Min)'")
print("   - 'Delivery Duration (min)'")
print("   - 'Estimated Duration'")
print("   - 'Duration'")
print("   - 'Delivery Time'")
print()
print("3. If names are different, update the training script to use consistent naming")
print("4. Retrain the model with the correct target column")
print("5. Update the app.py to use the same model")
