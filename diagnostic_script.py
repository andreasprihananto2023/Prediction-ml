import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import pickle

print("🔍 DIAGNOSTIC ANALYSIS FOR PERFECT R² SCORE")
print("=" * 60)

# Load data
try:
    data = pd.read_excel('Train Data.xlsx')
    print(f"✅ Data loaded: {data.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

print("\n📊 DATASET OVERVIEW")
print("-" * 30)
print(f"Dataset shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")

# Check for duplicates
duplicate_count = data.duplicated().sum()
print(f"Duplicate rows: {duplicate_count}")

# Basic statistics
print("\n📈 BASIC STATISTICS")
print("-" * 30)
print(data.describe())

# Check for missing values
print("\n❓ MISSING VALUES")
print("-" * 30)
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Add engineered features
data['Is Peak Hour'] = np.where(((data['Order Hour'] >= 11) & (data['Order Hour'] <= 14)) |
                                 ((data['Order Hour'] >= 17) & (data['Order Hour'] <= 20)), 1, 0)
data['Is Weekend'] = np.where(data['Order Month'].isin([6, 7, 8, 9]), 1, 0)

# Define features (WITHOUT target)
features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
            'Distance (km)', 'Topping Density', 'Traffic Level', 
            'Is Peak Hour', 'Is Weekend']

# Changed target to 'Delivery Duration'
target = 'Delivery Duration'

# Check if target column exists, if not try alternatives
if target not in data.columns:
    # Try common alternatives
    possible_targets = ['Delivery Duration', 'Estimated Duration (min)', 'Duration', 'Delivery Time']
    target_found = False
    
    for alt_target in possible_targets:
        if alt_target in data.columns:
            target = alt_target
            target_found = True
            print(f"✅ Using target column: '{target}'")
            break
    
    if not target_found:
        print(f"❌ Target column not found. Available columns: {data.columns.tolist()}")
        exit()
else:
    print(f"✅ Using target column: '{target}'")

print(f"\n🎯 TARGET VARIABLE ANALYSIS")
print("-" * 30)
print(f"Target variable: {target}")
print(f"Target statistics:")
print(data[target].describe())

# Check for constant or near-constant values
print(f"\nTarget unique values: {data[target].nunique()}")
print(f"Target variance: {data[target].var():.6f}")

# Correlation analysis
print(f"\n🔗 CORRELATION ANALYSIS")
print("-" * 30)
feature_data = data[features + [target]]
correlation_matrix = feature_data.corr()
print("Correlation with target:")
target_corr = correlation_matrix[target].sort_values(ascending=False)
print(target_corr)

# Check for perfect correlations (potential data leakage)
print(f"\n⚠️  CHECKING FOR DATA LEAKAGE")
print("-" * 30)
high_corr_features = target_corr[abs(target_corr) > 0.95]
if len(high_corr_features) > 1:  # >1 because target correlates with itself
    print("🚨 HIGH CORRELATION FEATURES (potential data leakage):")
    for feature, corr in high_corr_features.items():
        if feature != target:
            print(f"  {feature}: {corr:.6f}")
else:
    print("✅ No obvious data leakage detected")

# Prepare data
X = data[features].copy()
y = data[target].copy()

# Remove NaN values
mask = ~(X.isnull().any(axis=1) | y.isnull())
X_clean = X[mask]
y_clean = y[mask]

print(f"\n📊 DATA PREPARATION")
print("-" * 30)
print(f"Original data: {X.shape}")
print(f"After cleaning: {X_clean.shape}")

# Split data with different random states to test consistency
print(f"\n🔄 TESTING WITH DIFFERENT DATA SPLITS")
print("-" * 30)

results = []
for random_state in [42, 123, 999]:
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=random_state
    )
    
    # Train simple model
    rf_simple = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_simple.fit(X_train, y_train)
    
    # Predict
    y_pred_train = rf_simple.predict(X_train)
    y_pred_test = rf_simple.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results.append({
        'random_state': random_state,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae
    })
    
    print(f"Random State {random_state}:")
    print(f"  Train R²: {train_r2:.6f}, Test R²: {test_r2:.6f}")
    print(f"  Train MAE: {train_mae:.6f}, Test MAE: {test_mae:.6f}")

# Feature importance analysis
print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS")
print("-" * 30)
rf_analysis = RandomForestRegressor(n_estimators=100, random_state=42)
rf_analysis.fit(X_clean, y_clean)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_analysis.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.6f}")

# Check for data issues
print(f"\n🔍 CHECKING FOR DATA ISSUES")
print("-" * 30)

# Check if all target values are the same
if y_clean.nunique() == 1:
    print("🚨 ALL TARGET VALUES ARE THE SAME!")
    print(f"Constant value: {y_clean.iloc[0]}")

# Check if there are very few unique values
elif y_clean.nunique() < 10:
    print(f"⚠️  Very few unique target values: {y_clean.nunique()}")
    print("Unique values:", sorted(y_clean.unique()))

# Check for patterns in target values
print(f"\nTarget value distribution:")
print(y_clean.value_counts().head(10))

# Check if features perfectly determine target
print(f"\n🔍 CHECKING IF FEATURES PERFECTLY DETERMINE TARGET")
print("-" * 30)
feature_target_combinations = X_clean.copy()
feature_target_combinations['target'] = y_clean
grouped = feature_target_combinations.groupby(features)['target'].nunique()
perfect_combinations = grouped[grouped == 1].count()
total_combinations = len(grouped)

print(f"Total unique feature combinations: {total_combinations}")
print(f"Combinations with single target value: {perfect_combinations}")
print(f"Percentage of perfect combinations: {(perfect_combinations/total_combinations)*100:.2f}%")

if perfect_combinations == total_combinations:
    print("🚨 EACH FEATURE COMBINATION HAS EXACTLY ONE TARGET VALUE!")
    print("This explains the perfect R² score - the model can memorize the data perfectly.")

# Additional analysis: Check for mathematical relationships
print(f"\n🔍 CHECKING FOR MATHEMATICAL RELATIONSHIPS")
print("-" * 30)

# Check if target is a simple function of features
sample_data = X_clean.head(20).copy()
sample_data['target'] = y_clean.head(20)

print("Sample data to check for patterns:")
print(sample_data.to_string(index=False))

# Check if target might be calculated from features
print(f"\n🧮 CHECKING FOR CALCULATED TARGET")
print("-" * 30)

# Test some common formulas
test_formulas = []

# Formula 1: Simple weighted sum
formula1 = (X_clean['Pizza Complexity'] * 2 + 
           X_clean['Distance (km)'] * 3 + 
           X_clean['Traffic Level'] * 2 + 
           X_clean['Restaurant Avg Time'] * 0.5)
diff1 = np.abs(y_clean - formula1).mean()
test_formulas.append(('Weighted Sum', diff1))

# Formula 2: More complex
formula2 = (X_clean['Restaurant Avg Time'] + 
           X_clean['Distance (km)'] * 2 + 
           X_clean['Traffic Level'] * 3 + 
           X_clean['Pizza Complexity'] * 2 + 
           X_clean['Topping Density'] * 1.5)
diff2 = np.abs(y_clean - formula2).mean()
test_formulas.append(('Complex Formula', diff2))

# Formula 3: Base time + factors
formula3 = (20 + X_clean['Pizza Complexity'] * 3 + 
           X_clean['Distance (km)'] * 2 + 
           X_clean['Traffic Level'] * 2 + 
           X_clean['Topping Density'] * 1)
diff3 = np.abs(y_clean - formula3).mean()
test_formulas.append(('Base + Factors', diff3))

print("Testing potential formulas:")
for name, diff in test_formulas:
    print(f"  {name}: Mean absolute difference = {diff:.6f}")
    if diff < 0.001:
        print(f"    🚨 POTENTIAL EXACT FORMULA FOUND!")

# Residual analysis
print(f"\n📊 RESIDUAL ANALYSIS")
print("-" * 30)
rf_residual = RandomForestRegressor(n_estimators=100, random_state=42)
rf_residual.fit(X_clean, y_clean)
y_pred_residual = rf_residual.predict(X_clean)
residuals = y_clean - y_pred_residual

print(f"Residual statistics:")
print(f"  Mean: {residuals.mean():.6f}")
print(f"  Std: {residuals.std():.6f}")
print(f"  Min: {residuals.min():.6f}")
print(f"  Max: {residuals.max():.6f}")

# Check for zero residuals
zero_residuals = (np.abs(residuals) < 1e-10).sum()
print(f"  Zero residuals: {zero_residuals} out of {len(residuals)}")

if zero_residuals > len(residuals) * 0.8:
    print("  🚨 Most residuals are zero - model is memorizing!")

# Feature correlation matrix
print(f"\n🔗 FEATURE CORRELATION MATRIX")
print("-" * 30)
feature_corr = X_clean.corr()
print("High correlations between features:")
for i in range(len(features)):
    for j in range(i+1, len(features)):
        corr_val = feature_corr.iloc[i, j]
        if abs(corr_val) > 0.7:
            print(f"  {features[i]} <-> {features[j]}: {corr_val:.3f}")

# Recommendations
print(f"\n💡 RECOMMENDATIONS")
print("-" * 30)
if perfect_combinations == total_combinations:
    print("1. 🚨 Your data has a deterministic relationship - each input combination")
    print("   maps to exactly one output value. This suggests:")
    print("   - The data might be artificially generated")
    print("   - There might be a hidden formula connecting inputs to outputs")
    print("   - The target variable might be calculated from the features")
    print()
    print("2. 🔧 To create a more realistic model:")
    print("   - Add noise to the target variable")
    print("   - Use cross-validation to test generalization")
    print("   - Consider if this is real-world data or synthetic data")
    print("   - Try different validation strategies")

elif any(abs(corr_val) > 0.95 for feat, corr_val in target_corr.items() if feat != target):
    print("1. 🚨 High correlation detected - possible data leakage")
    print("2. 🔧 Review your features to ensure no future information is included")

else:
    print("1. ✅ No obvious data issues detected")
    print("2. 🔧 Consider using cross-validation for more robust evaluation")

# Additional specific recommendations
print(f"\n🎯 SPECIFIC RECOMMENDATIONS FOR YOUR DATA")
print("-" * 30)
print("1. 📊 Run this diagnostic script to understand your data better")
print("2. 🔧 Use the realistic_training.py script which adds noise and proper validation")
print("3. 📈 Consider cross-validation for more robust model evaluation")
print("4. 🧪 Test your model on completely new data if available")
print("5. 📝 Document your feature engineering process to avoid data leakage")

# Save diagnostic results
diagnostic_results = {
    'dataset_shape': data.shape,
    'target_column': target,
    'target_unique_values': y_clean.nunique(),
    'target_variance': y_clean.var(),
    'feature_target_correlation': target_corr.to_dict(),
    'perfect_combinations_percentage': (perfect_combinations/total_combinations)*100,
    'feature_importance': feature_importance.to_dict(),
    'residual_stats': {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'min': residuals.min(),
        'max': residuals.max(),
        'zero_residuals': zero_residuals
    },
    'test_results': results
}

# Save diagnostic results to file
with open('diagnostic_results.pkl', 'wb') as f:
    pickle.dump(diagnostic_results, f)

print(f"\n💾 Diagnostic results saved to 'diagnostic_results.pkl'")
print(f"🎉 Diagnostic analysis complete!")
print(f"\n📋 SUMMARY")
print("=" * 20)
print(f"Dataset: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"Target: {target} ({y_clean.nunique()} unique values)")
print(f"Perfect combinations: {(perfect_combinations/total_combinations)*100:.1f}%")
print(f"Average test R²: {np.mean([r['test_r2'] for r in results]):.6f}")
print(f"Average test MAE: {np.mean([r['test_mae'] for r in results]):.6f}")

if perfect_combinations == total_combinations:
    print("🚨 DATA IS DETERMINISTIC - Consider adding noise for realistic modeling")
else:
    print("✅ Data appears to have natural variation")