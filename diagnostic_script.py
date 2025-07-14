import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🔍 COMPREHENSIVE DIAGNOSTIC ANALYSIS FOR PIZZA DELIVERY MODEL")
print("=" * 70)

# Load data
try:
    data = pd.read_excel('Train Data.xlsx')
    print(f"✅ Data loaded successfully: {data.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

print("\n📊 DATASET OVERVIEW")
print("-" * 30)
print(f"Dataset shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Data types:\n{data.dtypes}")

# Check for duplicates
duplicate_count = data.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_count}")
if duplicate_count > 0:
    print("🚨 Found duplicate rows! This could affect model performance.")

# Basic statistics
print("\n📈 BASIC STATISTICS")
print("-" * 30)
print(data.describe())

# Check for missing values
print("\n❓ MISSING VALUES ANALYSIS")
print("-" * 30)
missing_values = data.isnull().sum()
total_missing = missing_values.sum()
if total_missing > 0:
    print("Missing values per column:")
    print(missing_values[missing_values > 0])
    print(f"Total missing values: {total_missing}")
    print(f"Percentage missing: {(total_missing / data.size) * 100:.2f}%")
else:
    print("✅ No missing values found")

# Add engineered features
print("\n🔧 FEATURE ENGINEERING")
print("-" * 30)
data['Is Peak Hour'] = np.where(((data['Order Hour'] >= 11) & (data['Order Hour'] <= 14)) |
                                 ((data['Order Hour'] >= 17) & (data['Order Hour'] <= 20)), 1, 0)
data['Is Weekend'] = np.where(data['Order Month'].isin([6, 7, 8, 9]), 1, 0)
print("✅ Added engineered features: 'Is Peak Hour', 'Is Weekend'")

# Define features
features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
            'Distance (km)', 'Topping Density', 'Traffic Level', 
            'Is Peak Hour', 'Is Weekend']

# Find target column
target = 'Delivery Duration (Min)'
if target not in data.columns:
    possible_targets = ['Delivery Duration (min)', 
                        'Delivery Time', 'Delivery Duration']
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
target_stats = data[target].describe()
print(f"Target statistics:\n{target_stats}")

# Check for outliers in target
Q1 = data[target].quantile(0.25)
Q3 = data[target].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data[(data[target] < lower_bound) | (data[target] > upper_bound)]
print(f"\nOutliers in target variable: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
if len(outliers) > 0:
    print(f"Outlier range: {outliers[target].min():.1f} - {outliers[target].max():.1f}")

# Check for constant or near-constant values
print(f"\nTarget unique values: {data[target].nunique()}")
print(f"Target variance: {data[target].var():.6f}")
print(f"Target coefficient of variation: {data[target].std() / data[target].mean():.6f}")

# Distribution analysis
print(f"\n📊 TARGET DISTRIBUTION")
print("-" * 30)
print("Value counts (top 10):")
print(data[target].value_counts().head(10))

# Check if target is constant
if data[target].nunique() == 1:
    print("🚨 TARGET IS CONSTANT! All values are the same.")
    print(f"Constant value: {data[target].iloc[0]}")

# Correlation analysis
print(f"\n🔗 CORRELATION ANALYSIS")
print("-" * 30)
feature_data = data[features + [target]]
correlation_matrix = feature_data.corr()
print("Correlation with target:")
target_corr = correlation_matrix[target].sort_values(ascending=False)
for feature, corr in target_corr.items():
    if feature != target:
        print(f"  {feature}: {corr:.6f}")

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

# Feature correlation matrix
print(f"\n🔗 INTER-FEATURE CORRELATION")
print("-" * 30)
feature_corr = data[features].corr()
print("High correlations between features (>0.7):")
high_corr_pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        corr_val = feature_corr.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append((features[i], features[j], corr_val))
            print(f"  {features[i]} <-> {features[j]}: {corr_val:.3f}")

if not high_corr_pairs:
    print("✅ No high correlations between features")

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
print(f"Rows removed: {X.shape[0] - X_clean.shape[0]}")

# Check for data uniqueness and determinism
print(f"\n🔍 DETERMINISM ANALYSIS")
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
    print("This explains perfect R² scores - the model can memorize the data.")
elif perfect_combinations > total_combinations * 0.8:
    print("⚠️  Data is highly deterministic (>80% perfect combinations)")
else:
    print("✅ Data shows natural variation")

# Multiple split validation
print(f"\n🔄 STABILITY TESTING WITH DIFFERENT DATA SPLITS")
print("-" * 30)

results = []
for random_state in [42, 123, 999, 2024, 555]:
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
        'test_mae': test_mae,
        'overfitting': train_r2 - test_r2
    })
    
    print(f"Random State {random_state}:")
    print(f"  Train R²: {train_r2:.6f}, Test R²: {test_r2:.6f}")
    print(f"  Train MAE: {train_mae:.6f}, Test MAE: {test_mae:.6f}")
    print(f"  Overfitting gap: {train_r2 - test_r2:.6f}")

# Stability analysis
print(f"\n📊 STABILITY ANALYSIS")
print("-" * 30)
test_r2_scores = [r['test_r2'] for r in results]
test_mae_scores = [r['test_mae'] for r in results]
overfitting_gaps = [r['overfitting'] for r in results]

print(f"Test R² - Mean: {np.mean(test_r2_scores):.6f}, Std: {np.std(test_r2_scores):.6f}")
print(f"Test MAE - Mean: {np.mean(test_mae_scores):.6f}, Std: {np.std(test_mae_scores):.6f}")
print(f"Overfitting Gap - Mean: {np.mean(overfitting_gaps):.6f}, Std: {np.std(overfitting_gaps):.6f}")

if np.std(test_r2_scores) < 0.01:
    print("✅ Model performance is very stable across different splits")
elif np.std(test_r2_scores) < 0.05:
    print("✅ Model performance is reasonably stable")
else:
    print("⚠️  Model performance varies significantly across splits")

# Cross-validation analysis
print(f"\n🔄 CROSS-VALIDATION ANALYSIS")
print("-" * 30)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = RandomForestRegressor(n_estimators=100, random_state=42)

# CV scores
cv_scores = cross_val_score(rf_cv, X_clean, y_clean, cv=cv, scoring='r2')
cv_mae_scores = cross_val_score(rf_cv, X_clean, y_clean, cv=cv, scoring='neg_mean_absolute_error')

print(f"Cross-validation R² scores: {cv_scores}")
print(f"CV R² - Mean: {cv_scores.mean():.6f}, Std: {cv_scores.std():.6f}")
print(f"CV MAE - Mean: {-cv_mae_scores.mean():.6f}, Std: {cv_mae_scores.std():.6f}")

# Feature importance analysis
print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS")
print("-" * 30)
rf_analysis = RandomForestRegressor(n_estimators=100, random_state=42)
rf_analysis.fit(X_clean, y_clean)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_analysis.feature_importances_
}).sort_values('importance', ascending=False)

print("Random Forest Feature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.6f}")

# Permutation importance
print(f"\n🔄 PERMUTATION IMPORTANCE")
print("-" * 30)
perm_importance = permutation_importance(rf_analysis, X_clean, y_clean, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame({
    'feature': features,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("Permutation Feature Importance:")
for idx, row in perm_importance_df.iterrows():
    print(f"  {row['feature']}: {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")

# Residual analysis
print(f"\n📊 RESIDUAL ANALYSIS")
print("-" * 30)
y_pred_full = rf_analysis.predict(X_clean)
residuals = y_clean - y_pred_full

print(f"Residual statistics:")
print(f"  Mean: {residuals.mean():.6f}")
print(f"  Std: {residuals.std():.6f}")
print(f"  Min: {residuals.min():.6f}")
print(f"  Max: {residuals.max():.6f}")
print(f"  Median: {residuals.median():.6f}")

# Check for zero residuals
zero_residuals = (np.abs(residuals) < 1e-10).sum()
near_zero_residuals = (np.abs(residuals) < 1e-6).sum()
print(f"  Zero residuals (< 1e-10): {zero_residuals} ({zero_residuals/len(residuals)*100:.1f}%)")
print(f"  Near-zero residuals (< 1e-6): {near_zero_residuals} ({near_zero_residuals/len(residuals)*100:.1f}%)")

if zero_residuals > len(residuals) * 0.5:
    print("  🚨 More than 50% of residuals are zero - model is memorizing!")
elif near_zero_residuals > len(residuals) * 0.8:
    print("  ⚠️  More than 80% of residuals are near zero - highly deterministic data")

# Formula testing
print(f"\n🧮 TESTING FOR MATHEMATICAL RELATIONSHIPS")
print("-" * 30)

# Test various potential formulas
test_formulas = []

# Formula 1: Restaurant time + distance-based delivery
formula1 = (X_clean['Restaurant Avg Time'] + 
           X_clean['Distance (km)'] * 2 + 
           X_clean['Traffic Level'] * 2)
diff1 = np.abs(y_clean - formula1).mean()
test_formulas.append(('Restaurant + Distance + Traffic', diff1))

# Formula 2: Weighted complexity
formula2 = (X_clean['Pizza Complexity'] * 3 + 
           X_clean['Distance (km)'] * 2 + 
           X_clean['Traffic Level'] * 2 + 
           X_clean['Restaurant Avg Time'] * 0.8 + 
           X_clean['Topping Density'] * 1.5)
diff2 = np.abs(y_clean - formula2).mean()
test_formulas.append(('Weighted Complexity', diff2))

# Formula 3: Base time + factors
formula3 = (15 + X_clean['Pizza Complexity'] * 2 + 
           X_clean['Distance (km)'] * 2.5 + 
           X_clean['Traffic Level'] * 3 + 
           X_clean['Topping Density'] * 1)
diff3 = np.abs(y_clean - formula3).mean()
test_formulas.append(('Base + Factors', diff3))

# Formula 4: Peak hour adjustment
formula4 = (X_clean['Restaurant Avg Time'] + 
           X_clean['Distance (km)'] * 2 + 
           X_clean['Traffic Level'] * 2 + 
           X_clean['Is Peak Hour'] * 5)
diff4 = np.abs(y_clean - formula4).mean()
test_formulas.append(('Peak Hour Adjusted', diff4))

print("Testing potential mathematical relationships:")
for name, diff in test_formulas:
    print(f"  {name}: Mean absolute difference = {diff:.6f}")
    if diff < 0.001:
        print(f"    🚨 POTENTIAL EXACT FORMULA FOUND!")
    elif diff < 1.0:
        print(f"    ⚠️  Very close approximation - possible relationship")

# Data generation pattern analysis
print(f"\n🔍 DATA GENERATION PATTERN ANALYSIS")
print("-" * 30)

# Check if data follows expected patterns
print("Expected vs Actual patterns:")

# Check if longer distances generally take more time
dist_time_corr = np.corrcoef(X_clean['Distance (km)'], y_clean)[0, 1]
print(f"Distance-Time correlation: {dist_time_corr:.3f} (expected: positive)")

# Check if higher traffic increases time
traffic_time_corr = np.corrcoef(X_clean['Traffic Level'], y_clean)[0, 1]
print(f"Traffic-Time correlation: {traffic_time_corr:.3f} (expected: positive)")

# Check if complex pizzas take longer
complexity_time_corr = np.corrcoef(X_clean['Pizza Complexity'], y_clean)[0, 1]
print(f"Complexity-Time correlation: {complexity_time_corr:.3f} (expected: positive)")

# Sample data inspection
print(f"\n📋 SAMPLE DATA INSPECTION")
print("-" * 30)
sample_data = X_clean.head(10).copy()
sample_data['target'] = y_clean.head(10)
print("First 10 rows:")
print(sample_data.to_string(index=False))

# Check for patterns in sample data
print(f"\nLooking for patterns in sample data...")
for i in range(min(5, len(sample_data))):
    row = sample_data.iloc[i]
    predicted_simple = (row['Restaurant Avg Time'] + 
                       row['Distance (km)'] * 2 + 
                       row['Traffic Level'] * 2)
    actual = row['target']
    print(f"Row {i+1}: Simple formula = {predicted_simple:.1f}, Actual = {actual:.1f}, Diff = {abs(predicted_simple - actual):.1f}")

# Advanced diagnostics
print(f"\n🔬 ADVANCED DIAGNOSTICS")
print("-" * 30)

# Check if target values are realistic for delivery times
min_delivery = y_clean.min()
max_delivery = y_clean.max()
mean_delivery = y_clean.mean()

print(f"Delivery time range: {min_delivery:.1f} - {max_delivery:.1f} minutes")
print(f"Mean delivery time: {mean_delivery:.1f} minutes")

# Realistic delivery time checks
if min_delivery < 5:
    print("⚠️  Minimum delivery time is very low (< 5 minutes)")
if max_delivery > 120:
    print("⚠️  Maximum delivery time is very high (> 2 hours)")
if mean_delivery < 15 or mean_delivery > 60:
    print("⚠️  Mean delivery time seems unusual for pizza delivery")

# Check for integer-only values
if all(y_clean == y_clean.astype(int)):
    print("📊 All target values are integers")
else:
    print("📊 Target values include decimals")

# Final recommendations
print(f"\n💡 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
print("-" * 50)

# Categorize the issue
if perfect_combinations == total_combinations:
    issue_type = "DETERMINISTIC"
    severity = "HIGH"
elif perfect_combinations > total_combinations * 0.8:
    issue_type = "HIGHLY_DETERMINISTIC"
    severity = "MEDIUM"
elif np.mean(test_r2_scores) > 0.99:
    issue_type = "OVERFITTING"
    severity = "MEDIUM"
else:
    issue_type = "NORMAL"
    severity = "LOW"

print(f"📋 ISSUE TYPE: {issue_type}")
print(f"🚨 SEVERITY: {severity}")
print()

if issue_type == "DETERMINISTIC":
    print("🔧 RECOMMENDATIONS:")
    print("1. 🚨 Your data is completely deterministic")
    print("2. 📊 Each input combination maps to exactly one output")
    print("3. 🔄 This explains the perfect R² scores")
    print("4. 💡 Solutions:")
    print("   - Add realistic noise to the target variable")
    print("   - Use the realistic_training.py script")
    print("   - Consider if this is synthetic/generated data")
    print("   - Validate on completely new data")
    print("5. ⚠️  Current model will not generalize to new data")

elif issue_type == "HIGHLY_DETERMINISTIC":
    print("🔧 RECOMMENDATIONS:")
    print("1. ⚠️  Your data is highly deterministic")
    print("2. 📊 Most input combinations have single outputs")
    print("3. 💡 Solutions:")
    print("   - Add small amount of noise for realism")
    print("   - Use cross-validation extensively")
    print("   - Test on external validation data")
    print("4. 🧪 Consider ensemble methods for better generalization")

elif issue_type == "OVERFITTING":
    print("🔧 RECOMMENDATIONS:")
    print("1. 📈 Model shows signs of overfitting")
    print("2. 🔄 Use cross-validation for evaluation")
    print("3. 💡 Solutions:")
    print("   - Reduce model complexity")
    print("   - Use regularization")
    print("   - Increase validation set size")
    print("   - Try different algorithms")

else:
    print("✅ RECOMMENDATIONS:")
    print("1. 🎉 Your data appears to have natural variation")
    print("2. 📊 Model performance looks reasonable")
    print("3. 💡 Continue with normal ML workflow:")
    print("   - Use cross-validation")
    print("   - Monitor for overfitting")
    print("   - Test on holdout data")
    print("   - Consider feature engineering")

# Technical recommendations
print(f"\n🛠️  TECHNICAL RECOMMENDATIONS:")
print("1. 🔄 Always use cross-validation for model evaluation")
print("2. 📊 Monitor both training and validation metrics")
print("3. 🧪 Test on completely new data when possible")
print("4. 📈 Consider ensemble methods for better generalization")
print("5. 🔍 Regular diagnostic checks like this analysis")

# Save comprehensive diagnostic results
diagnostic_results = {
    'dataset_info': {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'target_column': target,
        'duplicates': duplicate_count,
        'missing_values': missing_values.to_dict()
    },
    'target_analysis': {
        'unique_values': y_clean.nunique(),
        'variance': y_clean.var(),
        'mean': y_clean.mean(),
        'std': y_clean.std(),
        'min': y_clean.min(),
        'max': y_clean.max(),
        'outliers': len(outliers)
    },
    'correlation_analysis': {
        'feature_target_correlation': target_corr.to_dict(),
        'high_corr_features': [f for f, c in high_corr_features.items() if f != target],
        'inter_feature_correlations': [{'feature1': pair[0], 'feature2': pair[1], 'correlation': pair[2]} 
                                      for pair in high_corr_pairs]
    },
    'determinism_analysis': {
        'total_combinations': total_combinations,
        'perfect_combinations': perfect_combinations,
        'determinism_percentage': (perfect_combinations/total_combinations)*100
    },
    'stability_analysis': {
        'test_results': results,
        'r2_mean': np.mean(test_r2_scores),
        'r2_std': np.std(test_r2_scores),
        'mae_mean': np.mean(test_mae_scores),
        'mae_std': np.std(test_mae_scores)
    },
    'cross_validation': {
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'cv_mae_mean': -cv_mae_scores.mean(),
        'cv_mae_std': cv_mae_scores.std()
    },
    'feature_importance': {
        'random_forest': feature_importance.to_dict(),
        'permutation': perm_importance_df.to_dict()
    },
    'residual_analysis': {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'min': residuals.min(),
        'max': residuals.max(),
        'zero_residuals': zero_residuals,
        'near_zero_residuals': near_zero_residuals
    },
    'formula_testing': [{'name': name, 'difference': diff} for name, diff in test_formulas],
    'diagnosis': {
        'issue_type': issue_type,
        'severity': severity,
        'is_deterministic': perfect_combinations == total_combinations,
        'is_highly_deterministic': perfect_combinations > total_combinations * 0.8,
        'is_overfitting': np.mean(test_r2_scores) > 0.99
    }
}

# Save diagnostic results
with open('comprehensive_diagnostic_results.pkl', 'wb') as f:
    pickle.dump(diagnostic_results, f)

print(f"\n💾 Comprehensive diagnostic results saved to 'comprehensive_diagnostic_results.pkl'")
print(f"🎯 Run this analysis whenever you suspect model issues")
print(f"📊 Use the insights to improve your model development process")

print(f"\n🎉 COMPREHENSIVE DIAGNOSTIC ANALYSIS COMPLETE!")
print("=" * 50)
print(f"📋 FINAL SUMMARY:")
print(f"   Dataset: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"   Target: {target} ({y_clean.nunique()} unique values)")
print(f"   Determinism: {(perfect_combinations/total_combinations)*100:.1f}% perfect combinations")
print(f"   Stability: R² std = {np.std(test_r2_scores):.6f}")
print(f"   Issue Type: {issue_type}")
print(f"   Severity: {severity}")
print(f"   Recommendation: {'Use realistic_training.py with noise' if issue_type == 'DETERMINISTIC' else 'Continue with standard ML practices'}")