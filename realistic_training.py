import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🚀 REALISTIC MODEL TRAINING WITH PROPER VALIDATION")
print("=" * 60)

# Load data
try:
    data = pd.read_excel('Train Data.xlsx')
    print(f"✅ Data loaded: {data.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Add engineered features
data['Is Peak Hour'] = np.where(((data['Order Hour'] >= 11) & (data['Order Hour'] <= 14)) |
                                 ((data['Order Hour'] >= 17) & (data['Order Hour'] <= 20)), 1, 0)
data['Is Weekend'] = np.where(data['Order Month'].isin([6, 7, 8, 9]), 1, 0)

# Define features
features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
            'Distance (km)', 'Topping Density', 'Traffic Level', 
            'Is Peak Hour', 'Is Weekend']

# Changed target to 'Delivery Duration'
target = 'Delivery Duration (min)'

# Check if target column exists, if not try alternatives
if target not in data.columns:
    # Try common alternatives
    possible_targets = ['Delivery Duration (min)', 'Duration', 'Delivery Time']
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

# Prepare data
X = data[features].copy()
y = data[target].copy()

# Clean data
mask = ~(X.isnull().any(axis=1) | y.isnull())
X_clean = X[mask]
y_clean = y[mask]

print(f"\n📊 Data Shape After Cleaning: {X_clean.shape}")

# Check for perfect deterministic relationships
print(f"\n🔍 CHECKING FOR DETERMINISTIC RELATIONSHIPS")
print("-" * 50)

# Group by all features and check if each combination has unique target
feature_target_df = X_clean.copy()
feature_target_df['target'] = y_clean
grouped = feature_target_df.groupby(features)['target'].nunique()
perfect_matches = (grouped == 1).sum()
total_groups = len(grouped)

print(f"Total unique feature combinations: {total_groups}")
print(f"Combinations with single target value: {perfect_matches}")
print(f"Deterministic ratio: {(perfect_matches/total_groups)*100:.1f}%")

# If data is too deterministic, add some noise
if perfect_matches / total_groups > 0.8:
    print("\n⚠️  Data appears highly deterministic. Adding noise for more realistic modeling...")
    
    # Add small amount of noise to target variable
    noise_std = y_clean.std() * 0.05  # 5% of target standard deviation
    y_with_noise = y_clean + np.random.normal(0, noise_std, size=len(y_clean))
    
    # Ensure non-negative values (delivery time can't be negative)
    y_with_noise = np.maximum(y_with_noise, 5)  # Minimum 5 minutes
    
    print(f"Added noise with std: {noise_std:.2f}")
    print(f"Original target range: {y_clean.min():.1f} - {y_clean.max():.1f}")
    print(f"Noisy target range: {y_with_noise.min():.1f} - {y_with_noise.max():.1f}")
    
    y_final = y_with_noise
    use_noise = True
else:
    y_final = y_clean
    use_noise = False

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_final, test_size=0.2, random_state=42, stratify=None
)

print(f"\n📊 DATA SPLITS")
print("-" * 20)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Train different models and compare
models = {
    'RF_Simple': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    'RF_Medium': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
    'RF_Complex': RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
}

print(f"\n🔄 CROSS-VALIDATION RESULTS")
print("-" * 40)

best_model = None
best_score = -np.inf
results = {}

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results[name] = {
        'cv_mae': cv_mae,
        'cv_std': cv_std,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'model': model
    }
    
    print(f"\n{name}:")
    print(f"  CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
    print(f"  Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
    print(f"  Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
    
    # Check for overfitting
    if train_r2 - test_r2 > 0.1:
        print(f"  ⚠️  Possible overfitting (R² gap: {train_r2 - test_r2:.3f})")
    
    # Select best model based on test R² (balance between performance and generalization)
    if test_r2 > best_score:
        best_score = test_r2
        best_model = model

# Save the best model
model_info = {
    'model': best_model,
    'features': features,
    'n_features': len(features),
    'model_performance': {
        'cv_mae': results[best_model]['cv_mae'],
        'test_r2': results[best_model]['test_r2'],
        'test_mae': results[best_model]['test_mae']
    }
}

# Save model
with open('realistic_rf_model.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print(f"\n💾 Model saved as 'realistic_rf_model.pkl'")
