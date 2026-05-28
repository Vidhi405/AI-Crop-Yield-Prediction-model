"""
CROP YIELD PREDICTION - TRAINING SCRIPT
Trains separate XGBoost models for each crop
Includes non-linear fertilizer features (quadratic and log)
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

print("=" * 60)
print("CROP YIELD PREDICTION - MODEL TRAINING")
print("=" * 60)

# ============================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================

print("\n📂 Loading data...")

# Load datasets
crop_df = pd.read_csv('D:\\ProjectsTwo\\AICropYieldPrediction\\Datasets\\crop_yield(1).csv')
soil_df = pd.read_excel('D:\\ProjectsTwo\\AICropYieldPrediction\\Datasets\\State_major_soils.xlsx')


print(f"   Crop data: {crop_df.shape[0]:,} rows")
print(f"   Soil data: {soil_df.shape[0]} rows")

# Clean soil data
soil_clean = soil_df.copy()
soil_clean['State/UT'] = soil_clean['State/UT'].replace({
    'Bihar (including Jharkhand)': 'Bihar'
})

def extract_ph(ph_range):
    if pd.isna(ph_range):
        return 6.5
    try:
        parts = str(ph_range).split('–')
        if len(parts) == 2:
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        return float(ph_range)
    except:
        return 6.5

soil_clean['Soil_pH'] = soil_clean['Soil pH Range'].apply(extract_ph)
soil_clean = soil_clean[['State/UT', 'Soil_pH']]

# Clean crop data
crop_clean = crop_df.copy()
crop_clean.columns = crop_clean.columns.str.strip()

# Remove duplicates
crop_clean = crop_clean.drop_duplicates()

# Fix state names
state_fixes = {
    'Bihar (including Jharkhand)': 'Bihar',
    'Jammu & Kashmir': 'Jammu and Kashmir'
}
crop_clean['State'] = crop_clean['State'].replace(state_fixes)

# Fix season names
season_fixes = {
    'Kharif     ': 'Kharif',
    'Rabi       ': 'Rabi',
    'Whole Year ': 'Whole Year',
    'Autumn': 'Kharif',
    'Winter': 'Rabi'
}
crop_clean['Season'] = crop_clean['Season'].replace(season_fixes)
crop_clean['Season'] = crop_clean['Season'].str.strip()

# Remove invalid yields
crop_clean = crop_clean[crop_clean['Yield'] > 0]
crop_clean = crop_clean[crop_clean['Yield'] < 100]  # Cap extreme yields

# Remove invalid area
crop_clean = crop_clean[crop_clean['Area'] > 0]

# Cap outliers
for col in ['Fertilizer', 'Pesticide']:
    upper = crop_clean[col].quantile(0.99)
    crop_clean[col] = crop_clean[col].clip(upper=upper)

# Add soil pH
state_soil = soil_clean.set_index('State/UT')
crop_clean['Soil_pH'] = crop_clean['State'].map(state_soil['Soil_pH'])
crop_clean['Soil_pH'] = crop_clean['Soil_pH'].fillna(6.5)

# ============================================
# STEP 2: CREATE ENGINEERED FEATURES
# ============================================

print("\n🔧 Creating engineered features...")

# Fertilizer per hectare
crop_clean['Fertilizer_per_hectare'] = crop_clean['Fertilizer'] / crop_clean['Area']
crop_clean['Fertilizer_per_hectare'] = crop_clean['Fertilizer_per_hectare'].clip(
    upper=crop_clean['Fertilizer_per_hectare'].quantile(0.99)
)

# ========== NON-LINEAR FERTILIZER FEATURES ==========
# Quadratic term (captures the "too much fertilizer reduces yield" effect)
crop_clean['Fertilizer_squared'] = crop_clean['Fertilizer_per_hectare'] ** 2

# Log transform (captures diminishing returns)
crop_clean['Fertilizer_log'] = np.log1p(crop_clean['Fertilizer_per_hectare'])

# Cap extreme values
crop_clean['Fertilizer_squared'] = crop_clean['Fertilizer_squared'].clip(
    upper=crop_clean['Fertilizer_squared'].quantile(0.99)
)
crop_clean['Fertilizer_log'] = crop_clean['Fertilizer_log'].clip(
    upper=crop_clean['Fertilizer_log'].quantile(0.99)
)

print(f"   Added features: Fertilizer_per_hectare, Fertilizer_squared, Fertilizer_log")

print(f"\n✅ Cleaned data: {crop_clean.shape[0]:,} rows")

# ============================================
# STEP 3: ENCODE CATEGORICAL VARIABLES
# ============================================

print("\n📊 Encoding categorical variables...")

# Encode seasons
season_map = {'Kharif': 0, 'Rabi': 1, 'Whole Year': 2}
crop_clean['Season_Code'] = crop_clean['Season'].map(season_map)

# Encode states
le_state = LabelEncoder()
crop_clean['State_Code'] = le_state.fit_transform(crop_clean['State'])

# Encode crops
le_crop = LabelEncoder()
crop_clean['Crop_Code'] = le_crop.fit_transform(crop_clean['Crop'])

print(f"   States: {len(le_state.classes_)}")
print(f"   Crops: {len(le_crop.classes_)}")

# ============================================
# STEP 4: SELECT FEATURES
# ============================================

feature_cols = [
    'State_Code', 'Season_Code', 'Crop_Year',
    'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide',
    'Soil_pH', 
    'Fertilizer_per_hectare',
    'Fertilizer_squared',   # NEW: Captures optimal fertilizer level
    'Fertilizer_log'        # NEW: Captures diminishing returns
]

target_col = 'Yield'

print(f"\n📋 Features ({len(feature_cols)}): {feature_cols}")
print(f"🎯 Target: {target_col}")

# ============================================
# STEP 5: TRAIN SEPARATE MODELS FOR EACH CROP
# ============================================

print("\n🤖 Training crop-specific models...")
print("-" * 40)

# Create models directory
os.makedirs('models', exist_ok=True)

# Store all models
crop_models = {}
model_metrics = {}

# Get all crops
crops = crop_clean['Crop'].unique()
print(f"Training models for {len(crops)} crops\n")

for crop in crops:
    # Filter data for this crop
    crop_data = crop_clean[crop_clean['Crop'] == crop]
    
    # Skip if too few samples
    if len(crop_data) < 30:
        print(f"   ⚠️  {crop}: Only {len(crop_data)} samples - skipping")
        continue
    
    # Prepare features and target
    X = crop_data[feature_cols]
    y = crop_data[target_col]
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Store model and metrics
    crop_models[crop] = model
    model_metrics[crop] = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'samples': len(crop_data)
    }
    
    # Save individual model (clean filename)
    safe_crop_name = crop.lower().replace(' ', '_').replace('&', 'and').replace('/', '_')
    model_filename = f"models/{safe_crop_name}_model.pkl"
    pickle.dump(model, open(model_filename, 'wb'))
    
    # Print progress
    reliability_star = "⭐" if r2 >= 0.6 else "⚠️"
    print(f"   {reliability_star} {crop:25s} | R²: {r2:6.3f} | MAE: {mae:5.2f} | Samples: {len(crop_data):4d}")

# ============================================
# STEP 6: SAVE ENCODERS AND METADATA
# ============================================

print("\n💾 Saving encoders and metadata...")

# Save label encoders
encoders = {
    'crop': le_crop,
    'state': le_state,
    'season_map': season_map,
    'feature_cols': feature_cols  # Save feature columns for consistency
}
pickle.dump(encoders, open('models/label_encoders.pkl', 'wb'))

# Save feature columns separately (backward compatibility)
pickle.dump(feature_cols, open('models/feature_columns.pkl', 'wb'))

# Save model metrics
with open('models/model_metrics.json', 'w') as f:
    json.dump(model_metrics, f, indent=2)

# Save list of all crops
with open('models/all_crops.txt', 'w') as f:
    for crop in sorted(crop_models.keys()):
        f.write(f"{crop}\n")

# ============================================
# STEP 7: SUMMARY REPORT
# ============================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

print(f"\n📊 SUMMARY:")
print(f"   Total crops trained: {len(crop_models)}")
print(f"   Total crops skipped: {len(crops) - len(crop_models)}")
print(f"   Models saved in: /models/")
print(f"   Features used: {len(feature_cols)}")

print(f"\n🏆 TOP 10 BEST PERFORMING MODELS:")
top_models = sorted(model_metrics.items(), key=lambda x: x[1]['r2'], reverse=True)[:10]
for crop, metrics in top_models:
    stars = "⭐" * int(metrics['r2'] * 5)
    print(f"   {crop:25s}: R² = {metrics['r2']:.3f} {stars}")

print(f"\n⚠️ POOR PERFORMING MODELS (R² < 0.3):")
poor_models = [(crop, m) for crop, m in model_metrics.items() if m['r2'] < 0.3]
for crop, metrics in poor_models[:10]:
    print(f"   {crop:25s}: R² = {metrics['r2']:.3f}")

print(f"\n📁 Files created:")
print(f"   • models/<crop>_model.pkl (52 files)")
print(f"   • models/label_encoders.pkl")
print(f"   • models/feature_columns.pkl")
print(f"   • models/model_metrics.json")
print(f"   • models/all_crops.txt")

print("\n✅ Ready for recommendation system!")