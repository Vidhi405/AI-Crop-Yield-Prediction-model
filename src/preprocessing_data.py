"""
Crop Yield Prediction - Complete Data Cleaning Pipeline
Author: AI Assistant
Date: 2026
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD DATASETS
# ============================================

print("=" * 60)
print("STEP 1: LOADING DATASETS")
print("=" * 60)

# Load the datasets (update file paths as needed)
crop_df = pd.read_csv('D:\\ProjectsTwo\\AICropYieldPrediction\\Datasets\\crop_yield(1).csv')
soil_df = pd.read_excel('D:\\ProjectsTwo\\AICropYieldPrediction\\Datasets\\State_major_soils.xlsx')

print(f"✅ Crop dataset loaded: {crop_df.shape[0]:,} rows, {crop_df.shape[1]} columns")
print(f"✅ Soil dataset loaded: {soil_df.shape[0]} rows, {soil_df.shape[1]} columns")

# ============================================
# STEP 2: CLEAN SOIL DATASET
# ============================================

print("\n" + "=" * 60)
print("STEP 2: CLEANING SOIL DATASET")
print("=" * 60)

def clean_soil_dataset(df):
    """Clean and prepare soil data for merging"""
    soil_clean = df.copy()
    
    # Fix inconsistent state names
    soil_clean['State/UT'] = soil_clean['State/UT'].replace({
        'Bihar (including Jharkhand)': 'Bihar',
        'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands'
    })
    
    # Extract numeric pH from range (take midpoint)
    def extract_ph_range(ph_range):
        if pd.isna(ph_range):
            return 6.5
        try:
            parts = str(ph_range).split('–')
            if len(parts) == 2:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return (low + high) / 2
            return float(ph_range)
        except:
            return 6.5
    
    soil_clean['Soil_pH'] = soil_clean['Soil pH Range'].apply(extract_ph_range)
    
    # Create soil type categories
    soil_type_map = {
        'Red Soils': 'Red',
        'Alluvial Soils': 'Alluvial',
        'Black Soils': 'Black',
        'Desert Soils': 'Desert',
        'Coastal Alluvial Soils': 'Coastal Alluvial'
    }
    soil_clean['Soil_Type'] = soil_clean['Major Soil Type'].map(soil_type_map)
    
    # Keep only needed columns
    soil_clean = soil_clean[['State/UT', 'Soil_pH', 'Soil_Type']]
    
    print(f"✅ Soil dataset cleaned: {soil_clean.shape[0]} states with soil data")
    return soil_clean

soil_clean = clean_soil_dataset(soil_df)
print(soil_clean.head(10))

# ============================================
# STEP 3: CLEAN CROP DATASET
# ============================================

print("\n" + "=" * 60)
print("STEP 3: CLEANING CROP DATASET")
print("=" * 60)

def clean_crop_dataset(df, soil_df):
    """Main cleaning function for crop yield data"""
    crop_clean = df.copy()
    
    # 3.1 STANDARDIZE COLUMN NAMES
    crop_clean.columns = crop_clean.columns.str.strip()
    
    # 3.2 REMOVE DUPLICATES
    initial_rows = len(crop_clean)
    crop_clean = crop_clean.drop_duplicates()
    print(f"✅ Removed {initial_rows - len(crop_clean):,} duplicate rows")
    
    # 3.3 STANDARDIZE STATE NAMES
    state_fixes = {
        'Bihar (including Jharkhand)': 'Bihar',
        'Jammu & Kashmir': 'Jammu and Kashmir',
        'Andhra Pradesh (old)': 'Andhra Pradesh',
        'Telangana (old)': 'Telangana'
    }
    crop_clean['State'] = crop_clean['State'].replace(state_fixes)
    
    # 3.4 STANDARDIZE SEASON NAMES
    season_fixes = {
        'Kharif     ': 'Kharif',
        'Rabi       ': 'Rabi',
        'Whole Year ': 'Whole Year',
        'Autumn': 'Kharif',
        'Winter': 'Rabi',
        'Summer': 'Summer'
    }
    crop_clean['Season'] = crop_clean['Season'].replace(season_fixes)
    crop_clean['Season'] = crop_clean['Season'].str.strip()
    
    # 3.5 STANDARDIZE CROP NAMES (remove trailing spaces)
    crop_clean['Crop'] = crop_clean['Crop'].str.strip()
    
    # 3.6 REMOVE IMPOSSIBLE VALUES
    # Yield bounds by crop (tons/hectare)
    yield_bounds = {
        'Rice': (0, 12), 'Wheat': (0, 8), 'Maize': (0, 12),
        'Groundnut': (0, 5), 'Sugarcane': (0, 150), 'Cotton': (0, 5),
        'Potato': (0, 45), 'Onion': (0, 60), 'Jute': (0, 20),
        'Gram': (0, 4), 'Arhar/Tur': (0, 4), 'Bajra': (0, 5),
        'Barley': (0, 5), 'Rapeseed &Mustard': (0, 3),
        'Sesamum': (0, 2), 'Sunflower': (0, 3), 'Soyabean': (0, 4)
    }
    
    def is_reasonable_yield(row):
        crop = row['Crop']
        yield_val = row['Yield']
        if pd.isna(yield_val) or yield_val <= 0:
            return False
        if crop in yield_bounds:
            low, high = yield_bounds[crop]
            return low <= yield_val <= high
        return 0 < yield_val <= 50  # Default cap for unknown crops
    
    before = len(crop_clean)
    crop_clean = crop_clean[crop_clean.apply(is_reasonable_yield, axis=1)]
    print(f"✅ Removed {before - len(crop_clean):,} rows with unreasonable yields")
    
    # Area must be positive
    before = len(crop_clean)
    crop_clean = crop_clean[crop_clean['Area'] > 0]
    print(f"✅ Removed {before - len(crop_clean):,} rows with zero/negative area")
    
    # Production must be positive (or can be calculated)
    mask = (crop_clean['Production'] <= 0) & (crop_clean['Area'] > 0) & (crop_clean['Yield'] > 0)
    crop_clean.loc[mask, 'Production'] = crop_clean.loc[mask, 'Area'] * crop_clean.loc[mask, 'Yield']
    
    # 3.7 CAP EXTREME OUTLIERS (Fertilizer and Pesticide)
    for col in ['Fertilizer', 'Pesticide']:
        if col in crop_clean.columns:
            upper = crop_clean[col].quantile(0.99)
            lower = crop_clean[col].quantile(0.01)
            crop_clean[col] = crop_clean[col].clip(lower=lower, upper=upper)
    
    # 3.8 HANDLE MISSING VALUES
    # Critical columns - drop if missing
    critical_cols = ['Crop', 'State', 'Crop_Year', 'Area', 'Yield']
    before = len(crop_clean)
    crop_clean = crop_clean.dropna(subset=critical_cols)
    print(f"✅ Removed {before - len(crop_clean):,} rows with missing critical data")
    
    # Fill missing values with medians by crop
    for col in ['Annual_Rainfall', 'Fertilizer', 'Pesticide']:
        if col in crop_clean.columns:
            crop_clean[col] = crop_clean.groupby('Crop')[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    # 3.9 ADD SOIL DATA
    state_soil = soil_df.set_index('State/UT')
    crop_clean['Soil_pH'] = crop_clean['State'].map(state_soil['Soil_pH'])
    crop_clean['Soil_Type'] = crop_clean['State'].map(state_soil['Soil_Type'])
    
    # Fill missing soil data with defaults
    crop_clean['Soil_pH'] = crop_clean['Soil_pH'].fillna(6.5)
    crop_clean['Soil_Type'] = crop_clean['Soil_Type'].fillna('Alluvial')
    
    # 3.10 CREATE ENGINEERED FEATURES
    # Fertilizer and Pesticide per hectare
    crop_clean['Fertilizer_per_hectare'] = crop_clean['Fertilizer'] / crop_clean['Area']
    crop_clean['Pesticide_per_hectare'] = crop_clean['Pesticide'] / crop_clean['Area']
    
    # Cap extreme values in engineered features
    for col in ['Fertilizer_per_hectare', 'Pesticide_per_hectare']:
        upper = crop_clean[col].quantile(0.99)
        crop_clean[col] = crop_clean[col].clip(upper=upper)
    
    # Rainfall categories
    crop_clean['Rainfall_Category'] = pd.cut(
        crop_clean['Annual_Rainfall'],
        bins=[0, 750, 1500, 2500, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Yield efficiency (production per unit area - should be close to Yield)
    crop_clean['Yield_Efficiency'] = crop_clean['Production'] / crop_clean['Area']
    
    # 3.11 FIX KNOWN DATA ERRORS
    # Fix zero yield for Cardamom in West Bengal (1998)
    mask = (crop_clean['Crop'] == 'Cardamom') & \
           (crop_clean['State'] == 'West Bengal') & \
           (crop_clean['Crop_Year'] == 1998) & \
           (crop_clean['Yield'] == 0)
    if mask.any():
        median_yield = crop_clean[crop_clean['Crop'] == 'Cardamom']['Yield'].median()
        crop_clean.loc[mask, 'Yield'] = median_yield
    
    # Ensure no negative values
    crop_clean['Area'] = crop_clean['Area'].abs()
    crop_clean['Fertilizer'] = crop_clean['Fertilizer'].abs()
    crop_clean['Pesticide'] = crop_clean['Pesticide'].abs()
    
    # 3.12 REMOVE REMAINING OUTLIERS (3 standard deviations)
    def remove_outliers_by_crop(df, crop_col, target_col):
        """Remove outliers beyond 3 standard deviations for each crop"""
        cleaned = []
        for crop in df[crop_col].unique():
            crop_data = df[df[crop_col] == crop]
            mean = crop_data[target_col].mean()
            std = crop_data[target_col].std()
            if std > 0:
                crop_data = crop_data[
                    (crop_data[target_col] >= mean - 3*std) & 
                    (crop_data[target_col] <= mean + 3*std)
                ]
            cleaned.append(crop_data)
        return pd.concat(cleaned, ignore_index=True)
    
    before = len(crop_clean)
    crop_clean = remove_outliers_by_crop(crop_clean, 'Crop', 'Yield')
    print(f"✅ Removed {before - len(crop_clean):,} outliers (>3 standard deviations)")
    
    return crop_clean

# Apply cleaning
crop_clean = clean_crop_dataset(crop_df, soil_clean)

# ============================================
# STEP 4: VALIDATE CLEANED DATASET
# ============================================

print("\n" + "=" * 60)
print("STEP 4: VALIDATION REPORT")
print("=" * 60)

def validate_dataset(df):
    """Print comprehensive validation metrics"""
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   • Rows: {df.shape[0]:,}")
    print(f"   • Columns: {df.shape[1]}")
    print(f"   • Date range: {df['Crop_Year'].min()} - {df['Crop_Year'].max()}")
    print(f"   • States: {df['State'].nunique()}")
    print(f"   • Crops: {df['Crop'].nunique()}")
    print(f"   • Seasons: {df['Season'].unique().tolist()}")
    
    print(f"\n📊 MISSING VALUES:")
    missing = df.isnull().sum()
    print(f"   {missing[missing > 0] if (missing > 0).any() else 'No missing values!'}")
    
    print(f"\n📊 YIELD STATISTICS BY TOP 5 CROPS:")
    top_crops = df['Crop'].value_counts().head(5).index
    for crop in top_crops:
        data = df[df['Crop'] == crop]['Yield']
        print(f"   • {crop}: mean={data.mean():.2f}, median={data.median():.2f}, "
              f"min={data.min():.2f}, max={data.max():.2f}, n={len(data):,}")
    
    print(f"\n📊 SOIL pH DISTRIBUTION:")
    print(f"   • Range: {df['Soil_pH'].min():.1f} - {df['Soil_pH'].max():.1f}")
    print(f"   • Mean: {df['Soil_pH'].mean():.2f}")
    
    print(f"\n📊 RAINFALL DISTRIBUTION:")
    print(f"   • Range: {df['Annual_Rainfall'].min():.0f} - {df['Annual_Rainfall'].max():.0f} mm")
    print(f"   • Mean: {df['Annual_Rainfall'].mean():.0f} mm")
    
    return

validate_dataset(crop_clean)

# ============================================
# STEP 5: SAVE CLEANED DATASETS
# ============================================

print("\n" + "=" * 60)
print("STEP 5: SAVING CLEANED DATASETS")
print("=" * 60)

# Save to CSV files
crop_clean.to_csv('crop_yield_cleaned.csv', index=False)
soil_clean.to_csv('soil_data_cleaned.csv', index=False)

print("✅ Saved: crop_yield_cleaned.csv")
print("✅ Saved: soil_data_cleaned.csv")

# Also save a sample for quick inspection
crop_clean.sample(100).to_csv('crop_yield_sample.csv', index=False)
print("✅ Saved: crop_yield_sample.csv (100 random rows)")

# ============================================
# STEP 6: DATA EXPLORATION SUMMARY
# ============================================

print("\n" + "=" * 60)
print("STEP 6: DATA EXPLORATION SUMMARY")
print("=" * 60)

print(f"\n🏆 TOP 10 CROPS BY FREQUENCY:")
print(crop_clean['Crop'].value_counts().head(10))

print(f"\n🌾 TOP 10 CROPS BY AVERAGE YIELD (tons/hectare):")
avg_yield = crop_clean.groupby('Crop')['Yield'].mean().sort_values(ascending=False)
for crop, yield_val in avg_yield.head(10).items():
    print(f"   • {crop}: {yield_val:.2f}")

print(f"\n🏙️ TOP 10 STATES BY DATA VOLUME:")
print(crop_clean['State'].value_counts().head(10))

print(f"\n📈 YEARLY YIELD TREND (All Crops):")
yearly_avg = crop_clean.groupby('Crop_Year')['Yield'].mean()
print(f"   • 1997: {yearly_avg[1997]:.2f}")
print(f"   • 2005: {yearly_avg[2005]:.2f}")
print(f"   • 2010: {yearly_avg[2010]:.2f}")
print(f"   • 2015: {yearly_avg[2015]:.2f}")
print(f"   • 2019: {yearly_avg[2019]:.2f}")
print(f"   • Growth: {((yearly_avg[2019] - yearly_avg[1997]) / yearly_avg[1997] * 100):.1f}%")

# ============================================
# STEP 7: PREPARE FOR MACHINE LEARNING
# ============================================

print("\n" + "=" * 60)
print("STEP 7: ML PREPARATION - ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

def prepare_for_ml(df):
    """Encode categorical variables for ML models"""
    ml_df = df.copy()
    
    # Encode seasons
    season_map = {'Kharif': 0, 'Rabi': 1, 'Whole Year': 2, 'Summer': 3}
    ml_df['Season_Code'] = ml_df['Season'].map(season_map)
    
    # Encode soil types
    soil_map = {'Red': 0, 'Alluvial': 1, 'Black': 2, 'Desert': 3, 'Coastal Alluvial': 4}
    ml_df['Soil_Type_Code'] = ml_df['Soil_Type'].map(soil_map)
    
    # Encode rainfall categories
    rainfall_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
    ml_df['Rainfall_Category_Code'] = ml_df['Rainfall_Category'].map(rainfall_map)
    
    # Create state and crop codes (for models that need numeric)
    from sklearn.preprocessing import LabelEncoder
    le_state = LabelEncoder()
    le_crop = LabelEncoder()
    
    ml_df['State_Code'] = le_state.fit_transform(ml_df['State'])
    ml_df['Crop_Code'] = le_crop.fit_transform(ml_df['Crop'])
    
    # Select features for modeling
    feature_cols = [
        'Crop_Code', 'State_Code', 'Season_Code', 'Crop_Year',
        'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide',
        'Fertilizer_per_hectare', 'Pesticide_per_hectare',
        'Soil_pH', 'Soil_Type_Code', 'Rainfall_Category_Code'
    ]
    
    target_col = 'Yield'
    
    print(f"✅ Features for ML: {len(feature_cols)} columns")
    print(f"   Features: {feature_cols[:5]}... + {len(feature_cols)-5} more")
    print(f"   Target: {target_col}")
    
    return ml_df, feature_cols, target_col

ml_ready, features, target = prepare_for_ml(crop_clean)

# Save ML-ready dataset
ml_ready.to_csv('crop_yield_ml_ready.csv', index=False)
print("\n✅ Saved: crop_yield_ml_ready.csv (ready for ML models)")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("✅ CLEANING COMPLETE!")
print("=" * 60)

print("""
FILES GENERATED:
─────────────────────────────────────────────────
1. crop_yield_cleaned.csv     - Main cleaned dataset
2. soil_data_cleaned.csv      - Cleaned soil data  
3. crop_yield_sample.csv      - 100 random rows for inspection
4. crop_yield_ml_ready.csv    - ML-ready with encoded features

DATASET STATISTICS:
─────────────────────────────────────────────────
""")
print(f"   • Original rows: {len(crop_df):,}")
print(f"   • Cleaned rows: {len(crop_clean):,}")
print(f"   • Rows removed: {len(crop_df) - len(crop_clean):,} ({((len(crop_df)-len(crop_clean))/len(crop_df)*100):.1f}%)")
print(f"   • Features available: {len(features)}")

print("\n✅ Your data is now ready for machine learning!")
print("   Next step: Train your XGBoost model using crop_yield_ml_ready.csv")