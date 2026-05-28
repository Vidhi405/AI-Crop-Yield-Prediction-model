"""
CROP YIELD PREDICTION - RECOMMENDATION SYSTEM
Recommends best crops based on farm conditions
Includes non-linear fertilizer features for realistic predictions
"""

import pickle
import pandas as pd
import numpy as np
import os
import json

class CropRecommender:
    """
    Crop recommendation system using trained XGBoost models
    """
    
    # State-specific typical crops (what is commonly grown in each state)
    STATE_TYPICAL_CROPS = {
        'Punjab': ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane', 'Gram', 'Mustard', 'Potato', 'Onion'],
        'Haryana': ['Wheat', 'Rice', 'Bajra', 'Cotton', 'Sugarcane', 'Mustard', 'Gram', 'Barley'],
        'Uttar Pradesh': ['Wheat', 'Rice', 'Sugarcane', 'Potato', 'Mustard', 'Gram', 'Barley', 'Maize'],
        'Bihar': ['Rice', 'Wheat', 'Maize', 'Potato', 'Sugarcane', 'Gram', 'Pulses'],
        'West Bengal': ['Rice', 'Jute', 'Potato', 'Mustard', 'Sugarcane', 'Wheat'],
        'Karnataka': ['Ragi', 'Jowar', 'Maize', 'Groundnut', 'Sugarcane', 'Banana', 'Rice', 'Sunflower'],
        'Tamil Nadu': ['Rice', 'Groundnut', 'Sugarcane', 'Banana', 'Cotton', 'Ragi', 'Maize', 'Turmeric'],
        'Andhra Pradesh': ['Rice', 'Groundnut', 'Cotton', 'Sugarcane', 'Chilli', 'Tobacco', 'Maize'],
        'Telangana': ['Rice', 'Cotton', 'Maize', 'Groundnut', 'Red Gram', 'Sugarcane'],
        'Maharashtra': ['Cotton', 'Soybean', 'Sugarcane', 'Jowar', 'Bajra', 'Wheat', 'Gram'],
        'Gujarat': ['Cotton', 'Groundnut', 'Wheat', 'Bajra', 'Castor', 'Sugarcane', 'Mustard'],
        'Madhya Pradesh': ['Soybean', 'Wheat', 'Gram', 'Maize', 'Mustard', 'Sugarcane'],
        'Rajasthan': ['Bajra', 'Wheat', 'Mustard', 'Gram', 'Guar', 'Barley'],
        'Odisha': ['Rice', 'Ragi', 'Maize', 'Groundnut', 'Pulses', 'Sugarcane'],
        'Kerala': ['Rice', 'Tapioca', 'Coconut', 'Banana', 'Pepper', 'Rubber', 'Cashew'],
        'Assam': ['Rice', 'Tea', 'Jute', 'Sugarcane', 'Potato', 'Mustard'],
        'Himachal Pradesh': ['Wheat', 'Maize', 'Barley', 'Potato', 'Apple'],
        'Uttarakhand': ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Millet', 'Potato'],
        'Jharkhand': ['Rice', 'Maize', 'Gram', 'Pulses', 'Vegetables', 'Groundnut'],
        'Chhattisgarh': ['Rice', 'Maize', 'Gram', 'Pigeon Pea', 'Groundnut', 'Turmeric'],
        'Goa': ['Rice', 'Cashew', 'Coconut', 'Sugarcane', 'Banana'],
        'Delhi': ['Wheat', 'Rice', 'Maize', 'Mustard', 'Potato'],
    }
    
    # Crops that can be grown anywhere (fallback if state not in dictionary)
    DEFAULT_CROPS = ['Wheat', 'Rice', 'Maize', 'Potato', 'Onion', 'Tomato']
    
    def __init__(self, models_folder='models/'):
        """
        Load all trained models and encoders
        """
        print("Loading crop recommendation system...")
        
        # Load encoders
        self.encoders = pickle.load(open(f'{models_folder}label_encoders.pkl', 'rb'))
        
        # Get feature columns (now includes fertilizer_squared and fertilizer_log)
        self.feature_cols = pickle.load(open(f'{models_folder}feature_columns.pkl', 'rb'))
        
        # Load model metrics (to know which crops are reliable)
        try:
            with open(f'{models_folder}model_metrics.json', 'r') as f:
                self.metrics = json.load(f)
        except:
            self.metrics = {}
        
        # Load all crop models
        self.models = {}
        self.model_reliability = {}
        
        for model_file in os.listdir(models_folder):
            if model_file.endswith('_model.pkl'):
                # Extract crop name from filename
                crop_name = model_file.replace('_model.pkl', '').replace('_', ' ')
                # Capitalize properly
                crop_name = crop_name.title()
                
                # Load model
                self.models[crop_name] = pickle.load(open(f'{models_folder}{model_file}', 'rb'))
                
                # Store reliability (R² score) if available
                if crop_name in self.metrics:
                    self.model_reliability[crop_name] = self.metrics[crop_name]['r2']
                else:
                    self.model_reliability[crop_name] = 0
        
        print(f"✅ Loaded {len(self.models)} crop models")
        print(f"   Features used: {len(self.feature_cols)}")
        print(f"   Available crops: {', '.join(list(self.models.keys())[:10])}...")
    
    def predict_yield(self, crop, state, season, area, rainfall, fertilizer, pesticide, soil_ph, year):
        """
        Predict yield for a specific crop
        
        Parameters:
        - crop: Name of the crop (e.g., "Rice", "Wheat")
        - state: State name (e.g., "Punjab", "Karnataka")
        - season: "Kharif", "Rabi", or "Whole Year"
        - area: Area in hectares
        - rainfall: Annual rainfall in mm
        - fertilizer: Fertilizer used in kg
        - pesticide: Pesticide used in kg
        - soil_ph: Soil pH value (typically 5.0-8.0)
        - year: Crop year (e.g., 2024)
        
        Returns:
        - Predicted yield in tons/hectare
        """
        # Check if model exists for this crop
        if crop not in self.models:
            raise ValueError(f"No model available for crop: {crop}")
        
        # Encode categorical variables
        try:
            state_code = self.encoders['state'].transform([state])[0]
        except:
            print(f"Warning: State '{state}' not found in training data. Using default.")
            state_code = 0
        
        season_code = self.encoders['season_map'].get(season, 0)
        
        # Calculate fertilizer per hectare
        fert_per_hectare = fertilizer / area if area > 0 else 0
        
        # ========== NON-LINEAR FERTILIZER FEATURES ==========
        # Quadratic term (captures the "too much fertilizer reduces yield" effect)
        fert_squared = fert_per_hectare ** 2
        
        # Log transform (captures diminishing returns)
        fert_log = np.log1p(fert_per_hectare)
        
        # Prepare input features (must match training order)
        # Order must exactly match feature_cols from training
        input_data = [[
            state_code,
            season_code,
            year,
            area,
            rainfall,
            fertilizer,
            pesticide,
            soil_ph,
            fert_per_hectare,
            fert_squared,   # NEW: Quadratic term
            fert_log        # NEW: Log transform
        ]]
        
        # Predict
        model = self.models[crop]
        prediction = model.predict(input_data)[0]
        
        # Round to 2 decimal places
        return round(float(prediction), 2)
    
    def recommend_crops(self, state, season, area, rainfall, fertilizer, pesticide, soil_ph, year, top_n=5, min_reliability=0.5):
        """
        Recommend top N crops for given farm conditions
        
        Parameters:
        - state, season, area, rainfall, fertilizer, pesticide, soil_ph, year: Same as above
        - top_n: Number of top crops to recommend (default: 5)
        - min_reliability: Minimum R² score to consider a crop (default: 0.5)
        
        Returns:
        - List of dicts with crop, yield, and reliability
        """
        results = []
        
        # Get typical crops for this state (or use defaults if state not in dict)
        typical_crops = self.STATE_TYPICAL_CROPS.get(state, self.DEFAULT_CROPS)
        
        # Predict yield for each crop
        for crop, model in self.models.items():
            # FILTER 1: Only consider crops typical for this state
            if crop not in typical_crops:
                continue
            
            # FILTER 2: Only consider reliable models (R² >= min_reliability)
            reliability = self.model_reliability.get(crop, 0)
            if reliability < min_reliability:
                continue
                
            try:
                predicted_yield = self.predict_yield(
                    crop, state, season, area, rainfall, 
                    fertilizer, pesticide, soil_ph, year
                )
                results.append({
                    'crop': crop,
                    'yield': predicted_yield,
                    'reliability': round(reliability, 3)
                })
            except Exception as e:
                # Skip crops that cause errors
                continue
        
        # Sort by yield (highest first)
        results.sort(key=lambda x: x['yield'], reverse=True)
        
        return results[:top_n]
    
    def recommend_crops_simple(self, state, season, area, rainfall, fertilizer, pesticide, soil_ph, year, top_n=5):
        """
        Simplified recommendation without reliability filter
        (Backward compatible with original code)
        """
        results = []
        
        # Get typical crops for this state
        typical_crops = self.STATE_TYPICAL_CROPS.get(state, self.DEFAULT_CROPS)
        
        for crop, model in self.models.items():
            if crop not in typical_crops:
                continue
                
            try:
                predicted_yield = self.predict_yield(
                    crop, state, season, area, rainfall, 
                    fertilizer, pesticide, soil_ph, year
                )
                results.append((crop, predicted_yield))
            except Exception as e:
                continue
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    
    def recommend_detailed(self, state, season, area, rainfall, fertilizer, pesticide, soil_ph, year, top_n=5, min_reliability=0.5):
        """
        Get detailed recommendations with additional information
        """
        recommendations = self.recommend_crops(
            state, season, area, rainfall, fertilizer, pesticide, soil_ph, year, top_n, min_reliability
        )
        
        detailed = []
        for rank, rec in enumerate(recommendations, 1):
            detailed.append({
                'rank': rank,
                'crop': rec['crop'],
                'predicted_yield': rec['yield'],
                'unit': 'tons/hectare',
                'reliability': rec['reliability'],
                'total_production': round(rec['yield'] * area, 2) if area else None,
                'total_unit': 'tons'
            })
        
        return detailed
    
    def get_optimal_fertilizer(self, crop, state, season, area, rainfall, pesticide, soil_ph, year, fert_range=(0, 300, 10)):
        """
        Find optimal fertilizer amount for a given crop and location
        
        Parameters:
        - Same as predict_yield, but without fertilizer
        - fert_range: (min_fert, max_fert, step) in kg
        
        Returns:
        - Optimal fertilizer amount and corresponding yield
        """
        min_fert, max_fert, step = fert_range
        best_yield = 0
        best_fert = 0
        
        print(f"\n🔍 Finding optimal fertilizer for {crop} in {state}...")
        
        for fert in range(min_fert, max_fert + 1, step):
            try:
                yield_pred = self.predict_yield(
                    crop, state, season, area, rainfall, 
                    fert, pesticide, soil_ph, year
                )
                
                if yield_pred > best_yield:
                    best_yield = yield_pred
                    best_fert = fert
            except:
                continue
        
        return {
            'crop': crop,
            'optimal_fertilizer_kg': best_fert,
            'optimal_yield': best_yield,
            'unit': 'tons/hectare'
        }
    
    def get_reliability_report(self):
        """
        Get reliability report for all crops
        """
        reliable_crops = []
        unreliable_crops = []
        
        for crop, reliability in self.model_reliability.items():
            if reliability >= 0.6:
                reliable_crops.append((crop, reliability))
            else:
                unreliable_crops.append((crop, reliability))
        
        reliable_crops.sort(key=lambda x: x[1], reverse=True)
        unreliable_crops.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'reliable_crops': reliable_crops,
            'unreliable_crops': unreliable_crops,
            'total_crops': len(self.model_reliability),
            'reliable_count': len(reliable_crops),
            'unreliable_count': len(unreliable_crops)
        }


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    
    # Initialize recommender
    recommender = CropRecommender()
    
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Predict yield for a specific crop")
    print("=" * 60)
    
    # Example 1: Predict Rice yield in Punjab
    rice_yield = recommender.predict_yield(
        crop="Rice",
        state="Punjab",
        season="Kharif",
        area=100,
        rainfall=1200,
        fertilizer=150,
        pesticide=10,
        soil_ph=6.8,
        year=2024
    )
    print(f"\n🌾 Rice yield prediction for Punjab:")
    print(f"   Predicted yield: {rice_yield} tons/hectare")
    print(f"   Total production: {rice_yield * 100} tons")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Recommend best crops for a farm")
    print("=" * 60)
    
    # Example 2: Recommend crops for a farm in Punjab
    recommendations = recommender.recommend_detailed(
        state="Punjab",
        season="Rabi",
        area=50,
        rainfall=800,
        fertilizer=120,
        pesticide=8,
        soil_ph=7.2,
        year=2024,
        top_n=5,
        min_reliability=0.5
    )
    
    print(f"\n🌱 Top {len(recommendations)} crops for your farm in Punjab (Rabi season):")
    print("-" * 60)
    for rec in recommendations:
        stars = "⭐" * int(rec['reliability'] * 5)
        print(f"{rec['rank']}. {rec['crop']}: {rec['predicted_yield']} tons/hectare {stars}")
        if rec['total_production']:
            print(f"   → Estimated total: {rec['total_production']} tons")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Fertilizer Optimization (NEW!)")
    print("=" * 60)
    
    # Example 3: Find optimal fertilizer for Wheat in Punjab
    optimal = recommender.get_optimal_fertilizer(
        crop="Wheat",
        state="Punjab",
        season="Rabi",
        area=100,
        rainfall=800,
        pesticide=8,
        soil_ph=7.2,
        year=2024,
        fert_range=(0, 300, 20)  # Test from 0 to 300 kg in steps of 20
    )
    
    print(f"\n🌾 Optimal fertilizer recommendation for Wheat in Punjab:")
    print(f"   Optimal fertilizer: {optimal['optimal_fertilizer_kg']} kg/hectare")
    print(f"   Expected yield: {optimal['optimal_yield']} tons/hectare")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Compare different states")
    print("=" * 60)
    
    # Example 4: Compare Wheat yield across states
    states = ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh"]
    print(f"\n🌾 Wheat yield comparison (Rabi season):")
    print("-" * 60)
    for state in states:
        try:
            wheat_yield = recommender.predict_yield(
                crop="Wheat",
                state=state,
                season="Rabi",
                area=100,
                rainfall=900,
                fertilizer=120,
                pesticide=8,
                soil_ph=7.0,
                year=2024
            )
            print(f"   {state:20s}: {wheat_yield} tons/hectare")
        except:
            print(f"   {state:20s}: No model available")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Seasonal comparison")
    print("=" * 60)
    
    # Example 5: Compare Kharif vs Rabi for same farm
    seasons = ["Kharif", "Rabi"]
    print(f"\n🌱 Best crops by season for farm in Karnataka:")
    print("-" * 60)
    
    for season in seasons:
        recs = recommender.recommend_crops_simple(
            state="Karnataka",
            season=season,
            area=50,
            rainfall=1000,
            fertilizer=100,
            pesticide=5,
            soil_ph=6.5,
            year=2024,
            top_n=3
        )
        print(f"\n{season} season:")
        for crop, yield_pred in recs:
            print(f"   • {crop}: {yield_pred} tons/hectare")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Reliability Report")
    print("=" * 60)
    
    report = recommender.get_reliability_report()
    print(f"\n📊 Model Reliability Summary:")
    print(f"   Total crops: {report['total_crops']}")
    print(f"   Reliable crops (R² >= 0.6): {report['reliable_count']}")
    print(f"   Unreliable crops (R² < 0.6): {report['unreliable_count']}")
    
    print(f"\n🏆 Top 10 Most Reliable Crops:")
    for crop, reliability in report['reliable_crops'][:10]:
        stars = "⭐" * int(reliability * 5)
        print(f"   • {crop}: {reliability:.3f} {stars}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Fertilizer Impact Test (NEW!)")
    print("=" * 60)
    
    # Example 7: Show how yield changes with different fertilizer levels
    print(f"\n📈 Fertilizer impact on Wheat yield in Punjab:")
    print("-" * 60)
    print(f"{'Fertilizer (kg/ha)':<20} {'Predicted Yield (tons/ha)':<25}")
    print("-" * 60)
    
    for fert in [0, 50, 100, 150, 200, 250, 300]:
        try:
            yield_pred = recommender.predict_yield(
                crop="Wheat",
                state="Punjab",
                season="Rabi",
                area=100,
                rainfall=800,
                fertilizer=fert,
                pesticide=8,
                soil_ph=7.2,
                year=2024
            )
            # Add indicator for optimal
            marker = " ← OPTIMAL" if fert == optimal['optimal_fertilizer_kg'] else ""
            print(f"{fert:<20} {yield_pred:<25} {marker}")
        except:
            print(f"{fert:<20} Error")
    
    print("\n✅ Ready for production!")