"""
MODEL EVALUATION SCRIPT
Calculates overall accuracy using already cleaned data from training
"""

import pickle
import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("=" * 60)
print("MODEL EVALUATION - CROP YIELD PREDICTION")
print("=" * 60)

# ============================================
# STEP 1: LOAD MODELS AND ENCODERS
# ============================================

print("\n📂 Loading models and encoders...")

# Load encoders and feature columns
encoders = pickle.load(open('models/label_encoders.pkl', 'rb'))
feature_cols = pickle.load(open('models/feature_columns.pkl', 'rb'))

# Load model metrics from training
with open('models/model_metrics.json', 'r') as f:
    train_metrics = json.load(f)

print(f"✅ Loaded {len(train_metrics)} crop models")

# ============================================
# STEP 2: USE TRAINING METRICS (Already Calculated)
# ============================================

print("\n📊 Using metrics from training...")
print("-" * 60)

# Convert metrics to list
results = []
for crop, metrics in train_metrics.items():
    results.append({
        'crop': crop,
        'r2': metrics['r2'],
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'samples': metrics['samples']
    })
    
    # Print with star rating
    if metrics['r2'] >= 0.8:
        star = "⭐⭐⭐⭐⭐"
    elif metrics['r2'] >= 0.6:
        star = "⭐⭐⭐"
    elif metrics['r2'] >= 0.4:
        star = "⭐"
    else:
        star = "⚠️"
    
    print(f"   {star} {crop:25s} | R²: {metrics['r2']:6.3f} | MAE: {metrics['mae']:5.2f} | Samples: {metrics['samples']:4d}")

# ============================================
# STEP 3: CALCULATE OVERALL ACCURACY
# ============================================

print("\n" + "=" * 60)
print("OVERALL ACCURACY REPORT")
print("=" * 60)

# Filter valid results (R² > 0)
valid_results = [r for r in results if r['r2'] > 0]

if len(valid_results) == 0:
    print("No valid models found!")
    exit()

# Simple average
avg_r2 = sum(r['r2'] for r in valid_results) / len(valid_results)

# Weighted by sample size
total_samples = sum(r['samples'] for r in valid_results)
weighted_r2 = sum(r['r2'] * r['samples'] for r in valid_results) / total_samples

# Median R²
median_r2 = sorted([r['r2'] for r in valid_results])[len(valid_results)//2]

print(f"\n📊 SUMMARY STATISTICS:")
print(f"   Total crop models evaluated: {len(results)}")
print(f"   Valid models (R² > 0): {len(valid_results)}")
print(f"   Invalid models (R² ≤ 0): {len(results) - len(valid_results)}")
print(f"\n   📈 Average R² (simple): {avg_r2:.4f}")
print(f"   📈 Average R² (weighted by samples): {weighted_r2:.4f}")
print(f"   📈 Median R²: {median_r2:.4f}")

# ============================================
# STEP 4: PERFORMANCE BREAKDOWN
# ============================================

print(f"\n📊 PERFORMANCE BREAKDOWN:")

excellent = [r for r in valid_results if r['r2'] >= 0.8]
good = [r for r in valid_results if 0.6 <= r['r2'] < 0.8]
moderate = [r for r in valid_results if 0.4 <= r['r2'] < 0.6]
poor = [r for r in valid_results if 0 < r['r2'] < 0.4]

print(f"   ✅ Excellent (R² ≥ 0.8): {len(excellent)} crops")
print(f"   👍 Good (R² 0.6-0.8):   {len(good)} crops")
print(f"   📘 Moderate (R² 0.4-0.6): {len(moderate)} crops")
print(f"   ⚠️  Poor (R² < 0.4):     {len(poor)} crops")

# ============================================
# STEP 5: TOP AND BOTTOM PERFORMERS
# ============================================

print(f"\n🏆 TOP 10 BEST MODELS:")
top_10 = sorted(valid_results, key=lambda x: x['r2'], reverse=True)[:10]
for i, r in enumerate(top_10, 1):
    star_count = "⭐" * min(5, int(r['r2'] * 5))
    print(f"   {i:2d}. {r['crop']:25s} | R²: {r['r2']:.3f} {star_count}")

print(f"\n📉 BOTTOM 10 WORST MODELS (excluding negatives):")
bottom_10 = sorted(valid_results, key=lambda x: x['r2'])[:10]
for i, r in enumerate(bottom_10, 1):
    print(f"   {i:2d}. {r['crop']:25s} | R²: {r['r2']:.3f}")

# ============================================
# STEP 6: CONFIDENCE RECOMMENDATION FOR BACKEND
# ============================================

print(f"\n🎯 CONFIDENCE RECOMMENDATIONS FOR BACKEND:")
print("-" * 60)

print(f"\n   ✅ HIGH CONFIDENCE (R² ≥ 0.7) - {len([r for r in valid_results if r['r2'] >= 0.7])} crops:")
high_conf = [r['crop'] for r in valid_results if r['r2'] >= 0.7]
for crop in high_conf[:15]:
    print(f"      • {crop}")

print(f"\n   📘 MEDIUM CONFIDENCE (R² 0.5-0.7) - Use with caution:")
med_conf = [r['crop'] for r in valid_results if 0.5 <= r['r2'] < 0.7]
for crop in med_conf[:10]:
    print(f"      • {crop}")

print(f"\n   ⚠️  LOW CONFIDENCE (R² < 0.5) - Skip these:")
low_conf = [r['crop'] for r in valid_results if r['r2'] < 0.5]
for crop in low_conf[:10]:
    print(f"      • {crop}")

# ============================================
# STEP 7: FINAL VERDICT
# ============================================

print("\n" + "=" * 60)
print("✅ EVALUATION COMPLETE!")
print("=" * 60)

print(f"\n📊 FINAL VERDICT:")
print(f"   Your model has an AVERAGE R² of {avg_r2:.3f} across {len(valid_results)} crops.")
print(f"   This means the model explains {avg_r2*100:.1f}% of the yield variation on average.")

if avg_r2 >= 0.7:
    print(f"\n   🎉 EXCELLENT! The model is production-ready.")
elif avg_r2 >= 0.6:
    print(f"\n   👍 GOOD! The model is usable with caution for some crops.")
else:
    print(f"\n   ⚠️  NEEDS IMPROVEMENT. Consider adding more features or data.")

print(f"\n💡 Recommended min_reliability for backend: 0.6")
print(f"   This will show only the {len(good) + len(excellent)} reliable crops.")