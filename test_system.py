"""
Quick Testing Script for NASA Exoplanet Detection System
Demonstrates the system capabilities
"""

import requests
import json

# API endpoint
BASE_URL = 'http://localhost:5000'

print("="*80)
print("NASA EXOPLANET DETECTION SYSTEM - QUICK TEST")
print("="*80)

# Test 1: Manual Prediction
print("\n[TEST 1] Manual Prediction")
print("-" * 80)

# Sample data (typical confirmed exoplanet values)
sample_data = {
    'koi_period': 75.0,      # ~75 day orbit
    'koi_depth': 2500.0,     # Deep transit
    'koi_duration': 3.5,     # ~3.5 hour transit
    'koi_prad': 10.0,        # ~10 Earth radii (Jupiter-like)
    'koi_teq': 1200.0,       # Hot planet
    'koi_insol': 80.0,       # High insolation
    'koi_steff': 5800.0,     # Sun-like star
    'koi_slogg': 4.4,        # Sun-like gravity
    'koi_srad': 1.0,         # Sun-like radius
    'koi_impact': 0.2        # Low impact parameter
}

print(f"Input Parameters: {sample_data}")

try:
    response = requests.post(
        f'{BASE_URL}/predict',
        json=sample_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            r = result['result']
            print(f"\n✓ Prediction: {r['prediction']}")
            print(f"✓ Probability: {r['probability']:.4f}")
            print(f"✓ Confidence: {r['confidence']:.2f}%")
            print(f"\nModel Scores:")
            print(f"  - DNN: {r['model_scores']['dnn']:.4f}")
            print(f"  - CNN: {r['model_scores']['cnn']:.4f}")
            print(f"  - LSTM: {r['model_scores']['lstm']:.4f}")
            
            print(f"\nTop 3 Contributing Features:")
            for i, (feat, data) in enumerate(list(r['contributions'].items())[:3], 1):
                print(f"  {i}. {feat}: {data['contribution']:.4f}")
        else:
            print(f"✗ Error: {result['error']}")
    else:
        print(f"✗ Request failed with status code {response.status_code}")
except Exception as e:
    print(f"✗ Error: {str(e)}")
    print("Note: Make sure the Flask server is running (python app.py)")

# Test 2: Feature Importance
print("\n[TEST 2] Feature Importance")
print("-" * 80)

try:
    response = requests.get(f'{BASE_URL}/feature_importance')
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print("\nTop 5 Most Important Features:")
            for i, (feat, imp) in enumerate(zip(result['features'][:5], result['importance'][:5]), 1):
                print(f"  {i}. {feat}: {imp:.4f}")
        else:
            print(f"✗ Error: {result['error']}")
    else:
        print(f"✗ Request failed with status code {response.status_code}")
except Exception as e:
    print(f"✗ Error: {str(e)}")

# Test 3: Model Performance
print("\n[TEST 3] Model Performance")
print("-" * 80)

try:
    response = requests.get(f'{BASE_URL}/model_performance')
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            perf = result['performance']
            print("\nModel Performance Metrics:")
            print(f"\nDNN:")
            print(f"  - Accuracy: {perf['dnn']['accuracy']:.4f} ({perf['dnn']['accuracy']*100:.2f}%)")
            print(f"  - ROC-AUC: {perf['dnn']['roc_auc']:.4f}")
            print(f"\nCNN:")
            print(f"  - Accuracy: {perf['cnn']['accuracy']:.4f} ({perf['cnn']['accuracy']*100:.2f}%)")
            print(f"  - ROC-AUC: {perf['cnn']['roc_auc']:.4f}")
            print(f"\nLSTM:")
            print(f"  - Accuracy: {perf['lstm']['accuracy']:.4f} ({perf['lstm']['accuracy']*100:.2f}%)")
            print(f"  - ROC-AUC: {perf['lstm']['roc_auc']:.4f}")
            print(f"\n★ ENSEMBLE:")
            print(f"  - Accuracy: {perf['ensemble']['accuracy']:.4f} ({perf['ensemble']['accuracy']*100:.2f}%)")
            print(f"  - ROC-AUC: {perf['ensemble']['roc_auc']:.4f}")
        else:
            print(f"✗ Error: {result['error']}")
    else:
        print(f"✗ Request failed with status code {response.status_code}")
except Exception as e:
    print(f"✗ Error: {str(e)}")

print("\n" + "="*80)
print("TESTING COMPLETE!")
print("="*80)
print("\nNext Steps:")
print("1. Open http://localhost:5000 in your browser")
print("2. Try the manual input with sliders")
print("3. Upload the cumulative dataset CSV for batch analysis")
print("4. Explore the visualizations tab")
print("\n✓ System is ready for NASA Space Apps Challenge submission!")
