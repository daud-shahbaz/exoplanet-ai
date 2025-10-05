"""
NASA Exoplanet Detection Web Application
Flask Backend with Interactive Predictions and Visualizations
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create upload folder if doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Realistic feature ranges based on Kepler data
FEATURE_RANGES = {
    'koi_period': (0.5, 700),        # Orbital period in days
    'koi_depth': (10, 100000),       # Transit depth in ppm
    'koi_duration': (0.5, 20),       # Transit duration in hours
    'koi_prad': (0.5, 30),           # Planetary radius in Earth radii
    'koi_teq': (100, 3000),          # Temperature in Kelvin
    'koi_insol': (0.01, 2000),       # Insolation flux
    'koi_steff': (3000, 8000),       # Stellar temperature in K
    'koi_slogg': (3.5, 5.0),         # Surface gravity
    'koi_srad': (0.5, 3.0),          # Stellar radius in Solar radii
    'koi_impact': (0.0, 1.0)         # Impact parameter
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_input_data(data_dict):
    """Validate input data is within realistic ranges"""
    warnings = []
    errors = []
    
    for feat in feature_names:
        if feat in data_dict and feat in FEATURE_RANGES:
            val = data_dict[feat]
            min_val, max_val = FEATURE_RANGES[feat]
            
            # Allow 10% buffer for edge cases
            buffer = (max_val - min_val) * 0.1
            soft_min = min_val - buffer
            soft_max = max_val + buffer
            
            if val < soft_min or val > soft_max:
                # Critical error - way outside range
                errors.append(f"{feat}: {val:.2f} is critically outside range [{min_val}, {max_val}]")
            elif val < min_val or val > max_val:
                # Warning - slightly outside but acceptable
                warnings.append(f"{feat}: {val:.2f} is at edge of typical range [{min_val}, {max_val}]")
    
    return warnings, errors

# Load models and metadata
print("Loading models...")
model1 = tf.keras.models.load_model('models/dnn_model.keras')
model2 = tf.keras.models.load_model('models/cnn_model.keras')
model3 = tf.keras.models.load_model('models/lstm_model.keras')
scaler = joblib.load('models/scaler.pkl')

with open('models/metadata.json', 'r') as f:
    metadata = json.load(f)

feature_names = metadata['feature_names']
feature_importance = metadata['feature_importance']

print("✓ Models loaded successfully")
print(f"✓ Features: {len(feature_names)}")
print(f"✓ Ensemble Accuracy: {metadata['model_performance']['ensemble']['accuracy']:.2%}")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         features=feature_names,
                         importance=feature_importance,
                         performance=metadata['model_performance'])

@app.route('/predict', methods=['POST'])
def predict():
    """Predict exoplanet from manual input or file upload"""
    try:
        if 'file' in request.files and request.files['file'].filename != '':
            # File upload prediction
            file = request.files['file']
            
            # Validate file
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})
            
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Only CSV files are allowed'})
            
            # Save file securely
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                df = pd.read_csv(filepath, comment='#')
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'success': False, 'error': f'Failed to read CSV file: {str(e)}'})
            
            # Validate file size
            if len(df) > 10000:
                os.remove(filepath)
                return jsonify({'success': False, 'error': 'File too large. Maximum 10,000 rows allowed'})
            
            if len(df) == 0:
                os.remove(filepath)
                return jsonify({'success': False, 'error': 'CSV file is empty'})
            
            # Check if required features exist
            missing_features = [f for f in feature_names if f not in df.columns]
            if missing_features:
                os.remove(filepath)
                return jsonify({
                    'success': False,
                    'error': f'Missing required features: {", ".join(missing_features)}'
                })
            
            # Select required features and handle missing values
            X = df[feature_names].copy()
            
            # Track data quality
            missing_count = X.isnull().sum().sum()
            missing_pct = (missing_count / (len(X) * len(feature_names))) * 100
            
            # Fill missing values
            X = X.fillna(X.median())
            
            # Clean up file
            os.remove(filepath)
            X_scaled = scaler.transform(X)
            
            # Reshape for CNN/LSTM
            X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            
            # Ensemble prediction
            pred1 = model1.predict(X_scaled, verbose=0).flatten()
            pred2 = model2.predict(X_cnn, verbose=0).flatten()
            pred3 = model3.predict(X_cnn, verbose=0).flatten()
            
            ensemble_pred = 0.5 * pred1 + 0.3 * pred2 + 0.2 * pred3
            
            results = []
            for i, pred in enumerate(ensemble_pred):
                results.append({
                    'index': i,
                    'probability': float(pred),
                    'prediction': 'EXOPLANET DETECTED ✓' if pred > 0.5 else 'NOT AN EXOPLANET ✗',
                    'confidence': float(abs(pred - 0.5) * 2 * 100),
                    'model_scores': {
                        'dnn': float(pred1[i]),
                        'cnn': float(pred2[i]),
                        'lstm': float(pred3[i])
                    }
                })
            
            # Statistics
            detected = sum(1 for r in results if r['probability'] > 0.5)
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            return jsonify({
                'success': True,
                'results': results,
                'count': len(results),
                'detected': detected,
                'detection_rate': f'{detected/len(results)*100:.1f}%',
                'avg_confidence': f'{avg_confidence:.1f}%',
                'data_quality': {
                    'missing_values': int(missing_count),
                    'missing_percentage': f'{missing_pct:.2f}%'
                }
            })
        
        else:
            # Manual input prediction
            data = request.get_json()
            
            if not data:
                return jsonify({'success': False, 'error': 'No input data provided'})
            
            # Extract feature values with validation
            input_values = []
            input_dict = {}
            for feat in feature_names:
                val = data.get(feat, 0)
                try:
                    float_val = float(val)
                    input_values.append(float_val)
                    input_dict[feat] = float_val
                except ValueError:
                    return jsonify({'success': False, 'error': f'Invalid value for {feat}: {val}'})
            
            # Validate input ranges
            validation_warnings, validation_errors = validate_input_data(input_dict)
            
            # Reject if critical errors
            if validation_errors:
                return jsonify({
                    'success': False,
                    'error': 'Input validation failed: ' + '; '.join(validation_errors)
                })
            
            X = np.array(input_values).reshape(1, -1)
            X_scaled = scaler.transform(X)
            X_cnn = X_scaled.reshape(1, len(feature_names), 1)
            
            # Ensemble prediction with error handling
            try:
                pred1 = model1.predict(X_scaled, verbose=0)[0][0]
                pred2 = model2.predict(X_cnn, verbose=0)[0][0]
                pred3 = model3.predict(X_cnn, verbose=0)[0][0]
            except Exception as e:
                return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'})
            
            ensemble_pred = 0.5 * pred1 + 0.3 * pred2 + 0.2 * pred3
            
            # Calculate model agreement (better confidence metric)
            predictions = [pred1, pred2, pred3]
            pred_std = np.std(predictions)
            model_agreement = (1 - min(pred_std * 2, 1)) * 100  # Higher agreement = higher confidence
            
            # Calculate feature contributions
            contributions = {}
            for i, (feat, val) in enumerate(zip(feature_names, input_values)):
                importance_score = feature_importance.get(feat, 0)
                normalized_val = X_scaled[0, i]
                
                contributions[feat] = {
                    'value': float(val),
                    'normalized': float(normalized_val),
                    'importance': float(importance_score),
                    'contribution': float(importance_score * abs(normalized_val))
                }
            
            # Sort contributions by impact
            sorted_contributions = dict(sorted(contributions.items(), 
                                             key=lambda x: x[1]['contribution'], 
                                             reverse=True))
            
            result = {
                'probability': float(ensemble_pred),
                'prediction': 'EXOPLANET DETECTED ✓' if ensemble_pred > 0.5 else 'NOT AN EXOPLANET ✗',
                'confidence': float(abs(ensemble_pred - 0.5) * 2 * 100),
                'model_agreement': float(model_agreement),
                'model_scores': {
                    'dnn': float(pred1),
                    'cnn': float(pred2),
                    'lstm': float(pred3)
                },
                'contributions': sorted_contributions,
                'validation_warnings': validation_warnings
            }
            
            return jsonify({'success': True, 'result': result})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance data for visualization"""
    return jsonify({
        'success': True,
        'features': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    })

@app.route('/model_performance', methods=['GET'])
def get_model_performance():
    """Get model performance metrics"""
    return jsonify({
        'success': True,
        'performance': metadata['model_performance']
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("EXOPLANET DETECTION WEB APP")
    print("="*80)
    print("\n🚀 Starting Flask server...")
    print("🌍 Open browser at: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5000)
