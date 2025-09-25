from flask import Flask, render_template, request, jsonify
import json
import os
import pandas as pd
from datetime import datetime
from utils.predictor import MedicalPredictor

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

predictor = MedicalPredictor()

# Ensure directories exist
os.makedirs('user_data', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<disease>', methods=['POST'])
def predict_disease(disease):
    try:
        # Get form data
        form_data = request.json
        
        # Save user data to JSON file
        user_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_data/{disease}_{user_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(form_data, f, indent=2)
        
        # Get prediction
        result = predictor.predict(disease, form_data)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get_disease_inputs/<disease>')
def get_disease_inputs(disease):
    """Return the input fields required for a specific disease"""
    try:
        inputs = predictor.get_disease_inputs(disease)
        return jsonify(inputs)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5007)