import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class MedicalPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_models()
        
    def load_models(self):
     """Load trained ML models"""
    diseases = [
        'diabetes', 'heart', 'breast_cancer', 'lung_cancer', 'kidney_disease',
        'liver_disease', 'stroke', 'hypertension', 'parkinsons', 'alzheimer',
        'obesity', 'anemia', 'asthma', 'covid19', 'tuberculosis', 'thyroid',
        'skin_cancer', 'depression', 'pneumonia', 'gastrointestinal'
    ]
    
    for disease in diseases:
        path = f'models/{disease}_model.pkl'
        if os.path.exists(path):
            try:
                self.models[disease] = joblib.load(path)
            except:
                print(f"Could not load model for {disease}")
    
    def get_disease_inputs(self, disease):
        """Return input fields for each disease matching the frontend"""
        inputs_map = {
        'diabetes': [
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'gender', 'type': 'select', 'label': 'Gender', 'options': ['Male', 'Female', 'Other']},
            {'name': 'bmi', 'type': 'number', 'label': 'BMI', 'placeholder': 'Enter your BMI', 'min': 10, 'max': 60},
            {'name': 'blood_pressure', 'type': 'text', 'label': 'Blood Pressure', 'placeholder': 'e.g., 120/80'},
            {'name': 'glucose_level', 'type': 'number', 'label': 'Glucose Level', 'placeholder': 'mg/dL', 'min': 50, 'max': 300},
            {'name': 'family_history', 'type': 'select', 'label': 'Family History', 'options': ['No', 'Yes']},
            {'name': 'physical_activity', 'type': 'select', 'label': 'Physical Activity', 'options': ['Sedentary', 'Light', 'Moderate', 'Active']},
            {'name': 'diet', 'type': 'select', 'label': 'Diet Quality', 'options': ['Poor', 'Average', 'Good', 'Excellent']}
        ],
        'heart': [
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'gender', 'type': 'select', 'label': 'Gender', 'options': ['Male', 'Female', 'Other']},
            {'name': 'cholesterol', 'type': 'number', 'label': 'Cholesterol Level', 'placeholder': 'mg/dL', 'min': 100, 'max': 400},
            {'name': 'blood_pressure', 'type': 'text', 'label': 'Blood Pressure', 'placeholder': 'e.g., 120/80'},
            {'name': 'ecg', 'type': 'select', 'label': 'ECG Results', 'options': ['Normal', 'Abnormal']},
            {'name': 'heart_rate', 'type': 'number', 'label': 'Heart Rate', 'placeholder': 'bpm', 'min': 40, 'max': 200},
            {'name': 'smoking', 'type': 'select', 'label': 'Smoking History', 'options': ['Non-smoker', 'Former smoker', 'Current smoker']},
            {'name': 'diabetes_history', 'type': 'select', 'label': 'Diabetes History', 'options': ['No', 'Yes']}
        ],
        'breast_cancer': [
            {'name': 'tumor_size', 'type': 'number', 'label': 'Tumor Size (cm)', 'placeholder': 'Enter tumor size', 'min': 0, 'max': 10},
            {'name': 'biopsy_cell_data', 'type': 'text', 'label': 'Biopsy Cell Data', 'placeholder': 'Describe cell features'},
            {'name': 'family_history', 'type': 'select', 'label': 'Family History', 'options': ['No', 'Yes']},
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'hormonal_factors', 'type': 'text', 'label': 'Hormonal Factors', 'placeholder': 'Menstrual/Reproductive history'}
        ],
        'lung_cancer': [
            {'name': 'smoking_history', 'type': 'select', 'label': 'Smoking History', 'options': ['Non-smoker', 'Former smoker', 'Current smoker']},
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'gender', 'type': 'select', 'label': 'Gender', 'options': ['Male', 'Female', 'Other']},
            {'name': 'environmental_exposure', 'type': 'text', 'label': 'Environmental Exposure', 'placeholder': 'Pollution, Asbestos, etc.'},
            {'name': 'chronic_cough', 'type': 'select', 'label': 'Chronic Cough', 'options': ['No', 'Yes']},
            {'name': 'chest_pain', 'type': 'select', 'label': 'Chest Pain', 'options': ['No', 'Yes']}
        ],
        'kidney_disease': [
            {'name': 'blood_pressure', 'type': 'text', 'label': 'Blood Pressure', 'placeholder': 'e.g., 120/80'},
            {'name': 'creatinine', 'type': 'number', 'label': 'Creatinine Level', 'placeholder': 'mg/dL', 'min': 0, 'max': 10},
            {'name': 'albumin_urine', 'type': 'number', 'label': 'Albumin in Urine', 'placeholder': 'mg/dL', 'min': 0, 'max': 500},
            {'name': 'bun', 'type': 'number', 'label': 'Blood Urea Nitrogen (BUN)', 'placeholder': 'mg/dL', 'min': 0, 'max': 100},
            {'name': 'diabetes_history', 'type': 'select', 'label': 'Diabetes History', 'options': ['No', 'Yes']},
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120}
        ],
        'liver_disease': [
            {'name': 'bilirubin', 'type': 'number', 'label': 'Bilirubin Level', 'placeholder': 'mg/dL', 'min': 0, 'max': 10},
            {'name': 'alt', 'type': 'number', 'label': 'ALT Level', 'placeholder': 'U/L', 'min': 0, 'max': 200},
            {'name': 'ast', 'type': 'number', 'label': 'AST Level', 'placeholder': 'U/L', 'min': 0, 'max': 200},
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'alcohol_consumption', 'type': 'select', 'label': 'Alcohol Consumption', 'options': ['None', 'Occasional', 'Regular']},
            {'name': 'bmi', 'type': 'number', 'label': 'BMI', 'placeholder': 'Enter your BMI', 'min': 10, 'max': 60},
            {'name': 'hepatitis_history', 'type': 'select', 'label': 'Hepatitis Infection History', 'options': ['No', 'Yes']}
        ],
        'stroke': [
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'blood_pressure', 'type': 'text', 'label': 'Blood Pressure', 'placeholder': 'e.g., 120/80'},
            {'name': 'cholesterol', 'type': 'number', 'label': 'Cholesterol Level', 'placeholder': 'mg/dL', 'min': 100, 'max': 400},
            {'name': 'diabetes_history', 'type': 'select', 'label': 'Diabetes History', 'options': ['No', 'Yes']},
            {'name': 'physical_activity', 'type': 'select', 'label': 'Physical Activity', 'options': ['Sedentary', 'Light', 'Moderate', 'Active']},
            {'name': 'smoking', 'type': 'select', 'label': 'Smoking History', 'options': ['Non-smoker', 'Former smoker', 'Current smoker']},
            {'name': 'previous_stroke', 'type': 'select', 'label': 'Previous Stroke History', 'options': ['No', 'Yes']}
        ],
        'hypertension': [
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'bmi', 'type': 'number', 'label': 'BMI', 'placeholder': 'Enter your BMI', 'min': 10, 'max': 60},
            {'name': 'family_history', 'type': 'select', 'label': 'Family History', 'options': ['No', 'Yes']},
            {'name': 'sodium_intake', 'type': 'text', 'label': 'Sodium Intake', 'placeholder': 'mg/day'},
            {'name': 'alcohol_use', 'type': 'select', 'label': 'Alcohol Use', 'options': ['No', 'Occasional', 'Regular']},
            {'name': 'stress_levels', 'type': 'select', 'label': 'Stress Levels', 'options': ['Low', 'Moderate', 'High']},
            {'name': 'blood_pressure_readings', 'type': 'text', 'label': 'Blood Pressure Readings', 'placeholder': 'e.g., 120/80, 130/85'}
        ],
        'parkinsons': [
            {'name': 'tremor_frequency', 'type': 'number', 'label': 'Tremor Frequency', 'placeholder': 'Hz', 'min': 0, 'max': 20},
            {'name': 'voice_patterns', 'type': 'text', 'label': 'Voice Patterns', 'placeholder': 'Describe voice changes'},
            {'name': 'muscle_rigidity', 'type': 'select', 'label': 'Muscle Rigidity', 'options': ['Low', 'Moderate', 'High']},
            {'name': 'movement_speed', 'type': 'number', 'label': 'Movement Speed', 'placeholder': 'steps/sec or reaction time', 'min': 0, 'max': 10},
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'handwriting_analysis', 'type': 'text', 'label': 'Handwriting Analysis', 'placeholder': 'Describe writing changes'}
        ],
        'alzheimer': [
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'memory_test_scores', 'type': 'number', 'label': 'Memory Test Scores', 'placeholder': 'Score', 'min': 0, 'max': 100},
            {'name': 'family_history', 'type': 'select', 'label': 'Family History', 'options': ['No', 'Yes']},
            {'name': 'brain_scan_results', 'type': 'text', 'label': 'Brain Scan Results', 'placeholder': 'MRI/CT findings'},
            {'name': 'lifestyle', 'type': 'text', 'label': 'Lifestyle', 'placeholder': 'Diet, Exercise, etc.'}
        ],
        'obesity': [
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'gender', 'type': 'select', 'label': 'Gender', 'options': ['Male', 'Female', 'Other']},
            {'name': 'bmi', 'type': 'number', 'label': 'BMI', 'placeholder': 'Enter your BMI', 'min': 10, 'max': 60},
            {'name': 'diet', 'type': 'select', 'label': 'Diet Quality', 'options': ['Poor', 'Average', 'Good', 'Excellent']},
            {'name': 'exercise', 'type': 'select', 'label': 'Exercise Level', 'options': ['None', 'Light', 'Moderate', 'Active']},
            {'name': 'genetics', 'type': 'select', 'label': 'Genetics', 'options': ['No family history', 'Yes']},
            {'name': 'lifestyle', 'type': 'text', 'label': 'Lifestyle Factors', 'placeholder': 'Sedentary, Active, etc.'}
        ],
        'anemia': [
            {'name': 'hemoglobin', 'type': 'number', 'label': 'Hemoglobin Level', 'placeholder': 'g/dL', 'min': 0, 'max': 20},
            {'name': 'rbc_count', 'type': 'number', 'label': 'RBC Count', 'placeholder': 'million cells/µL', 'min': 0, 'max': 10},
            {'name': 'iron_level', 'type': 'number', 'label': 'Iron Level', 'placeholder': 'µg/dL', 'min': 0, 'max': 200},
            {'name': 'fatigue_symptoms', 'type': 'select', 'label': 'Fatigue Symptoms', 'options': ['None', 'Mild', 'Moderate', 'Severe']},
            {'name': 'diet_info', 'type': 'text', 'label': 'Diet Information', 'placeholder': 'Include iron intake'}
        ],
        'asthma': [
            {'name': 'family_history', 'type': 'select', 'label': 'Family History', 'options': ['No', 'Yes']},
            {'name': 'allergy_history', 'type': 'select', 'label': 'Allergy History', 'options': ['No', 'Yes']},
            {'name': 'pollution_exposure', 'type': 'text', 'label': 'Air Pollution Exposure', 'placeholder': 'Describe exposure'},
            {'name': 'coughing', 'type': 'select', 'label': 'Coughing Frequency', 'options': ['None', 'Occasional', 'Frequent']},
            {'name': 'wheezing', 'type': 'select', 'label': 'Wheezing', 'options': ['No', 'Yes']},
            {'name': 'lung_function_test', 'type': 'number', 'label': 'Lung Function Test (FEV1%)', 'placeholder': '%', 'min': 0, 'max': 100}
        ],
        'covid19': [
            {'name': 'temperature', 'type': 'number', 'label': 'Body Temperature', 'placeholder': '°F or °C', 'min': 95, 'max': 108},
            {'name': 'oxygen_level', 'type': 'number', 'label': 'Oxygen Level (SpO2)', 'placeholder': '%', 'min': 70, 'max': 100},
            {'name': 'cough', 'type': 'select', 'label': 'Cough Severity', 'options': ['None', 'Mild', 'Moderate', 'Severe']},
            {'name': 'fever', 'type': 'select', 'label': 'Fever Presence', 'options': ['No', 'Yes']},
            {'name': 'breathing_difficulty', 'type': 'select', 'label': 'Breathing Difficulty', 'options': ['No', 'Yes']},
            {'name': 'chest_pain', 'type': 'select', 'label': 'Chest Pain', 'options': ['No', 'Yes']},
            {'name': 'travel_contact_history', 'type': 'text', 'label': 'Travel/Contact History', 'placeholder': 'Describe exposure'}
        ],
        'tuberculosis': [
            {'name': 'chronic_cough', 'type': 'select', 'label': 'Chronic Cough', 'options': ['No', 'Yes']},
            {'name': 'chest_xray', 'type': 'text', 'label': 'Chest X-ray Results', 'placeholder': 'Describe findings'},
            {'name': 'weight_loss', 'type': 'select', 'label': 'Weight Loss', 'options': ['No', 'Yes']},
            {'name': 'night_sweats', 'type': 'select', 'label': 'Night Sweats', 'options': ['No', 'Yes']},
            {'name': 'blood_test', 'type': 'text', 'label': 'Blood Test Results', 'placeholder': 'Describe results'},
            {'name': 'sputum_test', 'type': 'text', 'label': 'Sputum Test Results', 'placeholder': 'Describe results'}
        ],
        'thyroid': [
            {'name': 'tsh', 'type': 'number', 'label': 'TSH Level', 'placeholder': 'µIU/mL', 'min': 0, 'max': 10},
            {'name': 't3', 'type': 'number', 'label': 'T3 Level', 'placeholder': 'ng/dL', 'min': 0, 'max': 300},
            {'name': 't4', 'type': 'number', 'label': 'T4 Level', 'placeholder': 'µg/dL', 'min': 0, 'max': 20},
            {'name': 'weight_change', 'type': 'text', 'label': 'Weight Changes', 'placeholder': 'Gain or loss'},
            {'name': 'heart_rate', 'type': 'number', 'label': 'Heart Rate', 'placeholder': 'bpm', 'min': 40, 'max': 200},
            {'name': 'fatigue', 'type': 'select', 'label': 'Fatigue', 'options': ['None', 'Mild', 'Moderate', 'Severe']},
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'gender', 'type': 'select', 'label': 'Gender', 'options': ['Male', 'Female', 'Other']}
        ],
        'skin_cancer': [
            {'name': 'lesion_image', 'type': 'file', 'label': 'Upload Lesion Image'},
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'sun_exposure', 'type': 'select', 'label': 'Sun Exposure', 'options': ['Low', 'Moderate', 'High']},
            {'name': 'family_history', 'type': 'select', 'label': 'Family History', 'options': ['No', 'Yes']}
        ],
        'depression': [
            {'name': 'sleep_patterns', 'type': 'text', 'label': 'Sleep Patterns', 'placeholder': 'Hours per night, quality'},
            {'name': 'mood_reports', 'type': 'text', 'label': 'Mood Reports', 'placeholder': 'Daily/weekly mood notes'},
            {'name': 'stress_levels', 'type': 'select', 'label': 'Stress Levels', 'options': ['Low', 'Moderate', 'High']},
            {'name': 'family_history', 'type': 'select', 'label': 'Family History', 'options': ['No', 'Yes']},
            {'name': 'lifestyle_habits', 'type': 'text', 'label': 'Lifestyle Habits', 'placeholder': 'Exercise, diet, etc.'}
        ],
        'pneumonia': [
            {'name': 'chest_xray', 'type': 'text', 'label': 'Chest X-ray Results', 'placeholder': 'Describe findings'},
            {'name': 'fever', 'type': 'number', 'label': 'Fever Temperature', 'placeholder': '°F or °C', 'min': 95, 'max': 108},
            {'name': 'cough', 'type': 'select', 'label': 'Cough Severity', 'options': ['None', 'Mild', 'Moderate', 'Severe']},
            {'name': 'breathing_rate', 'type': 'number', 'label': 'Breathing Rate', 'placeholder': 'breaths per minute', 'min': 0, 'max': 60},
            {'name': 'oxygen_levels', 'type': 'number', 'label': 'Oxygen Levels', 'placeholder': 'SpO2 %', 'min': 70, 'max': 100},
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120}
        ],
        'gastrointestinal': [
            {'name': 'age', 'type': 'number', 'label': 'Age', 'placeholder': 'Enter your age', 'min': 0, 'max': 120},
            {'name': 'abdominal_pain', 'type': 'select', 'label': 'Abdominal Pain', 'options': ['None', 'Mild', 'Moderate', 'Severe']},
            {'name': 'bloating', 'type': 'select', 'label': 'Bloating', 'options': ['None', 'Mild', 'Moderate', 'Severe']},
            {'name': 'endoscopy_results', 'type': 'text', 'label': 'Endoscopy Results', 'placeholder': 'Describe findings'},
            {'name': 'diet', 'type': 'text', 'label': 'Diet Information', 'placeholder': 'Meals, food habits'},
            {'name': 'stress_levels', 'type': 'select', 'label': 'Stress Levels', 'options': ['Low', 'Moderate', 'High']}
        ]
    }
        
        # Return inputs for the requested disease or empty list if not found
        return inputs_map.get(disease, [])
    
    def predict(self, disease, form_data):
        """Make prediction based on disease and form data"""
        if disease not in self.models:
            return self._simulate_prediction(disease, form_data)
        
        try:
            # Convert form data to feature array
            features = self._prepare_features(disease, form_data)
            
            # Make prediction
            prediction = self.models[disease].predict(features)[0]
            probability = self.models[disease].predict_proba(features)[0][1]
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': self._get_risk_level(probability),
                'factors': self._analyze_factors(disease, form_data),
                'recommendations': self._get_recommendations(disease, prediction, probability, form_data)
            }
            
        except Exception as e:
            print(f"Prediction error for {disease}: {e}")
            return self._simulate_prediction(disease, form_data)
    
    def _prepare_features(self, disease, form_data):
        """Prepare features for model prediction based on disease"""
        feature_maps = feature_maps = {
        'diabetes': ['age', 'bmi', 'glucose_level', 'blood_pressure'],
        'heart': ['age', 'gender', 'cholesterol', 'blood_pressure'],
        'breast_cancer': ['age', 'tumor_size', 'family_history', 'hormonal_factors'],
        'lung_cancer': ['age', 'smoking_history', 'chronic_cough', 'chest_pain'],
        'kidney_disease': ['age', 'creatinine', 'albumin_urine', 'bun'],
        'liver_disease': ['age', 'bilirubin', 'alt', 'ast'],
        'stroke': ['age', 'blood_pressure', 'cholesterol', 'diabetes_history'],
        'hypertension': ['age', 'bmi', 'family_history', 'blood_pressure_readings'],
        'parkinsons': ['age', 'tremor_frequency', 'muscle_rigidity', 'movement_speed'],
        'alzheimer': ['age', 'memory_test_scores', 'family_history', 'lifestyle'],
        'obesity': ['age', 'bmi', 'diet', 'exercise'],
        'anemia': ['hemoglobin', 'rbc_count', 'iron_level', 'fatigue_symptoms'],
        'asthma': ['family_history', 'allergy_history', 'coughing', 'wheezing'],
        'covid19': ['temperature', 'oxygen_level', 'cough', 'fever'],
        'tuberculosis': ['chronic_cough', 'weight_loss', 'night_sweats', 'blood_test'],
        'thyroid': ['age', 'tsh', 't3', 't4'],
        'skin_cancer': ['age', 'sun_exposure', 'family_history'],
        'depression': ['sleep_patterns', 'stress_levels', 'family_history', 'lifestyle_habits'],
        'pneumonia': ['age', 'fever', 'cough', 'breathing_rate'],
        'gastrointestinal': ['age', 'abdominal_pain', 'bloating', 'diet']
    }
        
        features = []
        if disease in feature_maps:
            for feature_name in feature_maps[disease]:
                value = form_data.get(feature_name, '0')
                # Convert text values to numerical
                if feature_name == 'blood_pressure' or feature_name == 'blood_pressure_readings':
                    # Extract systolic pressure (first number)
                    bp_parts = str(value).split('/')
                    features.append(float(bp_parts[0]) if bp_parts and bp_parts[0].isdigit() else 120.0)
                elif feature_name in ['gender', 'family_history', 'diabetes_history', 'chronic_cough', 'chest_pain', 
                                'fever', 'breathing_difficulty', 'weight_loss', 'night_sweats', 'previous_stroke']:
                    if feature_name == 'gender':
                      gender_map = {'Male': 1, 'Female': 0, 'Other': 0.5}
                      features.append(gender_map.get(value, 0.5))
                    else:
                     features.append(1 if value == 'Yes' else 0)
                elif feature_name in ['physical_activity', 'diet', 'exercise', 'smoking', 'alcohol_consumption', 
                                'stress_levels', 'muscle_rigidity', 'fatigue_symptoms', 'coughing', 'cough']:
                    option_maps = {
                    'physical_activity': {'Sedentary': 0, 'Light': 1, 'Moderate': 2, 'Active': 3},
                    'diet': {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3},
                    'exercise': {'None': 0, 'Light': 1, 'Moderate': 2, 'Active': 3},
                    'smoking': {'Non-smoker': 0, 'Former smoker': 1, 'Current smoker': 2},
                    'alcohol_consumption': {'None': 0, 'Occasional': 1, 'Regular': 2},
                    'stress_levels': {'Low': 0, 'Moderate': 1, 'High': 2},
                    'muscle_rigidity': {'Low': 0, 'Moderate': 1, 'High': 2},
                    'fatigue_symptoms': {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3},
                    'coughing': {'None': 0, 'Occasional': 1, 'Frequent': 2},
                    'cough': {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
                }
                feature_map = option_maps.get(feature_name, {})
                features.append(feature_map.get(value, 0))
                    
            else:
                    try:
                        features.append(float(value))
                    except:
                        features.append(0.0)
        else:
        # Default feature extraction for other diseases
          for key in ['age', 'bmi', 'glucose_level', 'cholesterol']:
            if key in form_data:
                try:
                    features.append(float(form_data[key]))
                except:
                    features.append(0.0)
        
        # Pad with zeros if needed
        while len(features) < 4:
            features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def _get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return 'Low'
        elif probability < 0.7:
            return 'Medium'
        else:
            return 'High'
    
    def _analyze_factors(self, disease, form_data):
        """Analyze contributing factors based on form data"""
        factors = []
        
        # Common health factors analysis
        common_factors = {
            'age': ('Age', 50, 30, 80),
            'bmi': ('BMI', 25, 18, 40),
            'glucose_level': ('Glucose Level', 100, 70, 200),
            'cholesterol': ('Cholesterol', 200, 150, 300),
            'blood_pressure': ('Blood Pressure', 120, 90, 180)
        }
        
        for factor_key, (factor_name, optimal, low, high) in common_factors.items():
            if factor_key in form_data:
                try:
                    value = float(form_data[factor_key])
                    # Calculate deviation from optimal
                    deviation = abs(value - optimal) / (high - low)
                    contribution = min(deviation * 100, 100)
                    factors.append({
                        'name': factor_name,
                        'contribution': round(contribution, 1),
                        'value': value,
                        'status': 'Optimal' if abs(value - optimal) < (optimal * 0.1) else 'Needs Attention'
                    })
                except:
                    continue
        
        return factors[:5]  # Return top 5 factors
    
    def _get_recommendations(self, disease, prediction, probability, form_data):
        """Get personalized recommendations"""
        base_recommendations = [
        "Consult with a healthcare professional for accurate diagnosis",
        "Schedule regular check-ups to monitor your health status",
        "Maintain a healthy lifestyle with balanced diet and exercise"
    ]
        disease_specific = {
        'diabetes': [
            "Monitor blood sugar levels regularly",
            "Maintain a balanced diet low in sugar and carbohydrates",
            "Engage in regular physical activity (30 minutes daily)",
            "Maintain healthy body weight",
            "Avoid smoking and limit alcohol consumption",
            "Get regular eye and foot examinations",
            "Monitor for symptoms like increased thirst, frequent urination"
        ],
        'heart': [
            "Monitor blood pressure regularly",
            "Reduce sodium intake in your diet",
            "Engage in cardiovascular exercises",
            "Avoid smoking and limit alcohol consumption",
            "Manage stress through relaxation techniques",
            "Maintain healthy cholesterol levels",
            "Get regular cardiac check-ups"
        ],
        'breast_cancer': [
            "Schedule regular mammograms as recommended",
            "Perform monthly self-examinations",
            "Maintain healthy weight and exercise regularly",
            "Limit alcohol consumption",
            "Discuss family history with your doctor",
            "Consider genetic counseling if high risk",
            "Be aware of breast changes and report immediately"
        ],
        'lung_cancer': [
            "Avoid smoking and secondhand smoke",
            "Use protective equipment in polluted environments",
            "Get regular chest X-rays if high risk",
            "Maintain good indoor air quality",
            "Report persistent cough to your doctor",
            "Monitor for symptoms like chest pain, coughing blood",
            "Consider low-dose CT screening if eligible"
        ],
        'kidney_disease': [
            "Monitor blood pressure regularly",
            "Control blood sugar if diabetic",
            "Reduce protein intake if recommended",
            "Avoid NSAIDs and other kidney-stressing medications",
            "Stay hydrated with adequate water intake",
            "Get regular kidney function tests",
            "Monitor for swelling, fatigue, changes in urination"
        ],
        'liver_disease': [
            "Avoid alcohol consumption",
            "Maintain healthy weight",
            "Get vaccinated for hepatitis if not immune",
            "Practice safe sex and avoid needle sharing",
            "Eat a balanced diet low in processed foods",
            "Get regular liver function tests",
            "Monitor for jaundice, abdominal pain, fatigue"
        ],
        'stroke': [
            "Control blood pressure and cholesterol",
            "Manage diabetes effectively",
            "Take prescribed medications regularly",
            "Recognize FAST symptoms (Face, Arms, Speech, Time)",
            "Maintain healthy weight and exercise",
            "Avoid smoking and limit alcohol",
            "Get regular cardiovascular check-ups"
        ],
        'hypertension': [
            "Reduce sodium intake to less than 2,300mg daily",
            "Follow DASH diet (Dietary Approaches to Stop Hypertension)",
            "Engage in regular aerobic exercise",
            "Limit alcohol to moderate levels",
            "Manage stress through meditation/yoga",
            "Monitor blood pressure at home",
            "Take medications as prescribed"
        ],
        'parkinsons': [
            "Engage in regular physical therapy and exercise",
            "Consider speech therapy for voice changes",
            "Modify home environment for safety",
            "Join support groups for emotional health",
            "Work with neurologist for medication management",
            "Practice balance and coordination exercises",
            "Maintain consistent daily routine"
        ],
        'alzheimer': [
            "Engage in mentally stimulating activities",
            "Maintain social connections and interactions",
            "Follow Mediterranean diet rich in omega-3",
            "Exercise regularly to improve blood flow to brain",
            "Manage cardiovascular risk factors",
            "Create memory aids and routines",
            "Consider cognitive training exercises"
        ],
        'obesity': [
            "Set realistic weight loss goals (1-2 pounds per week)",
            "Follow balanced diet with calorie control",
            "Engage in 150+ minutes of exercise weekly",
            "Keep food diary to track eating habits",
            "Get adequate sleep (7-9 hours nightly)",
            "Seek support from dietitian or weight loss program",
            "Avoid fad diets, focus on sustainable changes"
        ],
        'anemia': [
            "Increase iron-rich foods (red meat, spinach, beans)",
            "Consume vitamin C with iron meals for better absorption",
            "Consider iron supplements if recommended",
            "Treat underlying causes if identified",
            "Monitor hemoglobin levels regularly",
            "Rest adequately and manage fatigue",
            "Cook in cast-iron cookware to increase iron intake"
        ],
        'asthma': [
            "Identify and avoid asthma triggers",
            "Use inhalers as prescribed",
            "Create asthma action plan with your doctor",
            "Monitor peak flow readings regularly",
            "Keep rescue inhaler accessible at all times",
            "Get flu shot annually to prevent complications",
            "Consider allergy testing and treatment"
        ],
        'covid19': [
            "Isolate immediately and get tested",
            "Monitor oxygen saturation levels regularly",
            "Stay hydrated and get plenty of rest",
            "Seek emergency care if breathing difficulties worsen",
            "Follow local health department guidelines",
            "Notify close contacts for testing",
            "Continue monitoring symptoms for 14 days"
        ],
        'tuberculosis': [
            "Complete full course of antibiotics as prescribed",
            "Isolate during initial treatment phase if contagious",
            "Notify close contacts for screening",
            "Get regular follow-up chest X-rays",
            "Maintain good nutrition to support immune system",
            "Report any medication side effects immediately",
            "Avoid alcohol during treatment"
        ],
        'thyroid': [
            "Take thyroid medications as prescribed on empty stomach",
            "Get regular TSH level monitoring",
            "Maintain consistent medication timing",
            "Be aware of symptoms of over/under treatment",
            "Follow up with endocrinologist regularly",
            "Maintain iodine-rich diet if recommended",
            "Monitor for thyroid nodule changes"
        ],
        'skin_cancer': [
            "Use broad-spectrum sunscreen SPF 30+ daily",
            "Avoid peak sun hours (10am-4pm)",
            "Wear protective clothing and hats outdoors",
            "Perform monthly skin self-examinations",
            "Get annual skin checks by dermatologist",
            "Avoid tanning beds and artificial UV exposure",
            "Monitor existing moles for ABCDE changes"
        ],
        'depression': [
            "Establish consistent daily routine",
            "Engage in regular physical activity",
            "Practice mindfulness and relaxation techniques",
            "Maintain social connections and support network",
            "Consider therapy or counseling options",
            "Take medications as prescribed consistently",
            "Create a safety plan for crisis situations"
        ],
        'pneumonia': [
            "Complete full course of antibiotics if prescribed",
            "Get plenty of rest and stay hydrated",
            "Use humidifier to ease breathing",
            "Monitor temperature and breathing patterns",
            "Get pneumococcal vaccine if recommended",
            "Practice good hand hygiene to prevent spread",
            "Seek immediate care if symptoms worsen"
        ],
        'gastrointestinal': [
            "Keep food diary to identify trigger foods",
            "Eat smaller, more frequent meals",
            "Stay upright after eating to reduce reflux",
            "Manage stress through relaxation techniques",
            "Stay hydrated with water throughout day",
            "Consider probiotic foods for gut health",
            "Follow up with gastroenterologist for persistent symptoms"
        ]
    }
        
        specific = disease_specific.get(disease, [
            "Follow disease-specific prevention guidelines",
            "Monitor symptoms and report changes to your doctor",
            "Maintain regular medical check-ups"
        ])
        
        # Add risk-level specific recommendations
        risk_recommendations = {
        'Low': [
            "Continue with current healthy habits",
            "Maintain preventive care and regular screenings",
            "Stay informed about disease prevention strategies",
            "Focus on maintaining healthy lifestyle choices"
        ],
        'Medium': [
            "Increase frequency of health monitoring",
            "Consider lifestyle adjustments and improvements",
            "Discuss risk factors with healthcare provider",
            "Implement targeted prevention strategies"
        ],
        'High': [
            "Seek immediate medical consultation",
            "Implement comprehensive lifestyle changes",
            "Follow medical advice strictly",
            "Consider specialist referral if needed",
            "Establish regular monitoring schedule"
        ]
    }
        
        risk_level = self._get_risk_level(probability)
        risk_specific = risk_recommendations.get(risk_level, [])
        
        return base_recommendations + specific + risk_specific
    
    def _simulate_prediction(self, disease, form_data):
        """Fallback simulation for diseases without trained models"""
        # Calculate risk score based on available data
        risk_score = 0.1  # Base risk
        factors = []
        
        # Analyze common risk factors
        if 'age' in form_data:
            age = float(form_data.get('age', 40))
            age_risk = min((age - 30) / 100, 0.3)
            risk_score += age_risk
            factors.append({'name': 'Age', 'contribution': min(age_risk * 300, 100)})
        
        if 'bmi' in form_data:
            bmi = float(form_data.get('bmi', 25))
            bmi_risk = min(abs(bmi - 22) / 50, 0.3)
            risk_score += bmi_risk
            factors.append({'name': 'BMI', 'contribution': min(bmi_risk * 300, 100)})
        
        if 'family_history' in form_data and form_data['family_history'] == 'Yes':
            risk_score += 0.2
            factors.append({'name': 'Family History', 'contribution': 60})
        
        # Disease-specific risk adjustments
        disease_risks = {
            'diabetes': ['glucose_level', 'physical_activity'],
            'heart': ['cholesterol', 'blood_pressure', 'smoking'],
            'cancer': ['smoking_history', 'environmental_exposure']
        }
        
        risk_score = min(risk_score, 0.95)
        risk_score = max(risk_score, 0.05)
        
        probability = risk_score
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability': probability,
            'risk_level': self._get_risk_level(probability),
            'factors': factors[:3],
            'recommendations': self._get_recommendations(disease, prediction, probability, form_data)
        }

# Utility function to test the predictor
def test_predictor():
    """Test the medical predictor with sample data"""
    predictor = MedicalPredictor()
    
    # Test diabetes prediction
    sample_data = {
        'age': '45',
        'gender': 'Male',
        'bmi': '28',
        'blood_pressure': '130/85',
        'glucose_level': '110',
        'family_history': 'Yes',
        'physical_activity': 'Moderate',
        'diet': 'Average'
    }
    
    result = predictor.predict('diabetes', sample_data)
    print("Diabetes Prediction Result:")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.2f}")
    print(f"Risk Level: {result['risk_level']}")
    print("Factors:", result['factors'])
    print("Recommendations:", result['recommendations'][:3])

if __name__ == '__main__':
    test_predictor()