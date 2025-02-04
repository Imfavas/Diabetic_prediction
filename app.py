from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# Validator function
def validate_features(data):
    errors = []

    # Validate each field
    if not (0 <= data['pregnancies'] <= 20):
        errors.append("Pregnancies must be between 0 and 20.")
    if not (50 <= data['glucose'] <= 200):
        errors.append("Glucose level must be between 50 and 200.")
    if not (30 <= data['blood_pressure'] <= 120):
        errors.append("Blood Pressure must be between 30 and 120.")
    if not (0 <= data['skin_thickness'] <= 99):
        errors.append("Skin Thickness must be between 0 and 99.")
    if not (0 <= data['insulin'] <= 900):
        errors.append("Insulin must be between 0 and 900.")
    if not (10.0 <= data['bmi'] <= 60.0):
        errors.append("BMI must be between 10.0 and 60.0.")
    if not (0.0 <= data['dpf'] <= 2.5):
        errors.append("Diabetes Pedigree Function must be between 0.0 and 2.5.")
    if not (0 <= data['age'] <= 120):
        errors.append("Age must be between 0 and 120.")

    return errors


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values
        features = {
            'pregnancies': float(request.form['pregnancies']),
            'glucose': float(request.form['glucose']),
            'blood_pressure': float(request.form['blood_pressure']),
            'skin_thickness': float(request.form['skin_thickness']),
            'insulin': float(request.form['insulin']),
            'bmi': float(request.form['bmi']),
            'dpf': float(request.form['dpf']),
            'age': float(request.form['age']),
        }

        # Validate the input values
        errors = validate_features(features)
        if errors:
            return render_template('index.html', prediction_text="; ".join(errors))

        # Standardize the features
        feature_values = list(features.values())
        scaled_features = scaler.transform([feature_values])

        # Make prediction
        prediction = model.predict(scaled_features)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return render_template('index.html', prediction_text=f'Result: {result}')
    except ValueError:
        return render_template('index.html', prediction_text="Please enter valid numeric values.")
    except Exception as e:
        return render_template('index.html', prediction_text=f"An error occurred: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
