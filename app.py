from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
feature_columns = joblib.load('feature_columns.joblib')


def build_feature_row(form):
    family_history = 1 if form['family_history'] == 'Yes' else 0
    smoking = 1 if form['smoking'] == 'Yes' else 0
    alcohol = 1 if form['alcohol'] == 'Yes' else 0
    hormone_therapy = 1 if form['hormone_therapy'] == 'Yes' else 0
    menopause = 1 if form['menopause'] == 'Pre' else 0
    genetic_mutation = 1 if form['genetic_mutation'] == 'Positive' else 0
    diabetes = 1 if form['diabetes'] == 'Yes' else 0

    activity = form['physical_activity']
    physical_activity_low = 1 if activity == 'Low' else 0
    physical_activity_moderate = 1 if activity == 'Moderate' else 0

    breastfeeding = form['breastfeeding']
    breastfeeding_not_applicable = 1 if breastfeeding == 'Not Applicable' else 0
    breastfeeding_yes = 1 if breastfeeding == 'Yes' else 0

    row = {
        'Age': int(form['age']),
        'BMI': float(form['bmi']),
        'Family_History': family_history,
        'Smoking': smoking,
        'Alcohol_Consumption': alcohol,
        'Hormone_Therapy': hormone_therapy,
        'Menopause_Status': menopause,
        'Genetic_Mutation': genetic_mutation,
        'Blood_Pressure': int(form['blood_pressure']),
        'Cholesterol': int(form['cholesterol']),
        'Diabetes': diabetes,
        'Exercise_Days_Per_Week': int(form['exercise_days']),
        'Physical_Activity_Low': physical_activity_low,
        'Physical_Activity_Moderate': physical_activity_moderate,
        'Breastfeeding_History_Not Applicable': breastfeeding_not_applicable,
        'Breastfeeding_History_Yes': breastfeeding_yes,
    }

    return pd.DataFrame([row])[feature_columns]


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    input_df = build_feature_row(request.form)
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    result = {
        'prediction': 'Cancer Positive Risk' if prediction == 1 else 'Cancer Negative (Low Risk)',
        'probability': round(probability * 100, 1),
        'is_positive': bool(prediction == 1)
    }

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
