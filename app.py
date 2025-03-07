from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('age_predictor_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'Gender': int(request.form['gender']),
            'Blood Pressure (s/d)': float(request.form['blood_pressure']),
            'Smoking Status': int(request.form['smoking_status']),
            'Weight (kg)': float(request.form['weight']),
            'Height (cm)': float(request.form['height']),
            'BMI': float(request.form['bmi']),
            'Cholesterol Level (mg/dL)': float(request.form['cholesterol']),
            'Diet': int(request.form['diet']),
            'Physical Activity Level': int(request.form['physical_activity']),
            'Education Level': int(request.form['education']),
            'Blood Glucose Level (mg/dL)': float(request.form['blood_glucose']),
            'Cognitive Function': int(request.form['cognitive_function']),
            'Pollution Exposure': int(request.form['pollution_exposure']),
            'Medication Use': int(request.form['medication_use']),
            'Stress Levels': int(request.form['stress_levels']),
            'Family History': int(request.form['family_history']),
            'Bone Density (g/cm²)': float(request.form['bone_density']),
            'Vision Sharpness': int(request.form['vision_sharpness']),
            'Chronic Diseases': int(request.form['chronic_diseases']),
            'Sleep Patterns': int(request.form['sleep_patterns']),
            'Alcohol Consumption': int(request.form['alcohol_consumption']),
            'Mental Health Status': int(request.form['mental_health']),
            'Hearing Ability (dB)': float(request.form['hearing_ability']),
            'Sun Exposure': int(request.form['sun_exposure']),
            'Income Level': int(request.form['income_level'])
        }
        
        # Create a DataFrame with all required columns
        input_df = pd.DataFrame(columns=[
            'Gender', 'Blood Pressure (s/d)', 'Smoking Status', 'Weight (kg)', 
            'Height (cm)', 'BMI', 'Cholesterol Level (mg/dL)', 'Diet', 
            'Physical Activity Level', 'Education Level', 'Blood Glucose Level (mg/dL)', 
            'Cognitive Function', 'Pollution Exposure', 'Medication Use', 
            'Stress Levels', 'Family History', 'Bone Density (g/cm²)', 
            'Vision Sharpness', 'Chronic Diseases', 'Sleep Patterns', 
            'Alcohol Consumption', 'Mental Health Status', 'Hearing Ability (dB)', 
            'Sun Exposure', 'Income Level'
        ])
        
        # Fill in the user-provided values
        for key, value in user_input.items():
            input_df[key] = [value]
        
        # Fill missing columns with default values (e.g., 0)
        input_df = input_df.fillna(0)
        
        # Predict age
        predicted_age = model.predict(input_df)
        predicted_age = int(round(predicted_age[0]))  # Round and convert to integer
        
        return render_template('index.html', prediction=predicted_age)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)