from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Initialize Flask app
app = Flask(__name__)

# Read the dataset
data = pd.read_csv("crop_yield.csv") 

# Drop unnecessary columns
data1 = data.drop(['Crop_Year', 'Production', 'Annual_Rainfall'], axis=1)

# Initialize label encoders
label_encoders = {}
categorical_cols = ['Crop', 'Season', 'State']

# Modify this section to ensure label encoders are initialized before accessing them
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    # Remove leading and trailing spaces from unique values
    unique_values = [value.strip() for value in data1[col].unique()]
    label_encoders[col].fit(unique_values)
    data1[col] = label_encoders[col].transform(data1[col].str.strip())

data1['Stratify_Column'] = data1['Season'].astype(str)
data1['Fertilizer'] = (data1['Fertilizer'] / data1['Area'])
data1['Pesticide'] = (data1['Pesticide'] / data1['Area'])

# Splitting the dataset into features (X) and target variables (y)
X = data1.drop(['Fertilizer', 'Pesticide', 'Yield', 'Area', 'Stratify_Column'], axis=1)
y = data1[['Fertilizer', 'Pesticide', 'Yield']]

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    state = request.form['state']
    crop = request.form['crop']
    season = request.form['season']

    # Encode user input
    user_input = pd.DataFrame({'Crop': [crop], 'Season': [season], 'State': [state]})
    for col in categorical_cols:
        user_input[col] = label_encoders[col].transform(user_input[col])

    # Prediction
    predicted_values = model.predict(user_input)

    # Display predicted values
    predicted_fertilizer = round(predicted_values[0][0],2)
    predicted_pesticide = round(predicted_values[0][1],2)
    predicted_yield = round(predicted_values[0][2],2)

    return render_template('result.html', fertilizer=predicted_fertilizer, pesticide=predicted_pesticide, pred_yield=predicted_yield)

if __name__ == '__main__':
    app.run(debug=True)
