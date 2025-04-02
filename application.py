from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

model = pickle.load(open("Models/fraud_detection_model.pkl","rb"))
scaler = pickle.load(open("Models/scaler.pkl","rb"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_data():
    if request.method=='POST':
         # Retrieve input values from the form
        input_features = [float(request.form.get(f'V{i}')) for i in range(1, 29)]
            
        # Retrieve 'time' and 'amount'
        time = float(request.form.get('time'))
        amount = float(request.form.get('amount'))

        # Standardize 'amount' using the scaler
        new_amount = scaler.transform(np.array(amount).reshape(-1, 1))[0, 0]  # Extract scalar value

        # Append standardized amount to input features
        input_features.append(new_amount)

        # Convert to NumPy array and reshape for model input
        input_array = np.array([input_features])

        # Combine 'time' with the input array
        data = np.hstack((np.array([[time]]), input_array))
        results = model.predict(data)

        return render_template('home.html',results=results[0])
    
    else:
        return render_template("home.html")

if __name__ == '__main__':
    app.run()
