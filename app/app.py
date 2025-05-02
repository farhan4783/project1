from flask import Flask, request, jsonify # type: ignore
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Customer Churn Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([[
        data['call_minutes_month'],
        data['complaints'],
        data['recharge_frequency'],
        data['last_recharge_days']
    ]])
    prediction = model.predict(features)
    churn_probability = model.predict_proba(features)[0][1]
    output = {
        'churn_prediction': int(prediction[0]),
        'churn_probability': churn_probability
    }
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
