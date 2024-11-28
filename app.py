import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
# Load the saved model
model = joblib.load('iris_model.joblib')
app = Flask(__name__)
CORS(app) # Enable CORS for all routes
@app.route('/predict', methods=['POST'])
def predict():
	data = request.json['features']
	prediction = model.predict([data])[0]
	return jsonify({'prediction': int(prediction)})
if __name__ == "__main__":
	app.run(port=5000, debug=True)