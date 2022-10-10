import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app=Flask(__name__,template_folder='view')

# Load the pickle model
model = pickle.load(open("bodyfat_model.pkl", "rb"))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    event = request.json
    query_df = pd.DataFrame(event)
    prediction = model.predict(query_df)
    return jsonify({"Prediction": list(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
