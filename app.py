# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request

app = Flask(__name__)

# read our pickle file and label our final GBM model as model
model = pickle.load(open('defaulters_trained_model.pkl', 'rb'))


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = pd.DataFrame(data)  # converts shape from (x,) to (1, x)
        prediction = model.predict_proba(data)  # runs globally loaded model on the data
    return str(prediction[:,1])


if __name__ == '__main__':
    #load_model()  # load model at the beginning once only
    app.run(debug = True)
        
