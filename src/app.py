# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
import numpy as np
from flask import Flask, request, jsonify
import pickle

def build_model():
    # Importing the dataset
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    # Fitting Simple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    # Saving model to disk
    # pickle.dump(regressor, open('model.pkl','wb'))

    # # Loading model to compare the results
    # model = pickle.load( open('model.pkl','rb'))

    return regressor
    

app = Flask(__name__)

# build  the model
model = build_model()

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['exp'])]])

    # Take the first value of prediction
    output = prediction[0]

    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)


