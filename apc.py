# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 10:38:56 2025

@author: Motheo Moshageng
"""

#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle

#create flask app

app = Flask(__name__)

#load the model
logmodel = pickle.load(open('logmodel.pkl','rb'))

#define the home page
@app.route('/')
def home():
    return render_template('index.html')

#define the prediction
@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(X) for X in request.form.values() ] #convert string(input) values into float 
    features = [np.array(int_features)] #convert float values into an array
    prediction = logmodel.predict(features)
    
    return render_template('index.html', prediction_text = 'the customer has {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=False)
    