import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)
model = pickle.load(open('dtr.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( final_features )

    if prediction==1:
        pred='air pollution affected!!'
    else:
        pred="Congratulations! air pollution is Normal"
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0',port=5000)
    
