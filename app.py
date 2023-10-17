import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
app=Flask(__name__)
model = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
## to create predict api, takes json input
@app.route('/predict_api',methods =['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_trans_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output =model.predict(new_trans_data)
    print(output[0])
    return jsonify(output[0])
@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    new_trans_data = scaler.transform(np.array(data).reshape(1,-1))
    output =model.predict(new_trans_data)[0]
    return render_template("home.html",prediction_text='Price is {}'.format(output))

if __name__=='__main__':
    app.run(debug=True)





