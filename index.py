from flask import Flask, request, redirect, url_for, flash, render_template, jsonify
import os
from flask import send_file
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/ann", methods = ['POST'])
def checklogin():
    req_data = request.get_json()
    date = req_data['date'] 
    df = pd.DataFrame([date], columns=['Date'])
    df['Date'] = pd.to_datetime(date)
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    old_df = pd.read_csv('dataset.csv')
    values = old_df.values[-1].tolist()
    df['30d_avg'] = values[9]
    df['std'] = values[10]
    df['RSI'] = values[11]
    df['Williams%R'] = values[12]
    df.drop(columns=['Date'],axis=1, inplace=True)
    values = df.iloc[0]
    values = np.array(values).reshape(1, -1)
    loaded_model = pickle.load(open('ann_model.sav', 'rb'))
    result = loaded_model.predict(values)
    result = np.array2string(result)
    return jsonify(result)


if __name__ == '__main__':
   app.run(debug = True)