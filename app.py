from flask import Flask, render_template, request, jsonify, redirect, flash
import pandas as pd
import joblib

app = Flask(__name__)


def predict(index):
    data = pd.read_csv('./data/newest_data.csv')
    data.drop(columns=['Unnamed: 0', 'acq_date', 'acq_time', 'latitude', 'longitude'], inplace=True)
    try:
        loaded_model = joblib.load('sad_model.sav')
    except Exception as e:
        print(e)
    this_data = data.loc[[index]]
    result = loaded_model.predict_classes(this_data)
    if result == 0:
        percentage = '0%-120%'
    if result == 1:
        percentage = '120%-250%'
    if result == 2:
        percentage = '250%+'
    return percentage


@app.route('/', methods=['GET', 'POST'])
def root():
    return render_template('test.html')

@app.route('/result', methods=['GET', 'POST'])
def display_image():
    index_list = [1089, 1083, 100]
    index = index_list.pop()
    prediction = predict(index)
    render_template('result.html', prediction=prediction, index=index)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)