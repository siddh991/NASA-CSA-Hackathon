from flask import Flask, render_template, request, jsonify, redirect, flash
import pandas as pd
from random import randrange
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
        percentage = '120%-200%'
    if result == 2:
        percentage = '200%+'
    return percentage


@app.route('/', methods=['GET', 'POST'])
def root():
    return render_template('test.html')

@app.route('/result', methods=['GET', 'POST'])
def display_image():
    index_list = [1083, 100]
    index_image = {1083: {'before': 'sidd_NDVI_1.jpg', 'after': 'sidd_NDVI_2.jpg'},
                   100: {'before': 'index100-2019-05-21-2236.jpg', 'after': 'index100-2019-05-26-2204.jpg'}}
    index = index_list[randrange(1)]
    prediction = predict(index)
    before_img = 'static/' + index_image[index]['before']
    after_img = 'static/' + index_image[index]['after']
    return render_template('result.html', prediction=prediction, index=index, before_img=before_img, after_img=after_img)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)