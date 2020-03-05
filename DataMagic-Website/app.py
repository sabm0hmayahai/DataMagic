from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
import pandas
import csv
import tensorflow as tf
import os
import sys
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


######### Mongodb ##################
demo = MongoClient()
myclient = MongoClient('localhost', 27017)


db = myclient["HackBout"]  # db name
collection = db["Dataset"]  # collection name
feedb = db["Feedback"]  # feedback collection

app = Flask(__name__)


@app.route("/")
def menue():
    return render_template("home.html")


@app.route("/upload", methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        db.collection.drop()
        csvfilepath = request.form["file"]
        print(csvfilepath)

        with open(csvfilepath, "r") as csvfile:
            csvreader = csv.DictReader(csvfile)
            for csvrow in csvreader:
                print(csvrow)
                db.collection.insert_one(csvrow)
        return render_template("upload_select.html")
    else:
        return render_template("upload.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    v1 = request.form["v1"]
    v2 = request.form["v2"]
    v3 = request.form["v3"]
    v4 = request.form["v4"]
    v5 = request.form["v5"]
    v6 = request.form["v6"]
    # importing the data
    df = pd.read_excel("Data1.xlsx")

    # readying the data
    lt = list(range(df.shape[1]))
    df.columns = lt
    y = df.iloc[:, -3]
    date = df.iloc[:, 0]
    date = pd.to_datetime(date).dt.dayofyear
    product = df.iloc[:, 2]
    x = pd.concat([date, product])
    lt = ['date', 'product']
    x.columns = lt
    unq_lt = product.unique()
    product_grouping = x.groupby(['product', 'date']).sum()
    product_grouping.reset_index(inplace=True)
    product_grouping = product_grouping.iloc[:, -1]
    proc_data = []
    for i in range(len(unq_lt)):
        k = 1
        z = product_grouping[i*k:i*k+6].values
        k = k+1
        proc_data.append(z)

    # setting up the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=2, output_dim=64))
    model.add(tf.keras.layers.GRU(256, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(128))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), optimizer='sgd', metrics=['accuracy'])

    # sampling the data
    for i in range(len(unq_lt)):
        x = proc_data[i]
        n = 1
        x = x.astype('float32')
        sc_X = StandardScaler()
        x = x/max(x)
        x[x == 1] = 0.99
        x_sampled = np.zeros((len(x)-n-1, n))
        y = np.zeros((len(x)-n-1, 1))
        for i in range(0, len(x)-n-1):
            x_sampled[i] = x[i:i+n]
            y[i] = x[i+1]

    # evaluating the data
    model.fit(x_sampled, y, epochs=500)
    acc, loss = model.evaluate(x_sampled, y)
    return render_template("results.html")


@app.route("/results", methods=['GET', 'POST'])
def results():
    data = db.collection.find_one()
    features = []
    for key in data:
        features.append(key)
    distinct = db.collection.distinct("ITEM_NAME")
    values = db.collection.find({}, {QYANTITY: 1, _id: 0})
    dates = db.collection.aggregate([{$ group: {_id: "$Date"}}])
    totalDay = db.collection.distinct("QYANTITY")
    return render_template("results.html", features=features, distinct=distinct, dates=dates, totalDay=totalDay)


@app.route("/feedback", methods=['POST', 'GET'])
def feedback():
    if request.method == 'POST':
        data = request.form['feedback']
        print("Submitted !", data)
        db.feedb.insert_one({"feedback": data})
        return redirect(url_for('feedback'))
    else:
        return render_template("feedback.html")


if __name__ == "__main__":
    app.run()
