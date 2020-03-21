import tensorflow as tf
import os,sys
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# importing the data
df = pd.read_excel("Data1.xlsx")

# readying the data
lt = list(range(df.shape[1]))
df.columns = lt
y = df.iloc[:,-3]
date = df.iloc[:,0]
date = pd.to_datetime(date).dt.dayofyear
product = df.iloc[:,2]
x = pd.concat([date,product])
lt = ['date','product']
x.columns = lt
unq_lt = product.unique()
product_grouping = x.groupby(['product','date']).sum()
product_grouping.reset_index(inplace = True) 
product_grouping = product_grouping.iloc[:,-1]
proc_data = []
for i in range(len(unq_lt)):
    k=1
    z=product_grouping[i*k:i*k+6].values
    k=k+1
    proc_data.append(z)

# setting up the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=2, output_dim=64))
model.add(tf.keras.layers.GRU(256, return_sequences=True))
model.add(tf.keras.layers.SimpleRNN(128))
model.add(tf.keras.layers.Dense(1))
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='sgd',metrics=['accuracy'])

# sampling the data
for i in range(len(unq_lt)):
  x = proc_data[i]
  n = 1
  x = x.astype('float32')
  sc_X = StandardScaler()
  x = x/max(x)
  x[x==1]=0.99
  x_sampled = np.zeros((len(x)-n-1,n))
  y = np.zeros((len(x)-n-1,1))
  for i in range(0,len(x)-n-1):
    x_sampled[i] = x[i:i+n]
    y[i] = x[i+1]
  # evaluating the data
  model.fit(x_sampled, y, epochs=500)
  acc,loss = model.evaluate(x_sampled,y)