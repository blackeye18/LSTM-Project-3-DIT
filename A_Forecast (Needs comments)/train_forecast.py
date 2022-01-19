import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import random
import sys

#$python forecast.py –d <dataset> -n <number of time series selected> <OPTIONAL: name of model to save>

#from google.colab import drive
#drive.mount('/content/drive')
# input_path = "/content/drive/My Drive/project_3/input.csv"
#path = "/content/drive/My Drive/project_3/"
# file_name="nasd_input.csv"#to arxiko arxeio

path=""
file_name = "nasdaq2007_17.csv"  # to kainourio arxeio pou anevasan 3/1

N = 15

if len(sys.argv) < 5:
    print("Wrong no of arguments!")
    exit(1)

N=int(sys.argv[4])
file_name=sys.argv[2]

input_path = path + file_name
model_name = 'model_forecast_2.h5'

if len(sys.argv) > 5:
    model_name=sys.argv[5]

df = pd.read_csv(input_path, '\t', header=None)
print("Number of rows and columns:", df.shape)
dataset = df.iloc[:, 1:].values
print(dataset.shape)

if N>df.shape[0]:
    N=df.shape[0]
    print("Wrong n(n>size of dataset), new n=",N)

train_limit = round(0.8 * dataset.shape[1])
Y_test_set = np.array(list(range(train_limit + 1, dataset.shape[1] + 1)))



sequence = list(range(0, df.shape[0]))
random.shuffle(sequence)

print("Now training on ",N," training sets")
train_counter=0
for iteration in range(0, N):
    print("Iteration: ",iteration)
    train_counter=train_counter+1
    #arr = dataset[sequence[iteration]]
    arr = dataset[iteration]
    # repeat_arr[iteration]=dataset[sequence[iteration]]
    X_training_set = arr[:train_limit]
    X_test_set = arr[train_limit:]
    #print("X_training_set size ", X_training_set.size)
    #print("X_test_set size ", X_test_set.size)

    training_set = X_training_set
    training_set = training_set.reshape(-1, 1)
    #print("training_set.shape", training_set.shape)

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    prev_val = 60  # round(dataset.shape[1]/10)
    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    Y_train = []
    for i in range(prev_val, train_limit):
        X_train.append(training_set_scaled[i - prev_val:i, 0])
        Y_train.append(training_set_scaled[i, 0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #print("X_train.shape", X_train.shape)

    if iteration == 0:
        drop_num = 0.2
        unit_num = 50
        # keras.backend.clear_session()
        model = Sequential()
        # Adding the first LSTM layer and some Dropout regularisation
        model.add(LSTM(units=unit_num, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(drop_num))
        for i in range(0, 2):
            # Adding a second LSTM layer and some Dropout regularisation
            model.add(LSTM(units=unit_num, return_sequences=True))
            model.add(Dropout(drop_num))
        # Adding a fourth LSTM layer and some Dropout regularisation
        model.add(LSTM(units=unit_num))
        model.add(Dropout(drop_num + 0.1))
        # Adding the output layer
        model.add(Dense(units=1))
        # Compiling the RNN
        model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1, shuffle=False)
    # model.fit(X_train, Y_train, epochs = 100, batch_size = 32)

model.save(path + model_name)
print("Finished :) Trained with: ",train_counter)
