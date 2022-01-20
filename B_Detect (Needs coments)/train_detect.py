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
from sklearn.preprocessing import StandardScaler
import random
import sys
#$python train_detect.py –d <dataset> -n <number of time series selected> <OPTIONAL: name of model to save>
#from google.colab import drive
#import tensorflow as tf
#print(tf.__version__)
#sinartisi proetimasias dedomenwn gia dataset
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


#drive.mount('/content/drive')
# input_path = "/content/drive/My Drive/project_3/input.csv"
#path = "/content/drive/My Drive/project_3/"
# file_name="nasd_input.csv"

path=""
file_name = "nasdaq2007_17.csv"
model_name = 'model_detect_1.h5'
N = 359
#diavasma stoixeiwn aptin grammi entolwn
if len(sys.argv) < 5:
    print("Wrong no of arguments!")
    exit(1)

N=int(sys.argv[4])
file_name=sys.argv[2]
if len(sys.argv) > 5:
    model_name=sys.argv[5]


input_path = path + file_name

df = pd.read_csv(input_path, '\t', header=None)
print("Number of rows and columns:", df.shape)
# print(df.head(5))

if N>df.shape[0]:
    N=df.shape[0]
    print("Wrong n(n>size of dataset), new n=",N)

dataset = df.iloc[:, 1:].values
#print(dataset.shape)
train_size = int(0.8 * dataset.shape[1])
#test_size = (dataset.shape[1] - train_size)


sequence = list(range(0, df.shape[0]))##lista me ola ta rows tu dataset air8mika
random.shuffle(sequence)#anakatevume afti tin lista
print(df.shape[0])


N_train = N  # gia na kanei train mono me to dataset pu tha kanei predict
# N_train=df.shape[0]#gia na kanei train me olo to dataset

test_arr = []

TIME_STEPS = 30
ind_train = np.array(list(range(0, train_size)))#times apto 0 mexri to train_size
#ind_test = np.array(list(range(train_size + 1, dataset.shape[1] + 1)))
counterest=0
for iteration in range(0, N_train):
    print("Iteration: ",iteration)
    counterest=counterest+1
    #arr=dataset[iteration]
    arr = dataset[sequence[iteration]]
    train_df = pd.DataFrame(index=ind_train)#ftiaxnume DataFrame etsi wste o kwdikas mas na ine simvatos me auton tis istoselidas
    train_df['VALUES'] = arr[:train_size]

    scaler = StandardScaler()
    scaler = scaler.fit(train_df[['VALUES']])
    train_df['VALUES'] = scaler.transform(train_df[['VALUES']])


    X_train, y_train = create_dataset(train_df[['VALUES']], train_df.VALUES, TIME_STEPS)

    #print(X_train.shape)
    #print(X_train.shape[1], " ", X_train.shape[2])

    if iteration == 0::#mono stin prwti epanalipsi that ftia3ei to montelo
        model = keras.Sequential()
        model.add(keras.layers.LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
        for i in range(0, 1):
            # Adding a second LSTM layer and some Dropout regularisation
            model.add(keras.layers.LSTM(units=50, return_sequences=True))
            model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.LSTM(units=40, return_sequences=True))
        model.add(keras.layers.Dropout(rate=0.3))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
        model.compile(loss='mae', optimizer='adam')

    history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1, shuffle=False)#model fit se ka8e epanalipsi
    #kanume predict gia a3ilogisi tu dataset
    X_train_pred = model.predict(X_train)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
    #print("train_mae_loss ",train_mae_loss)

model.save(path + model_name)#apo8ikeuoume to montelo

print("Finished :)")