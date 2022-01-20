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
#$python detect.py –d <dataset> -n <number of time series selected> -mae <error value as double>

#from google.colab import drive
#import tensorflow as tf
#print(tf.__version__)
#sinartisi proetimasias dedomenwn gia to model
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
N = 359
mae = 0.6

if len(sys.argv) != 7:
    print("Wrong no of arguments!")
    exit(1)

N=int(sys.argv[4])
file_name=sys.argv[2]
mae=float(sys.argv[6])

input_path = path + file_name

df = pd.read_csv(input_path, '\t', header=None)
print("Number of rows and columns:", df.shape)

dataset = df.iloc[:, 1:].values
#print(dataset.shape)
train_size = int(0.8 * dataset.shape[1])
test_size = (dataset.shape[1] - train_size)

sequence = list(range(0, df.shape[0]))
random.shuffle(sequence)



N_train = N  # gia na kanei train mono me to dataset pu tha kanei predict
# N_train=df.shape[0]#gia na kanei train me olo to dataset

test_arr = []

TIME_STEPS = 30
ind_train = np.array(list(range(0, train_size)))
ind_test = np.array(list(range(train_size + 1, dataset.shape[1] + 1)))
counterest=0
#model_name = 'model_detect4.h5'
model_name='testdetect10_2.h5'
model = tf.keras.models.load_model(model_name)

plot_counts=0
percentage=0.1*N#ektypwnw to poly 10% tou dataset pou kanw test
max_plots=int(percentage)
if max_plots<1:
    max_plots=1
rand = 3
for iteration in range(0,N):
    arr=dataset[sequence[iteration]]
    test_df = pd.DataFrame(index=ind_test)
    test_df['VALUES']=arr[train_size:]

    scaler = StandardScaler()
    scaler = scaler.fit(test_df[['VALUES']])
    test_df['VALUES']=scaler.transform(test_df[['VALUES']])

    X_test,y_test=create_dataset(test_df[['VALUES']],test_df.VALUES,TIME_STEPS)

    THRESHOLD = mae
    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    ind_testX = np.array(list(range(TIME_STEPS + train_size + 1, dataset.shape[1] + 1)))
    test_score_df = pd.DataFrame(index=ind_testX)
    test_score_df['loss'] = test_mae_loss#apoliti diafora meta3i predicted timis kai kanonikis timis
    test_score_df['threshold'] = THRESHOLD#i timi mae pu dinete aptin grammi entolwn
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold#True an exi timi loss megaliteri tu THRESHOLD aliws False
    test_score_df['values'] = test_df[TIME_STEPS:]#pragmatikes times
    test_score_df['dates'] = np.array(list(range(TIME_STEPS + train_size + 1, dataset.shape[1] + 1)))
    anomalies = test_score_df[test_score_df.anomaly == True]
    print(anomalies.head())

    """
    #grafiki parastasi loss me dates
    plt.plot(test_score_df['dates'],test_score_df['loss'],color = "blue")
    plt.axhline(y=THRESHOLD, color='red', linestyle='-')
    plt.show()
    """

    if not anomalies.empty:
        if(rand%3==0 and plot_counts<max_plots):
        #if(plot_counts<10):
            plot_counts=plot_counts+1
            plt.plot(test_score_df['dates'], test_score_df['values'], color="blue")
            plt.plot(anomalies['dates'], anomalies['values'], 'o', color="red")#kokkines kukides opu to loss 3epernaei to THRESHOLD
            plt.ylabel('Values')
            plt.xlabel('Dates')
            plt.show()
        rand = random.randint(0, 10)
#print(counterest)
#print(plot_counts)
print("Finished :) No of plots: ",plot_counts)