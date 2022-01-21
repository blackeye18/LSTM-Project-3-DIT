#Georgios Georgiou sdi1800220 - Panagiotis Mavrommatis sdi1800115
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

#$python forecast.py –d <dataset> -n <number of time series selected>

#from google.colab import drive
#drive.mount('/content/drive')
# input_path = "/content/drive/My Drive/project_3/input.csv"
#path = "/content/drive/My Drive/project_3/"
# file_name="nasd_input.csv"#to arxiko arxeio

path=""
file_name = "nasdaq2007_17.csv" # to kainourio arxeio pou anevhke 3/1
input_path = path + file_name

N = 15 #arxikes times
#diavazume tis times aptin grammi entolwn
if len(sys.argv) != 5:
    print("Wrong no of arguments!")
    exit(1)

N=int(sys.argv[4])
file_name=sys.argv[2]

input_path = path + file_name

df = pd.read_csv(input_path, '\t', header=None)
print("Number of rows and columns:", df.shape)
dataset = df.iloc[:, 1:].values
#print(dataset.shape)
#pairnume 80% train kai 20%test
train_limit = round(0.8 * dataset.shape[1])
Y_test_set = np.array(list(range(train_limit + 1, dataset.shape[1] + 1)))

sequence = list(range(0, df.shape[0]))#lista me ola ta rows tu dataset air8mika
random.shuffle(sequence)#anakatevume afti tin lista


print("Now Loading model that has been trained with the whole dataset")

#model_name = 'model_forecast.h5'
#model_name ='newforecastsmalltest.h5'
model_name = 'best_model_forecast.h5'
#model_name= "model_forecast4layers.h5"

model = tf.keras.models.load_model(model_name)
prev_val = 60  # round(dataset.shape[1]/10)
train_counter=0
test_counter=0
total_mean=0
percentage=0.1*N #ektypwnw se plot to poly 10% tou dataset pou kanw test
max_plots=int(percentage)
if max_plots<1:
    max_plots=1
plot_counts=0
count=0
rand = 3
for iteration in range(0, N):
    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    test_counter=test_counter+1
    dataset_train = df.iloc[sequence[iteration], 1:train_limit + 1]
    dataset_test = df.iloc[sequence[iteration], train_limit + 1:]
    dataset_total = pd.concat((dataset_train, dataset_test), axis=0)

    # print("database shape",dataset_total.shape)
    # print("len(dataset_total) ",len(dataset_total),"len(dataset_test)",len(dataset_test))
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - prev_val:].values
    inputs = inputs.reshape(-1, 1)

    inputs = sc.fit_transform(inputs)
    X_test = []
    for i in range(prev_val, dataset.shape[1] - (train_limit - prev_val)):
        X_test.append(inputs[i - prev_val:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print(X_test.shape)

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    from sklearn.metrics import mean_squared_error

    mean_sq_err = mean_squared_error(dataset_test.values, predicted_stock_price)

    print("iteration ", iteration, "with mean squared error ", mean_sq_err)
    # Visualising the results

    if mean_sq_err<100:
        if (rand % 3 == 0 and plot_counts < max_plots):
            plot_counts = plot_counts+1
            plt.plot(Y_test_set, dataset_test.values, color="red", label="Real Value")
            plt.plot(Y_test_set, predicted_stock_price, color="blue", label="Predicted Value")
            plt.xticks(np.arange(train_limit, len(dataset_total) + 10, 50))
            plt.title(df.iloc[sequence[iteration], 0])
            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.legend()
            plt.show()

        rand = random.randint(0, 10)#an o arithmos einai mod3=0 tote ektypwnw thn grafikh

value = input("Type 1 to train and test each stock separately or 0 to exit!\n")
value = int(value)

if value == 1:
    print("Now training and testing each stock separately")
    rand = 3
    plot_counts=0

    for iteration in range(0, N):
        # arr=dataset[sequence[iteration]]
        # arr=repeat_arr[iteration]
        arr = dataset[sequence[iteration]]
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

        drop_num = 0.2
        unit_num = 50

        model2 = Sequential()
        # Adding the first LSTM layer and some Dropout regularisation
        model2.add(LSTM(units=unit_num, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model2.add(Dropout(drop_num))
        for i in range(0, 1):
            # Adding a second LSTM layer and some Dropout regularisation
            model2.add(LSTM(units=unit_num, return_sequences=True))
            model2.add(Dropout(drop_num))
        # Adding a fourth LSTM layer and some Dropout regularisation
        model2.add(LSTM(units=unit_num))
        model2.add(Dropout(drop_num + 0.1))
        # Adding the output layer
        model2.add(Dense(units=1))
        # Compiling the RNN
        model2.compile(optimizer='adam', loss='mean_squared_error')

        # Fitting the RNN to the Training set
        model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1, shuffle=False)
        # model.fit(X_train, Y_train, epochs = 100, batch_size = 32)

        dataset_train = df.iloc[sequence[iteration], 1:train_limit + 1]
        dataset_test = df.iloc[sequence[iteration], train_limit + 1:]
        dataset_total = pd.concat((dataset_train, dataset_test), axis=0)

        # print("database shape",dataset_total.shape)
        # print("len(dataset_total) ",len(dataset_total),"len(dataset_test)",len(dataset_test))
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - prev_val:].values
        inputs = inputs.reshape(-1, 1)

        inputs = sc.transform(inputs)
        X_test = []
        for i in range(prev_val, dataset.shape[1] - (train_limit - prev_val)):
            X_test.append(inputs[i - prev_val:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        print(X_test.shape)

        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        from sklearn.metrics import mean_squared_error

        mean_sq_err = mean_squared_error(dataset_test.values, predicted_stock_price)

        # Visualising the results
        print("iteration ", iteration, "with mean squared error ", mean_sq_err)

        if mean_sq_err < 100:
            if (rand % 3 == 0 and plot_counts < max_plots):
                plt.plot(Y_test_set, dataset_test.values, color="red", label="Real Value")
                plt.plot(Y_test_set, predicted_stock_price, color="blue", label="Predicted Value")
                plt.xticks(np.arange(train_limit, len(dataset_total) + 10, 50))
                plt.title(df.iloc[iteration, 0])
                plt.xlabel("Time")
                plt.ylabel("Values")
                plt.legend()
                plt.show()
            rand = random.randint(0, 10)

print("Finished! :)")
