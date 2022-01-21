import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import *
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import random
import sys

#$python train_reduce.py –d <dataset> -n <number of time series selected> <OPTIONAL: name of model to save>

#from google.colab import drive

#sinartisi proetimasias dedomenwn gia to model
def create_dataset(data, sc, windows=10, ):
    size = data.shape[0]
    windows_lst, resultX_lst, resultY_lst = [], [], []
    for i in range(0, size):
        v = data[i]
        windows_lst.append(v)
        if len(windows_lst) == windows:
            training_set_scaled = sc.fit_transform(windows_lst)
            resultX_lst.append(training_set_scaled)
            resultY_lst.append(windows_lst)
            windows_lst = []
    return np.array(resultX_lst), np.array(resultY_lst)

#sinartisi ektipwshs history
def plot_history(history):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")

#sinartisi ektipwsis dataset prin kai meta to autoencoder
def plot_examples(stock_input, stock_decoded):
    n = stock_input.shape[1]
    plt.figure(figsize=(20, 4))
    test_samples = train_size
    for i, idx in enumerate(list(np.arange(0, stock_input.shape[0], n))):
        # print("i=",i,"  idx=",idx)

        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)
    plt.show()


#drive.mount('/content/drive')
# input_path = "/content/drive/My Drive/project_3/input.csv"
#path = "/content/drive/My Drive/project_3/"
# file_name="nasd_input.csv"

path=""
file_name = "nasdaq2007_17.csv"
N = 359
if len(sys.argv) < 5:
    print("Wrong no of arguments!")
    exit(1)

N=int(sys.argv[4])
file_name=sys.argv[2]

encoder_name="encoder_1.h5"
if len(sys.argv) > 5:
    encoder_name=sys.argv[5]

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
test_size = (dataset.shape[1] - train_size)


sequence = list(range(0, df.shape[0]))#lista me ola rows tu dataset se aritmus
random.shuffle(sequence)#anakatevume tin lista


N_train = N  # gia na kanei train mono me to dataset pu tha kanei predict
# N_train=df.shape[0]#gia na kanei train me olo to dataset
test_arr = []

train_count=0

debug_flag=0 #gia na emfanizei grafikes parastaseis gia debug, dhladh gia na vlepoume an oi times pou allazoume exoyn kalytera apotelesmata h oxi.
#debug_flag=1#gia na mhn emfanizei grafikes parastaseis


for iteration in range(0, N_train):
    train_count=train_count+1
    arr = dataset[sequence[iteration]]

    encoding_dim = 3
    windows = 10  # encoding_dim*4

    train_df = arr[:train_size].reshape(-1, 1)
    sc = MinMaxScaler(feature_range=(0, 1))
    trainX, trainY = create_dataset(train_df, sc, windows)
    print(trainX.shape)
    print(trainY.shape)

    test_df = arr[train_size:].reshape(-1, 1)
    sc = MinMaxScaler(feature_range=(0, 1))
    testX, testY = create_dataset(test_df, sc, windows)

    if iteration == 0:#ftiaxnume to montelo mono stin prwti epanalipsi
        input_window = Input(shape=(windows, 1))
        x = Conv1D(16, 3, activation="relu", padding="same")(input_window)  # 10 dims
        # x = BatchNormalization()(x)
        x = MaxPooling1D(2, padding="same")(x)  # 5 dims
        x = Conv1D(1, 3, activation="relu", padding="same")(x)  # 5 dims
        # x = BatchNormalization()(x)
        encoded = MaxPooling1D(2, padding="same")(x)  # 3 dims

        encoder = Model(input_window, encoded)#encoder edw ta dedomena ine simpiesmena se mikroteri diastasi

        #################kanume compile ton encoder
        encoder.compile(optimizer='adam', loss='binary_crossentropy')
        # 3 dimensions in the encoded layer
        x = Conv1D(1, 3, activation="relu", padding="same")(encoded)  # 3 dims
        # x = BatchNormalization()(x)
        x = UpSampling1D(2)(x)  # 6 dims
        x = Conv1D(16, 2, activation='relu')(x)  # 5 dims
        # x = BatchNormalization()(x)
        x = UpSampling1D(2)(x)  # 10 dims
        decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)  # 10 dims
        autoencoder = Model(input_window, decoded)
        autoencoder.summary()

        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    history = autoencoder.fit(trainX, trainX, epochs=100, batch_size=1024, shuffle=True, validation_data=(testX, testX))

    # encoded_values = encoder.predict(testX)# entoli gia predictions encoded
    # print("encoded_values ",encoded_values.shape)
    testX_predicted = autoencoder.predict(testX)#xrisimopoioume ton autoencoder gia a3iologisi tu montelu

    if debug_flag == 0:
        plot_history(history)
        print("testX shape", testX.shape)
        plot_examples(testX, testX_predicted)
###kanume save ton encoder
encoder.save(path + encoder_name )
print("Finished :) Trained with: ",train_count)