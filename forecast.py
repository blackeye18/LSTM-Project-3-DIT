import sys
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

##from google.colab import drive
#drive.mount('/content/drive')
#input_path = "/content/drive/My Drive/project_3/input.csv"
#path = "/content/drive/My Drive/project_3/"
#file_name="nasdaq2007_17.csv"
path=""

if len(sys.argv)!=3:
  print("Wrong number of arguments! Exiting..")
  exit()
file_name=sys.argv[1]
N=int(sys.argv[2])
print("file_name: ",file_name," N: ",N)

input_path=path+file_name


df=pd.read_csv(input_path,'\t',header=None)
print("Number of rows and columns:", df.shape)
#print(df.head(5))
dataset=df.iloc[:,1:].values
print(dataset.shape)
train_limit=round(0.8*dataset.shape[1])
Y_training_set= np.array(list(range(1,train_limit +1)))
Y_test_set= np.array(list(range(train_limit+1,dataset.shape[1]+1)))
print("Y_training_set size ",Y_training_set.size)
print("Y_test_set size ",Y_test_set.size)
#N=3
import random
sequence=list(range(0,df.shape[0]))
random.shuffle(sequence)

for iteration in range(0,N):
  arr=dataset[sequence[iteration]]
  X_training_set =arr[:train_limit]
  X_test_set = arr[train_limit:]
  print("X_training_set size ",X_training_set.size)
  print("X_test_set size ",X_test_set.size)
  
  training_set=X_training_set
  training_set=training_set.reshape(-1,1)
  print("training_set.shape",training_set.shape)

  # Feature Scaling
  sc = MinMaxScaler(feature_range = (0, 1))
  training_set_scaled = sc.fit_transform(training_set)
  prev_val=round(dataset.shape[1]/10)
  # Creating a data structure with 60 time-steps and 1 output
  X_train = []
  Y_train = []
  for i in range(prev_val, train_limit):
    X_train.append(training_set_scaled[i-prev_val:i,0])
    Y_train.append(training_set_scaled[i,0])
  
  X_train, Y_train = np.array(X_train), np.array(Y_train)
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  
  print("X_train.shape",X_train.shape)
  drop_num=0.2
  unit_num=50
  keras.backend.clear_session()
  model = Sequential()
  #Adding the first LSTM layer and some Dropout regularisation
  model.add(LSTM(units = unit_num, return_sequences = True, input_shape =(X_train.shape[1], 1)))
  model.add(Dropout(drop_num))
  for i in range(0,0):
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = unit_num, return_sequences = True))
    model.add(Dropout(drop_num+0.1))
  """
  # Adding a third LSTM layer and some Dropout regularisation
  model.add(LSTM(units = unit_num, return_sequences = True))
  model.add(Dropout(drop_num))
  """
  # Adding a fourth LSTM layer and some Dropout regularisation
  model.add(LSTM(units = unit_num))
  model.add(Dropout(drop_num+0.1))
  # Adding the output layer
  model.add(Dense(units = 1))

  # Compiling the RNN
  model.compile(optimizer = 'adam', loss = 'mean_squared_error')

  # Fitting the RNN to the Training set
  model.fit(X_train, Y_train, epochs = 60, batch_size =32,verbose=1)
  #model.fit(X_train, Y_train, epochs = 100, batch_size = 32)
  dataset_train = df.iloc[sequence[iteration],1:train_limit+1]
  dataset_test = df.iloc[sequence[iteration],train_limit+1:]

  dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
  
  #print("database shape",dataset_total.shape)
  #print("len(dataset_total) ",len(dataset_total),"len(dataset_test)",len(dataset_test))
  inputs = dataset_total[len(dataset_total) - len(dataset_test) - prev_val:].values
  inputs = inputs.reshape(-1,1)


  inputs = sc.transform(inputs)
  X_test = []
  for i in range(prev_val, dataset.shape[1] - (train_limit-prev_val)):
    X_test.append(inputs[i-prev_val:i, 0])
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  print(X_test.shape)

  predicted_stock_price = model.predict(X_test)
  predicted_stock_price = sc.inverse_transform(predicted_stock_price)


  
  from sklearn.metrics import mean_squared_error
  mean_sq_err=mean_squared_error(dataset_test.values, predicted_stock_price)

  # Visualising the results
  print("iteration ",iteration,"with mean squared error ",mean_sq_err)
  plt.plot(Y_test_set,dataset_test.values, color = "red", label = "Real Value")
  plt.plot(Y_test_set,predicted_stock_price, color = "blue", label = "Predicted Value")
  plt.xticks(np.arange(train_limit,len(dataset_total)+10,50))
  plt.title(df.iloc[iteration,0])
  plt.xlabel("Time")
  plt.ylabel("Values")
  plt.legend()
  plt.show()
  #if iteration==5:
    #break