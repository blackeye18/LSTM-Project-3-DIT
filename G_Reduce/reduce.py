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
import csv
import sys
#from google.colab import drive

#drive.mount('/content/drive')
#path = "/content/drive/My Drive/project_3/"
#sinartisi proetimasias dedomenwn gia to predict
def create_dataset(data,sc,windows=10,):
  size=data.shape[0]
  windows_lst,resultX_lst,resultY_lst=[],[],[]
  for i in range(0,size):
    v=data[i]
    windows_lst.append(v)
    if len(windows_lst) == windows:
      training_set_scaled = sc.fit_transform(windows_lst)
      resultX_lst.append(training_set_scaled)
      resultY_lst.append(windows_lst)
      windows_lst=[]
  return np.array(resultX_lst),np.array(resultY_lst)
#sinartisi ektipwmatos simpiesmenwn dedomenwn se csv arxeio
def print_encoded(df,encoder_model,output_name):
  dataset=df.iloc[:,:].values
  iteration=0
  f = open(path+output_name, 'w')
  writer = csv.writer(f,delimiter='\t')

  for arr in dataset:
    """
    if iteration > 5:
      break
    """
    name=arr[0]
    data=arr[1:]
    encoding_dim = 3
    ##########alla3a ta windows se 10
    windows=10#encoding_dim*4

    data=data.reshape(-1,1)
    sc = MinMaxScaler(feature_range = (0, 1))
    dataX,dataY=create_dataset(data,sc,windows)

    encoded_data=encoder_model.predict(dataX)

    row=[]
    row.append(name)#prwta ine to onoma
    for x in encoded_data:
      for y in x:
        #print("y:",str(y).lstrip('[').rstrip(']'))
        row.append(str(y).lstrip('[').rstrip(']'))
    #for itemn in row:
      #writer.writerow(itemn)
    #new_lst = (','.join(row))
    #print(row)
    writer.writerow(row)#ektipwnume tin ka8e grammi sto arxeio csv
    iteration+=1

  print(output_name," length of row after",len(row),"number of rows after",iteration)
  f.close()

#$python reduce.py –d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>

path=""

dataset="nasdaq2007_17.csv"#onoma dataset prin to encoding
queryset="nasdaq2007_17.csv"#onoma queryset prin to encoding
output_dataset_file="output_dataset_file.csv"#onoma dataset meta to encoding
output_query_file="output_query_file.csv"#onoma queryset meta to encoding

#orismata aptin grammi entolwn
if len(sys.argv) != 9:
    print("Wrong no of arguments!")
    exit(1)

dataset=sys.argv[2]
queryset=sys.argv[4]
output_dataset_file=sys.argv[6]
output_query_file=sys.argv[8]

data_df=pd.read_csv(path+dataset,'\t',header=None)
query_df=pd.read_csv(path+queryset,'\t',header=None)
print("shape of dataset before",data_df.shape)
print("shape of queryset before",query_df.shape)
########kanw load ton encoder
encoder_model=keras.models.load_model(path+"encoder1.h5")#kanume load ton encoder
#tipwnume ta encoded dedomena sta arxeia
print_encoded(data_df,encoder_model,output_dataset_file)
print_encoded(query_df,encoder_model,output_query_file)

