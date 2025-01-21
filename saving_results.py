# %%
from __future__ import absolute_import, division, print_function
from inspect import trace

import pathlib

import os
from trace import Trace

# manually specify the GPUs to use
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pk
import tables
import datetime

import time
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Activation, Conv1D
import keras.backend as K
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau, CSVLogger, ModelCheckpoint, EarlyStopping
print(tf.__version__)

from scipy.io import readsav
from scipy.signal import deconvolve
from scipy.ndimage.interpolation import shift



import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from matplotlib import colors
import matplotlib
# matplotlib.use('Agg') #comment for jupiter plotting

#TURBO CHARGE YOUR PYTHON -- Make it parallel
import multiprocessing


weights_peak = np.full(52, 0.3)

weights_tail = np.full(180, 1)

weights = np.concatenate((weights_peak, weights_tail))
weights = weights**1

#print (weights)
#plt.plot(weights)
#plt.show()

print (weights.shape)


import time

start = time.time()

print("The time used to execute this is given below")


#print(weights1)

# %%
def convolve(signal, filterProfile): 

    return np.convolve(signal, filterProfile)

def customLoss(yTrue, yPred):
    
    #weights1 = K.constant(weights)
    #weights1 = weights
    # baseline = np.mean()
    return K.mean(K.square(yTrue - yPred)) #* weights1
    #return K.mean(K.square(yTrue - yPred))

def customLoss_MP(yTrue, yPred):

    weights1 = K.constant(weights_MP)
    return K.mean(K.square(yTrue - yPred) * weights1)

def customLoss1(yTrue, yPred):

    weights11 = K.constant(weights[0:40])
    return K.mean(K.square(yTrue - yPred) * weights11)


def load_spectral_profiles(directory,filename):
    hdul     = fits.open(directory+filename)
    spectrum = hdul[0].data
    print(hdul.header)
    return spectrum 

def build_simple_model1():
  #build the NN needed for the problem accomodating a single
  #convolutional layer and two densely connected hidden layer
  kSzConv1D = 5
  waveNumd  = 232
  model = keras.Sequential([
    layers.Conv1D(50, (3,),
        activation='relu',strides=1,input_shape=[waveNumd,1]),
    layers.Conv1D(40, (3,),
        activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(25, (3,),
        activation='relu'),
    layers.Conv1D(15, (4,),
        activation='relu'),
    layers.MaxPooling1D(2),
    #layers.GlobalAveragePooling1D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(40,activation=tf.nn.relu),
    layers.Dense(40,activation=tf.nn.relu),  
    #layers.GlobalAveragePooling1D(),
    #layers.Dense(20, activation='sigmoid'),    
    #layers.Dense(10, activation='sigmoid'),
    layers.Dense(waveNumd)])
  optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9,
          beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  #optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1,
  #                                          initial_accumulator_value=0.1, epsilon=1e-07)


  model.compile(loss=customLoss,
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

def build_conv_model():
  #build the NN needed for the problem accomodating a single
  #convolutional layer and two densely connected hidden layer
  kSzConv1D = 5
  waveNumd  = 232
  model = keras.Sequential([
    layers.Conv1D(15, (3,),
        activation='relu', strides=1, input_shape=[waveNumd, 1]),
    layers.Conv1D(30, (5,),
        activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(15, (3,),
        activation='relu'),
    layers.Conv1D(25, (3,),
        activation='relu'),
    #layers.MaxPooling1D(2),
    #layers.GlobalAveragePooling1D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    #layers.Dense(30,activation=tf.nn.relu),
    layers.Dense(30,activation=tf.nn.relu),  
    #layers.GlobalAveragePooling1D(),
    #layers.Dense(20, activation='sigmoid'),    
    #layers.Dense(10, activation='sigmoid'),
    layers.Dense(waveNumd)])
  #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
  #        beta_2=0.999, epsilon=0.1, decay=0.0, amsgrad=False)
  #optimizer = tf.keras.optimizers.SGD(lr=0.01, nesterov=True)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, #original
          beta_2=0.999, epsilon=0.1, decay=0.0, amsgrad=False)       
  #optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1,
  #                                          initial_accumulator_value=0.1, epsilon=1e-07)
  model.compile(loss='mean_squared_error',# # customLoss
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

def build_conv_model2():
    #build the NN needed for the problem accomodating a single
    #convolutional layer and two densely connected hidden layer
    kSzConv1D = 3
    waveNumd  = 232
    y1 = layers.Input(shape=[waveNumd,1])
    y = layers.Conv1D(5, (kSzConv1D, ), activation='relu')(y1)
    y = layers.MaxPooling1D(2)(y)
    y = layers.Conv1D(10, (kSzConv1D, ), activation='relu')(y)
    y = layers.MaxPooling1D(2)(y)
    y = layers.Conv1D(20, (kSzConv1D, ), activation='relu')(y)
    y = layers.MaxPooling1D(2)(y)
    #y = layers.Dense(100,activation='relu')(y)
    y = layers.UpSampling1D()(y)
    y = layers.Conv1D(20, (kSzConv1D, ), activation='relu')(y)
    #y = layers.Dropout(0.2)(y)
    y = layers.UpSampling1D()(y)
    y = layers.Conv1D(10, (kSzConv1D, ), activation='relu')(y)
    y = layers.UpSampling1D()(y)
    y = layers.Conv1D(5, (kSzConv1D, ), activation='relu')(y)
    #layers.GlobalAveragePooling1D(),
    #y = layers.Flatten()(y)
    #y = layers.Dropout(0.5)(y)
    y = layers.Flatten()(y)
    y = layers.Dense(waveNumd, activation=tf.nn.relu)(y)
    y2 = layers.Add()([y, y1[:,  :, 0]])
    y = layers.Dense(waveNumd, activation=tf.nn.relu)(y2)

    #layers.GlobalAveragePooling1D(),
    #layers.Dense(20, activation='sigmoid'),    
    #layers.Dense(10, activation='sigmoid'),
    
    x = layers.Dense(waveNumd, activation='linear')(y2)
    optimizer = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9,# lr= 1e-3
            beta_2=0.999, epsilon=0.1, decay=0.1, amsgrad=False) #1e-8
    model = keras.models.Model(inputs=y1, outputs=x)
    model.compile(loss=customLoss, optimizer=optimizer, # 'mean_squared_error'
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

def build_conv_model_MP(waveNumd, ratio):
    '''
    build the NN needed for the problem accomodating a single
    convolutional layer and two densely connected hidden layer
    '''
    kSzConv1D = 3
    # waveNumd  = 40
    
    y1 = layers.Input(shape=[int(waveNumd/ratio),1])
    y = layers.Conv1D(5,  (kSzConv1D, ))(y1)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.MaxPooling1D(2)(y)
    y = layers.Conv1D(10, (kSzConv1D, ))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.MaxPooling1D(2)(y)
    #y = layers.Conv1D(20, (kSzConv1D,), activation='relu')(y)
    #y = layers.MaxPooling1D(2)(y)
    # y = layers.Conv1D(20, (kSzConv1D, ), activation='relu')(y)
    # y = layers.Dense(waveNumd,activation='relu')(y)
    #y = layers.UpSampling1D()(y)
    #y = layers.Conv1D(20,(kSzConv1D,), activation='relu')(y)
    # y = layers.Dropout(0.2)(y)
    # y = layers.Dropout(0.2)(y)
    y = layers.UpSampling1D()(y)
    y = layers.Conv1D(10,(kSzConv1D,))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.UpSampling1D()(y)
    y = layers.Conv1D(5,(kSzConv1D,))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    # layers.GlobalAveragePooling1D(),
    # y = layers.Flatten()(y)
    y = layers.Flatten()(y)
    # y = layers.Dense(waveNumd, activation='sigmoid')(y)
    x = layers.Dense(waveNumd,activation='linear')(y)
    
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9,
            beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model2 = keras.models.Model(inputs=y1, outputs=x)
    model2.compile(loss=customLoss_MP,
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    return model2

def build_dense_model():
    waveNumd  = 100
    numDense1 = 256
    y1 = layers.Input(shape=[waveNumd,1])
    y = layers.Flatten()(y1)
    y = layers.Dense(numDense1,activation='relu')(y)
    #y = layers.Dense(numDense1,activation='relu')(y)
    y = layers.Dense(numDense1,activation='relu')(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(numDense1,activation='relu')(y)
    y = layers.Dense(numDense1,activation='relu')(y)
    #y = layers.Dense(256,activation='relu')(y)
    #y = layers.Flatten()(y)
    y = layers.Dense(waveNumd,activation='sigmoid')(y)
    y = layers.Add()([y,y1[:,:,0]])
    x = layers.Dense(waveNumd,activation='linear')(y)
    
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9,
            beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model = keras.models.Model(inputs=y1, outputs=x)
    model.compile(loss=customLoss,
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    
    return model


def Lorentzian(x0,x,gamma):
    x2  = (x0-x)*(x0-x)
    exp = gamma/(np.pi*(x2+gamma*gamma))
    return exp


def Gaussian(x,mu,sigma):
    return np.exp((-1)*(x-mu)*(x-mu)/(2*sigma*sigma))/np.sqrt(2*np.pi*sigma*sigma)
#%%%
def cutting(df_final_con):
    last_event= 8000
    df_cortado = pd.DataFrame()
    for i in range(3, 50):
        if i == 50:
            break
        df_i = df_final_con.loc[df_final_con.loc[:, 'Delay'] == i]
        df_i = df_i.reset_index(drop=True)
        df_i = df_i.iloc[0:last_event]
        df_cortado = pd.concat([df_cortado, df_i], ignore_index=1)
    return df_cortado


# %%


data_path = "C:/Users/jmdeltoro/data_pile-up/"
comb = 'g_n'
path = (data_path  + 'result_PSA_orig/train_test_th_0_525_0_525/all_delays_new_max/g_n/df_g_n_alldelays.hdf') #AQUI EL PATH DE LOS DATOS DE ENTRADA
data = pd.read_hdf(path)
data = data.sample(frac=1, random_state=0).reset_index(drop=True)


#%%
#data=pd.concat([data1,data2,data3,data4,data5], ignore_index=1)


trace1=np.array([x for x in data['Trace1']]).reshape(-1,232,1).astype('float32')
trace2=np.array([x for x in data['Trace2']]).reshape(-1,232,1).astype('float32')
traceFinal=np.array([x for x in data['TraceFinal']]).reshape(-1,232,1).astype('float32')

# %%
print (data)

# %%

from IPython.display import Image 
from keras.models import load_model


# %%
import keras.losses
keras.losses.customLoss = customLoss

path_model = "C:/Users/jmdeltoro/pile-up-neda/models/model_out2_noNumexo_g_n_nb_2out_3to50delays_30000epochs_good_distribution_filtered_0475_06" # AQUI PATH A LA CARPETA DEL MODELO

model = tf.keras.models.load_model(path_model, custom_objects={'Loss':customLoss})

# %%

df_results = pd.DataFrame()

# %%
df_results = data

# %%
df_results["Trace1_result"] = np.nan


# %%
df_results["Trace2_result"] = np.nan
#%%

print(len(data))

# %%
df_results['Trace1_result'] = df_results['Trace1_result'].astype('object')
df_results['Trace2_result'] = df_results['Trace2_result'].astype('object')
for i in tqdm(range(0, len(data))):#len(data)
    eventNumber = i
    current_trace = traceFinal[eventNumber]
    #current_trace = np.roll(current_trace, -6)
    result, result2 = model.predict(current_trace.reshape(1,232,1))
    df_results.at[i, 'Trace1_result'] = result[0]
    df_results.at[i, 'Trace2_result'] = result2[0]
    #plt.plot(result[0])
    #plt.plot(result2[0])

# %%
#df_results = df_results.iloc[0:i+1]
df_results = df_results.reset_index(drop=True)
#print(df_results)
# %%

path = 'C:/Users/jmdeltoro/data_pile-up/reconstruction/reconstruction_signals_noNUMEXO_ng_model_'+ comb +'_6th_test_dataset_3to40_30000epochs_2outputs_good_distribution_filtered_0525_0525_with_0575_0575_gn_new_max.pickle' #AQUI ARCHIVO DE SALIDA. HA DE SER .PICKLE

df_results.to_pickle(path)

# %%


end = time.time()

print(end - start)
print("DONE")
# %%
