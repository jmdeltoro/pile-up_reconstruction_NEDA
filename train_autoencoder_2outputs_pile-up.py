# Code developed by Jose Manuel Deltoro Berrio
# University of Valencia - Escuela Técnica Superior d'Enginyeria (ETSE)
# Python Version: 3.9.6
# Tensorflow Version: 2.6.0
# Tensorflow Keras Version: 2.6.0

#%%
from __future__ import absolute_import, division, print_function
from sklearn.preprocessing import normalize, MinMaxScaler
import matplotlib as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pk
import tables
import datetime

import time
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow import keras
import keras.api._v2.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Conv1D
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau, CSVLogger, ModelCheckpoint, EarlyStopping
print(tf.__version__)

from scipy.io import readsav
from scipy.signal import deconvolve
from scipy.ndimage import shift

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from matplotlib import colors
import matplotlib
import multiprocessing

from keras_flops import get_flops
from IPython.display import Image 
from keras.models import load_model


def build_conv_model2_2out():
    #build the NN needed for the problem accomodating a single
    #convolutional layer and two densely connected hidden layer
    kSzConv1D = 3
    waveNumd  = 232
    y1 = layers.Input(shape=[waveNumd,1])
    y = layers.Conv1D(5, (kSzConv1D, ), activation='relu')(y1) #leackyrelu??  poner mas canales??? mas conexxiones???
    y = layers.MaxPooling1D(2)(y)
    y = layers.Conv1D(10, (kSzConv1D, ), activation='relu')(y)
    y = layers.MaxPooling1D(2)(y)
    y = layers.Conv1D(20, (kSzConv1D, ), activation='relu')(y)
    y_div = layers.MaxPooling1D(2)(y)
    #y = layers.Dense(100,activation='relu')(y)
    y = layers.UpSampling1D()(y_div)
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


    z = layers.UpSampling1D()(y_div)
    z = layers.Conv1D(20, (kSzConv1D, ), activation='relu')(z)
    #y = layers.Dropout(0.2)(y)
    z = layers.UpSampling1D()(z)
    z = layers.Conv1D(10, (kSzConv1D, ), activation='relu')(z)
    z = layers.UpSampling1D()(z)
    z = layers.Conv1D(5, (kSzConv1D, ), activation='relu')(z)
    #layers.GlobalAveragePooling1D(),
    #y = layers.Flatten()(y)
    #y = layers.Dropout(0.5)(y)
    z = layers.Flatten()(z)
    z = layers.Dense(waveNumd, activation=tf.nn.relu)(z)
    z2 = layers.Add()([z, y1[:,  :, 0]])
    z = layers.Dense(waveNumd, activation=tf.nn.relu)(z2)

    
    x1 = layers.Dense(waveNumd, activation='linear', name='out1')(y2)
    x2 = layers.Dense(waveNumd, activation='linear', name='out2')(z2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9,
            beta_2=0.999, epsilon=0.1, decay=0.1, amsgrad=False) 
    model = keras.models.Model(inputs=y1, outputs=(x1,x2))
    model.compile(loss='mean_squared_error', optimizer=optimizer, 
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

#%%
#Data for training  --------   gamma-neutron and neutron-gamma
data_path = "C:/Users/jmdeltoro/data_pile-up/"

path = (data_path  + 'result_PSA_orig/marcin/all_delays/g_n/df_gn_alldelays.pickle')
data1 = pd.read_pickle(path)

path = (data_path  + 'result_PSA_orig/marcin/all_delays/n_g/df_ng_alldelays.pickle')
data2 = pd.read_pickle(path)


#%%
# Unión de datasets y desorden de señales

df_train = pd.concat([data1, data2])
data = df_train.sample(frac=1).reset_index(drop=True)
data.info(verbose=True)
print(data)

### Formateo de datos para utilizarlos para el entrenamiento de la red
trace1=np.array([x for x in data['Trace1']]).reshape(-1,232,1).astype('float32')
trace2=np.array([x for x in data['Trace2']]).reshape(-1,232,1).astype('float32')
traceFinal=np.array([x for x in data['TraceFinal']]).reshape(-1,232,1).astype('float32')


#%%
# Separación de train, validation
train = 0.7
val = 0.3

assert train+val==1.0

trace1_train = trace1[:int(len(trace1)*train)]
trace1_val = trace1[int(len(trace1)*train):int(len(trace1)*train)+int(len(trace1)*val)]
trace1_test = trace1[int(len(trace1)*train)+int(len(trace1)*val):]

trace2_train = trace2[:int(len(trace2)*train)]
trace2_val = trace2[int(len(trace2)*train):int(len(trace2)*train)+int(len(trace2)*val)]
trace2_test = trace2[int(len(trace2)*train)+int(len(trace2)*val):]

traceFinal_train = traceFinal[:int(len(traceFinal)*train)]
traceFinal_val = traceFinal[int(len(traceFinal)*train):int(len(traceFinal)*train)+int(len(traceFinal)*val)]
traceFinal_test = traceFinal[int(len(traceFinal)*train)+int(len(traceFinal)*val):]


#%%
data_path = "C:/Users/jmdeltoro/data_pile-up/train_data/"
path_model = "C:/Users/jmdeltoro/pile-up-neda/models/model_out2_noNumexo_g_n_nb_2out_3to50delays_30000epochs_good_distribution_filtered_marcin"

loading_model = True

if loading_model :
    import keras.losses
    keras.losses.customLoss = customLoss
    model = tf.keras.models.load_model(path_model, custom_objects={'Loss':customLoss})
    print ("loading trained model = " + path_model)
else:
    model = build_conv_model2_2out()

model.summary()
tf.keras.utils.plot_model(model, to_file="model_diagram_out2_noNumexo_g_n_nb_2out_3to50delays_30000epochs_good_distribution_filtered_marcin.png", show_shapes=True)
Image('model_diagram_out2_noNumexo_g_n_nb_2out_3to50delays_30000epochs_good_distribution_filtered_marcin.png')

flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(flops)

#%%
EPOCHS = 10000
#EPOCHS = 1
optimizer = 'Adam'
#optimizer = 'SGD'

if optimizer == 'Adam':
    model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9,
              beta_2=0.999, epsilon=0.1, decay=0.1, amsgrad=False)

checkpoint1 = ModelCheckpoint(path_model, monitor="val_out1_mean_absolute_error", verbose=1, save_best_only=True, 
                             mode='min', save_freq="epoch")
early_stopping1 = EarlyStopping(monitor="val_out1_mean_absolute_error", patience=10, mode='min')

checkpoint2 = ModelCheckpoint(path_model, monitor="val_out2_mean_absolute_error", verbose=1, save_best_only=True, 
                             mode='min', save_freq="epoch")
early_stopping2 = EarlyStopping(monitor="val_out2_mean_absolute_error", patience=10, mode='min')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit( y = [trace1_train,trace2_train],
                     x = traceFinal_train,
                     callbacks=[early_stopping1, checkpoint1, early_stopping2, checkpoint2, tensorboard_callback],
                     epochs=EPOCHS, validation_split=0.20, verbose=1, shuffle=True,
                     batch_size=1024)

with open('model_out2_noNumexo_g_n_nb_2out_3to50delays_30000epochs_good_distribution2_filtered_marcin.pk', 'wb') as fd:
    pk.dump(history.history,fd)


fig = plt.figure() 
plt.plot(history.history['loss'][1:])
plt.plot(history.history['val_loss'][1:])
plt.yscale('log')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig('model_out2_noNumexo_g_n_nb_2out_3to50delays_30000epochs_good_distribution2_filtered_marcin.png')
