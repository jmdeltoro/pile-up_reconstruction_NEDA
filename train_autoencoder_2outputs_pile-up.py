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
#from tensorflow.keras.layers import Dense, Activation, Conv1D
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau, CSVLogger, ModelCheckpoint, EarlyStopping
print(tf.__version__)

from scipy.io import readsav
from scipy.signal import deconvolve
#from scipy.ndimage.interpolation import shift

from scipy.ndimage import shift

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from matplotlib import colors
import matplotlib
import multiprocessing


from keras_flops import get_flops
from IPython.display import Image 
from keras.models import load_model


weights_peak = np.full(52, 0.3)

weights_tail = np.full(180, 1)

weights = np.concatenate((weights_peak, weights_tail))
weights = weights**1


print (weights.shape)



#%%
def convolve(signal, filterProfile): 

    return np.convolve(signal, filterProfile)

def customLoss(yTrue, yPred):
    
    #weights1 = K.constant(weights)
    weights1 = weights
    # baseline = np.mean()
    return K.mean(K.square(yTrue - yPred)) * weights1
    #return K.mean(K.square(yTrue - yPred))

#def customLoss_MP(yTrue, yPred):
#   
#    weights1 = K.constant(weights_MP)
#    return K.mean(K.square(yTrue - yPred) * weights1)

def customLoss1(yTrue, yPred):

    weights11 = K.constant(weights[0:40])
    return K.mean(K.square(yTrue - yPred) * weights11)

#%%
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
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
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
  #optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, nesterov=True)
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9,# learning_rate= 1e-3
            beta_2=0.999, epsilon=0.1, decay=0.1, amsgrad=False) #1e-8
    model = keras.models.Model(inputs=y1, outputs=x)
    model.compile(loss=customLoss, optimizer=optimizer, # 'mean_squared_error'
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


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

    #layers.GlobalAveragePooling1D(),
    #layers.Dense(20, activation='sigmoid'),    
    #layers.Dense(10, activation='sigmoid'),
    
    x1 = layers.Dense(waveNumd, activation='linear', name='out1')(y2)
    x2 = layers.Dense(waveNumd, activation='linear', name='out2')(z2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9,# learning_rate= 1e-3
            beta_2=0.999, epsilon=0.1, decay=0.1, amsgrad=False) #1e-8
    model = keras.models.Model(inputs=y1, outputs=(x1,x2))
    model.compile(loss=customLoss, optimizer=optimizer, # 'mean_squared_error'
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

#%%
def cutting(df_final_con):
    last_event = 2000
    df_cortado = pd.DataFrame()
    for i in range(3, 51):
        df = df_final_con.loc[df_final_con.loc[:, 'Delay'] == i]
        df = df.reset_index(drop=True)
        df = df.iloc[0:last_event]
        df_cortado = pd.concat([df_cortado, df], ignore_index=1)
    return df_cortado




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

#%%



data_path = "C:/Users/jmdeltoro/data_pile-up/train_data/"

#path_model = "/home/mlphd/pile-up-neda/models/model_out2_noNumexo_g_n_nb_2out_alldelays_50000epochs"
path_model = "C:/Users/jmdeltoro/pile-up-neda/models/model_out2_noNumexo_g_n_nb_2out_3to50delays_30000epochs_good_distribution_filtered_marcin"

loading_model = True

if loading_model :
    import keras.losses
    keras.losses.customLoss = customLoss
    model = tf.keras.models.load_model(path_model, custom_objects={'Loss':customLoss})
    print ("loading trained model = " + path_model)
else:
    model = build_conv_model2_2out()
    #model = build_conv_model2()
#model = build_simple_model1()

#model = build_dense_model()

model.summary()
tf.keras.utils.plot_model(model, to_file="model_diagram_out2_noNumexo_g_n_nb_2out_3to50delays_30000epochs_good_distribution_filtered_marcin.png", show_shapes=True)
Image('model_diagram_out2_noNumexo_g_n_nb_2out_3to50delays_30000epochs_good_distribution_filtered_marcin.png')

flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(flops)

#%%
EPOCHS = 40000
#EPOCHS = 1
optimizer = 'Adam'
#optimizer = 'SGD'

if optimizer == 'Adam':
    model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9,
              beta_2=0.999, epsilon=0.1, decay=0.1, amsgrad=False)
    #model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
    #      beta_2=0.999, epsilon=0.1, decay=1.0, amsgrad=False)

#if optimizer == 'SGD': 
#    model.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, nesterov=True)

#checkpoint1 = ModelCheckpoint(path_model, monitor="val_out1_mean_absolute_error", verbose=1, save_best_only=True, 
#                             mode='min', save_freq="epoch")

checkpoint1 = ModelCheckpoint(path_model, monitor="val_out1_mean_absolute_error", verbose=1, save_best_only=True, 
                             mode='min', save_freq="epoch")


#early_stopping1 = EarlyStopping(monitor="val_out1_mean_absolute_error", patience=50, mode='min')

early_stopping1 = EarlyStopping(monitor="val_out1_mean_absolute_error", patience=10, mode='min')



checkpoint2 = ModelCheckpoint(path_model, monitor="val_out2_mean_absolute_error", verbose=1, save_best_only=True, 
                             mode='min', save_freq="epoch")

early_stopping2 = EarlyStopping(monitor="val_out2_mean_absolute_error", patience=10, mode='min')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


history = model.fit( y = [trace1_train,trace2_train],
                     x = traceFinal_train,
                     #validation_data=([trace1_val, trace2_val],traceFinal_val),
                     callbacks=[early_stopping1, checkpoint1, early_stopping2, checkpoint2, tensorboard_callback],
                     epochs=EPOCHS, validation_split=0.20, verbose=1, shuffle=True,
                     batch_size=1024)

#validation_split=0.20,
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
#plt.show()


#%%

import keras.losses
keras.losses.customLoss = customLoss
#path_model = "models/model_conv1_model_20epoch_custom_loss_bigdataset_lr01"
                     
#model = tf.keras.models.load_model(path_model, custom_objects={'Loss':customLoss})

eventNumber = 232

fig = plt.figure()      
plt.xlabel("samples")
plt.ylabel("value")
plt.ylim(0,15000)
plt.plot(np.array(range(232)),traceFinal[eventNumber].reshape(232))#, s=1)
# plt.show()
# fig.savefig('TraceFinal.png')

# fig = plt.figure()      
plt.xlabel("samples")
plt.ylabel("value")
plt.ylim(0,15000)
plt.plot(np.array(range(232)),trace1[eventNumber].reshape(232))#, s=1)
#print (trace1[eventNumber])
# plt.show()
# fig.savefig('Trace1.png')

result, result2 = model.predict(traceFinal[eventNumber].reshape(1,232,1))
# fig = plt.figure()      
plt.xlabel("samples")
plt.ylabel("value")
plt.ylim(0,15000)
plt.plot(np.array(range(232)),result.reshape(232))#, s=1)
plt.legend(['Entrada','Out','Pred'])
#plt.show()
fig.savefig(path_model + '/Comparison_model_out2_noNumexo_g_n_nb_2out_alldelays_3to50_30000epochs_good_distribution_filtered_marcin.png')
# %%
