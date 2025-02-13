##################################################################################################################################################
# -Versiones:                                                                                                                                    #
#    -Python = 3.6.12                                                                                                                            #
#    -Tensorflow = 2.6.0                                                                                                                         #       
##################################################################################################################################################


#%%
from __future__ import absolute_import, division, print_function
from inspect import trace

import pathlib

import os
from trace import Trace

import matplotlib as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pk
import tables
import datetime

import time
from tqdm import tqdm

import tflite_runtime as tf
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(
    model_path = r"C:\Users\rmart\pile-up_reconstruction_NEDA\pile-up_reconstruction_NEDA\test_pinqz2\converted_model.tflite",  #Asegurarse que el path es el correcto para el .tflite
    num_threads=1,
    )
#%%
try:
    interpreter.allocate_tensors()
except:
    pass
    
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
if input_details[0]['dtype'] == np.float32:
    floating_model = True
else: floating_model = False

#Inicialización terminada, empieza la preparación de los datos de entrada

start = time.time()

print("The time used to execute this is given below")

data_path = r"C:\Users\rmart\pile-up_reconstruction_NEDA\pile-up_reconstruction_NEDA\test_pinqz2"
comb = 'g_n'
path = (data_path  + '\dataset_test_200eventos_g-n.pkl') #AQUI EL PATH DE LOS DATOS DE ENTRADA
data = pd.read_pickle(path)
data = data.sample(frac=1, random_state=0).reset_index(drop=True)


#data=pd.concat([data1,data2,data3,data4,data5], ignore_index=1)


trace1=np.array([x for x in data['Trace1']]).reshape(-1,232,1).astype('float32')
trace2=np.array([x for x in data['Trace2']]).reshape(-1,232,1).astype('float32')
traceFinal=np.array([x for x in data['TraceFinal']]).reshape(-1,232,1).astype('float32')

print("data:")
print (data)

#llamamos al interprete de TFlite 



df_results = pd.DataFrame()

df_results = data

df_results["Trace1_result"] = np.nan

df_results["Trace2_result"] = np.nan

print(len(data))

df_results['Trace1_result'] = df_results['Trace1_result'].astype('object')
df_results['Trace2_result'] = df_results['Trace2_result'].astype('object')

print(input_details[0]['shape'])

for i in tqdm(range(0, len(data))):#len(data)
    eventNumber = i
    current_trace = traceFinal[eventNumber]
    #print(current_trace)
    current_trace = np.reshape(current_trace, (1, 232, 1))
    #current_trace = np.roll(current_trace, -6)

    interpreter.set_tensor(input_details[0]['index'], current_trace)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    result2 = interpreter.get_tensor(output_details[1]['index'])

    df_results.at[i, 'Trace1_result'] = result[0]
    df_results.at[i, 'Trace2_result'] = result2[0]
    #plt.plot(result[0])
    #plt.plot(result2[0])
    

#df_results = df_results.iloc[0:i+1]
df_results = df_results.reset_index(drop=True)

#path = r'C:\Users\rmart\Documents\pruebaTFlite' + comb + '\reconsturccion_g-n.pickle' #AQUI ARCHIVO DE SALIDA. HA DE SER .PICKLE

df_results.to_pickle(r'C:\Users\rmart\pile-up_reconstruction_NEDA\pile-up_reconstruction_NEDA\test_pinqz2\reconsturccion_g-n.pickle')

end = time.time()

print(end - start)
print("DONE")
# %%
