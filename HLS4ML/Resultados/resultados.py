#%%
import numpy as np
import os
import pandas as pd

path_dataset = r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Datasets"
path_resultados = r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Resultados"

#%% Cargar valores reales
data_gn = pd.read_pickle(os.path.join(path_dataset, "1000_eventos_g_n.pickle"))
data_ng = pd.read_pickle(os.path.join(path_dataset, "1000_eventos_n_g.pickle"))

traza_real = {'gn': data_gn, 'ng': data_ng}

#%% Cargar predicciones RaspberryPi
resultados_rsppi_fastrtl = []
resultados_rsppi_new_struc = []
resultados_rsppi_mod3 = []

delay_rsppi_fastrtl = []
delay_rsppi_new_struct = []
delay_rsppi_mod3 = []

for j in ('gn', 'ng'):
    for i in ('1', '2'):
        data = np.load(os.path.join(path_resultados, f'RaspberryPi/resultados{i}_fastrtl_{j}.npy'))
        resultados_rsppi_fastrtl.append(data)
        delay = np.load(os.path.join(path_resultados, f'RaspberryPi/tiempos_fastrtl_{j}.npy'))
        delay_rsppi_fastrtl.append(delay)

        data = np.load(os.path.join(path_resultados, f'RaspberryPi/resultados{i}_new_struct_{j}.npy'))
        resultados_rsppi_new_struc.append(data)
        delay = np.load(os.path.join(path_resultados, f'RaspberryPi/tiempos_new_struct_{j}.npy'))
        delay_rsppi_new_struct.append(delay)

        data = np.load(os.path.join(path_resultados, f'RaspberryPi/resultados{i}_mod3_{j}.npy'))
        resultados_rsppi_mod3.append(data)
        delay = np.load(os.path.join(path_resultados, f'RaspberryPi/tiempos_mod3_{j}.npy'))
        delay_rsppi_mod3.append(delay)

#%% Cargar predicciones keras
resultados_keras_fastrtl = []
resultados_keras_new_struc = []
resultados_keras_mod3 = []

delay_keras_fastrtl = []
delay_keras_new_struct = []
delay_keras_mod3 = []

for j in ('gn', 'ng'):
    for i in ('1', '2'):
        data = np.load(os.path.join(path_resultados, f'Keras/resultados{i}_modelo_400_epochs_fast_rtl_{j}.npy'))
        resultados_keras_fastrtl.append(data)
        delay = np.load(os.path.join(path_resultados, f'Keras/tiempos_modelo_400_epochs_fast_rtl_{j}.npy'))
        delay_keras_fastrtl.append(delay)

        data = np.load(os.path.join(path_resultados, f'Keras/resultados{i}_modelo_10000_epochs_new_structure_modification_{j}.npy'))
        resultados_keras_new_struc.append(data)
        delay = np.load(os.path.join(path_resultados, f'Keras/tiempos_modelo_10000_epochs_new_structure_modification_{j}.npy'))
        delay_keras_new_struct.append(delay)

        data = np.load(os.path.join(path_resultados, f'Keras/resultados{i}_modelo_400_epochs_mod3_{j}.npy'))
        resultados_keras_mod3.append(data)
        delay = np.load(os.path.join(path_resultados, f'Keras/tiempos_modelo_400_epochs_mod3_{j}.npy'))
        delay_keras_mod3.append(delay)

#%% Cargar predicciones hls4ml

resultados_hls4ml_fastrtl = []
resultados_hls4ml_new_struc = []
resultados_hls4ml_mod3 = []

delay_hls4ml_fastrtl = []
delay_hls4ml_new_struct = []
delay_hls4ml_mod3 = []

for j in ('gn', 'ng'):
    for i in ('1', '2'):
        data = np.load(os.path.join(path_resultados, f'hls4ml/resultados{i}_fastrtl_{j}.npy'))
        resultados_hls4ml_fastrtl.append(data)
        delay = np.zeros((1000, 232)) # La sim de HLS4ML no simula tiempos, se usa 0 para mantener uniformidad
        delay_hls4ml_fastrtl.append(delay)

        data = np.zeros((1000, 232)) # En hls4ml no funciona este modelo, se usa 0 para mantener uniformidad
        resultados_hls4ml_new_struc.append(data)
        delay = np.zeros((1000, 232)) # La sim de HLS4ML no simula tiempos, se usa 0 para mantener uniformidad
        delay_hls4ml_new_struct.append(delay)

        data = np.load(os.path.join(path_resultados, f'hls4ml/resultados{i}_mod3_{j}.npy'))
        resultados_hls4ml_mod3.append(data)
        delay = np.zeros((1000, 232)) # La sim de HLS4ML no simula tiempos, se usa 0 para mantener uniformidad
        delay_hls4ml_mod3.append(delay)

#%% Mostrar una traza de cada modelo/plataforma/dataset y calcular MSE de todos

import matplotlib.pyplot as plt

# Definir modelos, plataformas y datasets
modelos = {
    'Keras': {
        'out1_gn': {
            'FastRTL': resultados_keras_fastrtl[0],
            'Mod3': resultados_keras_mod3[0],
            'NewStruct': resultados_keras_new_struc[0],
        },
        'out2_gn': {
            'FastRTL': resultados_keras_fastrtl[1],
            'Mod3': resultados_keras_mod3[1],
            'NewStruct': resultados_keras_new_struc[1],
        },
        'out1_ng': {
            'FastRTL': resultados_keras_fastrtl[2],
            'Mod3': resultados_keras_mod3[2],
            'NewStruct': resultados_keras_new_struc[2],
        },
        'out2_ng': {
            'FastRTL': resultados_keras_fastrtl[3],
            'Mod3': resultados_keras_mod3[3],
            'NewStruct': resultados_keras_new_struc[3],
        },
        'delay_gn': [delay_keras_fastrtl[0], delay_keras_mod3[0], delay_keras_new_struct[0]],
        'delay_ng': [delay_keras_fastrtl[1], delay_keras_mod3[1], delay_keras_new_struct[1]]
    },
    'RaspberryPi': {
        'out1_gn': {
            'FastRTL': resultados_rsppi_fastrtl[0],
            'Mod3': resultados_rsppi_mod3[0],
            'NewStruct': resultados_rsppi_new_struc[0],
        },
        'out2_gn': {
            'FastRTL': resultados_rsppi_fastrtl[1],
            'Mod3': resultados_rsppi_mod3[1],
            'NewStruct': resultados_rsppi_new_struc[1],
        },
        'out1_ng': {
            'FastRTL': resultados_rsppi_fastrtl[2],
            'Mod3': resultados_rsppi_mod3[2],
            'NewStruct': resultados_rsppi_new_struc[2],
        },
        'out2_ng': {
            'FastRTL': resultados_rsppi_fastrtl[3],
            'Mod3': resultados_rsppi_mod3[3],
            'NewStruct': resultados_rsppi_new_struc[3],
        },
        'delay_gn': [delay_rsppi_fastrtl[0], delay_rsppi_mod3[0], delay_rsppi_new_struct[0]],
        'delay_ng': [delay_rsppi_fastrtl[1], delay_rsppi_mod3[1], delay_rsppi_new_struct[1]]
    },

    'hls4ml': {
        'out1_gn': {
            'FastRTL': resultados_hls4ml_fastrtl[0],
            'Mod3': resultados_hls4ml_mod3[0],
            'NewStruct': resultados_hls4ml_new_struc[0],
        },
        'out2_gn': {
            'FastRTL': resultados_hls4ml_fastrtl[1],
            'Mod3': resultados_hls4ml_mod3[1],
            'NewStruct': resultados_hls4ml_new_struc[1],
        },
        'out1_ng': {
            'FastRTL': resultados_hls4ml_fastrtl[2],
            'Mod3': resultados_hls4ml_mod3[2],
            'NewStruct': resultados_hls4ml_new_struc[2],
        },
        'out2_ng': {
            'FastRTL': resultados_hls4ml_fastrtl[3],
            'Mod3': resultados_hls4ml_mod3[3],
            'NewStruct': resultados_hls4ml_new_struc[3],
        },
        'delay_gn': [delay_hls4ml_fastrtl[0], delay_hls4ml_mod3[0], delay_hls4ml_new_struct[0]],
        'delay_ng': [delay_hls4ml_fastrtl[1], delay_hls4ml_mod3[1], delay_hls4ml_new_struct[1]]
    }
    
}

for plataforma in modelos:
    for key in modelos[plataforma]:
        # Saltar las claves de delay si existen
        if 'delay' in key:
            continue
        for modelo in modelos[plataforma][key]:
            arr = modelos[plataforma][key][modelo]
            print(f"{plataforma} | {key} | {modelo} -> shape: {arr.shape}")
#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Definición de datasets y salidas
datasets = ['gn', 'ng']
salidas = ['out1', 'out2']
nombres_modelos = ['FastRTL', 'Mod3', 'NewStruct']
nombres_plataformas = ['Keras', 'RaspberryPi', 'hls4ml'] #recordar añadir 'hls4ml'

# Cargar valores reales para cada dataset y salida
# Si tienes dos salidas reales distintas, reemplaza el segundo elemento por la columna correspondiente
y_true = {
    'out1_gn': traza_real['gn'].Trace1,
    'out2_gn': traza_real['gn'].Trace2,
    'out1_ng': traza_real['ng'].Trace1,
    'out2_ng': traza_real['ng'].Trace2
}

idx = 300  # Índice del evento a mostrar

for plataforma in nombres_plataformas:
    for d, dataset in enumerate(datasets):
        for s, salida in enumerate(salidas):
            key = f'{salida}_{dataset}'
            for modelo in nombres_modelos:
                preds = modelos[plataforma][key][modelo]
                y = np.stack(y_true[key].values)
                # Plot
                '''
                plt.figure(figsize=(8, 4))
                plt.plot(y[idx], label=f'True {salida} ({dataset})')
                plt.plot(preds[idx].reshape(-1), label=f'{plataforma}-{modelo} {salida} ({dataset})')
                plt.title(f'{plataforma} - {modelo} - {salida} - {dataset}')
                plt.legend()
                plt.tight_layout()
                plt.show()
                '''
                # Metricas:

                mse = mean_squared_error(y, preds.squeeze())
                nrmse = np.sqrt(mse) / (np.max(y) - np.min(y))
                mae = mean_absolute_error(y, preds.squeeze())
                r2 = r2_score(y, preds.squeeze())
               
                print(f'---------------------{plataforma}-{modelo}-{salida}-{dataset}-------------------')
                print(f"MSE {plataforma}-{modelo}-{salida}-{dataset}: {mse}")
                print(f"MAE {plataforma}-{modelo}-{salida}-{dataset}: {mae}")
                print(f"MSE Normalizado {plataforma}-{modelo}-{salida}-{dataset}: {nrmse}")
                print(f"r2 {plataforma}-{modelo}-{salida}-{dataset}: {r2}")
                print("---------------------------------------------------------------------------------")

for plataforma in nombres_plataformas:
    for d, dataset in enumerate(datasets):
        for delay_key in [f'delay_{dataset}']:
            delays = modelos[plataforma][delay_key]
            for i, modelo in enumerate(nombres_modelos):
                delay_arr = delays[i]
                mean_delay = np.mean(delay_arr)

                print(f'---------------------{plataforma}-{modelo}-{salida}-{dataset}-------------------')
                print(f"Media delay {plataforma} | {modelo} | {dataset}: {mean_delay}")
                print("---------------------------------------------------------------------------------")

# %%
