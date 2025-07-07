#%%
import numpy as np
import os
import pandas as pd

path_dataset = r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Datasets"
path_resultados = r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Resultados"

#%% Cargar valores reales
data_gn = pd.read_pickle(os.path.join(path_dataset, "1000_eventos_g_n.pickle"))
data_ng = pd.read_pickle(os.path.join(path_dataset, "1000_eventos_g_n.pickle"))

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
        data = np.load(os.path.join(path_resultados, f'hls4ml/resultados{i}_modelo_400_epochs_fast_rtl_{j}.npy'))
        resultados_hls4ml_fastrtl.append(data)
        delay = np.load(os.path.join(path_resultados, f'hls4ml/tiempos_modelo_400_epochs_fast_rtl_{j}.npy'))
        delay_hls4ml_fastrtl.append(delay)

        data = np.load(os.path.join(path_resultados, f'hls4ml/resultados{i}_modelo_10000_epochs_new_structure_modification_{j}.npy'))
        resultados_hls4ml_new_struc.append(data)
        delay = np.load(os.path.join(path_resultados, f'hls4ml/tiempos_modelo_10000_epochs_new_structure_modification_{j}.npy'))
        delay_hls4ml_new_struct.append(delay)

        data = np.load(os.path.join(path_resultados, f'hls4ml/resultados{i}_modelo_400_epochs_mod3_{j}.npy'))
        resultados_hls4ml_mod3.append(data)
        delay = np.load(os.path.join(path_resultados, f'hls4ml/tiempos_modelo_400_epochs_mod3_{j}.npy'))
        delay_hls4ml_mod3.append(delay)

#%%Plotear resultados
import matplotlib.pyplot as plt

# Selecciona el índice del evento a mostrar
idx = 300

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(y_true1[idx], label='True Trace1')
plt.plot(preds_keras_out1[idx].reshape(232,1), label='Predicted Trace1')
plt.title('Salida 1')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_true2[idx], label='True Trace2')
plt.plot(preds_keras_out2[idx].reshape(232,1), label='Predicted Trace2')
plt.title('Salida 2')
plt.legend()

plt.tight_layout()
plt.show()

#%% Calcular MSE para cada salida
mse_keras1 = np.mean((preds_keras_out1.squeeze() - y_true1)**2)
mse_keras2 = np.mean((preds_keras_out2.squeeze() - y_true2)**2)

#%% Calcular media de tiempo de predicción
mean_delay_keras = np.mean(dalay_keras)

#%% Imprimir resultados
print(f"MSE salida 1 (Keras): {mse_keras1}")
print(f"MSE salida 2 (Keras): {mse_keras2}")
print(f"Tiempo medio de predicción (Keras): {mean_delay_keras} s")
# %%
