#%%
import numpy as np
import os

path_dataset = r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Datasets"
path_keras = r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Resultados"

#%% Cargar valores reales
import pandas as pd
data_gn = pd.read_pickle(os.path.join(path_dataset, "dataset_test_200eventos_g-n.pkl"))
data_ng = pd.read_pickle(os.path.join(path_dataset, "dataset_test_200eventos_n-g.pkl"))

y_true1 = np.concatenate([np.stack(data_gn['Trace1'].values), np.stack(data_ng['Trace1'].values)])
y_true2 = np.concatenate([np.stack(data_gn['Trace2'].values), np.stack(data_ng['Trace2'].values)])

#%% Cargar predicciones y datos reales (ajusta los nombres de archivo según corresponda)

preds_keras_ng_1 = np.load(os.path.join(path_keras, 'Keras/predicciones_kerasout1_n-g.npy'))
preds_keras_ng_2 = np.load(os.path.join(path_keras, 'Keras/predicciones_kerasout2_n-g.npy'))

preds_keras_gn_1 = np.load(os.path.join(path_keras, 'Keras/predicciones_kerasout1_g-n.npy'))
preds_keras_gn_2 = np.load(os.path.join(path_keras, 'Keras/predicciones_kerasout2_g-n.npy'))

preds_keras_out1 = np.concatenate([preds_keras_ng_1, preds_keras_gn_1])
preds_keras_out2 = np.concatenate([preds_keras_ng_2, preds_keras_gn_2])

#%% Cargar delays keras
delay_keras_ng = np.load(os.path.join(path_keras, 'Keras/delay_n-g.npy'))
delay_keras_gn = np.load(os.path.join(path_keras, 'Keras/delay_g-n.npy'))
dalay_keras = np.concatenate([delay_keras_ng, delay_keras_gn])

#%%Plotear resultados
import matplotlib.pyplot as plt

# Selecciona el índice del evento a mostrar
idx = 6

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(preds_keras_ng_2[idx].reshape(232,), label='True Trace1')
plt.plot(preds_keras_gn_2[idx].reshape(232,), label='Predicted Trace1')
plt.title('Salida 1')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_true2[idx], label='True Trace2')
plt.plot(preds_keras_out2[idx].reshape(232,), label='Predicted Trace2')
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