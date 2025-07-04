#%%
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

saved_model_dir = r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Modelos\modelo_400_epochs_fast_rtl"
model = load_model(os.path.join(saved_model_dir, 'model.h5'))

data_path = r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Datasets\dataset_test_200eventos_g-n.pkl"
data = pd.read_pickle(data_path)

predicts_out1 = []
predicts_out2 = []

delay_time = []

for i in range(len(data.TraceFinal)):
    trace = data.TraceFinal[i]

    start_time = time.time()
    result, result2 = model.predict(trace.reshape(1,232,1))
    stop_time = time.time()

    predicts_out1.append(result)
    predicts_out2.append(result2)

    delay_time.append(stop_time - start_time)

predicts_out1 = np.array(predicts_out1)
predicts_out2 = np.array(predicts_out2)
delay_time = np.array(delay_time)

np.save(r'C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Resultados\Keras\predicciones_kerasout1_g-n.npy', predicts_out1)
np.save(r'C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Resultados\Keras\predicciones_kerasout2_g-n.npy', predicts_out2)
np.save(r'C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Resultados\Keras\delay_g-n.npy', delay_time)
# %%
