#%%
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

path_resultados = r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Resultados\Keras"



data_gn = pd.read_pickle(r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Datasets\1000_eventos_g_n.pickle")
data_ng = pd.read_pickle(r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Datasets\1000_eventos_n_g.pickle")
data = {'gn': data_gn, 'ng': data_ng}


for k in ('modelo_400_epochs_fast_rtl', 'modelo_400_epochs_mod3', 'modelo_10000_epochs_new_structure_modification'):
    saved_model_dir = r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Modelos"
    model = load_model(os.path.join(saved_model_dir, k, 'model.h5'))

    predicts_1 = []
    predicts_2 = []

    delay = []

    for j in ('ng', 'gn'):
        for i in range(len(data[j].TraceFinal)):
            trace = data[j].TraceFinal[i]

            start_time = time.time()
            result, result2 = model.predict(trace.reshape(1,232,1))
            stop_time = time.time()

            predicts_1.append(result)
            predicts_2.append(result2)

            delay.append(stop_time - start_time)

        predicts_out1 = np.array(predicts_1)
        predicts_out2 = np.array(predicts_2)
        delay_time = np.array(delay)

        np.save(os.path.join(path_resultados, f'resultados1_{k}_{j}.npy'), predicts_out1)
        np.save(os.path.join(path_resultados, f'resultados2_{k}_{j}.npy'), predicts_out2)
        np.save(os.path.join(path_resultados, f'tiempos_{k}_{j}.npy'), delay_time)
# %%
