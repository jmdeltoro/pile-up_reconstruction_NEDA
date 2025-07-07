from tensorflow.keras.models import load_model
import hls4ml
import pandas as pd
import numpy as np
import os

#%% Cargar el modelo guardado
model = load_model(os.path.join('C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Modelos\modelo_10000_epochs_new_structure_modification', 'model.h5'))

from tensorflow.keras.models import Model
modelo1out = Model(inputs=model.input, outputs=model.output[0])

config = hls4ml.utils.config_from_keras_model(modelo1out, granularity='name', default_precision='fixed<16,6>')

config['Model']['ReuseFactor'] = 100

config['Model']['Strategy'] = 'Resource'

config['Model']['BramFactor'] = 1000000000

config['ClockPeriod'] = 10

'''config['LayerName']['conv1d_7']['Strategy'] = 'Latency' #para la salidda 2 es conv1d_12
config['LayerName']['out1']['Strategy'] = 'Latency'  #para la salida 2 es out2'''

hls_model = hls4ml.converters.convert_from_keras_model(modelo1out,
                                                       hls_config=config,
                                                       output_dir='hls4ml_test_original_predicciones',
                                                       backend='Vivado',
                                                       part='xczu7ev-ffvc1156-2-e',
                                                       io_type='io_stream',
                                                       )

hls_model.compile()

# Cargar el dataset en formato pickle
data_path = "pile-up_reconstruction_NEDA/HLS4ML/Datasets/dataset_test_200eventos_g-n.pkl"
data = pd.read_pickle(data_path)

# Extraer una traza (por ejemplo, la columna 'TraceFinal' de la primera fila)
# Ajusta el nombre de la columna y el índice según tus necesidades
all_preds = []
for idx, trace in enumerate(data.TraceFinal):
    X_test = np.array([trace])
    y_hls = hls_model.predict(np.ascontiguousarray(X_test.astype(float)))
    all_preds.append(y_hls)

all_preds = np.array(all_preds)
np.save('pile-up_reconstruction_NEDA/HLS4ML/Resultados/hls4ml/predicciones_hls4mlout1_n-g.npy', all_preds)