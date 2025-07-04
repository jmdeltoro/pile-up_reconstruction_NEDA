from tensorflow.keras.models import load_model
import hls4ml
import pandas as pd
import numpy as np
import os

#%% Cargar el modelo guardado
model = load_model(os.path.join(r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Modelos\modelo_400_epochs_fast_rtl", 'model.h5'))

from tensorflow.keras.models import Model
modelo1out = Model(inputs=model.input, outputs=model.output[0])

config = hls4ml.utils.config_from_keras_model(modelo1out, granularity='name', default_precision='fixed<16,6>')

config['Model']['ReuseFactor'] = 100

config['Model']['Strategy'] = 'Resource'

config['Model']['BramFactor'] = 1000000000

config['ClockPeriod'] = 10

config['LayerName']['conv1d_7']['Strategy'] = 'Latency'
config['LayerName']['out1']['Strategy'] = 'Latency'

hls_model = hls4ml.converters.convert_from_keras_model(modelo1out,
                                                       hls_config=config,
                                                       output_dir='hls4ml_test_fastrtl',
                                                       backend='Vivado',
                                                       part='xczu7ev-ffvc1156-2-e',
                                                       io_type='io_stream',
                                                       )

hls_model.compile()

# Cargar el dataset en formato pickle
data_path = r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Resultados\reconsturccion_g-n_PC.pkl"
data = pd.read_pickle(data_path)

# Extraer una traza (por ejemplo, la columna 'TraceFinal' de la primera fila)
# Ajusta el nombre de la columna y el índice según tus necesidades
X_test = np.array([data.TraceFinal[0]])

y_hls = hls_model.predict(np.ascontiguousarray(X_test))

print(y_hls)