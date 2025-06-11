#%%
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
import tensorflow as tf
co = {}
_add_supported_quantized_objects(co)
import os

saved_model_dir = "/home/rmart/NEDA/modelo_400_epochs_mod3/modelo_400_epochs_mod3"


#%% Cargar el modelo guardado
model = tf.keras.models.load_model(saved_model_dir, compile=True)

model.save(os.path.join(saved_model_dir, 'model.h5'), save_format="h5")

model = load_model(os.path.join(saved_model_dir, 'model.h5'))

#%% cambio el modelo para que tenga una sola salida
n_out = 0 #para seleccionar que salida utilizar

from tensorflow.keras.models import Model
modelo1out = Model(inputs=model.input, outputs=model.output[n_out])

#%%
#os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

config = hls4ml.utils.config_from_keras_model(modelo1out, granularity='name', default_precision='int<16>')

#%%
config['Model']['ReuseFactor'] = 100

config['Model']['Strategy'] = 'Resource'

config['Model']['BramFactor'] = 1000000000

config['ClockPeriod'] = 10
'''
config['LayerName']['dense']['ReuseFactor'] = 234320  #234320
config['LayerName']['dense']['Strategy'] = 'Resource'

config['LayerName']['dense_relu']['ReuseFactor'] = 234320  #234320

config['LayerName']['out1']['ReuseFactor'] = 234320  #234320'''


#%%
hls_model = hls4ml.converters.convert_from_keras_model(modelo1out,
                                                       hls_config=config,
                                                       output_dir='hls4ml_test_modificaciones_mod3',
                                                       backend='Vivado',
                                                       part='xczu7ev-ffvc1156-2-e',
                                                       io_type='io_stream',
                                                       )
hls_model.compile()
#hls4ml.utils.plot_model(hls_model, show_shapes=False, show_precision=False, to_file='modelo_nuevo.png')

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_pickle('/home/rmart/NEDA/pile-up_reconstruction_NEDA/test_pinqz2/1000_eventos_g_n.pickle')
trace = data.Trace1
data = data.TraceFinal

#%%
results = pd.DataFrame()
results['Pile-Up'] = data
results['KERAS'] = np.nan
results['HLS4ML'] = np.nan

results['KERAS'] = results['KERAS'].astype('object')
results['HLS4ML'] = results['HLS4ML'].astype('object')

for i in (906, 909, 972):
    eventNumber = i
    t = np.linspace(0, len(data[i]))
    current_trace = data[eventNumber]
    y_keras = model.predict(current_trace.reshape(1,232,1))
    y_hls = hls_model.predict(current_trace.astype(float))
    results.at[i, 'KERAS'] = y_keras
    results.at[i, 'HLS4ML'] = y_hls

    plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})  # Tipografía legible

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=False, dpi=200)
    fig.suptitle(f'Evento {i}', fontsize=16, fontweight='bold')

    # Primer subplot: señal original y KERAS
    axs[0].plot(data[i], label='Input pile-up signal', color='tab:blue', linewidth=1.5)
    axs[0].plot(y_hls, linestyle='--', label='HLS4ML first pulse reconstruction', color='tab:red', linewidth=2)
    axs[0].plot(y_keras[n_out].reshape(232,), linestyle='--', label='TFLite first pulse reconstruction', color='tab:orange', linewidth=2)
    axs[0].plot(trace[i].reshape(232,), label='Original pulse without pile-up', color='tab:green', linewidth=1)
    axs[0].set_xlabel('Samples', fontsize=16)
    axs[0].set_ylabel('Amplitude', fontsize=16)
    axs[0].set_title('Output TFLite and HLS4ML', fontsize=18)
    axs[0].legend(loc='lower right', fontsize=16)
    axs[0].grid(True, linestyle=':', alpha=0.7)

    plt.show()

#%%

# %%
