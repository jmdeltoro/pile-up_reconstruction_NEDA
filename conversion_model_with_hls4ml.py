from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
import tensorflow as tf
co = {}
_add_supported_quantized_objects(co)
import os

os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']


#%% conversion a h.5
saved_model_dir = "./model"

# Cargar el modelo con TensorFlow Keras
model = tf.keras.models.load_model(saved_model_dir, compile=True)

# Guardarlo en formato .h5
model.save("model.h5", save_format="h5")

model = load_model('model/model.h5')
# %% cambio el modelo para que tenga una sola salida
from tensorflow.keras.models import Model
modelo1out = Model(inputs=model.input, outputs=model.output[0])

#%%

config = hls4ml.utils.config_from_keras_model(modelo1out, granularity='name')

hls_model = hls4ml.converters.convert_from_keras_model(modelo1out,
                                                       hls_config=config,
                                                       output_dir='hls4ml_prj_pynq-z2',
                                                       backend='VivadoAccelerator',
                                                       board='pynq-z2')

hls_model.build(bitfile=True)