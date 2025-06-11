
#%%
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
import tensorflow as tf
co = {}
_add_supported_quantized_objects(co)
import os

saved_model_dir = "/home/rmart/NEDA/pile-up_reconstruction_NEDA/modelo_10000_epochs_new_structure_modification"


#%% Modelo ejemplo
'''with open('/home/rmart/pile-up_reconstruction_NEDA/KERAS_dense_16x500x500x500x500x500x5.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

loaded_model.load_weights('/home/rmart/pile-up_reconstruction_NEDA/KERAS_dense_16x500x500x500x500x500x5_weights.h5')'''
'''
#%%Modelo 2ª iteración
model = load_model('/home/rmart/NEDA/pile-up_reconstruction_NEDA/modelo_700_epochs_new_structure/model.h5')'''

#%% Modelo 3ª iteración, cambiado con flatten
# conversion a h.5
saved_model_dir = "/home/rmart/NEDA/modelo_400_epochs_fast_rtl/modelo_400_epochs_fast_rtl"

# Cargar el modelo con TensorFlow Keras
model = tf.keras.models.load_model(saved_model_dir, compile=True)

#%% Guardarlo en formato .h5
model.save(os.path.join(saved_model_dir, 'model.h5'), save_format="h5")

#%% Cargar el modelo guardado
model = load_model(os.path.join(saved_model_dir, 'model.h5'))

#%% cambio el modelo para que tenga una sola salida
from tensorflow.keras.models import Model
modelo1out = Model(inputs=model.input, outputs=model.output[0])

#%%
os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

config = hls4ml.utils.config_from_keras_model(modelo1out, granularity='name', default_precision='fixed<16,6>')

config['Model']['ReuseFactor'] = 100

config['Model']['Strategy'] = 'Resource'

config['Model']['BramFactor'] = 1000000000

config['ClockPeriod'] = 10

#%%
hls_model = hls4ml.converters.convert_from_keras_model(modelo1out,
                                                       hls_config=config,
                                                       output_dir='hls4ml_test_fastrtl',
                                                       backend='Vivado',
                                                       part='xczu7ev-ffvc1156-2-e',
                                                       io_type='io_stream',
                                                       )


#%%+
hls_model.compile()
# Visualizar el modelo
#hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='modelo_fastrtl.png')

#%%
os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
hls_model.write()  
input("Press Enter to continue...")


hls_model.build(csim=True, export=True)



'''#####################################################################################################
# ESCRIBIR CADA VEZ QUE SE ABRA EL TERMIAL                                                          
export XILINX_VIVADO=/tools/Xilinx/Vivado/2019.2
if [ -n "${PATH}" ]; then
  export PATH=/tools/Xilinx/Vivado/2019.2/bin:$PATH
else
  export PATH=/tools/Xilinx/Vivado/2019.2/bin
fi                                                                                     
                                                                                                   
 export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/                                                    
#####################################################################################################'''