##################################################################################################################################################
# -Versiones:                                                                                                                                    #
#    -Python = 3.6.12                                                                                                                            #
#    -Tensorflow = 2.6.0                                                                                                                         #       
##################################################################################################################################################

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import keras.losses

def customLoss(yTrue, yPred):
    
    #weights1 = K.constant(weights)
    #weights1 = weights
    # baseline = np.mean()
    return K.mean(K.square(yTrue - yPred))

keras.losses.customLoss = customLoss
keras_model = load_model(os.path.join(r"C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Modelos\modelo_400_epochs_mod3", 'model.h5')) #path of the keras model

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

with open(r'C:\Users\rmart\pile-up_reconstruction_NEDA\HLS4ML\Modelos\modelo_400_epochs_mod3\converted_model.tflite', 'wb') as f: ##name or path where save the converted model
  f.write(tflite_model) 
