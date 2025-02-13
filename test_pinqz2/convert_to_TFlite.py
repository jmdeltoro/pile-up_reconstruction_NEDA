##################################################################################################################################################
# -Versiones:                                                                                                                                    #
#    -Python = 3.6.12                                                                                                                            #
#    -Tensorflow = 2.6.0                                                                                                                         #       
##################################################################################################################################################

import tensorflow as tf
import keras.losses

def customLoss(yTrue, yPred):
    
    #weights1 = K.constant(weights)
    #weights1 = weights
    # baseline = np.mean()
    return K.mean(K.square(yTrue - yPred))

keras.losses.customLoss = customLoss
keras_model = tf.keras.models.load_model(r'C:\Users\rmart\pile-up_reconstruction_NEDA\pile-up_reconstruction_NEDA\modelo_final_entrenado_autoencoder\saved_model.pb', custom_objects={'Loss':customLoss}) #path of the keras model

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

with open('converted_model.tflite', 'wb') as f: ##name or path where save the converted model
  f.write(tflite_model) 
