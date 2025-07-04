import time
import tflite_runtime.interpreter as tflite
import pandas as pd
import pickle as pk
import numpy as np #version 1.26, tflite necesita numpy con una version anterior a la 2

interpreter = tflite.Interpreter(model_path="/home/rmart/pruebas_raspberypi/converted_model_fastrtl.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open('/home/rmart/pruebas_raspberypi/dataset_test_200eventos_n-g.pkl', 'rb') as file:
    input_data = pk.load(file)
    
output1_data = np.zeros(len(input_data.TraceFinal))
output2_data = np.zeros(len(input_data.TraceFinal))
tiempo_inferencia = []

for i in range(len(input_data.TraceFinal)):
    trace = input_data.TraceFinal[i]
    trace = trace.astype(np.float32)
    trace = trace.reshape(1,232,1)
    
    inicio = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], trace)
    interpreter.invoke()
    fin = time.perf_counter()
    
    tiempo_inferencia.append(fin-inicio)
    
    output1_data = interpreter.get_tensor(output_details[0]['index'])
    output2_data = interpreter.get_tensor(output_details[1]['index'])
        
np.savetxt("/home/rmart/pruebas_raspberypi/resultados1_n-g.csv", output1_data.reshape(232), delimiter=',')
np.savetxt("/home/rmart/pruebas_raspberypi/resultados2_n-g.csv", output2_data.reshape(232), delimiter=',')
np.savetxt("/home/rmart/pruebas_raspberypi/tiempos_n-g.csv", tiempo_inferencia, delimiter=',')
