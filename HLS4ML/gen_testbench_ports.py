
#%%
NUM_SAMPLES = 232
DATA_WIDTH = 16
#%% Declaraciones de señales
print("// ==== Declaraciones de señales ====\n")

print(f"// Entradas del DUT")
print(f"reg [{DATA_WIDTH - 1}:0] input_data [{NUM_SAMPLES - 1}:0];")
print(f"reg                input_valid [{NUM_SAMPLES - 1}:0];")
print(f"wire               input_ready [{NUM_SAMPLES - 1}:0];\n")

print(f"// Salidas del DUT")
print(f"wire [{DATA_WIDTH - 1}:0] output_data [{NUM_SAMPLES - 1}:0];")
print(f"wire               output_valid [{NUM_SAMPLES - 1}:0];")
print(f"reg                output_ready [{NUM_SAMPLES - 1}:0];\n")

#%% Conexiones del DUT
print("// ==== Conexiones del DUT ====\n")

for i in range(NUM_SAMPLES):
    print(f"    .input_1_V_data_{i}_V_0_tdata(input_data[{i}]),")
    print(f"    .input_1_V_data_{i}_V_0_tvalid(input_valid[{i}]),")
    print(f"    .input_1_V_data_{i}_V_0_tready(input_ready[{i}]),\n")

for i in range(NUM_SAMPLES):
    print(f"    .layer25_out_V_data_{i}_V_0_tdata(output_data[{i}]),")
    print(f"    .layer25_out_V_data_{i}_V_0_tvalid(output_valid[{i}]),")
    print(f"    .layer25_out_V_data_{i}_V_0_tready(output_ready[{i}]),\n")

#%% Asignaciones de datos
print("// ==== Asignaciones de datos ====\n")
for i in range(NUM_SAMPLES):
    print(f"input_data[{i}] = data_in[{i}];")

# %% Asignadción de valid
print("// ==== Asignación de valid ====\n")
for i in range(NUM_SAMPLES):
    print(f"input_valid[{i}] = 1'b1;")

#%%
