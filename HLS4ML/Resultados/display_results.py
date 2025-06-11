#%%
import pandas as pd
import pickle as pk
import matplotlib as m
import matplotlib.pyplot as plt
import numpy as np

data_path = r"C:/Users/rmart/pile-up_reconstruction_NEDA/pile-up_reconstruction_NEDA/test_pinqz2"

comb_OUT = 'g_n'
path_OUT = (data_path + '/reconsturccion_g-n_PC.pkl')
print(path_OUT)
data_OUT = pd.read_pickle(path_OUT)
data_OUT = data_OUT.sample(frac=1, random_state=0).reset_index(drop=True)

#%%

for i in range(0, len(data_OUT.TraceFinal)):
    fig_handle = plt.figure()

    x = np.linspace(0, 1000, len(data_OUT.TraceFinal[0]))
    y_IN = data_OUT.TraceFinal[i]

    y_OUT_1 = data_OUT.Trace1_result[i]
    y_OUT_2 = data_OUT.Trace2_result[i]

    plt.plot(x,abs(y_IN),)
    plt.plot(x, abs(y_OUT_1),)
    plt.plot(x, abs(y_OUT_2))
    plt.show()




# %%
