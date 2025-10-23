#uncomment one of the following :

from torch_nn import mlp #use torch 
# from mlp_custom import mlp #use my own neural network

import seaborn as sns
import matplotlib.pyplot as plt

from get_data import read_json



input,target = read_json()

errors = []
for i in range(6):

    NET = mlp([45,30])
    NET.Train(input,target,100)
    errors.append(NET.final_net_error)
    plt.figure()
    sns.lineplot(x = NET.epochs,y=NET.Loss)

print("final errors:")
for i in range(len(errors)):
    print(f'{i} errors : {errors[i]}')

plt.show()
