#uncomment one of the following :

from torch_nn import mlp #use torch 
# from mlp_custom import mlp #use my own neural network

import seaborn as sns
import matplotlib.pyplot as plt

import time

from get_data import read_json
from get_data import split_data


input,target = read_json()

input_train, target_train, input_test,target_test = split_data(input,target,0.6)

errors = []

print(len(input))
print(len(input_train))
print(len(input_test))

time.sleep(5)
for i in range(6):

    NET = mlp([65,65,65])
    NET.Train(input_train,target_train,100,0.005)
    errors.append(NET.final_net_error)
    plt.figure()
    sns.lineplot(x = NET.epochs,y=NET.Loss)
    NET.Train(input_test,target_test,100,0.005)
    sns.lineplot(x = NET.epochs,y = NET.Loss)

print("final errors:")
for i in range(len(errors)):
    print(f'{i} errors : {errors[i]}')

plt.show()
