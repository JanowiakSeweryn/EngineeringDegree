#uncomment one of the following :

from torch_nn import mlp #use torch 
# from mlp_custom import mlp #use my own neural network

import seaborn as sns
import matplotlib.pyplot as plt

import time

from get_data import read_json
from get_data import split_data
from get_data import GESTURES
import os
import numpy as np

DYNAMIC = False #for time gestures 

NET_FILE = "weights"
module_dir = os.path.dirname(__file__)
file_path = os.path.join(module_dir, f"{NET_FILE}.pth")


input,target = read_json()

input_train, target_train, input_test,target_test = split_data(input,target,0.8)

input_train, target_train, input_val,target_val = split_data(input_train,target_train,0.75)

errors = []

print(len(input))
print(len(input_train))
print(len(input_test))
print(len(input_val))



#retursn true positives, true negatives etc
def TestTFPN(custom_net):


    tp = [0]*len(GESTURES)
    tn = [0]*len(GESTURES)
    fp = [0] *len(GESTURES)
    fn = [0] *len(GESTURES)
    
    for i in range(len(GESTURES)):
        for x,y in zip(input_test,target_test):
            
            gest = GESTURES[np.argmax(y)] #target gesture as string
            
            custom_net.input_change(x)
            y_pred = custom_net.predict()

            net_gest = GESTURES[custom_net.gesture_detected_index] #net prediction gesture as a string
            
            #positives: 
            if GESTURES[i] == gest :
                if net_gest == gest: tp[i] += 1 #true positve
                else: fn[i]+= 1 # false negative
            else:
                if net_gest == gest: tn[i] += 1 #true positve
                else: fp[i] += 1 # false negative

    return tp, tn, fp, fn


for i in range(1):

    NET = mlp([40,32])
    NET.Train(input_train,target_train,1000,0.005)
    errors.append(NET.final_net_error)
    plt.figure()
    sns.lineplot(x = NET.epochs,y=NET.Loss)
    #validation
    NET.Train(input_val,target_val,1000,0.005)
    errors.append(NET.final_net_error)
    sns.lineplot(x = NET.epochs,y=NET.Loss)



print("final errors:")
for i in range(len(errors)):
    print(f'{i} errors : {errors[i]}')

TP,TN,FP,FN = TestTFPN(NET)

if len(input_test) != TP[0]+TN[0]+FP[0]+FN[0]: print("error!")
else: print("data analised properly!")

TPR = {}
TNR = {}

lines = []

for i,gest in enumerate(GESTURES):
    tpr = TP[i]/(TP[i]+FN[i])
    tnr = TN[i]/(TN[i]+TP[i])

    TPR[gest] = tpr
    TNR[gest] = tnr

    print(f"{gest} TPR={tpr}, TNR={tnr}")
    lines.append(f"{gest} TPR={tpr}, TNR={tnr}")

np.argmin(TPR)

lines.append(" ")
print()
lines.append(f"best TPR: {max(TPR.items(), key=lambda x: x[1])}")
lines.append(f"worst TPR: {min(TPR.items(), key=lambda x: x[1])}")
lines.append(f"best TNR: {max(TNR.items(), key=lambda x: x[1])}")
lines.append(f"worst TNR: {min(TNR.items(), key=lambda x: x[1])}")

with open("results.txt", "w") as f:
    for l in lines:
        f.write(l + "\n")



NET.save_weights(dynamic=DYNAMIC)

plt.show()


