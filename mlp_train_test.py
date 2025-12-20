#uncomment one of the following :

# from torch_nn import mlp #use torch 
from mlp_custom import mlp #use my own neural network

import seaborn as sns
import matplotlib.pyplot as plt

import time

from get_data import read_json
from get_data import split_data
from get_data import GESTURES
import os
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

DYNAMIC = False #for time gestures 

NET_FILE = "weights"
module_dir = os.path.dirname(__file__)
file_path = os.path.join(module_dir, f"{NET_FILE}.pth")


input_data,target_data = read_json()

input_train, target_train, input_test,target_test = split_data(input_data,target_data,0.8)

input_train, target_train, input_val,target_val = split_data(input_train,target_train,0.75)

errors = []

print(len(input_data))
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
            y_p = custom_net.predict()

            net_gest = GESTURES[custom_net.gesture_detected_index] #net prediction gesture as a string
            
            #positives: 
            if GESTURES[i] == gest :
                if net_gest == gest: tp[i] += 1 #true positve
                else: fn[i]+= 1 # false negative
            else:
                if net_gest == gest: tn[i] += 1 #true positve
                else: fp[i] += 1 # false negative

    return tp, tn, fp, fn



def get_predictions(custom_net):
    y_true = []
    y_pred = []
    for x,y in zip(input_test,target_test):
        gest = GESTURES[np.argmax(y)] #target gesture as string
        custom_net.input_change(x)
        y_p = custom_net.predict()
        net_gest = GESTURES[custom_net.gesture_detected_index] 

        y_true.append(gest)
        y_pred.append(net_gest)

    return y_true, y_pred

        
for i in range(1):

    NET = mlp([30,],solver="adam") 
    # NET.batch_size=50
    NET.Train(input_train,target_train,500,0.001)

    print(len(NET.epochs),len(NET.Loss))
 
    plt.figure()
    sns.lineplot(x = NET.epochs,y=NET.Loss, label='Train')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
  #   validation
    NET.Validate(input_val,target_val,500,0.001)
    errors.append(NET.final_net_error)

    print(len(NET.epochs),len(NET.Loss))

    sns.lineplot(x = NET.epochs,y=NET.Loss, label='Validation')
    plt.legend()

NET.save_weights()

print("final errors:")
for i in range(len(errors)):
    print(f'{i} errors : {errors[i]}')


def DispConfussionMatrix(custom_net,filename):
    TP,TN,FP,FN = TestTFPN(custom_net)



    if len(input_test) != TP[0]+TN[0]+FP[0]+FN[0]: print("error!")
    else: print("data analised properly!")

    TPR = {}
    TNR = {}
    FNR = {}
    FPR = {}


    lines = []

    for i,gest in enumerate(GESTURES):
        tpr = TP[i]/(TP[i]+FN[i])
        tnr = TN[i]/(TN[i]+TP[i])

        fpr = FN[i]/(FN[i]+TP[i])
        fnr = FN[i]/(FN[i]+TP[i])

        TPR[gest] = tpr
        TNR[gest] = tnr

        FNR[gest] = fnr
        FPR[gest] = fpr

        print(f"{gest} TPR={tpr}, TNR={tnr}")
        lines.append(f"{gest} TPR={tpr}, TNR={tnr}")


    y_true, y_pred = get_predictions(NET)

    cm = confusion_matrix(y_true, y_pred, labels=GESTURES)

    df = pd.DataFrame(cm,index=GESTURES,columns=GESTURES)

    plt.figure()
    sns.heatmap(df, annot=True, cmap="Blues",vmax=0.2*len(input_data)/(len(GESTURES)))
    plt.title("Performance Metrics per Class")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')


    lines.append(" ")
    print()
    lines.append(f"best TPR: {max(TPR.items(), key=lambda x: x[1])}")
    lines.append(f"worst TPR: {min(TPR.items(), key=lambda x: x[1])}")
    lines.append(f"best TNR: {max(TNR.items(), key=lambda x: x[1])}")
    lines.append(f"worst TNR: {min(TNR.items(), key=lambda x: x[1])}")

    with open(filename, "w") as f:
        for l in lines:
            f.write(l + "\n")

DispConfussionMatrix(NET,"results1.txt")

# NET2 = mlp([40,32])

# NET2.load_weights()

# DispConfussionMatrix(NET2,"results2.txt")

print(f"acc for train: {NET.get_acc(input_train,target_train)}")
print(f"acc for validation: {NET.get_acc(input_val,target_val)}")
print(f"acc for testing: {NET.get_acc(input_test,target_test)}")

# NET.save_weights(dynamic=DYNAMIC)

plt.show()


