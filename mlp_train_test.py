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
import pandas as pd

from sklearn.metrics import confusion_matrix

DYNAMIC = False #for time gestures *dynamic gestures doesn't work*

UPDATE_WEIGHTS = False

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

print("\nSplit details (target_test):")
temp_target = [np.argmax(t) for t in target_data]
for i, g in enumerate(GESTURES):
    print(f"{g}: {temp_target.count(i)}")
print("--------------------------------")


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

errors_validate = []
errors_train = []

H = [[5,5,5],[15,15,15],[30,35,30],[5,15],[15,25],[50,25],[64,32],[16,64,48]]
solvers = ["adam"]

# for solver in solvers:
#     if solver  == "sgd": lr = 0.1; max_epoch = 2000
#     else: lr = 0.001; max_epoch = 500
    
#     for h in H:
#         print(f"training with {solver} and {h}")
#         NET = mlp(h,solver=solver) 
#         # NET.batch_size=50
#         NET.Train(input_train,target_train,max_epoch,lr)
#         print(f"train error: {NET.final_net_error}")

#         #   validation
#         NET.Validate(input_val,target_val,max_epoch,lr)    
#         print(f"validate error: {NET.final_net_error}")


NET = mlp([30,35,30],solver="adam") 
# NET.batch_size=50
NET.Train(input_train,target_train,500,0.001)
errors_train.append(NET.final_net_error)

plt.figure()
sns.lineplot(x = NET.epochs,y=NET.Loss, label='Train')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#   validation
NET.Validate(input_val,target_val,500,0.001)
errors_validate.append(NET.final_net_error)

    # sns.lineplot(x = NET.epochs,y=NET.Loss, label='Validation')
    # plt.legend()


# NET.save_weights()

print("final errors:")
for i in range(len(errors_train)):
    print(f'{i} errors train: {errors_train[i]} | {errors_validate[i]} errors validate: {errors_validate[i]}')


def DispConfussionMatrix(custom_net,filename):


    TP,TN,FP,FN = TestTFPN(custom_net)


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

    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, cmap="Blues", fmt='d', vmax=0.2*len(input_data)/(len(GESTURES)))
    plt.title("Performance Metrics per Class")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()


    lines.append(" ")
    print()
    lines.append(f"best TPR: {max(TPR.items(), key=lambda x: x[1])}")
    lines.append(f"worst TPR: {min(TPR.items(), key=lambda x: x[1])}")
    lines.append(f"best TNR: {max(TNR.items(), key=lambda x: x[1])}")
    lines.append(f"worst TNR: {min(TNR.items(), key=lambda x: x[1])}")

    with open(filename, "w") as f:
        for l in lines:
            f.write(l + "\n")


def DispROC(custom_net):
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle

    y_score = []
    for x in input_test:
        custom_net.input_change(x)
        probs = custom_net.predict()
        
        # Handle different return types (tensor vs numpy)
        if hasattr(probs, 'detach'):
             probs = probs.detach().numpy()
        elif isinstance(probs, list):
             probs = np.array(probs)
        
        # If probs is (1, N), squeeze it
        if hasattr(probs, 'shape') and len(probs.shape) > 1:
            probs = probs.squeeze()
            
        y_score.append(probs)
    
    y_score = np.array(y_score)
    y_test = np.array(target_test)
    n_classes = len(GESTURES)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(12, 10))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(GESTURES[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

DispConfussionMatrix(NET,"results1.txt")

DispROC(NET)

print(f"acc for train: {NET.get_acc(input_train,target_train)}")
print(f"acc for validation: {NET.get_acc(input_val,target_val)}")
print(f"acc for testing: {NET.get_acc(input_test,target_test)}")

if UPDATE_WEIGHTS:
    NET.save_weights(dynamic=DYNAMIC)

plt.show()


