from knn_classifier import ml_model  # KNN classifier with mlp interface

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

DYNAMIC = False  # for time gestures 

NET_FILE = "knn_weights"
module_dir = os.path.dirname(__file__)
file_path = os.path.join(module_dir, f"{NET_FILE}.pkl")


input_data, target_data = read_json()

input_train, target_train, input_test, target_test = split_data(input_data, target_data, 0.8)

input_train, target_train, input_val, target_val = split_data(input_train, target_train, 0.75)

errors = []

print(f"Total samples: {len(input_data)}")
print(f"Training samples: {len(input_train)}")
print(f"Test samples: {len(input_test)}")
print(f"Validation samples: {len(input_val)}")


# Returns true positives, true negatives etc
def TestTFPN(custom_net):
    tp = [0] * len(GESTURES)
    tn = [0] * len(GESTURES)
    fp = [0] * len(GESTURES)
    fn = [0] * len(GESTURES)
    
    for i in range(len(GESTURES)):
        for x, y in zip(input_test, target_test):
            
            gest = GESTURES[np.argmax(y)]  # target gesture as string
            
            custom_net.input_change(x)
            y_p = custom_net.predict()

            net_gest = GESTURES[custom_net.gesture_detected_index]  # net prediction gesture as a string
            
            # positives: 
            if GESTURES[i] == gest:
                if net_gest == gest:
                    tp[i] += 1  # true positive
                else:
                    fn[i] += 1  # false negative
            else:
                if net_gest == gest:
                    tn[i] += 1  # true negative
                else:
                    fp[i] += 1  # false positive

    return tp, tn, fp, fn


def get_predictions(custom_net):
    y_true = []
    y_pred = []
    for x, y in zip(input_test, target_test):
        gest = GESTURES[np.argmax(y)]  # target gesture as string
        custom_net.input_change(x)
        y_p = custom_net.predict()
        net_gest = GESTURES[custom_net.gesture_detected_index] 

        y_true.append(gest)
        y_pred.append(net_gest)

    return y_true, y_pred


# Test different k values
k_values = [1, 3, 5, 7, 9, 11]
best_k = 5
best_acc = 0

print("\n=== Testing different k values ===")
for k in k_values:
    NET = ml_model(hidden_sizes=k, solver="adam")  # k is passed as hidden_sizes
    NET.Train(input_train, target_train, 1, 0.001)  # KNN doesn't need epochs or lr
    
    # Validate
    acc = NET.get_acc(input_val, target_val)
    print(f"k={k}: Validation accuracy = {acc:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"\nBest k value: {best_k} with accuracy: {best_acc:.2f}%")

# Train final model with best k
print(f"\n=== Training final KNN model with k={best_k} ===")
NET = ml_model(hidden_sizes=best_k, solver="adam")
NET.Train(input_train, target_train, 500, 0.001)

print(f"Epochs: {len(NET.epochs)}, Loss history: {len(NET.Loss)}")

# Plot training loss (all zeros for KNN)
plt.figure()
sns.lineplot(x=NET.epochs, y=NET.Loss, label='Train (KNN has no loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Validation
NET.Validate(input_val, target_val, 500, 0.001)
errors.append(NET.final_net_error)

print(f"Epochs: {len(NET.epochs)}, Loss history: {len(NET.Loss)}")

sns.lineplot(x=NET.epochs, y=NET.Loss, label='Validation')
plt.legend()
plt.title(f'KNN Classifier (k={best_k})')

# Save model
NET.save_weights()

print("\nFinal errors:")
for i in range(len(errors)):
    print(f'{i} errors : {errors[i]}')


def DispConfusionMatrix(custom_net, filename):
    TP, TN, FP, FN = TestTFPN(custom_net)

    if len(input_test) != TP[0] + TN[0] + FP[0] + FN[0]:
        print("error!")
    else:
        print("data analysed properly!")

    TPR = {}
    TNR = {}
    FNR = {}
    FPR = {}

    lines = []

    for i, gest in enumerate(GESTURES):
        tpr = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
        tnr = TN[i] / (TN[i] + TP[i]) if (TN[i] + TP[i]) > 0 else 0

        fpr = FN[i] / (FN[i] + TP[i]) if (FN[i] + TP[i]) > 0 else 0
        fnr = FN[i] / (FN[i] + TP[i]) if (FN[i] + TP[i]) > 0 else 0

        TPR[gest] = tpr
        TNR[gest] = tnr

        FNR[gest] = fnr
        FPR[gest] = fpr

        print(f"{gest} TPR={tpr:.4f}, TNR={tnr:.4f}")
        lines.append(f"{gest} TPR={tpr:.4f}, TNR={tnr:.4f}")

    y_true, y_pred = get_predictions(NET)

    cm = confusion_matrix(y_true, y_pred, labels=GESTURES)

    df = pd.DataFrame(cm, index=GESTURES, columns=GESTURES)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, cmap="Blues", fmt='d', vmax=0.2 * len(input_data) / (len(GESTURES)))
    plt.title(f"KNN Confusion Matrix (k={NET.k})")
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
    lines.append(f"k value: {NET.k}")

    with open(filename, "w") as f:
        for l in lines:
            f.write(l + "\n")


print("\n=== Generating Confusion Matrix ===")
DispConfusionMatrix(NET, "knn_results.txt")

# Test loading weights
# NET2 = mlp(hidden_sizes=best_k)
# NET2.load_weights()
# DispConfusionMatrix(NET2, "knn_results2.txt")

print("\n=== Final Accuracy Results ===")
print(f"acc for train: {NET.get_acc(input_train, target_train):.2f}%")
print(f"acc for validation: {NET.get_acc(input_val, target_val):.2f}%")
print(f"acc for testing: {NET.get_acc(input_test, target_test):.2f}%")

plt.show()
