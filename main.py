import seaborn as sns
import matplotlib.pyplot as plt
from mlp import mlp
from hand import HandDetect
from hand import cv2
import os
import time
import sys


sns.set_theme()


WIN_WIDTH = 1920
WIN_HEIGHT = 1080

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,WIN_HEIGHT)

detector = HandDetect(False,2,0.5,0.5)
data_1 = []

def get_landmarks_input(data_1):

    result_input = []

    if len(data_1) > 0:
        index_val = [pt[0] for pt in data_1]
        x_vals = [pt[1] for pt in data_1]  
        y_vals = [pt[2] for pt in data_1]  

        ymax = max(y_vals)

        for i, val in enumerate(y_vals):
            #reversing the y-axis to easy to analize
            y_vals[i] = (ymax-y_vals[i])
            #normalize the [0] value (wrist) to be begging of y and x axis
            
        y_wrist = y_vals[0]
        x_wrist = x_vals[0]
        for i,val in enumerate(y_vals):
            y_vals[i] = y_vals[i] - y_wrist
            x_vals[i] = x_vals[i] - x_wrist
        
        for i in range(len(x_vals)):
            result_input.append(x_vals[i])
            result_input.append(y_vals[i])#

    return result_input


def get_data_png(filename):

    image = cv2.imread(filename)
    image = detector.findfinger(image)
    data = detector.handlm_Pos()

    return get_landmarks_input(data)



def create_data_set():

    current_dir = os.getcwd()

    input = []
    target = []

    Data_set = [input,target]

    for file in os.listdir(f'{current_dir}/fist_closed'):
        
        input.append(get_data_png(f'fist_closed/{file}')), target.append([0,1])
        Data_set.append([input,target])

    for file in os.listdir(f'{current_dir}/fist_open'):
        input.append(get_data_png(f'fist_open/{file}')), target.append([1,0])
        Data_set.append([input,target])

    return Data_set

NET = mlp([64,32])

Data_set = create_data_set()

input = Data_set[0]
target = Data_set[1]

for t in input:
    print(len(t))




sys.exit()

NET.Train(input,target,2000)

# NET.PlotEntropyError()

while True:

    data_1 = []
    ret, frame = cap.read()
    src = cv2.flip(frame,1)
    frame = src
    frame = detector.findfinger(frame)
    
    data_1 = detector.handlm_Pos()

    cv2.imshow("camera",frame)

    if cv2.waitKey(25) == ord('q'):
        break

    if len(data_1) > 0:
        NET.input_change(get_landmarks_input(data_1))
        NET.predict()

cap.release()
cv2.destroyAllWindows()

print("end")
