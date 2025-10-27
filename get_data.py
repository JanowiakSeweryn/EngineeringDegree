#creates data set for single hand detection 
#the file uses the create_data_set()  funtion to create data sets:
#data sets have 2 outputs: targets (array to neural network for classification problem)

from hand import HandDetect
from hand import cv2
import math
import json
import os
import random

#every time when adding/removing gesture add set to true
#after that set to False if you want fast way of runnng other programs
Update_Data_Set = True

#main list of gestures

GESTURES = [
    "fist_open",
    "fist_closed",
    "fist_flipped",
    "german_3" ,
    "uk_3",
    "kon",
    "zero" 
]


detector = HandDetect(False,2,0.5,0.5)

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
        

        for i in range(21):
            result_input.append(x_vals[i])
            result_input.append(y_vals[i])
            # result_input.append(math.sqrt(x_vals[i]*x_vals[i] + y_vals[i]*y_vals[i]))
        

        return result_input

    else: return False

def get_data_png(filename):

    image = cv2.imread(filename)
    image = detector.findfinger(image)
    data = detector.handlm_Pos()

    return get_landmarks_input(data)


def create_data_set():

    current_dir = os.getcwd()

    input_data = []
    target_data = []

    target_gesture = []

    gesture_index = 0

    #initialize all targets as list [0,0,0...number of gestures]
    for i in range(len(GESTURES)):
        target_gesture.append(0)
    
    #here add new gesture
    for gesture in GESTURES:
        
        target_gesture[gesture_index] = 1 

        for file in os.listdir(f'{current_dir}/{gesture}'):
            sample = get_data_png(f'{gesture}/{file}')
            if sample and  isinstance(sample,list):
                input_data.append(sample)
                target_data.append(target_gesture.copy()) #

        print(f"gesture {GESTURES[gesture_index]} added")

        target_gesture[gesture_index] = 0
        gesture_index += 1


    #shuffle the data
    combined = list(zip(input_data, target_data))

    random.shuffle(combined)

    input_data, target_data = zip(*combined)
    Data = {
        "input": input_data ,
        "target": target_data,
    }

    with open("data_set.json","w") as f:
        json.dump(Data,f,indent=4)

    
def read_json():

    with open("data_set.json","r") as f:
        data = json.load(f)
    
    input_data = data["input"]
    target_data = data["target"]
            

    return input_data, target_data

def split_data(input,target,train_size):

    test_size = round(len(input)*train_size)
    x_train = input[0:test_size]
    x_test = input[test_size:len(input)]

    y_train = target[0:test_size]
    y_test = target[test_size:len(input)]

    return x_train, y_train, x_test, y_test



#UPDATE DATA SET 
if(Update_Data_Set):
    create_data_set() 