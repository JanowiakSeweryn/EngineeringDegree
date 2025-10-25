#Creates a folder in current directory with png data set

#create data set for one gesture:
#1 Set GESTURE_NAME constant to your gesture name 
#2 Set lenght (how many png files) of yout data set
#3 Start propram 
#4 when ready to take photos click the s key on your keyboard
#5 program automaticaly exit after compliting task
GESTURE_NAME = "fist_flipped" 
DATA_SIZE = 100

import mediapipe as mp
import cv2
import os

WIN_WIDTH = 1920
WIN_HEIGHT = 1080

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,WIN_HEIGHT)

current_dir = os.getcwd()
frame_number = 0


# create directory
GESTURE_DIR = os.path.join(current_dir,f'{GESTURE_NAME}') 

os.makedirs(GESTURE_DIR)

start = False
while(True):
    ret, frame = cap.read()

    src = cv2.flip(frame,1)
    frame = src
    cv2.imshow("camera",frame)


    if cv2.waitKey(25) == ord('s'):
       start = True

    if start:
        if(frame_number < DATA_SIZE):
            frame_number += 1
            print(frame_number)
            filename = f'{current_dir}/{GESTURE_NAME}/fo_{frame_number}.png'
            cv2.imwrite(filename,frame)
        else:
            break
    


