#Creates a folder in current directory with png data set

#create data set for one gesture:
#1 Set GESTURE_NAME constant to your gesture name 
#2 Set lenght (how many png files) of yout data set
#3 Start propram 
#4 when ready to take photos click the s key on your keyboard
#5 program automaticaly exit after compliting task
GESTURE_NAME = "zero" 
DATA_SIZE = 350

import mediapipe as mp
import cv2
import os
import sys

WIN_WIDTH = 1920
WIN_HEIGHT = 1080
FPS = 60

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FPS,FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,WIN_HEIGHT)

current_dir = os.getcwd()
frame_number = 0


# create directory
GESTURE_DIR = os.path.join(current_dir,f'{GESTURE_NAME}') 

if not os.path.isdir(GESTURE_DIR):
    os.makedirs(GESTURE_DIR)

start = False

i = 0
while(True):
    ret, frame = cap.read()

    src = cv2.flip(frame,1)
    frame = src
    cv2.imshow("camera",frame)


    if cv2.waitKey(1) == ord('s'):
       start = True
    
    if start:
        if(frame_number < DATA_SIZE):
            frame_number += 1

            sys.stdout.write(f'{100*frame_number/DATA_SIZE}% done')
            sys.stdout.flush()
            print()

            filename = f'{current_dir}/{GESTURE_NAME}/fo_{frame_number}.png'
            cv2.imwrite(filename,frame)
        else:
            break



