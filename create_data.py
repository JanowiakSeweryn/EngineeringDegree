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


# for file in os.listdir(f'{current_dir}/fist_closed'):
#     print(file)

start = False

while(True):
    ret, frame = cap.read()

    src = cv2.flip(frame,1)
    frame = src
    cv2.imshow("camera",frame)


    if cv2.waitKey(25) == ord('s'):
       start = True

    if start:
        if(frame_number < 100):
            frame_number += 1
            print(frame_number)
            filename = f'{current_dir}/rand_gest/fo_{frame_number}.png'
            cv2.imwrite(filename,frame)
        else:
            break
    


