#uncomment one of the following :
from torch_nn import mlp #use torch 
# from mlp_custom import mlp #use my own neural network


#media pipe class to detect hand
from hand import HandDetect
from hand import cv2

from get_data import get_landmarks_input 
from get_data import read_json


#gestures name
from get_data import GESTURES 

import os 




class gesture_detection:

    def __init__(self):

        WIN_WIDTH = 315
        WIN_HEIGHT = 240

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS,60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,WIN_HEIGHT)

        self.detector = HandDetect(False,2,0.5,0.5)
        self.data_1 = []

        input,target = read_json()
        self.NET = mlp([40,32])
        self.NET.Train(input,target,150,0.01)

        self.ret, self.frame = self.cap.read()

    def clasify(self):
        data_1 = []
        self.ret, self.frame = self.cap.read()
        src = cv2.flip(self.frame,1)
        self.frame = src
        self.frame = self.detector.findfinger(self.frame)
        # cv2.imshow("camera",self.frame)
        data_1 = self.detector.handlm_Pos()
    

        if len(data_1) > 0:
            self.NET.input_change(get_landmarks_input(data_1))
            self.NET.predict()

            # NET.disp() #displays softmax of full output for all gestures 

            print(GESTURES[self.NET.gesture_detected_index]) #displays name of the gesture
            
            return GESTURES[self.NET.gesture_detected_index]
        
    def destroy_cap(self):
        self.cap.release()
        cv2.destroyAllWindows()