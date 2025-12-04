#uncomment one of the following :
# from torch_nn import mlp #use torch 
from mlp_custom import mlp #use my own neural network


#media pipe class to detect hand
from hand import HandDetect
from hand import cv2

from get_data import get_landmarks_input 
from get_data import read_json
from collections import deque

#gestures name
from get_data import GESTURES 
from get_data import DYNAMIC_GESTURES

import os 


WIN_WIDTH = 315
WIN_HEIGHT = 240

FRAMES_DG = 30

# NET_FILE = "weights"

class gesture_detection:

    def __init__(self,dynamic=False):

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS,60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,WIN_HEIGHT)

        self.detector = HandDetect(False,2,0.5,0.5)
        self.data_1 = []
        self.dynamic = dynamic

        self.iter_dynamic = 0
        self.prev_data = []

        input,target = read_json()
        self.NET = mlp([40,32])
        # self.NET.create_layers(len(input[0]),len(target[0]) )
        self.NET.load_weights()

        # self.NET.Train(input,target,150,0.01)

        self.ret, self.frame = self.cap.read()

    def clasify(self,frame = None):
    
        data_1 = []
        if frame is None:
            self.ret, self.frame = self.cap.read()
            src = cv2.flip(self.frame,1)
            self.frame = src
            self.frame = self.detector.findfinger(self.frame)
        else:
            self.frame = self.detector.findfinger(frame)
        
        # rect_coords = self.detector.draw_hand_rect(frame)


        # cv2.imshow("camera",self.frame)
        data_1 = self.detector.handlm_Pos()

        # self.detector.display_text(frame, GESTURES[self.NET.gesture_detected_index], rect_coords)



        if len(data_1) > 0 and not self.dynamic :

            self.NET.input_change(get_landmarks_input(data_1,self.dynamic))
            self.NET.predict()

            # NET.disp() #displays softmax of full output for all gestures 

            # print(GESTURES[self.NET.gesture_detected_index]) #displays name of the gesture
            
            return GESTURES[self.NET.gesture_detected_index]
        
        else:
            if not len(data_1) > 30:
                if self.iter_dynamic <= 30:
                    for d in self.prev_data:
                        self.iter_dynamic +=1 
                        self.data_dynamic.append(d)
                else:
                    self.iter_dynamic -=1
                    self.data_dynamic = self.data_dynamic[42:]


            if len(data_1) > 30 and self.data_dynamic:
                
                if self.iter_dynamic <=30:
                    self.iter_dynamic +=1
                    self.prev_data = data_1
                    for d in data_1:
                        self.data_dynamic.append(d)

                else:
                    self.data_dynamic = self.data_dynamic[42,:]
                    self.iter_dynamic -=1
    
                if len(self.data_dynamic) > 42*FRAMES_DG:
                    self.NET.input_change(get_landmarks_input(self.data_dynamic))
                    self.NET.predict()

                    # NET.disp() #displays softmax of full output for all gestures 

                    print(DYNAMIC_GESTURES[self.NET.gesture_detected_index]) #displays name of the gesture
                    
                    return DYNAMIC_GESTURES[self.NET.gesture_detected_index]
            
            # else:
            #     self.NET.input_change(get_landmarks_input(data_1))
            #     self.NET.predict()

            #     # NET.disp() #displays softmax of full output for all gestures 

            #     print(GESTURES[self.NET.gesture_detected_index]) #displays name of the gesture
                
            #     return GESTURES[self.NET.gesture_detected_index]


        
        
    def destroy_cap(self):
        self.cap.release()
        cv2.destroyAllWindows()