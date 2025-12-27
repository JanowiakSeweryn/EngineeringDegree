#uncomment one of the following :
# from torch_nn import mlp as ml_model #use torch 
from mlp_custom import mlp as ml_model #use my own neural network
# from knn_classifier import ml_model

# media pipe class to detect hand
from hand import HandDetect
from hand import cv2

from get_data import get_landmarks_input 
from get_data import read_json
from collections import deque

#gestures name
from get_data import GESTURES 
from get_data import DYNAMIC_GESTURES

import os 


WIN_WIDTH = 256
WIN_HEIGHT = 256

FRAMES_DG = 30

# NET_FILE = "weights"

class gesture_detection:

    def __init__(self,dynamic=False):

        self.cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

        self.cap.set(cv2.CAP_PROP_FPS,60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,WIN_HEIGHT)

        self.detector = HandDetect(False,2,0.5,0.5)
        self.data_1 = []
        self.dynamic = dynamic

        self.iter_dynamic = 0
        self.prev_data = []

        input,target = read_json()
        # self.NET = ml_model([30,35,30])
        self.NET = ml_model([40,32])
        # self.NET.create_layers(len(input[0]),len(target[0]) )
        self.NET.load_weights()

        # self.NET.Train(input,target,150,0.01)

        self.ret, self.frame = self.cap.read()

    def clasify(self,frame = None):
    
        data_1 = []
        if frame is None:
            self.ret, self.frame = self.cap.read()
            
            # Check if frame was captured successfully
            if not self.ret or self.frame is None:
                print("Warning: Failed to capture frame from camera")
                return None
            
            src = cv2.flip(self.frame,1)
            self.frame = src
            self.frame = self.detector.findfinger(self.frame)
        else:
            # Check if provided frame is valid
            if frame is None:
                print("Warning: Provided frame is None")
                return None
            self.frame = self.detector.findfinger(frame)
        
        # rect_coords = self.detector.draw_hand_rect(frame)


        # cv2.imshow("camera",self.frame)
        data_1 = self.detector.handlm_Pos()

        # self.detector.display_text(frame, GESTURES[self.NET.gesture_detected_index], rect_coords)


        if len(data_1) > 0 and not self.dynamic :
            
            # Initial prediction
            self.NET.input_change(get_landmarks_input(data_1, self.dynamic))
            self.NET.predict()
            initial_gesture = GESTURES[self.NET.gesture_detected_index]
            
            # Check handedness
            handedness = self.detector.get_handedness()
            
            # If Left hand is detected
            if handedness == "Left":
                # Exceptions logic: if it matches exceptions, return it.
                # Note: assumed user meant 'left_thumb' and 'right_thumb'
                if initial_gesture == "left_thumb" or initial_gesture == "right_thumb":
                    return initial_gesture
                else:
                    # Mirror the data (flip x coordinates)
                    # data_1 is list of (id, x, y)
                    data_mirrored = []
                    for item in data_1:
                        # item[1] is x. Mirror: 1.0 - x
                        data_mirrored.append((item[0], 1.0 - item[1], item[2]))
                    
                    # Predict again with mirrored data
                    self.NET.input_change(get_landmarks_input(data_mirrored, self.dynamic))
                    self.NET.predict()
                    return GESTURES[self.NET.gesture_detected_index]

            return initial_gesture
        else:
            return "hand_flipped"

                    
    def destroy_cap(self):
        self.cap.release()
        cv2.destroyAllWindows()