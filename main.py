

#media pipe class to detect hand
from hand import HandDetect
from hand import cv2

from get_data import get_landmarks_input 
from get_data import read_json,split_data

from mlp_custom import mlp as ml_model 
HIDDEN_LAYER = [40,32]

# from torch_nn import mlp as ml_model 
# HIDDEN_LAYER = [30,35,30]

# from knn_classifier import knn as ml_model 


#gestures name
from get_data import GESTURES 
import time

import sys

# Create a dictionary to hold arguments
args = {}

# sys.argv[0] is the script name, so we start at index 1
for arg in sys.argv[1:]:
    if '=' in arg:
        key, value = arg.split('=')
        args[key] = int(value)


# WIN_WIDTH = args.get('width')
# WIN_HEIGHT = args.get('height')

# WIN_WIDTH = 640
# WIN_HEIGHT = 480

WIN_WIDTH = 256
WIN_HEIGHT = 256

cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FPS,60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,WIN_HEIGHT)

detector = HandDetect(False,2,0.5,0.5)
data_1 = []


inputs,target = read_json()

print(len(inputs))
print(len(target))

input_train, target_train, input_test,target_test = split_data(inputs,target,0.5)


NET = ml_model(HIDDEN_LAYER)
# NET.create_layers(len(input_train[0]),len(target_train[0]))
NET.load_weights()
 
from collections import deque

# FPS averaging - store last N frame times
fps_history = deque(maxlen=30)
prev_time = time.time()
latency_ms = 0

while True:

    data_1 = []
    prev_camera_time = time.perf_counter()
    
    ret, frame = cap.read()

    src = cv2.flip(frame,1)
    frame = src

    # frame = detector.findfinger(frame)
    
    # # Draw rectangle around hand
    # rect_coords = detector.draw_hand_rect(frame)
    
    # data_1 = detector.handlm_Pos()

    # if len(data_1) > 0:
    #     NET.input_change(get_landmarks_input(data_1))
    #     NET.predict()

    #     # NET.disp() #displays softmax of full output for all gestures 

    #     gesture_name = GESTURES[NET.gesture_detected_index]
    #     print(gesture_name) #displays name of the gesture
        
    #     # Display gesture name at lower edge of rectangle
    #     detector.display_text(frame, gesture_name, rect_coords)
    

    # Calculate averaged FPS
    current_time = time.time()
    fps_history.append(current_time - prev_time)
    prev_time = current_time
    
    avg_fps = len(fps_history) / sum(fps_history) if fps_history else 0
    
    # Display averaged FPS on screen
    fps_text = f"FPS: {int(avg_fps)}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display Latency in ms
    latency_text = f"Latency: {latency_ms:.2f} ms"
    cv2.putText(frame, latency_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("camera",frame)
    after_camera_time = time.perf_counter()
    latency_ms = (after_camera_time - prev_camera_time)*1000

    if cv2.waitKey(1) == ord('q'):
        break
    if len(fps_history) > 100:
        fps_history.pop(0)

cap.release()
cv2.destroyAllWindows()

print("end")
