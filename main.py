CUSTOM_NET=False


#media pipe class to detect hand
from hand import HandDetect
from hand import cv2

from get_data import get_landmarks_input 
from get_data import read_json,split_data

# if CUSTOM_NET :from mlp_custom import mlp 
# else: from torch_nn import mlp #use torch #use my own neural network

from knn_classifier import mlp 
#gestures name
from get_data import GESTURES 

WIN_WIDTH = 480
WIN_HEIGHT = 340

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,WIN_HEIGHT)

detector = HandDetect(False,2,0.5,0.5)
data_1 = []


inputs,target = read_json()

print(len(inputs))
print(len(target))

input_train, target_train, input_test,target_test = split_data(inputs,target,0.5)

# iter = 0
# for i in input:
#     if isinstance(i,bool):
#         print(iter)
#         print("INVALID_DATA")
#     iter += 1

# NET = mlp([65,65,65])

NET = mlp([40,32])
NET.create_layers(len(input_train[0]),len(target_train[0]))
NET.load_weights()

# NET.Train(inputs,target,200,0.001)
 

while True:

    data_1 = []
    ret, frame = cap.read()
    src = cv2.flip(frame,1)
    frame = src
    frame = detector.findfinger(frame)
    
    # Draw rectangle around hand
    rect_coords = detector.draw_hand_rect(frame)
    
    data_1 = detector.handlm_Pos()

    if len(data_1) > 0:
        NET.input_change(get_landmarks_input(data_1))
        NET.predict()

        # NET.disp() #displays softmax of full output for all gestures 

        gesture_name = GESTURES[NET.gesture_detected_index]
        print(gesture_name) #displays name of the gesture
        
        # Display gesture name at lower edge of rectangle
        detector.display_text(frame, gesture_name, rect_coords)
    
    cv2.imshow("camera",frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("end")
