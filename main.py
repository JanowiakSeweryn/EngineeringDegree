#uncomment one of the following :
# from torch_nn import mlp #use torch 
from mlp_custom import mlp #use my own neural network

#media pipe class to detect hand
from hand import HandDetect
from hand import cv2


from get_data import get_landmarks_input 
from get_data import read_json

#gestures name
from get_data import GESTURES 

WIN_WIDTH = 1920
WIN_HEIGHT = 1080

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,WIN_HEIGHT)

detector = HandDetect(False,2,0.5,0.5)
data_1 = []


input,target = read_json()

NET = mlp([45,30])
NET.Train(input,target,150)
 

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

        NET.disp() #displays softmax of full output for all gestures 

        # print(GESTURES[NET.gesture_detected_index]) #displays name of the gesture

cap.release()
cv2.destroyAllWindows()

print("end")
