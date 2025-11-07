import mediapipe as mp
import cv2


class HandDetect:

    def __init__(self,mode,maxHands,detectionCon,trackCon):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.hand_points = []

        self.handsMp = mp.solutions.hands
        self.hands=self.handsMp.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20] #???

    def findfinger(self,frame,draw=True):
        imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
    
        if self.results.multi_hand_landmarks:
            
            for handLms in self.results.multi_hand_landmarks:
            
                if draw:
                    self.mpDraw.draw_landmarks(frame,handLms,self.handsMp.HAND_CONNECTIONS)

        return frame
    
    def handlm_Pos(self):
        
        #1 | 2 / 21 list cointains a landmark positions
        hand_landmark_pos = []
        iter = 0
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                lm_pos = []
                iter = iter+1
                for id,lm in enumerate(handLms.landmark):
                    lm_pos = (id, lm.x,lm.y)
                    hand_landmark_pos.append(lm_pos)
    
        return hand_landmark_pos



   
