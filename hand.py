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
        self.hands=self.handsMp.Hands(
            # min_detection_confidence=0.6,
            # min_tracking_confidence=0.6,
            # max_num_hands=2,
            # model_complexity=1
            )
        
        self.mpDraw = mp.solutions.drawing_utils
        # self.tipIds = [4, 8, 12, 16, 20] #???

    def findfinger(self,frame,draw=True):
        imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)


        if self.results.multi_hand_landmarks:
            
            for handLms in self.results.multi_hand_landmarks:
            
                if draw:
                    self.mpDraw.draw_landmarks(frame,handLms,self.handsMp.HAND_CONNECTIONS)

        return frame
    
    def handlm_Pos(self, mirror=False):
        """
        Get hand landmark positions
        
        Parameters:
        - mirror: If True, mirror left hand landmarks to match right hand orientation.
                 If False, preserve original handedness (useful for distinguishing left/right thumb gestures)
        """
        #1 | 2 / 21 list cointains a landmark positions
        hand_landmark_pos = []
        iter = 0
        if self.results.multi_hand_landmarks:
            for hand_idx, handLms in enumerate(self.results.multi_hand_landmarks):
                lm_pos = []
                iter = iter+1
                
                # Check if this is a left hand
                is_left_hand = False
                if self.results.multi_handedness and mirror:
                    handedness = self.results.multi_handedness[hand_idx]
                    # MediaPipe returns "Left" or "Right" from the user's perspective
                    if handedness.classification[0].label == "Left":
                        is_left_hand = True
                
                for id,lm in enumerate(handLms.landmark):
                    # Mirror x-coordinate for left hand only if mirror=True
                    x_coord = 1.0 - lm.x if is_left_hand else lm.x
                    lm_pos = (id, x_coord, lm.y)
                    hand_landmark_pos.append(lm_pos)
    
        return hand_landmark_pos

    def draw_hand_rect(self, frame):
        """
        Draw a rectangle from highest to lowest x,y values from hand landmarks
        Returns the rectangle coordinates (x_min, y_min, x_max, y_max) or None if no hand detected
        """
        if self.results.multi_hand_landmarks:
            h, w, c = frame.shape
            
            for handLms in self.results.multi_hand_landmarks:
                # Initialize min/max values
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                # Find bounding box from all landmarks
                for lm in handLms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Draw rectangle
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                return (x_min, y_min, x_max, y_max)
        
        return None

    def display_text(self, frame, text, rect_coords, font_scale=0.7, color=(0, 255, 0), thickness=2):
        """
        Display text on the camera screen at the lower edge of the rectangle
        
        Parameters:
        - frame: the video frame to draw on
        - text: the text message to display
        - rect_coords: tuple (x_min, y_min, x_max, y_max) from draw_hand_rect
        - font_scale: size of the text (default 0.7)
        - color: BGR color tuple (default green)
        - thickness: text thickness (default 2)
        """
        if rect_coords is not None:
            x_min, y_min, x_max, y_max = rect_coords
            
            # Calculate text size to center it horizontally
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Position text at lower edge of rectangle, centered horizontally
            text_x = x_min + (x_max - x_min - text_size[0]) // 2
            text_y = y_max + text_size[1] + 10  # 10 pixels below the rectangle
            
            # Draw text with background for better visibility
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return frame



   
