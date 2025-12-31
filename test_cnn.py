"""
Test CNN with raw camera images - no MediaPipe landmarks used.
Uses the full camera frame directly for gesture recognition.
"""

import cv2
import time

# Import CNN for raw camera image processing
from cnn_realtime import RealtimeCNN

# Gesture names
from get_data import GESTURES 

import sys

# Create a dictionary to hold arguments
args = {}

# sys.argv[0] is the script name, so we start at index 1
for arg in sys.argv[1:]:
    if '=' in arg:
        key, value = arg.split('=')
        args[key] = int(value)


WIN_WIDTH = args.get('width', 640)
WIN_HEIGHT = args.get('height', 480)

print(f"Resolution: {WIN_WIDTH}x{WIN_HEIGHT}")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_HEIGHT)

# Initialize CNN for raw camera images
# Number of gesture classes (excluding six-seven if not in training data)
gestures_used = [g for g in GESTURES if g != 'six-seven']
num_classes = len(gestures_used)
print(f"Number of gesture classes: {num_classes}")
print(f"Gestures: {gestures_used}")

CNN_NET = RealtimeCNN(num_classes=num_classes)
CNN_NET.load_weights()

# FPS tracking variables
prev_time = 0
fps = 0

print("\nStarting camera... Press 'q' to quit.\n")

while True:

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
        
    # Flip frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    
    # Feed raw camera frame to CNN (no MediaPipe)
    CNN_NET.input_change(frame)
    CNN_NET.predict()

    # Get gesture name
    gesture_name = gestures_used[CNN_NET.gesture_detected_index]
    print(gesture_name)
    
    # Display gesture name on frame
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Calculate and display FPS
    current_time = time.time()
    if prev_time != 0:
        fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Display FPS on screen
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("CNN Test - Raw Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("end")
