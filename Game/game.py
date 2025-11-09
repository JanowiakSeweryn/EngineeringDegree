import sys
sys.path.insert(0,"../")

from gestdetect import gesture_detection

from window_app import Win
import time
from level_class import sound_effect
import threading
import cv2



SONG ="remastered1.wav"
LEVEL_NAME = "Level1"
FPS = 60
frame_time = 1/FPS
BEAT_INTERVAL = FPS/120

frame = 0
bmp_next_time = 0
Window = Win()

frame_start = 0


Clasifier = gesture_detection()

run = True

current_gest = None
gest = None
current_frame = None

def gest_detect():
    global current_gest 
    global current_frame
    while(run):
        result = Clasifier.clasify()
        current_gest = result
        res_frame = Clasifier.frame
        current_frame = res_frame

threading.Thread(target=gest_detect,daemon=True).start()



song_file = f"sound/{SONG}"

Level = sound_effect(song_file.encode("utf-8"),f"levels/{LEVEL_NAME}.json")
Level.LoadLevel()
Level.LoadPng(Window.renderer,"sprites/of_01.jpeg")
Level.Loadblocs_png(Window.renderer)

def Play(renderer):
    if Window.Start: #and not Level.level_failed:
        click = 0
        if Window.Left : click = 1
        if Window.Up : click = 2
        if Window.Right : click = 3
        if Window.Down : click = 4

        if current_gest == "zero" : click = 1
        # if current_gest == "german_3" : click = 2
        if current_gest == "german_3" : click = 3
        # if current_gest == "fist_closed": click = 4

        Level.Play()
        Level.PlayLevel(click)
        Level.Draw_blocs(renderer)
        Level.FailedLevel() #check if current fails are enough to fail full level
    # if Level.level_failed:
    #     Level.disp(Window.renderer)



next_beat = time.perf_counter() + BEAT_INTERVAL


while(Window.run):

    frame_start = time.perf_counter()
    
    Window.Events()
        
    Window.Render_start()
    Play(Window.renderer)
    Window.Render_present()
    Window.Reset_Events()

    if current_frame is not None:
        cv2.imshow("camera",current_frame)
        cv2.waitKey(1)


    frame_end_time = time.perf_counter() - frame_start
    if (frame_end_time < frame_time):
        time.sleep(frame_time - frame_end_time)

Window.Destroy()
