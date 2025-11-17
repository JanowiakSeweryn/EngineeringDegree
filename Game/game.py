import sys
sys.path.insert(0,"../")

from gestdetect import gesture_detection
from window_app import Win
from menu import menu
from level_class import sound_effect

import threading
import cv2
import time
import sdl2

from game_render import Window
from game_render import frame_time
from game_render import SCENES
from game_render import gest_detect
from game_render import InitializeGame
from game_render import InitializeFrame
from game_render import ShowFrame

threading.Thread(target=gest_detect,daemon=True).start()

scene_index = 0
prev_scene_index = 0 
change_scene = False

InitializeGame()

def get_index():

    global scene_index
    global change_scene
    global prev_scene_index


    if Window.Start or Window.Pause or Window.Return2Main:
        change_scene = True

    if change_scene:
        current_index = scene_index

        if Window.Return2Main : scene_index = 0
        if Window.Start: scene_index = 1
        if Window.Pause: scene_index = 2

        if current_index == scene_index:
            b = prev_scene_index
            prev_scene_index = scene_index
            scene_index = b
        else:
            prev_scene_index = current_index
            
        time.sleep(0.1)
        change_scene = False
    


while(Window.run):

    frame_start = time.perf_counter()

    InitializeFrame()
    get_index()
    SCENES[scene_index]()
    ShowFrame()



    frame_end_time = time.perf_counter() - frame_start
    if (frame_end_time < frame_time):
        time.sleep(frame_time - frame_end_time)
    

Window.Destroy()


