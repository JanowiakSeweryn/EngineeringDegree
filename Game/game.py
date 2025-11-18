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
from game_render import frame_time, ClickButton
from game_render import SCENES 
from game_render import gest_detect
from game_render import InitializeGame
from game_render import InitializeFrame
from game_render import ShowFrame
from game_render import TriggerButtons
from game_render import RestartLevel

threading.Thread(target=gest_detect,daemon=True).start()

scene_index = "mainscene"
prev_scene_index = "mainscene"

change_scene = False

InitializeGame()

def get_index():

    global scene_index
    global change_scene
    global prev_scene_index


    if Window.Event_trigger["Start"] or Window.Event_trigger["Pause"] or Window.Event_trigger["Return2Main"] :
        change_scene = True

    if Window.Event_trigger["Resume"]:
        change_scene = True

    if change_scene:
        current_index = scene_index

        if Window.Event_trigger["Return2Main"] : scene_index = "mainscene"
        if Window.Event_trigger["Start"]: scene_index = "playscene"
        if Window.Event_trigger["Pause"]: scene_index = "pausescene"
        if Window.Event_trigger["Resume"]: scene_index = "playscene"

        if current_index == scene_index:
            b = prev_scene_index
            prev_scene_index = scene_index
            scene_index = b
        else:
            prev_scene_index = current_index
            
        time.sleep(0.1)
        change_scene = False
    

def TriggerOptions(option):
    global scene_index

    if option is not None:
        if option == "play": Window.Event_trigger["Start"] = True
        if option == "pause": Window.Event_trigger["Pause"] = True
        if option == "resume" :Window.Event_trigger["Resume"] = True
        if option == "return2home" : Window.Event_trigger["Return2Main"] = True
        if option == "restart": 
            RestartLevel()
            scene_index = "playscene"

while(Window.run):

    frame_start = time.perf_counter()

    InitializeFrame()

    get_index()

    SCENES[scene_index]()

    current_option = TriggerButtons(scene_index)

               
    ShowFrame()

    TriggerOptions(current_option)

    frame_end_time = time.perf_counter() - frame_start
    if (frame_end_time < frame_time):
        time.sleep(frame_time - frame_end_time)
    

Window.Destroy()


