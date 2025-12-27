import sys
sys.path.insert(0,"../")

from gestdetect import gesture_detection
from window_app import Win
from menu import menu
from level_class import LevelClass

import threading
import cv2
import time
import sdl2

from game_render import Window
from game_render import frame_time, current_level
from game_render import SCENES 
from game_render import gest_detect
from game_render import InitializeGame
from game_render import InitializeFrame
from game_render import ShowFrame
from game_render import TriggerButtons
from game_render import RestartLevel

from game_render import main_screen_theme,menu_theme,prev_time,fps_history
    

threading.Thread(target=gest_detect,daemon=True).start()


# scene_index = "selectlevelscene"
scene_index = "openingscene"
# scene_index = "mainscene"
prev_scene_index = "mainscene"

change_scene = False

InitializeGame()

def get_sceene_index():

    global scene_index
    global change_scene
    global prev_scene_index

    if Window.Event_trigger["Start"] or Window.Event_trigger["Pause"] or Window.Event_trigger["Return2Main"] :
        change_scene = True

    if Window.Event_trigger["Resume"] or Window.Event_trigger["selectlevel"]:
        change_scene = True
    
    if Window.Event_trigger.get("LevelSuccess", False) or Window.Event_trigger.get("LevelFailed", False):
        change_scene = True

    if change_scene:
        current_index = scene_index

        if Window.Event_trigger["Return2Main"] : scene_index = "mainscene"
        if Window.Event_trigger["Start"]: scene_index = "playscene"
        if Window.Event_trigger["Pause"]: scene_index = "pausescene"
        if Window.Event_trigger["Resume"]: scene_index = "playscene"
        if Window.Event_trigger["selectlevel"]: scene_index = "selectlevelscene"
        if Window.Event_trigger.get("LevelSuccess", False): scene_index = "levelsuccessscene"
        if Window.Event_trigger.get("LevelFailed", False): scene_index = "levelfailedscene"

        if current_index == scene_index:
            b = prev_scene_index
            prev_scene_index = scene_index
            scene_index = b
        else:
            prev_scene_index = current_index
        
        # Apply fade transition only for specific scene changes:
        # 1. Select level → Play level
        # 2. Play level → Any menu (pause, success, failed, main)
        apply_fade = False
        
        if current_index == "selectlevelscene" and scene_index == "playscene":
            apply_fade = True
        elif current_index == "playscene" and scene_index in ["pausescene", "levelsuccessscene", "levelfailedscene", "mainscene"]:
            apply_fade = True
        
        if apply_fade:
            # Perform fade transition (0.5 seconds, only affects main thread)
            Window.FadeTransition(duration=0.5)
        
        # Reset level completion events after scene transition is processed
        Window.Event_trigger["LevelSuccess"] = False
        Window.Event_trigger["LevelFailed"] = False
            
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

        if option ==  "selectlevel" :
            Window.Event_trigger["selectlevel"] = True
            print("SELECTED OPTION!")
        if option == "exit": Window.run = False

    if scene_index == "selectlevelscene":
        if option is not None and isinstance(option,str): Window.Event_trigger["Start"] = True

def Music_themes(index):
    
    if index == "mainscene" or index == "selectlevelscene" or index == "levelsuccessscene":
        main_screen_theme.PlayMusic()
    else:
        main_screen_theme.StopMusic()
                
    
    if index == "pausescene" or index == "levelfailedscene":
        menu_theme.PlayMusic()
    else:
        menu_theme.StopMusic()



while(Window.run):

    frame_start = time.perf_counter()

    InitializeFrame()

    get_sceene_index()

    SCENES[scene_index]()

    current_option = TriggerButtons(scene_index)

    Music_themes(scene_index)


    ShowFrame()

    TriggerOptions(current_option)

    frame_end_time = time.perf_counter() - frame_start
    if (frame_end_time < frame_time):
        time.sleep(frame_time - frame_end_time)
    

Window.Destroy()



