#IDEA:
#create list of function each for rendering selected "view":
#for main screen, gamplay screen, setting screen etc
#we can change between scenes by changing index of the list (in the game.py file) :0

#buttons in scenes:

import sys
sys.path.insert(0,"../")

from gestdetect import gesture_detection
from window_app import Win
from menu import menu
from level_class import sound_effect

from options import MENU_OPTION,MAIN_SCREEN_OPTION

import threading
import cv2
import time
import sdl2

SONG ="remastered1.wav"
LEVEL_NAME = "Level1"
FPS = 60
frame_time = 1/FPS
frame = 0

Window = Win()

song_file = f"sound/{SONG}"

#objects:
Menu = menu(1000,600)
Main_screen = menu(1200,1080)
Level = sound_effect(song_file.encode("utf-8"),f"levels/{LEVEL_NAME}.json")
Clasifier = gesture_detection()

#load sprites buttons 
def InitializeGame():
    Menu.CreateOptions(MENU_OPTION,Window.renderer)
    Main_screen.CreateOptions(MAIN_SCREEN_OPTION,Window.renderer)
    Main_screen.color = sdl2.ext.Color(100,0,100)
    Main_screen.paused = True
    Level.LoadLevel()
    Level.LoadPng(Window.renderer,"sprites/of_01.jpeg")
    Level.Loadblocs_png(Window.renderer)

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

#*******
# threading.Thread(target=gest_detect,daemon=True).start()

#InitializedFrame and ShowFrame is used for all scenes and are const 
def InitializeFrame():
    Window.Events()
    Window.Render_start()

def ShowFrame():
    Window.Render_present()
    Window.Reset_Events()

    if current_frame is not None:
        cv2.imshow("camera",current_frame)
        cv2.waitKey(1)


def PlayScene():

    # InitializeFrame()

    click = 0

    if current_gest == "zero" or current_gest == "kon" : click = 1
    if current_gest == "uk_3" : click = 2
    if current_gest == "german_3" : click = 3
    if current_gest == "fist_closed": click = 4
    
    if Window.Event_trigger["Left"] : click = 1
    if Window.Event_trigger["Up"] : click = 2
    if Window.Event_trigger["Right"] : click = 3
    if Window.Event_trigger["Down"] : click = 4

    Level.PlayMusic()
    Level.PlayLevel(click)
    Level.Draw_blocs(Window.renderer)
    Level.FailedLevel() #check if current fails are enough to fail full level
    # if Level.level_failed:
    #     Level.disp(Window.renderer)
    #     Level.PauseMusic()
    
    ShowFrame()

def PauseSceneRender():
    
    # InitializeFrame()
    Level.PauseMusic()
    Menu.Select_button(Window.Event_trigger["Left"], 
        Window.Event_trigger["Right"],
        Window.Event_trigger["ClickButton"])
    Menu.Render(Window.renderer)
    
    # ShowFrame()

def MainSceneRender():
    # InitializeFrame()
    Level.ResetLevel()
    Level.StopMusic()
    Main_screen.Select_button(Window.Event_trigger["Left"],
        Window.Event_trigger["Right"],
        Window.Event_trigger["ClickButton"])  
    
    Main_screen.Render(Window.renderer)

    # ShowFrame()

def RestartLevel():
    Level.ResetLevel()
    Level.StopMusic()
    

def TriggerButtons(scene):

    inp = 0 #user input
    if scene == "mainscene": inp = Main_screen.Trigger_button()
    if scene == "pausescene": inp = Menu.Trigger_button()

    print(f"inp = {inp}")
    return inp


SCENES = {
        "mainscene": MainSceneRender,
        "playscene": PlayScene,
        "pausescene" : PauseSceneRender,
        }

