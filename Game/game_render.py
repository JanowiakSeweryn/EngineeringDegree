#IDEA:
#create list of function each for rendering selected "view":
#for main screen, gamplay screen, setting screen etc
#we can change between scenes by changing index of the list (in the game.py file) :0

#buttons in scenes:

import sys
sys.path.insert(0,"../")

from gestdetect import gesture_detection
from window_app import Win, WINDOW_HEIGHT, WINDOW_WIDTH
from menu import menu
from sound import sound_play
from animation import center_rect
from Levels import LEVELS

from options import MENU_OPTION,MAIN_SCREEN_OPTION,GESTURES_INPUT,SELECT_LEVEL_OPTION,OPENING_SCREEN_OPTION,LEVEL_SUCCESS_OPTION,LEVEL_FAILED_OPTION

import cv2
import time
import sdl2
import numpy as np
 
import sdl2.sdlmixer as mix

SONG ="remastered1.wav"
LEVEL_NAME = "Level1"
FPS = 60
frame_time = 1/FPS
frame = 0

Window = Win()

song_file = f"sound/{SONG}"

#scenes:
Menu = menu(WINDOW_WIDTH,WINDOW_HEIGHT)
Main_screen = menu(WINDOW_WIDTH,WINDOW_HEIGHT)
Select_level_scene = menu(WINDOW_WIDTH,WINDOW_HEIGHT)

opening_scene = menu(WINDOW_WIDTH,WINDOW_HEIGHT)
opening_scene.render_option = False

Level_success_scene = menu(WINDOW_WIDTH,WINDOW_HEIGHT)
Level_failed_scene = menu(WINDOW_WIDTH,WINDOW_HEIGHT)

#classifiers
Clasifier = gesture_detection(dynamic=False)

#sound_effects:
butt_chunk = sound_play(2,b"sound/change_button.wav")
main_screen_theme = sound_play(3,b"sound/hand_trouble_main.wav")

main_screen_theme.start_playing = True
menu_theme = sound_play(4,b"sound/pause.wav")
menu_theme.start_playing = True

intro_theme = sound_play(5,b"sound/intro_song.wav")



current_level = 0 

def LoadLevels():
    for lvl in LEVELS:
        lvl.LoadLevel()
        lvl.LoadPng(Window.renderer,"sprites/of_01.jpeg")
        lvl.Loadblocs_png(Window.renderer)

#load sprites buttons 
def InitializeGame():

    sdl2.sdlttf.TTF_Init()
    font = sdl2.sdlttf.TTF_OpenFont(b"fonts/ARIAL.TTF", 72)
    if not font:
        print("Font load error:", sdl2.sdlttf.TTF_GetError())
        sys.exit()

    Menu.CreateOptions(MENU_OPTION,Window.renderer)
    
    Main_screen.button.center = True
    Main_screen.button.rect = center_rect(Main_screen.button.rect)
    Main_screen.CreateOptions(MAIN_SCREEN_OPTION,Window.renderer)
    Main_screen.color = sdl2.ext.Color(100,0,100)
    
    Main_screen.LoadBackgroundPNG("sprites/main_screen.png",Window.renderer)
    Select_level_scene.CreateOptions(SELECT_LEVEL_OPTION,Window.renderer)
    Select_level_scene.color = sdl2.ext.Color(150,150,0)
    Select_level_scene.LoadText(Window.renderer,font)

    opening_scene.LoadBackgroundPNG("sprites/opening_scene.png",Window.renderer)
    opening_scene.CreateOptions(OPENING_SCREEN_OPTION,Window.renderer)
    opening_scene.color = sdl2.ext.Color(255,255,255)

    Level_success_scene.CreateOptions(LEVEL_SUCCESS_OPTION,Window.renderer)
    Level_success_scene.color = sdl2.ext.Color(120,120,120)   
    Level_success_scene.LoadText(Window.renderer,font)

    Level_failed_scene.CreateOptions(LEVEL_FAILED_OPTION,Window.renderer)
    Level_failed_scene.color = sdl2.ext.Color(120,120,120)   
    Level_failed_scene.LoadText(Window.renderer,font)

    LoadLevels()


run = True
current_gest = None
prev_gest = None
prevframe_gest = None
gest = None
current_frame = None
ClickButton = False


current_gest_dynamic = None

#this funcion  run in diffrient 
def gest_detect():
    global current_gest 
    global current_frame
    global prev_gest
    global prevframe_gest
    global ClickButton
    global current_gest_dynamic

    tick_start = 0
    tick_time = 0
    tick_reaction_time = 1.0#in seconds

    iter = 0

    gest_inputs = ["fist_open","left_thumb","right_thumb","uk_3"]

    while(Window.run):

        tick_start = time.perf_counter()

        prev_gest = current_gest

        result = Clasifier.clasify()
        current_gest = result
    
        res_frame = Clasifier.frame
        current_frame = res_frame

        for g in gest_inputs:
            if current_gest == g:
                if prev_gest == current_gest: iter += tick_time
                else: iter = 0
                
                if g == "fist_open" or g == "uk_3": tick_reaction_time = 1.2
                else: tick_reaction_time = 0.8
                if(iter > tick_reaction_time):
                    iter = 0
                    GESTURES_INPUT[g] = True
        
        tick_time = time.perf_counter() - tick_start

        # print(current_gest_dynamic)

def CV_buttons():

    global intro

    # print("deb::")
    # for f in GESTURES_INPUT:
    #     print(f)
    
    if current_gest_dynamic is not None:
        if current_gest_dynamic == "six-seven":
            print("6767676767676767!!!")

    if current_gest is not None:
        if GESTURES_INPUT[current_gest]:
    
            GESTURES_INPUT[current_gest] = False
            if current_gest == "fist_open" : Window.Event_trigger["ClickButton"] = True
            if current_gest == "left_thumb" : Window.Event_trigger["Left"] = True
            if current_gest == "right_thumb" : Window.Event_trigger["Right"] = True
            if current_gest == "uk_3" : Window.Event_trigger["Pause"] = True

            butt_chunk.start_playing = True

    butt_chunk.PlayMusic()


#InitializedFrame and ShowFrame is used for all scenes and are const 
def InitializeFrame():

    CV_buttons()
    Window.Events()
    Window.Render_start()

def ShowFrame():
    Window.Render_present()
    Window.Reset_Events()

#displays window 
    if current_frame is not None:
        rect_coords = Clasifier.detector.draw_hand_rect(current_frame)
        cv2.imshow("camera",current_frame)
        Clasifier.detector.display_text(current_frame, current_gest, rect_coords)

        cv2.waitKey(1)

def SelectLevelScene():
    global current_level

    # InitializeFrame()
    LEVELS[current_level].ResetLevel()
    LEVELS[current_level].StopMusic()

    if not mix.Mix_Playing(3):
        main_screen_theme.start_playing = True

    Select_level_scene.Select_button(Window.Event_trigger["Left"], 
        Window.Event_trigger["Right"],
        Window.Event_trigger["ClickButton"])
    Select_level_scene.Render(Window.renderer)

    current_level = Select_level_scene.option.index(Select_level_scene.current_option)

intro_theme.start_playing = True

def OpeningScene():
    
    if current_gest == "fist_open": 
        opening_scene.animate_intro()
        if not mix.Mix_Playing(5):
            intro_theme.start_playing = True
        intro_theme.PlayMusic()
    else:
        intro_theme.StopMusic()
        opening_scene.intro_rect.y = WINDOW_HEIGHT

    opening_scene.Select_button(Window.Event_trigger["Left"], 
        Window.Event_trigger["Right"],
        Window.Event_trigger["ClickButton"])
    
    opening_scene.Render(Window.renderer)


level_completed = False
def PlayScene():

    global level_completed

    # InitializeFrame()

    click = 0

    if current_gest == "zero" or current_gest == "kon" : click = 1
    if current_gest == "peace" : click = 2
    if current_gest == "german_3" : click = 3
    if current_gest == "fist_closed": click = 4
    
    if Window.Event_trigger["Left"] : click = 1
    if Window.Event_trigger["Up"] : click = 2
    if Window.Event_trigger["Right"] : click = 3
    if Window.Event_trigger["Down"] : click = 4

    LEVELS[current_level].PlayMusic()
    LEVELS[current_level].PlayLevel(click)
    LEVELS[current_level].Draw_blocs(Window.renderer)
    LEVELS[current_level].FailedLevel() #check if current fails are enough to fail full level
    
    # Check for level success or failure
    if LEVELS[current_level].level_failed:
        print(">>> Triggering LevelFailed event in game_render")
        LEVELS[current_level].PauseMusic()
        Window.Event_trigger["LevelFailed"] = True
        LEVELS[current_level].level_failed = False
        
    if LEVELS[current_level].level_succeded:
        print(">>> Triggering LevelSuccess event in game_render")
        LEVELS[current_level].PauseMusic()
        Window.Event_trigger["LevelSuccess"] = True
        LEVELS[current_level].level_succeded = False
    
    ShowFrame()

def PauseSceneRender():
    
    # InitializeFrame()

    if not mix.Mix_Playing(4):
        menu_theme.start_playing = True

    LEVELS[current_level].PauseMusic()

    Menu.Select_button(Window.Event_trigger["Left"], 
        Window.Event_trigger["Right"],
        Window.Event_trigger["ClickButton"])
    Menu.Render(Window.renderer)
    
    # ShowFrame()


def MainSceneRender():

    # InitializeFrame()
    LEVELS[current_level].ResetLevel()
    LEVELS[current_level].StopMusic()

    if not mix.Mix_Playing(3):
        main_screen_theme.start_playing = True

    main_screen_theme.PlayMusic()

    Main_screen.Select_button(Window.Event_trigger["Left"],
        Window.Event_trigger["Right"],
        Window.Event_trigger["ClickButton"])  
    
    Main_screen.Render(Window.renderer)

    # ShowFrame()

def RestartLevel():
    LEVELS[current_level].ResetLevel()
    LEVELS[current_level].StopMusic()

def LevelSuccessScene():
    LEVELS[current_level].StopMusic()
    
    if not mix.Mix_Playing(3):
        main_screen_theme.start_playing = True
    
    main_screen_theme.PlayMusic()
    
    Level_success_scene.Select_button(Window.Event_trigger["Left"],
        Window.Event_trigger["Right"],
        Window.Event_trigger["ClickButton"])
    
    Level_success_scene.Render(Window.renderer)

def LevelFailedScene():
    LEVELS[current_level].StopMusic()
    
    if not mix.Mix_Playing(4):
        menu_theme.start_playing = True
    
    menu_theme.PlayMusic()
    
    Level_failed_scene.Select_button(Window.Event_trigger["Left"],
        Window.Event_trigger["Right"],
        Window.Event_trigger["ClickButton"])
    
    Level_failed_scene.Render(Window.renderer)
    

def TriggerButtons(scene):

    inp = 0 #user input
    if scene == "mainscene": inp = Main_screen.Trigger_button()
    if scene == "pausescene": inp = Menu.Trigger_button()
    if scene == "selectlevelscene" : inp = Select_level_scene.Trigger_button()
    if scene == "openingscene" : inp = opening_scene.Trigger_button()
    if scene == "levelsuccessscene" : inp = Level_success_scene.Trigger_button()
    if scene == "levelfailedscene" : inp = Level_failed_scene.Trigger_button()

    # print(f"inp = {inp}")
    return inp


SCENES = {
        "openingscene" : OpeningScene,
        "mainscene": MainSceneRender,
        "playscene": PlayScene,
        "pausescene" : PauseSceneRender,
        "selectlevelscene" : SelectLevelScene,
        "levelsuccessscene" : LevelSuccessScene,
        "levelfailedscene" : LevelFailedScene,
        }

