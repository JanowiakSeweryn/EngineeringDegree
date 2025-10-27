import sdl2
import sdl2.ext
import time
from gestdetect import gesture_detection
import threading 
from gestdetect import cv2

FPS = 60
WIDTH = 800
HEIGHT = 600

frame_delay = 1/FPS


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
        time.sleep(0.1)

threading.Thread(target=gest_detect,daemon=True).start()

sdl2.ext.init()

window = sdl2.ext.Window("win",size=(WIDTH,HEIGHT))
renderer = sdl2.ext.Renderer(window)
window.show()

rect = sdl2.SDL_Rect(400,200,200,200)

color_fo = sdl2.ext.Color(255,0,0)
color_ff = sdl2.ext.Color(155,0,0)
color_fc = sdl2.ext.Color(0,0,255)
color_g3 = sdl2.ext.Color(0,255,0)
color_rg = sdl2.ext.Color(255,255,0)

ccolor = color_fo

frame = 0


while(run):
    frame_start = time.perf_counter()

    events = sdl2.ext.get_events()
    for event in events:
        if(event.type == sdl2.SDL_QUIT):
            run = False
    frame +=1

    print(frame)
    
    if frame % 60 == 0:
        frame = 0
        
        if current_gest == "german_3":
            ccolor = color_g3
        if current_gest == "fist_open":
            ccolor = color_fo
        if current_gest == "fist_closed":
            ccolor = color_fc
        if current_gest == "rand_gest":
            ccolor = color_rg
        if current_gest == "fist_flipped":
            ccolor = color_ff

    renderer.clear(sdl2.ext.Color(0,0,0))
    renderer.fill([rect],ccolor)
    renderer.present()

    if current_frame is not None:
        cv2.imshow("camera",current_frame)

    frame_time = time.perf_counter() - frame_start

    if frame_time < frame_delay:
        time.sleep(frame_delay - frame_time)
    else:
        print("to slow !")

Clasifier.destroy_cap()

