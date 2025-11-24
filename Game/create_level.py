from window_app import Win
import time
from level_class import LevelClass


LEVEL_NAME = "Level3"
LEVEL_SONG = "CarTrouble.mp3"

#level dificulty 
#when level is very dynamic it requas LEVEL easablility to very small
#it number of additional frames to to "threshould" the level 
#read more at README code

LEVEL_EASEALITY = 10
song_file = f"sound/{LEVEL_SONG}"

Level = LevelClass(song_file.encode("utf-8"),f"levels/{LEVEL_NAME}.json")

FPS = 60
frame_time = 1/FPS
BEAT_INTERVAL = FPS/120

frame = 0
bmp_next_time = 0
Window = Win()
frame_start = 0


def Create_Level(): 

    click = 0
    if Window.Event_trigger["Left"] : click = 1
    if Window.Event_trigger["Up"] : click = 2
    if Window.Event_trigger["Right"] : click = 3
    if Window.Event_trigger["Down"] : click = 4

    Level.SetLevel(click)

Window.run = True



    
while(Window.run):
    frame_start = time.perf_counter()

    Window.Events()
    Window.Render_start()
    if Window.Event_trigger["Start"]:
        print(frame)
        frame += 1
        
    if(frame >= 60):
        Level.PlayMusic()
        Create_Level()

    Window.Render_present()

    Window.Reset_Events()

    frame_end_time = time.perf_counter() - frame_start
    if (frame_end_time < frame_time):
        time.sleep(frame_time - frame_end_time)


Level.SaveLevel(LEVEL_EASEALITY)

#clearing the audio
Level.clean()

Window.Destroy()
