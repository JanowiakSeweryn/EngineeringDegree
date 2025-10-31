from window_app import Win
import time
from sound import sound_effect

FPS = 60
frame_time = 1/FPS
BEAT_INTERVAL = FPS/120

frame = 0
bmp_next_time = 0
Window = Win()
frame_start = 0

Level_song = sound_effect(b"remastered1.wav")

def Create_Level(): 


    click = 0
    if Window.Right : click = 1
    if Window.Left : click = 2

    Level_song.SetLevel(click)


while(Window.run):
    frame_start = time.perf_counter()
    Window.Events()

    if Window.Start:
        print(frame)
        frame += 1
        if(frame >= 60):
            Level_song.play()
            Create_Level()
    Window.Rendering()



    Window.Reset_Events()

    frame_end_time = time.perf_counter() - frame_start
    if (frame_end_time < frame_time):
        time.sleep(frame_time - frame_end_time)
   
Level_song.SaveLevel()
Level_song.clean()
Window.Destroy()
