from window_app import Win
import time
from level_class import sound_effect


LEVEL_NAME = "Level1"
LEVEL_SONG = "remastered1.wav"


song_file = f"sound/{LEVEL_SONG}"

Level = sound_effect(song_file.encode("utf-8"),f"levels/{LEVEL_NAME}.json")

FPS = 60
frame_time = 1/FPS
BEAT_INTERVAL = FPS/120

frame = 0
bmp_next_time = 0
Window = Win()
frame_start = 0


def Create_Level(): 

    click = 0
    if Window.Left : click = 1
    if Window.Up : click = 2
    if Window.Right : click = 3
    if Window.Down : click = 4

    Level.SetLevel(click)

while(Window.run):
    frame_start = time.perf_counter()
    Window.Events()

    Window.Render_start()
    if Window.Start:
        print(frame)
        frame += 1
        if(frame >= 60):
            Level.Play()
            Create_Level()


    Window.Render_present()

    Window.Reset_Events()

    frame_end_time = time.perf_counter() - frame_start
    if (frame_end_time < frame_time):
        time.sleep(frame_time - frame_end_time)
   
Level.SaveLevel(10)
Level.clean()
Window.Destroy()
