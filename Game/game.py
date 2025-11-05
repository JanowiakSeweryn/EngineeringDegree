from window_app import Win
import time
from level_class import sound_effect


SONG ="remastered1.wav"
LEVEL_NAME = "Level1"
FPS = 60
frame_time = 1/FPS
BEAT_INTERVAL = FPS/120

frame = 0
bmp_next_time = 0
Window = Win()

frame_start = 0


song_file = f"sound/{SONG}"

Level = sound_effect(song_file.encode("utf-8"),f"levels/{LEVEL_NAME}.json")
Level.LoadLevel()
Level.LoadPng(Window.renderer,"sprites/of_01.jpeg")

def Play(renderer):
    click = 0
    if Window.Start and not Level.level_failed:
        if Window.Right : click = 1
        if Window.Left : click = 2
        Level.Play()
        Level.PlayLevel(click)
        Level.Draw_blocs(renderer)
        Level.FailedLevel() #check if current fails are enough to fail full level
    if Level.level_failed:
        Level.disp(Window.renderer)



next_beat = time.perf_counter() + BEAT_INTERVAL


while(Window.run):

    frame_start = time.perf_counter()
    
    Window.Events()
        
    Window.Render_start()
    Play(Window.renderer)
    Window.Render_present()
    Window.Reset_Events()


    frame_end_time = time.perf_counter() - frame_start
    if (frame_end_time < frame_time):
        time.sleep(frame_time - frame_end_time)

Window.Destroy()
