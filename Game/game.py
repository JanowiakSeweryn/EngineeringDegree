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

level_song = sound_effect(b"remastered1.wav")

level_song.LoadLevel()

next_beat = time.perf_counter() + BEAT_INTERVAL


    

while(Window.run):

    frame_start = time.perf_counter()
    
    Window.Events()
        
    Window.Rendering()

    Window.Reset_Events()


    frame_end_time = time.perf_counter() - frame_start
    if (frame_end_time < frame_time):
        time.sleep(frame_time - frame_end_time)
  
    
    

Window.Destroy()
