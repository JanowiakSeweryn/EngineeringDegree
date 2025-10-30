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

chunk = sound_effect(b"bmp.wav")

   
next_beat = time.perf_counter() + BEAT_INTERVAL

def play_bit():
    global frame
    global bmp_next_time
    global next_beat

  
    if frame_start >= next_beat:
        
        next_beat += BEAT_INTERVAL  # schedule next beat
        chunk.play()
    
    # sleep just enough to maintain ~60fp


while(Window.run):
    frame_start = time.perf_counter()
    Window.Events()
    Window.Rendering()

    print(frame)
    play_bit()

    frame_end_time = time.perf_counter() - frame_start
    if (frame_end_time < frame_time):
        # time.sleep(frame_time - frame_end_time)
        time.sleep(max(0, (1/60) - (time.perf_counter() - frame_start)))
    

Window.Destroy()
