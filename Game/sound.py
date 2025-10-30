import sdl2.sdlmixer as mix
import sys

class sound_effect:
    def __init__(self,filename):
        

        mix.Mix_OpenAudio(44100,mix.MIX_DEFAULT_FORMAT,2,2048)
        self.sound = mix.Mix_LoadWAV(filename)

        if not self.sound:
            print(f"Error, file : {filename} not found!")
            sys.exit()
        pass

    def play(self):
        mix.Mix_PlayChannel(-1,self.sound,0)
    
    def clean(self):
        mix.Mix_FreeChunk(self.sound)
        mix.Mix_CloseAudio()

