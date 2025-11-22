import sdl2
import sdl2.sdlmixer as mix
import sys
import json
from rectangle import decoration
import copy
import os

class sound_play:

    def __init__(self,channel,filename):
        self.start_playing = False
        self.paused_playing = False
        self.sound_channel = channel
        self.frame = 0

        mix.Mix_OpenAudio(44100,mix.MIX_DEFAULT_FORMAT,2,2048)

        self.sound = mix.Mix_LoadWAV(filename)

        if not self.sound:
            print(f"Error, file : {filename} not found!")
            sys.exit()


    def PlayMusic(self):


        if self.start_playing :
            mix.Mix_PlayChannel(self.sound_channel,self.sound,0)
            self.start_playing = False

        # if not self.start_playing:
        #     self.frame += 1
        #     if self.


    def PauseMusic(self):
        if not self.start_playing:
            mix.Mix_Pause(self.sound_channel)
            self.start_playing = True
            self.paused_playing = True

    def StopMusic(self):
        mix.Mix_Pause(self.sound_channel)
        self.start_playing = True
        self.paused_playing = False

    def clean(self):
        mix.Mix_FreeChunk(self.sound)
        mix.Mix_CloseAudio()
