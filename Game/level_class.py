import sdl2
import sdl2.sdlmixer as mix
import sys
import json
from rectangle import decoration
import copy

class sound_effect:
    def __init__(self,sound_filename,level_filename):
        

        mix.Mix_OpenAudio(44100,mix.MIX_DEFAULT_FORMAT,2,2048)
        self.sound = mix.Mix_LoadWAV(sound_filename)
        self.sound_channel = 1
        self.start_playing = True
        self.paused_playing = False
        self.Level_pattern = [] 
        self.Level_play = []
        self.level_index = 0
        self.draw_block = False
        self.count = 0
        self.current_block = decoration(0,0,0,1,1)

        color = sdl2.ext.Color(255,0,0)
        self.block_right = decoration(color,800,0,150,150)
        self.block_up = decoration(color,500,0,150,150)
        self.block_down = decoration(color,300,0,150,150)
        self.block_left = decoration(color,100,0,150,150)

        #if player have click the right pattern 
        self.Succeded = False
        self.Failed = False
        
        self.succes_rate = 0
        self.fail_rate = 0


        #mistakes can hapen 1 time per 10 second of the level
        self.fail_threshold = 0
        self.level_failed = False
        self.filename = level_filename

        self.blocs = [] #list of blocs to draw

        if not self.sound:
            print(f"Error, file : {sound_filename} not found!")
            sys.exit()
        pass

    def Loadblocs_png(self,renderer):
        self.block_down.load_png("sprites/down.png",renderer)
        self.block_up.load_png("sprites/up.png",renderer)
        self.block_left.load_png("sprites/left.png",renderer)
        self.block_right.load_png("sprites/right.png",renderer)
        print("PNG INITIALIZED")

    def PlayMusic(self):

        if not self.paused_playing:
            if self.start_playing :
                mix.Mix_PlayChannel(self.sound_channel,self.sound,0)
                self.start_playing = False
        else:
            if self.start_playing:
                mix.Mix_Resume(self.sound_channel)
                self.start_playing = False
            
    
    def PauseMusic(self):
        if not self.start_playing:
            mix.Mix_Pause(self.sound_channel)
            self.start_playing = True
            self.paused_playing = True

    def SetLevel(self,impuls):
        self.Level_pattern.append(impuls) 

    def SaveLevel(self,threshold):
        self.Blocs_pattern = self.Level_pattern

        for i in range(len(self.Level_pattern)):
            if (self.Level_pattern[i] != 0):
                for j in range(threshold):
                    self.Level_pattern[i-j] = self.Level_pattern[i]
                    self.Level_pattern[i-j] = self.Level_pattern[i]
    
        self.Level = {
            "Blocs_pattern" : self.Blocs_pattern,
            "Level_pattern" : self.Level_pattern,
        }

        with open(self.filename,"w") as f:
            json.dump(self.Level,f,indent=4)

    def LoadLevel(self):
        with open(self.filename,"r") as f:
            data = json.load(f)
        self.Level_play = data["Level_pattern"]
        self.fail_threshold = 0.01667*len(self.Level_play)*0.1
    
    def FailedLevel(self):
        if self.fail_rate > self.fail_threshold:
            self.level_failed = True


    def PlayLevel(self,click):
        if self.level_index <  len(self.Level_play):

            if len(self.blocs) != 0:
                self.current_block = max(self.blocs,key=lambda d: d.rect.y)
            
            if click != 0 and self.Level_play[self.level_index] != 0 :
                if self.Level_play[self.level_index] == click:
                    self.Succeded = True

                    print(f'succes {self.succes_rate}')
                    self.succes_rate += 1

                    self.current_block.color = sdl2.ext.Color(0,255,0)
                    self.current_block.checked = True
                    
                else:
                    if self.Level_play[self.level_index+1 == 0]:
                        self.Failed = True
                        self.current_block.color = sdl2.ext.Color(55,0,0)
                        self.current_block.checked = True
                        self.fail_rate += 1
                
            if self.current_block.failed():
                self.Failed = True
                self.Succeded = False
                self.current_block.checked = True
                self.fail_rate+=1
                self.current_block.color = sdl2.ext.Color(55,0,0)
                print(f"FAILED!{self.fail_rate}")
                
            self.level_index += 1

    def Destroy_blocs(self):
        self.blocs = [b for b in self.blocs if not b.reset()]

    def Create_blocs(self):
        if self.level_index+30 < len(self.Level_play):
            if not self.Level_play[self.level_index+30] == 0 and not self.draw_block :
                x = self.get_block_pos(self.Level_play[self.level_index+30])
                self.blocs.append(x)
                self.draw_block = True
            if self.Level_play[self.level_index+30] == 0 :
                self.draw_block = False
 
    def get_block_pos(self,index):
        if index == 1:
            return copy.copy(self.block_left)
        if index == 4:
            return copy.copy(self.block_down)
        if index == 2:
            return  copy.copy(self.block_up)
        if index == 3:
            return  copy.copy(self.block_right)
        
        
    def LoadPng(self,renderer,filename):
        factory = sdl2.ext.SpriteFactory(sdl2.ext.TEXTURE, renderer=renderer)
        self.sprite = factory.from_image(filename)
    
    def disp(self,renderer):
        renderer.copy(self.sprite,None)
        
    def Draw_blocs(self,renderer):
        self.Create_blocs()

        
        self.Failed = False
        self.Succeded = False

        for b in self.blocs:
            b.draw(renderer)

        self.Destroy_blocs()
        
    
    def clean(self):
        mix.Mix_FreeChunk(self.sound)
        mix.Mix_CloseAudio()

        

