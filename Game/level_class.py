import sdl2
import sdl2.sdlmixer as mix
import sys
import json
from rectangle import decoration

class sound_effect:
    def __init__(self,sound_filename,level_filename):
        

        mix.Mix_OpenAudio(44100,mix.MIX_DEFAULT_FORMAT,2,2048)
        self.sound = mix.Mix_LoadWAV(sound_filename)
        self.start_playing = True
        self.Level_pattern = [] 
        self.Level_play = []
        self.level_index = 0
        self.draw_block = False
        self.count = 0

        #if player have click the right pattern 
        self.Succeded = False
        self.Failed = False
        self.succes_rate = 0
        self.filename = level_filename

        self.blocs = [] #list of blocs to draw

        if not self.sound:
            print(f"Error, file : {sound_filename} not found!")
            sys.exit()
        pass

    def Play(self):
        if(self.start_playing == True):
            mix.Mix_PlayChannel(-1,self.sound,0)
            self.start_playing = False

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

    def PlayLevel(self,click):
        if click != 0 and self.Level_play[self.level_index] != 0 :
            if self.Level_play[self.level_index] == click:
                self.Succeded = True
                print(f'succes {self.succes_rate}')
                self.succes_rate += 1
        
            if (self.Level_play[self.level_index] != click ):
                self.Failed = True

        self.level_index += 1
    
    def Create_blocs(self):
        if self.level_index+30 < len(self.Level_play):
            if not self.Level_play[self.level_index+30] == 0 and not self.draw_block :
                # print("nighterss!!!!")
                x = self.get_block_pos(self.Level_play[self.level_index+30])
                color = sdl2.ext.Color(255,0,0) 
                self.blocs.append(decoration(color,x,0,150,150))
                self.draw_block = True
            if self.Level_play[self.level_index+30] == 0 :
                self.draw_block = False
        
    def Destroy_blocs(self):
        self.blocs = [b for b in self.blocs if not b.reset()]

    def get_block_pos(self,index):
        if index == 1:
            return 800
        if index == 2:
            return 100
        
    def Draw_blocs(self,renderer):
        self.Create_blocs()
        if len(self.blocs) != 0:
            if self.Succeded:
                current_block = max(self.blocs,key=lambda d: d.rect.y)
                current_block.color = sdl2.ext.Color(0,255,0)
            if self.Failed:
                current_block = max(self.blocs,key=lambda d: d.rect.y)
                current_block.color = sdl2.ext.Color(55,0,0)
        
        self.Failed = False
        self.Succeded = False
        for b in self.blocs:
            b.draw(renderer)
        self.Destroy_blocs()
        
    
    def clean(self):
        mix.Mix_FreeChunk(self.sound)
        mix.Mix_CloseAudio()

        


