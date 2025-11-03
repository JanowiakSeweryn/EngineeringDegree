import sdl2
import sdl2.sdlmixer as mix
import sys
import json
from rectangle import decoration

class sound_effect:
    def __init__(self,filename):
        

        mix.Mix_OpenAudio(44100,mix.MIX_DEFAULT_FORMAT,2,2048)
        self.sound = mix.Mix_LoadWAV(filename)
        self.start_playing = True
        self.Level = [] 
        self.Level_play = []
        self.level_index = 0
        self.draw_block = False
        self.count = 0



        self.blocs = [] #list of blocs to draw

        if not self.sound:
            print(f"Error, file : {filename} not found!")
            sys.exit()
        pass

    def play(self):
        if(self.start_playing == True):
            mix.Mix_PlayChannel(-1,self.sound,0)
            self.start_playing = False

    def SetLevel(self,impuls):
        self.Level.append(impuls) 

    def SaveLevel(self):
        for i in range(len(self.Level)):
            if (self.Level[i] != 0):
                for j in range(5):
                    self.Level[i-j] = self.Level[i]
                    self.Level[i-j] = self.Level[i]

        with open("Level.json","w") as f:
            json.dump(self.Level,f,indent=4)

    def LoadLevel(self):
        with open("Level.json","r") as f:
            data = json.load(f)
        self.Level_play = data

    def PlayLevel(self,click):
    
        if self.Level_play[self.level_index] == click and self.Level_play[self.level_index] != 0:
            print("SUCCES")
    
        # if (self.Level_play[self.level_index] != click) : print("failed!!")

        self.level_index += 1
    
    def Create_blocs(self):
        if not self.Level_play[self.level_index+30] == 0:
            if self.count >= 0: self.draw_block = True
            self.count+=1

        if self.draw_block and self.count <= 10:
            print("nighterss!!!!")
            x = self.get_block_pos(self.Level_play[self.level_index])
            color = sdl2.ext.Color(255,0,0)
            self.blocs.append(decoration(color,x,0,200,200))
            self.count == 0
            self.draw_block = False
        
    def Destroy_blocs(self):
        self.blocs = [b for b in self.blocs if not b.reset()]

    def get_block_pos(self,index):
        if index == 1:
            return 100
        else:
            return 800
        
    def Draw_blocs(self,renderer):
        self.Create_blocs()
        for b in self.blocs:
            b.draw(renderer)
        # self.Destroy_blocs()
        
    
    def clean(self):
        mix.Mix_FreeChunk(self.sound)
        mix.Mix_CloseAudio()

        


