import sdl2
import sdl2.sdlmixer as mix
import sys
import json
from rectangle import decoration,BAR_HEIGH
from rectangle import BLOCK_FALL_TIME
import copy
import os


class LevelClass:
    def __init__(self,sound_filename,level_filename):
        

        self.mixer = mix.Mix_OpenAudio(44100,mix.MIX_DEFAULT_FORMAT,2,2048)
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

        self.flash_rect = sdl2.SDL_Rect(0,0,1920,1080)
        self.start_flash = False
        self.flash_frame = 0
    

        #if player have click the right pattern 
        self.Succeded = False
        self.Failed = False
        
        self.succes_rate = 0
        self.fail_rate = 0

        #mistakes can hapen 1 time per 10 second of the level
        self.fail_threshold = 0
        self.level_failed = False
        self.level_succeded = False
        self.filename = level_filename

        self.blocs = [] #list of blocs to draw

        if not self.sound:
            print(f"Error, file : {sound_filename} not found!")
            sys.exit()
        pass

    def Loadblocs_png(self,renderer):
        self.block_down.load_png("sprites/closed_fist.png",renderer)
        self.block_up.load_png("sprites/peace.png",renderer)
        self.block_left.load_png("sprites/zero.png",renderer)
        self.block_right.load_png("sprites/german_3.png",renderer)

        self.CreateProgressBarr(4*150+100)
        self.CreateHealhBar(renderer)

        print("PNG INITIALIZED")

    def Flash(self,renderer,frame):
        flash_time = 20
        alfa = ((flash_time - frame)/flash_time)*155
        flash_color = sdl2.ext.Color(0,255,0,alfa)

        if frame < flash_time:
            renderer.fill([self.flash_rect],flash_color)


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

    def StopMusic(self):
        mix.Mix_Pause(self.sound_channel)
        self.start_playing = True
        self.paused_playing = False

    def SetLevel(self,impuls):
        self.Level_pattern.append(impuls) 

    def SaveLevel(self,threshold):
        self.Blocs_pattern = self.Level_pattern
        click_length = 0 

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
        
        #number of failed blokcs to check to fail the level
        self.fail_threshold = 0.01667*len(self.Level_play)*0.1 + 3
    
    def FailedLevel(self,renderer):
        if self.fail_rate > self.fail_threshold:
            print(f"LEVEL FAILED! fail_rate: {self.fail_rate} > fail_threshold: {self.fail_threshold}")
            self.level_failed = True
            self.fail_rate = 0
            self.CreateHealhBar(renderer)
            


    def PlayLevel(self,click):

        if self.level_index <  len(self.Level_play):

            if len(self.blocs) != 0:
                self.current_block = max(self.blocs,key=lambda d: d.rect.y)
            
            # if click != 0 and self.Level_play[self.level_index] != 0 :
            # Check if block is in valid zone (bottom of block must be at least at y=750)
            block_in_valid_zone = self.current_block.rect.y + self.current_block.rect.h >= 750
            
            if click != 0 and self.Level_play[self.level_index] != 0 and not self.current_block.checked and block_in_valid_zone:
                if self.Level_play[self.level_index] == click :
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
        else:
            # Level has reached the end (all frames played)
            print(f"Level ended! fail_rate: {self.fail_rate}, fail_threshold: {self.fail_threshold}")
            if self.fail_rate < self.fail_threshold:
                #succeded level
                print("LEVEL SUCCEEDED!")
                self.level_succeded = True
    
    def ResetLevel(self):
        self.level_index = 0
        self.blocs.clear()

    def Destroy_blocs(self):
        self.blocs = [b for b in self.blocs if not b.reset()]

    def Create_blocs(self):
        frame_speed=BLOCK_FALL_TIME
        if self.level_index+frame_speed < len(self.Level_play):
            if not self.Level_play[self.level_index+frame_speed] == 0 and not self.draw_block :
                x = self.get_block_pos(self.Level_play[self.level_index+frame_speed])
                self.blocs.append(x)
                self.draw_block = True
            if self.Level_play[self.level_index+frame_speed] == 0 :
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
        if os.path.exists(filename):
            factory = sdl2.ext.SpriteFactory(sdl2.ext.TEXTURE, renderer=renderer)
            self.sprite = factory.from_image(filename)
    
    def disp(self,renderer):
        renderer.copy(self.sprite,None)
        
    def Draw_blocs(self,renderer):
        self.Create_blocs()
        self.ShowHealthBar(renderer)


        if self.Succeded:
            self.start_flash = True

        
        self.Failed = False
        self.Succeded = False

        self.ShowProgressBarr(700,renderer)


        for b in self.blocs:
            b.draw(renderer)

        self.Destroy_blocs()
    
    def CreateProgressBarr(self,progress_width):
        self.bar =decoration(sdl2.ext.Color(155,155,155),150,20,progress_width,20)
        self.progress_bar = decoration(sdl2.ext.Color(0,255,0),150,20,0,20)
    
    def CreateHealhBar(self,renderer):
        self.health_number = round(self.fail_threshold)
        self.health_bar = []
        print(self.health_number)
        for i in range(self.health_number):
            self.health_bar.append(decoration(sdl2.ext.Color(255,0,0),1200,20+100*i,100,100))
            self.health_bar[i].load_png("sprites/health.png",renderer)

    def ShowHealthBar(self,renderer):
        if self.Failed and len(self.health_bar) > 0: self.health_bar.pop()
        for h in self.health_bar:
            h.draw(renderer,move=False,draw_rect=False)
    
    def ShowProgressBarr(self,progress_width,renderer):
        self.bar.draw(renderer,move=False)
        self.progress_bar.rect2.w = int(progress_width*self.level_index/len(self.Level_play))
        self.progress_bar.draw(renderer,move=False)
    
    def clean(self):
        mix.Mix_FreeChunk(self.sound)
        mix.Mix_CloseAudio()

        

