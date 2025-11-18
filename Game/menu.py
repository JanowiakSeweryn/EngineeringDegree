import sdl2
import sdl2.sdlmixer as mix
from rectangle import decoration
import copy
from button import button

class menu:
    def __init__(self,w,h):

        self.width = w 
        self.heigh = h
        self.x = 0
        self.y = 0

        self.selected_button = 0

        self.rect = sdl2.SDL_Rect(self.x,self.y,self.width,self.heigh)
        self.color = sdl2.ext.Color(55,55,55)

        color = sdl2.ext.Color(155,0,0,255)
        self.button = button(color,800,0,150,150)
        self.buttons = []

        self.paused = False
        self.switch = False
        self.exe_button = False

        self.option = []
        self.current_option = None

        # #filenames of icons 
        # self.filenames = ["sprites/down.png"]


    #create option list in game_render.py
    #each option have the option name as key to trigger this setting
    #each option must have the same name of option.png or else it won't work

    def CreateOptions(self,option_names,renderer):
        for option in option_names:
            self.option.append(option)
            self.button.load_png(f"sprites/{option}.png",renderer)
            self.buttons.append(copy.copy(self.button))

    def Select_button(self,event_left,event_right,event_exe):
        
        input = 0

        if self.exe_button:
            self.exe_button = False

        for b in self.buttons:

            b.select(self.selected_button)

        
        s_button = [b for b in self.buttons if b.selected]

        input = s_button[0].index

        self.current_option = self.option[input]

        
        if event_exe : self.exe_button = True
        if event_left: self.selected_button -= 1
        if event_right : self.selected_button += 1
        if self.selected_button < 0: self.selected_button = len(self.buttons)-1
        if self.selected_button >= len(self.buttons) : self.selected_button = 0

        # print(self.selected_button)
    def Trigger_button(self):

        if self.exe_button:
            return self.current_option
    
        else:
            return None

    def pause(self,window_key):

        if not self.paused:
            if window_key :
                self.paused = True
        else:
            if window_key:
                self.paused = False


    def Render(self,renderer):
        renderer.fill([self.rect],self.color)
        for b in self.buttons:
            b.draw(renderer)
        
    
        

        





    

