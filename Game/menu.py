import sdl2
import sdl2.sdlmixer as mix
from rectangle import decoration
import copy
from button import button

class menu:
    def __init__(self,w,h):
        self.width = w 
        self.heigh = h
        self.x = 100
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

        self.option = [0,1,2]
        self.current_option = 0

        # #filenames of icons 
        # self.filenames = ["sprites/down.png"]

    def create_buttons(self,renderer):
        for i in range(3):
            self.button.load_png("sprites/down.png",renderer)
            self.buttons.append(copy.copy(self.button))
        
        for i in range(3):
            print(self.buttons[i].index)

    def Select_button(self,event_left,event_right,event_exe):
        
        input = 0

        if self.exe_button:
            self.exe_button = False


        for b in self.buttons:

            b.select(self.selected_button)

            if b.selected: break
            input+=1


        self.current_option = self.option[input]
              
        
        if event_exe : self.exe_button = True
        if event_left: self.selected_button -= 1
        if event_right : self.selected_button += 1
        if self.selected_button < 0: self.selected_button = len(self.buttons)-1
        if self.selected_button >= len(self.buttons) : self.selected_button = 0

        print(self.selected_button)


            
    def pause(self,window_key):

        if not self.paused:
            if window_key :
                self.paused = True
        else:
            if window_key:
                self.paused = False
        

    def Render(self,renderer):
        if self.paused:
            renderer.fill([self.rect],self.color)
            for b in self.buttons:
                b.draw(renderer)
            
    
        

        





    

