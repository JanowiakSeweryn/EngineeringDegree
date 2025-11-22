import sdl2
import sdl2.sdlmixer as mix
from rectangle import decoration
import copy
from button import button
from sdl2 import SDL_Color
from rectangle import get_sprite
from animation import intro_animation
from text import GetTexture
from window_app import WINDOW_HEIGHT, WINDOW_WIDTH



class menu:
    def __init__(self,w,h):

        self.width = w 
        self.heigh = h
        self.x = 0
        self.y = 0

        self.selected_button = 0

        self.rect = sdl2.SDL_Rect(self.x,self.y,self.width,self.heigh)
        self.color = sdl2.ext.Color(55,55,55)

        button_color = sdl2.ext.Color(155,155,155,255)
        self.button = button(button_color,800,0,200,200)
        self.buttons = []

        self.paused = False
        self.switch = False
        self.exe_button = False

        self.option = []
        self.current_option = None
        self.render_option = True #displaays buttons 

        self.background_sprite = None
        #for the textures
        self.render_text = False
        self.text_textures = []

        self.intro_rect = sdl2.SDL_Rect(0,WINDOW_HEIGHT,WINDOW_WIDTH,WINDOW_HEIGHT*3)

    #create option list in game_render.py
    #each option have the option name as key to trigger this setting
    #each option must have the same name of option.png or else it won't work

    def CreateOptions(self,option_names,renderer):
        number_of_option = 0
        for option in option_names:
        
            self.option.append(option)
            self.button.load_png(f"sprites/{option}.png",renderer)
            self.buttons.append(copy.copy(self.button))
            number_of_option += 1

    #positioning buttons
        if number_of_option != 1:
            m = 250 #diff betwen 0 and first button on the left center x times 2
            marg_size = WINDOW_WIDTH - m
            init_x = (WINDOW_WIDTH - marg_size)*0.5

            marg = int((WINDOW_WIDTH-m)/(number_of_option-1)) 

            self.buttons[0].rect.x = int(init_x)

            for i in range(number_of_option) :

                self.buttons[i].rect.x = int((init_x) + i*marg - self.buttons[0].rect.w*0.5)
                self.buttons[i].rect.y = int(WINDOW_HEIGHT*0.5 - self.buttons[i].rect.h*0.5)


    
    def LoadBackgroundPNG(self,filename,renderer):
        self.background_sprite = get_sprite(filename,renderer)

    #load text affter call of the createOptioin, otherwise no text will be renderered:
    def LoadText(self,renderer,font):
        for option in self.option:
            self.text_textures.append(GetTexture(renderer,font,option[:-5],color=SDL_Color(0,0,0)))
        self.render_text = True
    
    def animate_intro(self):
        self.intro_rect = intro_animation(self.intro_rect)
    


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

        if not self.render_option:
            renderer.fill([self.intro_rect],sdl2.ext.Color(0,230,0))
        
        if self.background_sprite is not None:
            renderer.copy(self.background_sprite.texture,None,self.rect)

        if self.render_option:
            for b in self.buttons:
                b.draw(renderer)

        if len(self.text_textures) > 0:
            for index, t in enumerate(self.text_textures):
                sdl2.SDL_RenderCopy(renderer.sdlrenderer,t,None,self.buttons[index].rect)
        
    

        





    

