import sdl2
import sdl2.ext
import time
from rectangle import decoration
from level_class import LevelClass


WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 1080
BPM = 120

class Win:
    def __init__(self):

        sdl2.ext.init()
        sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO)

        self.frame = 0

        self.window = sdl2.ext.Window("game",size=(WINDOW_WIDTH,WINDOW_HEIGHT))
        self.renderer = sdl2.ext.Renderer(self.window)
        self.events = sdl2.ext.get_events()


        self.window.show() 
        self.rect = sdl2.SDL_Rect(0,750,1000,200)
        self.color_1 = sdl2.ext.Color(255,0,0)
        self.color_2 = sdl2.ext.Color(0,255,0)
        self.iter = 0 

        self.current_color = [self.color_1, self.color_2]

        self.run = True
        self.change_color = False

        self.Event_trigger = {
            "Start" :False,
            "Left" :False,
            "Right" :False,
            "Up": False,
            "Down":False,
            "Pause":False,
            "ClickButton" :False,
            "Return2Main": False ,
            "Resume" : False,
            "selectlevel": False
        }

        self.blocks = []


    def Events(self):
        self.events = sdl2.ext.get_events()

        for event in self.events:
            if(event.type == sdl2.SDL_QUIT):
                self.run = False
            if(event.type == sdl2.SDL_KEYDOWN):

                if event.key.keysym.sym == sdl2.SDLK_RIGHT:
                    self.change_color = True
                    self.Event_trigger["Right"] = True

                if event.key.keysym.sym == sdl2.SDLK_LEFT:
                    self.change_color = True
                    self.Event_trigger["Left"] = True

                if event.key.keysym.sym == sdl2.SDLK_DOWN:
                    self.change_color = True
                    self.Event_trigger["Down"] = True

                if event.key.keysym.sym == sdl2.SDLK_UP:
                    self.change_color = True
                    self.Event_trigger["Up"] = True

                if event.key.keysym.sym == sdl2.SDLK_s:
                    self.Event_trigger["Start"] = True

                if event.key.keysym.sym == sdl2.SDLK_p:
                    self.Event_trigger["Pause"] = True
        
                if event.key.keysym.sym == sdl2.SDLK_SPACE:
                    self.Event_trigger["ClickButton"] = True

                if event.key.keysym.sym == sdl2.SDLK_m:
                    self.Event_trigger["Return2Main"] = True

                if event.key.keysym.sym == sdl2.SDLK_p:
                    self.Event_trigger["selectlevel"] = True

    def Render_start(self):
        
        self.renderer.clear(sdl2.ext.Color(0,0,0))

        if(self.change_color):
            self.iter += 1
            self.iter = self.Relu(self.iter)
            self.change_color = False

        self.renderer.fill([self.rect],sdl2.ext.Color(155,155,155))

    def Render_present(self):
        self.renderer.present()
        self.frame += 1
    

    def Destroy(self):
        self.renderer.destroy()
        self.window.close()

        sdl2.ext.quit()
    
    def Relu(self,number):
        if(number >= len(self.current_color)):
            return 0
        else:
            return number
    
    def Reset_Events(self):

        for ev in self.Event_trigger:
            self.Event_trigger[ev] = False

    


        
    





