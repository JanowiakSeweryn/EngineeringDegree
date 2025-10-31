import sdl2
import sdl2.ext
import time

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 1080
BPM = 120
FPB = 3600/BPM # frames per bit (game runs at 60 frames per second )

class Win:
    def __init__(self):

        sdl2.ext.init()
        sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO)

        self.frame = 0

        self.window = sdl2.ext.Window("game",size=(WINDOW_WIDTH,WINDOW_HEIGHT))
        self.renderer = sdl2.ext.Renderer(self.window)
        self.events = sdl2.ext.get_events()


        self.window.show() 
        self.rect = sdl2.SDL_Rect(0,0,200,200)
        self.color_1 = sdl2.ext.Color(255,0,0)
        self.color_2 = sdl2.ext.Color(0,255,0)
        self.iter = 0

        self.current_color = [self.color_1, self.color_2]

        self.run = True
        self.change_color = False

        self.Start = False
        self.Left = False
        self.Right = False

    def Events(self):
        self.events = sdl2.ext.get_events()

        for event in self.events:
            if(event.type == sdl2.SDL_QUIT):
                self.run = False
            if(event.type == sdl2.SDL_KEYDOWN):
                if event.key.keysym.sym == sdl2.SDLK_RIGHT:
                    self.change_color = True
                    self.Right = True
                if event.key.keysym.sym == sdl2.SDLK_LEFT:
                    self.change_color = True
                    self.Left = True
                if event.key.keysym.sym == sdl2.SDLK_s:
                    self.Start = True
                
        
    def Rendering(self):
        
        self.renderer.clear(sdl2.ext.Color(0,0,0))

        if(self.change_color):
            self.iter += 1
            self.iter = self.Relu(self.iter)
            self.change_color = False

        self.renderer.fill([self.rect],self.current_color[self.iter])

        self.renderer.present()

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
        self.Right = False
        self.Left = False
    


        
    





