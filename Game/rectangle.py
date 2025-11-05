import sdl2
import sdl2.ext

class decoration:
    def __init__(self,color,x,y,width,height):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = sdl2.SDL_Rect(self.x,self.y,self.width,self.height)
        self.speed = 20 # (600 pixels per 30 frames )
        pass
    
    def move(self):
        self.rect.y += self.speed
    
    def reset(self):
        if self.rect.y >= 800 : return True
        else: return False
        
    def draw(self,renderer):
        renderer.fill(self.rect,self.color)
        self.move()
        
