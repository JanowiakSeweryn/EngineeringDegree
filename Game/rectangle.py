import sdl2
import sdl2.ext

HEIGH = 1080
BLOCK_SIZE  = 150
BAR_HEIGH = 750 + BLOCK_SIZE
class decoration:
    def __init__(self,color,x,y,width,height):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = sdl2.SDL_Rect(self.x,self.y,self.width,self.height)
        self.speed = 20 # (600 pixels per 30 frames )

        self.checked = False #if player checked 
        self.threshold_check = 800

        pass
    
    def move(self):
        self.rect.y += self.speed
    
    def failed(self):
        if self.rect.y >=  950 and not self.checked : return True
        else: return False
    
    def reset(self):
        if self.rect.y >= HEIGH or (self.checked and self.rect.y > BAR_HEIGH ): return True
        else: return False
        
    def draw(self,renderer):
        renderer.fill(self.rect,self.color)
    
        self.move()
        
