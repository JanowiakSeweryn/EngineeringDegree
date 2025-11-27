import sdl2
import sdl2.ext
import os

HEIGH = 1080
BLOCK_SIZE  = 150
BAR_HEIGH = 750 + BLOCK_SIZE

#how man pixels per 1 frame to block to get down to 
BLOCK_FALL_TIME = 45
BLOCK_SPEED = int(600/BLOCK_FALL_TIME)
BLOCK_REACT_TIME = 30 #0.5 half second to check the box


class decoration:
    def __init__(self,color,x,y,width,height):

        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = sdl2.SDL_Rect(self.x,self.y,self.width,self.height)
        self.speed = int(BLOCK_SPEED)

        self.checked = False #if player checked 
        self.threshold_check = 800

        self.rect_size = 10
        self.rect2 = sdl2.SDL_Rect(self.x-self.rect_size,self.y-self.rect_size,
                                   self.width+2*self.rect_size,self.height+2*self.rect_size)
        self.sprite = 0
        self.index = 0


        pass
    def __copy__(self):
        new_obj = type(self)(self.color,self.x,self.y,self.width,self.height)  # manually call the constructor (__init__)
        # new_obj.x = self.x
        new_obj.index = self.index
        self.index += 1
        new_obj.sprite = self.sprite
        
        return new_obj


    def load_png(self,filename,renderer):
          #load png:
        if os.path.exists(filename):
            factory = sdl2.ext.SpriteFactory(sdl2.ext.TEXTURE, renderer=renderer)
            self.sprite = factory.from_image(filename)

    def move(self):
        self.rect.y += self.speed
        self.rect2.y += self.speed
    
    def failed(self):
        if self.rect.y >=  BAR_HEIGH+150 and not self.checked : return True
        else: return False
    
    def reset(self):
        # if self.checked: return True
        if self.rect.y >= HEIGH or (self.checked and self.rect.y > BAR_HEIGH ): return True
        else: return False
    
    def center_rect(self, x, y, w, h, size):
        
        return x- size, y-size, w+2*size, h+2*size 

        
    def draw(self,renderer):
        renderer.fill(self.rect2,self.color)
        if type(self.sprite) is not int:
            renderer.copy(self.sprite.texture,None,self.rect)
        self.move()


def get_sprite(filename,renderer):
    
    if os.path.exists(filename):
        factory = sdl2.ext.SpriteFactory(sdl2.ext.TEXTURE, renderer=renderer)
        sprite = factory.from_image(filename)
        return sprite
    
    else: 
        return None
        
