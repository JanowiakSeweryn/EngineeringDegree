
import sdl2
import sdl2.sdlmixer as mix
from rectangle import decoration

class button(decoration):
    def __init__(self, color, x, y, width, height):
        self.selected = False
        self.init_color = color
        super().__init__(color, x, y, width, height)

    def __copy__(self):
        return super().__copy__()
    
    def select(self,input_select):
        if input_select == self.index:
            self.selected = True
            self.color = self.init_color + sdl2.ext.Color(100,55,55,100)
        else:
            self.color = self.init_color
            self.selected = False            


    def draw(self, renderer):
        self.rect.x = self.index*300

        self.rect2.x, self.rect2.y , self.rect2.w ,self.rect2.h = self.center_rect(self.rect.x, self.rect.y, self.rect.w, self.rect.h, 20)
        
        renderer.fill(self.rect2,self.color)
        if type(self.sprite) is not int:
            renderer.copy(self.sprite.texture,None,self.rect)
    
