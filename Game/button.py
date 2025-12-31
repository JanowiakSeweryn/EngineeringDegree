
import sdl2
from rectangle import decoration
from animation import center_rect

class button(decoration):
    def __init__(self, color, x, y, width, height):
        self.selected = False
        self.init_color = color
        super().__init__(color, x, y, width, height)
        self.scale_factor = 1.1
        self.center = False

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
        if self.center:
            self.rect = center_rect(self.rect)
        
        # Create a scaled rect for selected button animation
        render_rect = sdl2.SDL_Rect(self.rect.x, self.rect.y, self.rect.w, self.rect.h)
        if self.selected:
            # Make the button slightly bigger (10% larger)

            new_w = int(self.rect.w * self.scale_factor)
            new_h = int(self.rect.h * self.scale_factor)
            render_rect.x = self.rect.x - (new_w - self.rect.w) // 2
            render_rect.y = self.rect.y - (new_h - self.rect.h) // 2
            render_rect.w = new_w
            render_rect.h = new_h
            
        self.rect2.x, self.rect2.y , self.rect2.w ,self.rect2.h = self.center_rect(render_rect.x, render_rect.y, render_rect.w, render_rect.h, 20)
        
        renderer.fill(self.rect2,self.color)
        if type(self.sprite) is not int:
            renderer.copy(self.sprite.texture,None,render_rect)
    
