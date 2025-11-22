from window_app import WINDOW_HEIGHT, WINDOW_WIDTH
import sdl2


def center_rect(rect):

    rect.x = int(0.5*(WINDOW_HEIGHT - float(rect.w)))
    rect.y = int(0.5*(WINDOW_HEIGHT - rect.h))

    # rect.x = 0
    # rect.y = 100

    # result = sdl2.SDL_Rect(rect.x,rect.y,rect.w,rect.h)

    return rect

def intro_animation(rect):

    speed = int(WINDOW_HEIGHT/75)
    rect.y -= speed

    return rect


