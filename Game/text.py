import sdl2

def GetTexture(renderer,font,text_message,color):

    message = f"{text_message}"
    text_surface = sdl2.sdlttf.TTF_RenderText_Solid(
        font,
        message.encode("utf-8"),
        color  
    )


    text_texture = sdl2.SDL_CreateTextureFromSurface(renderer.sdlrenderer, text_surface)
    sdl2.SDL_FreeSurface(text_surface)

    return text_texture