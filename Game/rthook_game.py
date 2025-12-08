import sys
import os

if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)
    os.environ['PYSDL2_DLL_PATH'] = sys._MEIPASS + os.pathsep + os.path.join(sys._MEIPASS, 'sdl2')