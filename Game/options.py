#file with ARRAYS of option:

import sys
sys.path.insert(0,"../")

from get_data import GESTURES

OPTIONS = {
    "return2mainscene" : False,
    "return2playscene": False,
    "pause": False,
    "resume": False,
    "play" : False,
}

MENU_OPTION = [
    "pause",
    "resume",
    "restart",
    "return2home"
]

MAIN_SCREEN_OPTION = [
    "play"
]


gest_iter = {}

for g in GESTURES:
    gest_iter[g] = False

GESTURES_INPUT = gest_iter






