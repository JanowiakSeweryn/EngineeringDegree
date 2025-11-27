from level_class import LevelClass

LEVELS_DICT = {
    #level name  #level songs:
    'Level1' : "remastered1.wav",
    'Level2' : "diddy.mp3",
    'Level3' : "CarTrouble.mp3",
    'Level4' : "CarTrouble.mp3",
}



levels = []
for level_name, level_song in LEVELS_DICT.items():
    song_file = f"sound/{level_song}"
    levels.append(LevelClass(song_file.encode("utf-8"),f"levels/{level_name}.json"))

LEVELS = levels