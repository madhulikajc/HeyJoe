import os
from pydub import AudioSegment

WAV_PATH = "/Users/mjain/Desktop/HeyJoe_data/raw_data_wav/"

def trim_bg_10():
    ten_seconds = 10 * 1000
    for filename in os.listdir(WAV_PATH + "backgrounds/new"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(WAV_PATH + "backgrounds/new/"+filename)
            background = background[:ten_seconds]
            background.export(WAV_PATH + "backgrounds_10/"+filename, format="wav")

trim_bg_10()
