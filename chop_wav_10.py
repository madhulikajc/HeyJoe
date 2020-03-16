import os
from pydub import AudioSegment

## This file creates ten second increments from 2 long recordings I made to test
## the trigger word detection Hey Joe algorithms
## I will then hand label these recordings (in order to mimic real world test data for the dev set)
## This is different from the training set which was auto created by splicing trigger words
## and negative words with backgrounds

WAV_PATH = "/Users/mjain/Desktop/HeyJoe_data/raw_data_wav/dev_set/new/"

ten_seconds = 10 * 1000
for filename in os.listdir(WAV_PATH):
    if filename.endswith("wav"):
        recording = AudioSegment.from_wav(WAV_PATH+filename)
        # chop up the recording into ten second chunks to create the dev set
        duration = recording.duration_seconds
        print(filename, " ", duration)
        for i in range(int(duration // 10)):
            current_chunk = recording[i * ten_seconds : (i+1) * ten_seconds]
            current_chunk.export(WAV_PATH + "dev_ex_" + str(i) + filename, format="wav")



