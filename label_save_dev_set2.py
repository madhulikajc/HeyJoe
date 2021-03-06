import os
from pydub import AudioSegment
from scipy.io import wavfile

import numpy as np
from td_utils import *
from heyjoe_helper_functions import *

## This file hand labels the training/dev set created from actual speech 
## containing hey joe instructions
## This is the second round of hand labeled dev set - create another 25
## See README file for details on what files were used in these hand labeled dev sets
## This code saves the hand labeled sets as X_hand_label_train2.npy and Y..

WAV_PATH = "/Users/mjain/Desktop/HeyJoe_data/raw_data_wav/dev_set/001/"
TRAINING_PATH = "/Users/mjain/Desktop/HeyJoe_data/Training_Data/"

num_examples = 25 #10 with Hey joe activates and 15 without
Ty = 1375
Tx = 5511
n_freq = 101

Y = np.zeros((num_examples, Ty, 1)) 
X = np.zeros((num_examples, Tx, n_freq))

ten_seconds = 10 * 1000

# Handlabel all training files
filename = WAV_PATH + "18.wav"
x = graph_spectrogram(filename)
X[0] = x.T
Y[0] = (insert_ones(Y[0].T, (5511-50)/5511 * ten_seconds)).T

filename = WAV_PATH + "19.wav"
x = graph_spectrogram(filename)
X[1] = x.T
Y[1] = (insert_ones(Y[1].T, 1950/5511 * ten_seconds)).T

filename = WAV_PATH + "20.wav"
x = graph_spectrogram(filename)
X[2] = x.T
Y[2] = (insert_ones(Y[2].T, 2820/5511 * ten_seconds)).T

filename = WAV_PATH + "21.wav"
x = graph_spectrogram(filename)
X[3] = x.T
Y[3] = (insert_ones(Y[3].T, 2290/5511 * ten_seconds)).T

filename = WAV_PATH + "22.wav"
x = graph_spectrogram(filename)
X[4] = x.T
Y[4] = (insert_ones(Y[4].T, 4360/5511 * ten_seconds)).T

filename = WAV_PATH + "23.wav"
x = graph_spectrogram(filename)
X[5] = x.T
Y[5] = (insert_ones(Y[5].T, 4800/5511 * ten_seconds)).T

filename = WAV_PATH + "24.wav"
x = graph_spectrogram(filename)
X[6] = x.T
Y[6] = (insert_ones(Y[6].T, 2950/5511 * ten_seconds)).T

filename = WAV_PATH + "25.wav"
x = graph_spectrogram(filename)
X[7] = x.T
Y[7] = (insert_ones(Y[7].T, 3350/5511 * ten_seconds)).T

filename = WAV_PATH + "26.wav"
x = graph_spectrogram(filename)
X[8] = x.T
Y[8] = (insert_ones(Y[8].T, 4150/5511 * ten_seconds)).T

filename = WAV_PATH + "27.wav"
x = graph_spectrogram(filename)
X[9] = x.T
Y[9] = (insert_ones(Y[9].T, 475/5511 * ten_seconds)).T



######

WAV_PATH_NEG = "/Users/mjain/Desktop/HeyJoe_data/raw_data_wav/dev_set/000/"


# Hand label the zeros, already initialized to zeros, just save the Xs Use files 31.wav to 45.wav
j=10
for i in range(31, 46):
    filename = WAV_PATH_NEG + str(i) + ".wav"
    x = graph_spectrogram(filename)
    X[j] = x.T
    j = j + 1



np.save(TRAINING_PATH + 'X_hand_label_train2', X)
np.save(TRAINING_PATH + 'Y_hand_label_train2', Y)




