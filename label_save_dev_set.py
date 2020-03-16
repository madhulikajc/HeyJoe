import os
from pydub import AudioSegment
from scipy.io import wavfile

import numpy as np
from td_utils import *
from heyjoe_helper_functions import *

## This file hand labels the training/dev set created from actual speech 
## containing hey joe instructions
## This code also saves the hand labeled sets as X_hand_label_train.npy and Y..
## This only does a few of the dev set, as we want to use the remaining in a dev/test set later

WAV_PATH = "/Users/mjain/Desktop/HeyJoe_data/raw_data_wav/dev_set/001/"

num_examples = 25 #10 with Hey joe activates and 15 without
Ty = 1375
Tx = 5511
n_freq = 101

Y = np.zeros((num_examples, Ty, 1)) 
X = np.zeros((num_examples, Tx, n_freq))

ten_seconds = 10 * 1000

# Handlabel all training files
filename = WAV_PATH + "1.wav"
x = graph_spectrogram(filename)
X[0] = x.T
Y[0] = (insert_ones(Y[0].T, (5511-50)/5511 * ten_seconds)).T

filename = WAV_PATH + "2.wav"
x = graph_spectrogram(filename)
X[1] = x.T
Y[1] = (insert_ones(Y[1].T, 1420/5511 * ten_seconds)).T

filename = WAV_PATH + "3.wav"
x = graph_spectrogram(filename)
X[2] = x.T
Y[2] = (insert_ones(Y[2].T, 1600/5511 * ten_seconds)).T

filename = WAV_PATH + "4.wav"
x = graph_spectrogram(filename)
X[3] = x.T
Y[3] = (insert_ones(Y[3].T, 1830/5511 * ten_seconds)).T

filename = WAV_PATH + "5.wav"
x = graph_spectrogram(filename)
X[4] = x.T
Y[4] = (insert_ones(Y[4].T, 56/5511 * ten_seconds)).T
Y[4] = (insert_ones(Y[4].T, 2150/5511 * ten_seconds)).T

filename = WAV_PATH + "6.wav"
x = graph_spectrogram(filename)
X[5] = x.T
Y[5] = (insert_ones(Y[5].T, 3200/5511 * ten_seconds)).T

filename = WAV_PATH + "7.wav"
x = graph_spectrogram(filename)
X[6] = x.T
Y[6] = (insert_ones(Y[6].T, 4230/5511 * ten_seconds)).T

filename = WAV_PATH + "8.wav"
x = graph_spectrogram(filename)
X[7] = x.T
Y[7] = (insert_ones(Y[7].T, 3890/5511 * ten_seconds)).T

filename = WAV_PATH + "9.wav"
x = graph_spectrogram(filename)
X[8] = x.T
Y[8] = (insert_ones(Y[8].T, 3770/5511 * ten_seconds)).T

filename = WAV_PATH + "10.wav"
x = graph_spectrogram(filename)
X[9] = x.T
Y[9] = (insert_ones(Y[9].T, 990/5511 * ten_seconds)).T



######

WAV_PATH_NEG = "/Users/mjain/Desktop/HeyJoe_data/raw_data_wav/dev_set/000/"

# Hand label the zeros, already initialized to zeros, just save the Xs
for i in range(15):
    filename = WAV_PATH_NEG + str(i) + ".wav"
    x = graph_spectrogram(filename)
    X[10+i] = x.T



np.save('X_hand_label_train', X)
np.save('Y_hand_label_train', Y)




