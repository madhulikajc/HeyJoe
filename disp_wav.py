import os
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from td_utils import *
import sys

Tx = 5511
n_freq = 101
Ty = 1375

WAV_PATH = "/Users/mjain/Desktop/HeyJoe_data/raw_data_wav/"

#wav_file="../train.wav"
wav_file = WAV_PATH + "dev_set/001/" + sys.argv[1]
#print(sys.argv[1])
#print(sys.argv[2])
#print(sys.argv[3])

# recording = AudioSegmentfrom_wav("aa_test.wav")
# x = graph_spectrogram("aa_test.wav")

rate, data = get_wav_info(wav_file)
nfft = 200 # Length of each window segment
fs = 8000 # Sampling frequencies
noverlap = 120 # Overlap between windows
nchannels = data.ndim
if nchannels == 1:
    pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
elif nchannels == 2:
    pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
#return pxx
#plt.show()
print(pxx.shape)
plt.plot(range(5511), pxx.T)
#plt.plot(range(3500,4000), pxx.T[3500:4000, :])
plt.xlim(int(sys.argv[2]), int(sys.argv[3]))
plt.show()
