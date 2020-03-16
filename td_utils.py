import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


WAV_PATH = "/Users/mjain/Desktop/HeyJoe_data/raw_data_wav/"

# Load raw audio files for speech synthesis
def load_raw_audio():
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir(WAV_PATH + "activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav(WAV_PATH + "activates/"+filename)
            activates.append(activate)
    for filename in os.listdir(WAV_PATH + "backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(WAV_PATH + "backgrounds/"+filename)
            backgrounds.append(background)
    for filename in os.listdir(WAV_PATH + "negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav(WAV_PATH + "negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds
