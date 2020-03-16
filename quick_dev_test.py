import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment

import numpy as np
from td_utils import *

from tensorflow.keras.models import Model, load_model


## Hand labels 1-3 examples quickly to see how well the model is doing, creates X and Y
## Also creates predictions on an example so we can see how well the model is doing without
## precision and recall

num_examples = 1

Tx = 5511   # number of timesteps in the spectrogram of a 10 second wave file
n_freq = 101 # number of frequencies we input from the spectrogram to the model

Ty = 1375

X_dev = np.zeros((num_examples, Tx, n_freq))
Y_dev = np.zeros((num_examples, Ty, 1))

WAV_PATH = "/Users/mjain/Desktop/HeyJoe_data/raw_data_wav/dev_set/001/"


filename3 = WAV_PATH + "16.wav"
#filename1 = "train.wav"

#x = graph_spectrogram(filename1)
#X_dev[0] = x.T
#x = graph_spectrogram(filename2)
#X_dev[0] = x.T
x = graph_spectrogram(filename3)
X_dev[0] = x.T

# Y_dev[0, 412:462, 0] = 1


# 3/10 * Ty = 11111 for 50 spaces
# pick two others and label accordingly

# load model
# test the model on group of handlabeled examples
# print result

model = load_model('Models/my_model17_14002a2b_30e_100b_hl_sagem.h5')


#loss, acc = model.evaluate(X_dev, Y_dev)
#print(acc)

predictions = model.predict(X_dev)
#print(predictions)
print(predictions.shape)
print("200")
print(predictions[0, 200:300, 0])
print("300")
print(predictions[0, 300:400, 0])
print("400")
print(predictions[0, 400:500, 0])
print("500")
print(predictions[0, 500:600, 0])
print("600")
print(predictions[0, 600:700, 0])
print("700")
print(predictions[0, 700:800, 0])
print("800")
print(predictions[0, 800:900, 0])
print("900")
print(predictions[0, 900:1000, 0])
print("1000")
print(predictions[0, 1000:1100, 0])
print("1100")
print(predictions[0, 1100:1200, 0])
print("1200")
print(predictions[0, 1200:1300, 0])




