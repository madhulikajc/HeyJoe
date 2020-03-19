import numpy as np
from pydub import AudioSegment

from td_utils import *
from heyjoe_helper_functions import *


# Global Variables

TRAINING_PATH = "/Users/mjain/Desktop/HeyJoe_data/Training_Data/"
RAW_AUDIO_PATH = "/Users/mjain/Desktop/HeyJoe_data/raw_data_wav/"

# the number of time steps input to sequence model from spectrogram
Tx = 5511 

# number of frequencies input to the model at each time step of the spectrogram
n_freq = 101 

# number of time steps in the output of our model
Ty = 1375 


examples_pb = 100 #number of training examples created per background

# Load Audio Segments using pydub
activates, negatives, backgrounds = load_raw_audio(RAW_AUDIO_PATH)
print(len(backgrounds))

def create_training_set():

    num_bg = len(backgrounds)    # number of backgrounds
    num_examples = num_bg * examples_pb  # total number of training examples we will create

    X = np.zeros((num_examples, Tx, n_freq))
    Y = np.zeros((num_examples, Ty, 1))

    for i in range(num_bg):
        for j in range(examples_pb):
            current_example = i * examples_pb + j
            x, y = create_training_example(backgrounds[i], activates, negatives, current_example)
            X[current_example] = x.T
            Y[current_example] = y.T
            print(current_example)

    print(X.shape)
    print(Y.shape)
    np.save(TRAINING_PATH + "X14002c", X)
    np.save(TRAINING_PATH + "Y14002c", Y)

create_training_set()
