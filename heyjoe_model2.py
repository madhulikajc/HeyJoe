import numpy as np

from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam


## Assumes that heyjoe_model1.py has already been run and Coursera weights were transferred to the
## correct model architecture. This picks up from where that left off, and trains for more iterations



Tx = 5511
n_freq = 101
Ty = 1375

model = load_model('Models/my_model8.h5')
model.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


# Load X and Y constructed examples with heyjoe and other words in them, for training (new examples)
X = np.load("Training_Data/X1400a.npy")
Y = np.load("Training_Data/Y1400a.npy")


model.fit(X, Y, batch_size = 5, epochs=30)


model.save('./Models/my_model9.h5')    # save an iteration, next time load this filename and save another iteration


# Test separately by spot checking - see raw_data/quick_dev_test

#loss, acc = model.evaluate(X_dev, Y_dev)
#print("Dev set accuracy", acc)  # Decent metric to make sure algorithm is working but not super 
                                # helpful for evaluation because a lot of 0s will be marked correctly
                                # Use precision/recall eventually

