import numpy as np

from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam


# Test model on a dev/test set

TRAINING_PATH = "/Users/mjain/Desktop/HeyJoe_data/Training_Data/"

Tx = 5511
n_freq = 101
Ty = 1375

# model = load_model('Models/my_model17_14002a2b_30e_100b_hl_sagem.h5')
model = load_model('Models/my_model17_14002a_30e_100b_hl_sagem.h5')  # This one performs better on dev set

# model.summary()


# Load X and Y hand labeled real world examples unseen by model training
# containing hey joe and other words


X_dev = np.load(TRAINING_PATH + "X_hand_label_train2.npy")
Y_dev = np.load(TRAINING_PATH + "Y_hand_label_train2.npy")


#loss, acc = model.evaluate(X_dev, Y_dev)
#print("Dev set accuracy", acc)  # Decent metric to make sure algorithm is working but not super 
                                # helpful for evaluation because a lot of 0s will be marked correctly
                   
# Calculate precision and recall

Y_p = model.predict(X_dev)
print("Y_predictions shape: ", Y_p.shape)


# Convert to True and False (0s and 1s)
Y_p = (Y_p >= 0.5)

tp = np.sum((Y_p == True) & (Y_dev == 1))
fp = np.sum((Y_p == True) & (Y_dev == 0))

print("True positives: ", tp)
print("False positives: ", fp)

precision = tp/(tp+fp)

fn = np.sum((Y_p == False) & (Y_dev == 1))
print("False Negatives: ", fn)

recall = tp/(tp+fn)

print("Precision: ", precision)
print("Recall: ", recall)


