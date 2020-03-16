import numpy as np

from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam


## Assumes that heyjoe_model1.py has already been run and Coursera weights were transferred to the
## correct model architecture. This picks up from where that left off, and trains for more iterations
## This is a modification to heyjoe_model2.py and combines a hand labeled real world training set with
## a synthetic data set for a new training round 


Tx = 5511
n_freq = 101
Ty = 1375

model = load_model('Models/my_model17_1400c_30_plushandlabel25.h5')
model.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


# Load X and Y constructed examples with heyjoe and other words in them, for training (new examples)
Xhl = np.load("Training_Data/X_hand_label_train.npy")
Yhl = np.load("Training_Data/Y_hand_label_train.npy")

Xa = np.load("Training_Data/X1400d.npy")
Ya = np.load("Training_Data/Y1400d.npy")

print(Xhl.shape)
#print(Yhl.shape)
print(Xa.shape)

X = np.concatenate((Xhl, Xa))
Y = np.concatenate((Yhl, Ya))

# X<anything>.shape should be [#concatenated_examples, Tx, n_freq] 
# Y<anything>.shape should be [#concatenated_examples, Ty, 1]

#print(X.shape)
#print(Y.shape)


# On my computer minibatch size = 10 works well, but on Amazon AWS Sagemaker GPU instance
# a larger minibatch size of 100 or 200 delivers the real speed improvement

model.fit(X, Y, batch_size = 10, epochs=30, shuffle = True)


model.save('Models/my_model18_1400d_30_plushandlabel25.h5')    # save an iteration, next time load this filename and save another iteration



# Comments only:
# Useful code fragments

# Potentially concatenate in such a way that I drop in the hand labeled data set examples one by one, 
# into the synthetic data set, and save the resulting data set.
# Right now setting shuffle=True is fine, it's pretty fast on
# Amazon Sagemaker instance, but a bit slow on my computer

#X = np.concatenate((Xa[0:50], Xhl[0:1], Xa[50:100], Xhl[1:2], Xa[100:150], Xhl[2:3], Xa[150:200], Xhl[3:4], Xa[200:250], Xhl[4:5], Xa[250:300], Xhl[5:6], Xa[300:350], Xhl[6:7], Xa[350:400], Xhl[7:8], Xa[400:450], Xhl[8:9], Xa[450:500], Xhl[9:10], Xa[500:550], Xhl[10:11],  Xa[550:600], Xhl[11:12], Xa[600:650], Xhl[12:13], Xa[650:700], Xhl[13:14], Xa[700:750], Xhl[14:15], Xa[750:800], Xhl[15:16], Xa[800:850], Xhl[16:17], Xa[850:900], Xhl[17:18], Xa[900:950], Xhl[18:19], Xa[950:1000], Xhl[19:20], Xa[1000:1050], Xhl[20:21], Xa[1050:1100], Xhl[21:22], Xa[1100:1150], Xhl[22:23],  Xa[1150:1200], Xhl[23:24], Xa[1200:1250], Xhl[24:25], Xa[1250:1400]))

#Y = 


#loss, acc = model.evaluate(X_dev, Y_dev)
#print("Dev set accuracy", acc)  # Decent metric to make sure algorithm is working but not super 
                                # helpful for evaluation because a lot of 0s will be marked correctly
                                # Use precision/recall eventually

