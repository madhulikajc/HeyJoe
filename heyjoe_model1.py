import numpy as np

from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam



def model(input_shape):
    """
    Function creating the Trigger Word Detection model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    ## Model designed for Trigger Word Detection
    
    X_input = Input(shape = input_shape)
    
    X = Conv1D(filters=256, kernel_size=15, strides=4)(X_input)                   
    X = BatchNormalization()(X)                                

    X = Activation("relu")(X)                               
    X = Dropout(rate=0.2)(X)                              # dropout


    X = GRU(units=128, use_bias=True, return_sequences=True, kernel_initializer='VarianceScaling', recurrent_activation = 'hard_sigmoid', implementation = '2', reset_after=False)(X) 

    X = Dropout(rate=0.2)(X)                            # dropout
    X = BatchNormalization()(X)   
    

    X = GRU(units=128, use_bias=True, return_sequences = True, kernel_initializer='VarianceScaling', recurrent_activation = 'hard_sigmoid', implementation = '2', reset_after=False)(X)  
    X = Dropout(rate=0.2)(X)                                 # dropout
    X = BatchNormalization()(X)                                
    X = Dropout(rate=0.2)(X)                                 # dropout 
    

    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) 

    model = Model(inputs = X_input, outputs = X)
    
    return model  

########

Tx = 5511
n_freq = 101
Ty = 1375

model = model(input_shape = (Tx, n_freq))
model.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

# print("My model get config - last printout was coursera model getconfig")
# print(model.get_config())

# Simply loading the Coursera GPU trained model for transfer learning does not work because
# the keep_prob parameter of Dropout layers changed from keep_prob to dropout rate from version 2
# to version 3. Below lines of code were used to identify the model structure so I could use
# that in my model and use the Coursera weights. Commented out as it is not needed anymore. 

# coursera_model = load_model('./tr_model.h5')
# coursera_model.summary()
# print(coursera_model.outputs)
# print(coursera_model.get_config())


# Load X and Y constructed examples with heyjoe and other words in them, for training
X = np.load("X700.npy")
Y = np.load("Y700.npy")


# Load the weights from the Coursera model trained on the trigger word "activate", transfer learning
model.load_weights('coursera_weights')

model.fit(X, Y, batch_size = 10, epochs=10)


model.save('./my_model1.h5')    # save an iteration, next time load this filename and save another iteration. This model can serve as a basis for loading and training. 




# Load X and Y dev set, also auto-constructed examples for now, use real world examples
# and hand labeled Ys down the road
#X_dev= np.load("X70.npy")
#Y_dev= np.load("Y70.npy")


#loss, acc = model.evaluate(X_dev, Y_dev)
#print("Dev set accuracy", acc)  # Decent metric to make sure algorithm is working but not super 
                                # helpful for evaluation because a lot of 0s will be marked correctly
                                # Use precision/recall
