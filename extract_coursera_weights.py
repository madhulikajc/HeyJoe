from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam


model = load_model('./tr_model.h5')  # Coursera model that I wrote the code for  
                                     # trained on a GPU, model for trigger word detection

# Extract weights from this model trained on a GPU and use on a newly created model
# (this is because dropout arguments went from being dropout rate to keep_probs)

model.save_weights('coursera_weights', save_format='tf')


