import numpy as np
import tensorflow as tf

LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell

state = (tf.zeros([1,LSTM_CELL_SIZE]),)*2
print(state)

lstm = tf.keras.layers.LSTM(LSTM_CELL_SIZE, return_sequences=True, return_state=True)

lstm.states=state

print(lstm.states)
