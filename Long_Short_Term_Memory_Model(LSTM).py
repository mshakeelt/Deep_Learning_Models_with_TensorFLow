import numpy as np
import tensorflow as tf

LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell

state = (tf.zeros([1,LSTM_CELL_SIZE]),)*2
print(state)

lstm = tf.keras.layers.LSTM(LSTM_CELL_SIZE, return_sequences=True, return_state=True)

lstm.states=state

print(lstm.states)

#Batch size x time steps x features.
sample_input = tf.constant([[3,2,2,2,2,2]],dtype=tf.float32)

batch_size = 1
sentence_max_length = 1
n_features = 6

new_shape = (batch_size, sentence_max_length, n_features)

inputs = tf.constant(np.reshape(sample_input, new_shape), dtype = tf.float32)

output, final_memory_state, final_carry_state = lstm(inputs)

print('Output : ', tf.shape(output))

print('Memory : ',tf.shape(final_memory_state))

print('Carry state : ',tf.shape(final_carry_state))