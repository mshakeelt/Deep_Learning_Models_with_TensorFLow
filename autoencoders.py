import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#>>>>>>>>>>>>>>>>>>>>>>MINST dataloading<<<<<<<<<<<<<<<<<<<<<<<
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

y_train = y_train.astype('float32') / 255.
y_test = y_test.astype('float32') / 255.

x_image_train = tf.reshape(x_train, [-1,28,28,1])  
x_image_train = tf.cast(x_image_train, 'float32') 

x_image_test = tf.reshape(x_test, [-1,28,28,1]) 
x_image_test = tf.cast(x_image_test, 'float32') 

print(x_train.shape)

flatten_layer = tf.keras.layers.Flatten()
x_train = flatten_layer(x_train)

print(x_train.shape)

#>>>>>>>>>>>>>>>>>>>model parameters initialization<<<<<<<<<<<<<<
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10
global_step = tf.Variable(0)
total_batch = int(len(x_train) / batch_size)

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
encoding_layer = 32 # final encoding bottleneck features
n_input = 784 # MNIST data input (img shape: 28*28)

