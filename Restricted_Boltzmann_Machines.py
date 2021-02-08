import urllib.request
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import tile_raster_images
import matplotlib.pyplot as plt

with urllib.request.urlopen("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0120EN-SkillsNetwork/labs/Week4/data/utils.py") as url:
    response = url.read()
target = open('utils.py', 'w')
target.write(response.decode('utf-8'))
target.close()

#>>>>>>>>>>>>>>>>>>>>RBM Layers<<<<<<<<<<<<<<<<<<<
v_bias = tf.Variable(tf.zeros([7]), tf.float32)     # Bias unit for visible layer
h_bias = tf.Variable(tf.zeros([2]), tf.float32)     # Bias unit for hidden layer

W = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(7, 2)).astype(np.float32))   #Weights between visible and hidden layers

