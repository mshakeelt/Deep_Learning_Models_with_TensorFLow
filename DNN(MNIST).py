import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("categorical labels")
print(y_train[0:5])

# make labels one hot encoded
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

print("one hot encoded labels")
print(y_train[0:5])

width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem

x_image_train = tf.reshape(x_train, [-1,28,28,1])  
x_image_train = tf.cast(x_image_train, 'float32') 

x_image_test = tf.reshape(x_test, [-1,28,28,1]) 
x_image_test = tf.cast(x_image_test, 'float32') 

#creating new dataset with reshaped inputs
train_ds2 = tf.data.Dataset.from_tensor_slices((x_image_train, y_train)).batch(50)
test_ds2 = tf.data.Dataset.from_tensor_slices((x_image_test, y_test)).batch(50)

#Change the data size depending upon main memory
#x_image_train = tf.slice(x_image_train,[0,0,0,0],[5000, 28, 28, 1])
#y_train = tf.slice(y_train,[0,0],[5000, 10])

#>>>>>>>>>>>>>>>>>>First layer<<<<<<<<<<<<<<<<<<<<<<

W_conv1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.1, seed=0))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

def convolve1(x):
    return(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

def h_conv1(x): 
    return(tf.nn.relu(convolve1(x)))

def conv1(x):
    return tf.nn.max_pool(h_conv1(x), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#>>>>>>>>>>>>>>>>>>Second layer<<<<<<<<<<<<<<<<<<<<<<

W_conv2 = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

def convolve2(x): 
    return(tf.nn.conv2d(conv1(x), W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

def h_conv2(x):  
    return tf.nn.relu(convolve2(x))

def conv2(x):  
    return(tf.nn.max_pool(h_conv2(x), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))

#>>>>>>>>>>>>>>>>>>Flattening Second layer<<<<<<<<<<<<<<<<<<<<<<

def layer2_matrix(x): 
    return tf.reshape(conv2(x), [-1, 7 * 7 * 64])

#>>>>>>>>>>>>>>>>>>Fully Connected Layer<<<<<<<<<<<<<<<<<<<<<<

W_fc1 = tf.Variable(tf.random.truncated_normal([7 * 7 * 64, 1024], stddev=0.1, seed = 2))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs

def fcl(x): 
    return tf.matmul(layer2_matrix(x), W_fc1) + b_fc1

def h_fc1(x): 
    return tf.nn.relu(fcl(x))

#>>>>>>>>>>>>>>>>>>Dropout Layer<<<<<<<<<<<<<<<<<<<<<<

keep_prob=0.5
def layer_drop(x): 
    return tf.nn.dropout(h_fc1(x), keep_prob)

#>>>>>>>>>>>>>>>>>>last (Fully Connected) Layer<<<<<<<<<<<<<<<<<<<<<<

W_fc2 = tf.Variable(tf.random.truncated_normal([1024, 10], stddev=0.1, seed = 2)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

def fc(x): 
    return tf.matmul(layer_drop(x), W_fc2) + b_fc2

def y_CNN(x): 
    return tf.nn.softmax(fc(x))

#>>>>>>>>>>>>>>>>>>Loss Function<<<<<<<<<<<<<<<<<<<<<<

def cross_entropy(y_label, y_pred):
    return (-tf.reduce_sum(y_label * tf.math.log(y_pred + 1.e-10)))

#>>>>>>>>>>>>>>>>>>Optimizer Function<<<<<<<<<<<<<<<<<<<<<<

optimizer = tf.keras.optimizers.Adam(1e-4)

#>>>>>>>>>>>>>>>>>>Training<<<<<<<<<<<<<<<<<<<<<<

variables = [W_conv1, b_conv1, W_conv2, b_conv2, 
             W_fc1, b_fc1, W_fc2, b_fc2, ]

def train_step(x, y):
    with tf.GradientTape() as tape:
        current_loss = cross_entropy( y, y_CNN( x ))
        grads = tape.gradient( current_loss , variables )
        optimizer.apply_gradients( zip( grads , variables ) )
        return current_loss.numpy()

correct_prediction = tf.equal(tf.argmax(y_CNN(x_image_train), axis=1), tf.argmax(y_train, axis=1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))


loss_values=[]
accuracies = []
epochs = 1

for i in range(epochs):
    j=0
    # each batch has 50 examples
    for x_train_batch, y_train_batch in train_ds2:
        j+=1
        current_loss = train_step(x_train_batch, y_train_batch)
        if j%50==0: #reporting intermittent batch statistics
            correct_prediction = tf.equal(tf.argmax(y_CNN(x_train_batch), axis=1),
                                  tf.argmax(y_train_batch, axis=1))
            #  accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
            print("epoch ", str(i), "batch", str(j), "loss:", str(current_loss),
                     "accuracy", str(accuracy)) 
            
    current_loss = cross_entropy( y_train, y_CNN( x_image_train )).numpy()
    loss_values.append(current_loss)
    correct_prediction = tf.equal(tf.argmax(y_CNN(x_image_train), axis=1),
                                  tf.argmax(y_train, axis=1))
    #  accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
    accuracies.append(accuracy)
    print("end of epoch ", str(i), "loss", str(current_loss), "accuracy", str(accuracy) )  

#>>>>>>>>>>>>>>>>>>Evaluation<<<<<<<<<<<<<<<<<<<<<<

j=0
accuracies=[]
# evaluate accuracy by batch and average...reporting every 100th batch
for x_train_batch, y_train_batch in train_ds2:
        j+=1
        correct_prediction = tf.equal(tf.argmax(y_CNN(x_train_batch), axis=1),
                                  tf.argmax(y_train_batch, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
        accuracies.append(accuracy)
        if j%100==0:
            print("batch", str(j), "accuracy", str(accuracy) ) 
import numpy as np
print("accuracy of entire set", str(np.mean(accuracies)))            
