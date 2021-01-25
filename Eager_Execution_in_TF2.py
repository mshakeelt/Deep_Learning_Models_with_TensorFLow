import tensorflow as tf
import numpy as np
print(tf.executing_eagerly())

from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
""" disable_eager_execution()
print(tf.executing_eagerly())

import numpy as np
a = tf.constant(np.array([1., 2., 3.]))
print(type(a))

b = tf.constant(np.array([4.,5.,6.]))
c = tf.tensordot(a, b, 1)
print(type(c))

print(c)

session = tf.compat.v1.Session()
output = session.run(c)
session.close()
print(output) """

enable_eager_execution()
print(tf.executing_eagerly())

x = [[4]]
m = tf.matmul(x, x)
print("Result, {}".format(m))

a = tf.constant(np.array([1., 2., 3.]))
print(type(a))

print(a.numpy())

b = tf.constant(np.array([4.,5.,6.]))
c = tf.tensordot(a, b,1)
print(type(c))

print(c.numpy())

def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(1, max_num.numpy()+1):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num.numpy())
    counter += 1

fizzbuzz(15)




