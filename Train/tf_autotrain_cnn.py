import tensorflow as tf
import os

import sys
sys.path.append('Train_Function/')

from TF_machine import Tensorflow_Machine

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

''' Collect Mnist DATA '''
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# model 1
sess = tf.InteractiveSession()

tm1 = Tensorflow_Machine(sess, "model1",
                         input_file = "models_attribute/model1.txt",
                         layer_file = "models_layers/model1.txt")

tm1.training_model(training_epochs=100)

print (tm1.test_model())

sess.close()

# model 2
#sess = tf.InteractiveSession()

#tm2 = Tensorflow_Machine(sess, "model2",
#                         input_file = "models_attribute\model2.txt",
#                         layer_file = "models_layers\model2.txt")

#tm2.training_model(training_epochs = 50)

#print (tm2.test_model())

#sess.close()

# model 3
#sess = tf.InteractiveSession()

#tm3 = Tensorflow_Machine(sess, "model3",
#                         input_file = "models_attribute\model3.txt",
#                         layer_file = "models_layers\model3.txt")

#tm3.training_model(training_epochs = 50)

#print (tm3.test_model())

#sess.close()
