import tensorflow as tf
import os

import sys
sys.path.append('Train_Function/')

from TF_machine import Tensorflow_Machine

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

''' Collect Mnist DATA '''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

sess = tf.InteractiveSession()

tm1 = Tensorflow_Machine(sess, "model_mnist",
                         input_file="models_attribute/model_mnist.txt",
                         layer_file="models_layers/model_mnist.txt",
                         mnist=mnist)

tm1.training_model(training_epochs=1000)

tm1.test_model()

sess.close()
