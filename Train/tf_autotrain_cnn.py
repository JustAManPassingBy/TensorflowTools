import tensorflow as tf
import os

import sys
sys.path.append('Train_Function/')
sys.path.append('Collect_Prediction/')

from TF_machine import Tensorflow_Machine
from CP_machine import CP_Machine


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

''' Collect Mnist DATA '''
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# model 1
sess = tf.Session()

tm1 = Tensorflow_Machine(sess, "model1",
                         input_file="models_attribute/model1.txt",
                         layer_file="models_layers/model1.txt")

tm1.training_model(training_epochs=-1)

tm1_result, answer = tm1.test_model()

tm1.destory_all()

sess = tf.Session()

# model 2
tm2 = Tensorflow_Machine(sess, "model2",
                         input_file = "models_attribute/model2.txt",
                         layer_file = "models_layers/model2.txt")

tm2.training_model(training_epochs=-1)

tm2_result, _ = tm2.test_model()

tm2.destory_all()

# model 3
sess = tf.InteractiveSession()

tm3 = Tensorflow_Machine(sess, "model3",
                         input_file="models_attribute/model3.txt",
                         layer_file="models_layers/model3.txt")

tm3.training_model(training_epochs=-1)

tm3_result, _ = tm3.test_model()

tm3.destory_all()


# Make result
results = list()

results.append(tm1_result)
results.append(tm2_result)
results.append(tm3_result)

num_result = 3

cpm = CP_Machine(answer,
                 results,
                 2,
                 num_result)

cpm.report()
