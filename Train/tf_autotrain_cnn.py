import tensorflow as tf
import os
import numpy as np

import sys
sys.path.append('Train_Function/')
sys.path.append('Collect_Prediction/')

from TF_machine import Tensorflow_Machine
from CP_machine import CP_Machine


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

''' Collect Mnist DATA '''
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

results = list()

'''
# model 1
sess = tf.Session()

tm1 = Tensorflow_Machine(sess, "model1",
                         input_file="models_attribute/model1.txt",
                         layer_file="models_layers/model1.txt")

tm1.training_model(training_epochs=1000)

tm1_result, answer = tm1.test_model()

tm1.destory_all()

sess = tf.Session()

# model 2
tm2 = Tensorflow_Machine(sess, "model2",
                         input_file = "models_attribute/model2.txt",
                         layer_file = "models_layers/model2.txt")

tm2.training_model(training_epochs=1000)

tm2_result, _ = tm2.test_model()

tm2.destory_all()

# model 3
sess = tf.InteractiveSession()

tm3 = Tensorflow_Machine(sess, "model3",
                         input_file="models_attribute/model3.txt",
                         layer_file="models_layers/model3.txt")

tm3.training_model(training_epochs=5000)

tm3_result, _ = tm3.test_model()

tm3.destory_all()

results.append(tm1_result)
results.append(tm2_result)
results.append(tm3_result)

num_result = 3
'''

# 9 consultants
for i in range(1, 10):
    model_name = "model" + str(i)
    input_file_name = "models_attribute/model" + str(i) + ".txt"
    layer_file_name = "models_layers/model" + str(i) + ".txt"
    
    if (i % 3 == 0):
        training_count = 5000
    else :
        training_count = 1500

    sess = tf.InteractiveSession()

    current_model = Tensorflow_Machine(sess, model_name, input_file_name, layer_file_name)

    current_model.training_model(training_epochs=training_count)

    result, answer = current_model.test_model()

    results.append(result)

    current_model.destory_all()

num_result = 9
y_size = 1

cpm = CP_Machine(answer,
                 results,
                 y_size,
                 num_result)

cpm.report()
