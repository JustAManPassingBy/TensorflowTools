import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import datetime
import threading
import time
from collections import namedtuple

from tf_functions import get_data_with_float32, print_data, get_raw_data_from_csv, get_raw_data_from_tsv, print_result, print_cost, print_all_layer_function
from tf_trainfunctions import Tensorflow_Machine


''' Collect Mnist DATA '''
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# get session
sess = tf.InteractiveSession()

tm = Tensorflow_Machine(sess, "test_model")

tm.training_model(training_epochs = 500)

print(tm.test_model())
