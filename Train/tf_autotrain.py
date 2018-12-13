import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import datetime

from tf_functions import cost_predictor, get_data_with_float32, print_data, get_raw_data_from_csv

# Collect mnist data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

## todo standardization (data standard)
## overfitting : Regularization

## print(Array.ndim) == print rank
## print(Array.shape) == print shape


######################################################
######## NEEDS TO CONSIDER VARIABLES
# set random seed
tf.set_random_seed(2416)

# parameter
my_learning_rate = 1e-4
my_regularization_rate = 0
training_epochs = 1000
dataset_size = 8
testdata_size = 8
batch_size = 4
print_interval = 100

input_arraysize = 6
output_arraysize = 1

#direct Bridge
direct_bridge = True

# Layer  input , layer '1' , layer '2'  ...  layer 'k' , output
# layer size must not less than 2 (input / output)
layer_size=[input_arraysize, input_arraysize, 16, output_arraysize]

# s323ave & restore variables
# set "NULL" if don't have it
# Example : savepath="/tmp/model.ckpt"
# Example : savepath="NULL"
savepath="NULL"
restorepath="NULL"

# dropout ratio
dropout_ratio = 0.6

# collecting cost list size
cost_list_size = 50

goal_descend_relation = int(cost_list_size / 2)

######## END OF CONSIDER VARIABLES
######################################################
# check direct bridge
if (direct_bridge is True) and (input_arraysize != layer_size[1]) :
    print("Error : In direct bridge, first hidden layer size is same with input layer")
    print("Input : " + str(input_arraysize) + " / Hidden : " + str(layer_size[1]))
    exit
 
# dropout probability variable
keep_prob = tf.placeholder(tf.float32)

# automatical parts
total_layer= len(layer_size)

if total_layer <= 1 :
    print(total_layer + "Your layer is too small XD")
    exit

#total_batch = int(mnist.train.num_examples / batch_size)
total_batch = int(dataset_size / batch_size)

list_for_auto_control = list()

# Set input, output placeholder
# [NONE , 784] means  in 784 variables for one dimension can be of any size(dimensions)
X = tf.placeholder(tf.float32, [None, layer_size[0]])
Y = tf.placeholder(tf.float32, [None, layer_size[total_layer - 1]])

### Input
### First layer - layer 1 ~ layer k - 1 - layer k
# get variable : get initialize node values with xavier initializer (size is [layer size[i] , layer size[i + 1])
# varaible     : get constant variable
# relu         : Declining function (x > 0 ? x : 0.01x)
# dropout      : Probability that specific node is dropped in learning (Re_estimate for every study)
for i in range(0, total_layer - 1) :
    if (direct_bridge is False) or (i != 0) :
        # Set Weight , Bias
        W = tf.get_variable(('W'+str(i)), shape=[layer_size[i], layer_size[i + 1]],
                           initializer=tf.contrib.layers.xavier_initializer())
        B = tf.Variable(tf.random_normal([layer_size[i + 1]]), name=('B'+str(i)))
    else :
        # Set Weight, Bias for direct bridge
        W = tf.eye(layer_size[i], name=('W0'))

        #B = tf.constant(0.1, shape=[layer_size[i + 1]])


    # Layer Result
    # case of input node
    # Todo : how can i get save value in tf.nn
    if (i == 0) :
        if (direct_bridge is False) :
            L = tf.nn.relu(tf.matmul(X, W) + B)
            L = tf.nn.dropout(L, keep_prob=dropout_ratio)
        else :
            L = tf.nn.relu(tf.matmul(X, W))   
    # case of hidden nodes
    elif (i != total_layer - 2) :
        L = tf.nn.relu(tf.matmul(PREVL, W) + B)

        L = tf.nn.dropout(L, keep_prob=dropout_ratio)

    PREVL = L

# set hypothesis
# hypothesis [0.9 0.1 0.0 0.0 ...] // O.9 might be an answer
#hypothesis = tf.matmul(L, W) + B
hypothesis = tf.sigmoid(tf.matmul(L, W) + B)

# Cost is difference between label & hypothesis(Use softmax for maximize difference
# [0.9 0.1 0.0 0.0 ...] ===(softmax)==> [1 0 0 0 ...] (Soft max is empthsize special answer)
# Square = {for all X in [X], init M = 0 , M += (WX - Y)^2 , finally M /= sizeof([X]) 
# label is true difference
# reduce mean is average of all matrix's elements
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# reqularization
l2reg = my_regularization_rate * tf.reduce_mean(tf.square(W))

# define optimzer : To minimize cost / with learning rate / Use adam optimizer
# https://smist08.wordpress.com/tag/adam-optimizer/ For find more optimizer
#optimizer = tf.train.AdamOptimizer(learning_rate=my_learning_rate).minimize(cost - l2reg)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=my_learning_rate).minimize(cost - l2reg)

# saver
saver = tf.train.Saver()

# get session
sess = tf.InteractiveSession()

# if restore path exist
if restorepath == "NULL" :
    sess.run(tf.global_variables_initializer())
    print("Initialize variables")
else :
    try :
        saver.restore(sess, restorepath)
    except ValueError :
        print("Invaild path : ", restorepath, " :: Initialize path")
        sess.run(tf.global_variables_initializer())
        print("Initialize variables")
    except:
        print("Fail to restore from previous checkpoint")
        print("It might be happened in shrinking or expanding layers")
        isyes = input("type [init] if you want to initilize all processes : ")
        if (isyes.lower() == "init") :
            sess.run(tf.global_variables_initializer())
            print("Initialize variables")
        else :
            exit
    else :
        print ("restore done")

# collect input data
Xarr = list()
Yarr = list()
Xarr, Yarr = get_raw_data_from_csv(Xarr, Yarr, "America_NASDAQ.csv", drop_yarr = False, skipfirstline = True)

# batch
X_batches, Y_batches = tf.train.batch([Xarr, Yarr], batch_size=batch_size, enqueue_many=True)

# coordinate
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Train model
for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(0, total_batch) :
        # add info into batch count
        X_batch, Y_batch = sess.run([X_batches, Y_batches])

        feed_dict = {X: X_batch, Y: Y_batch, keep_prob: dropout_ratio}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    if (epoch % print_interval) == 0 :
        print('Epoch' , '{:6d}'.format(epoch), 'done. Cost :', '{:.9f}'.format(avg_cost))
        #tf.Print(hypothesis, [hypothesis])

        if savepath != "NULL" :
            saver.save(sess, savepath)
    
coord.request_stop()
coord.join(threads)

print("Learning Done")

# if save path exist
if savepath != "NULL" :
    saver.save(sess, savepath)
    print("save done")


# get test value
Xtest = list()
Ytest = list()

Xtest, Ytest = get_raw_data_from_csv(Xtest, Ytest, "America_NASDAQ.csv", drop_yarr = True, skipfirstline = True)

predict_val, _ = sess.run([hypothesis, Y], feed_dict={X : Xtest, Y : Ytest, keep_prob: 1})

print_data(predict_val, "test.csv")

