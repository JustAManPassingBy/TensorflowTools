import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import datetime
import sys
import threading
import time

from tf_functions import get_data_with_float32, print_data, get_raw_data_from_csv, get_raw_data_from_tsv, print_result, print_cost
from tf_trainfunctions import cost_predictor, create_cnn_layer, create_layer, summary_histogram

# Processing
print("Processing")

''' Collect Mnist DATA '''
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

############### INPUT LAYER ###################

''' Variables '''
tf.set_random_seed(764)

my_learning_rate = 1e-5

my_regularization_rate = 0

dropout_ratio = 1.0

# training counts(epochs)
training_epochs = 10000

# number of input datasets for train
dataset_size = 1376

# number of input datasets for test
testdata_size = 126

# number of batches (data counts for training once)
batch_size = 50

# Print / Graph inverval (print cycles)
print_interval = 100
graph_interval = 5
summary_interval = 10

# each input data's node size
input_arraysize = 48

# each output data's node size
output_arraysize = 2

# collecting cost list size (TBD)
cost_list_size = 50


''' files '''
# train file name
train_file="train.txt"

# test file name
test_file="test.txt"

## save & restore variables 
# set "NULL" if don't have it
# Example : savepath='/tmp/model.ckpt' savepate='NULL'
# window " , linux '
#savepath="/tmp/model.ckpt"
savepath = restorepath="/tmp/model.ckpt"
#savepath = restorepath = "NULL"
snapshotmincostpath="/tmp/minmodel.ckpt"

# show costs
showcost = True
showcost_filename = "showcost.txt"

# print all layer
printalllayer = True
# set stdout if you want to get results with standard output
printalllayer_filename = "alllayer.txt"


''' options '''
# snapshot min cost (not need in dropout)
snapshotmincost = False

# printgraph
printgraph = True

# variable learning rate
my_initial_learning_rate=1e-3
decay_steps = 100000
decay_rate = 0.98

# thread
num_thread = 1


''' Layers '''
#direct Bridge
direct_bridge = False

# Layer  input , layer '1' , layer '2'  ...  layer 'k' , output
if (direct_bridge is True) :
    layer_size=[input_arraysize, input_arraysize,86, 72, 32, 13, output_arraysize]
else :
    layer_size=[input_arraysize, 40, output_arraysize]
    #layer_size = list()
    #layer_size.append(input_arraysize)

    #for i in range(input_arraysize - 1, output_arraysize, -1) :
    #    layer_size.append(i)

    #layer_size.append(output_arraysize)

############### PROGRAM LAYER ###################
# 20% Variation create
print(" 20% Done")


''' Check Validity '''
# check direct bridge
if (direct_bridge is True) and (input_arraysize != layer_size[1]) :
    print("Error : In direct bridge, first hidden layer size is same with input layer")
    print("Input : " + str(input_arraysize) + " / Hidden : " + str(layer_size[1]))
    exit

goal_descend_relation = int(cost_list_size / 2)
 
# dropout probability variable
keep_prob = tf.placeholder(tf.float64)

# automatical parts
total_layer= len(layer_size)

if total_layer <= 1 :
    print(total_layer + "Your layer is too small XD")
    exit

#total_batch = int(mnist.train.num_examples / batch_size) // for mnist
total_batch = int(dataset_size / batch_size)

list_for_auto_control = list()


''' Create tensorflow sess, variables, input_data '''
# get session
sess = tf.InteractiveSession()

# Set input, output placeholder
X = tf.placeholder(tf.float64, [None, layer_size[0]])
Y = tf.placeholder(tf.float64, [None, layer_size[total_layer - 1]])

# 40% Validity check
print(" 40% Done")


''' Make & Get array for train data '''
# Make list
Xarr = list()
Yarr = list()

# collect input data
#Xarr, Yarr = get_raw_data_from_csv(Xarr, Yarr, "America_NASDAQ.csv", drop_yarr = False, skipfirstline = True)
Xarr, Yarr = get_raw_data_from_tsv(Xarr, Yarr, train_file, X_size = dataset_size, Y_size = 2, drop_yarr = False, skipfirstline = False)

# create Batches(Slice of train data)
## Normal batch
X_batches, Y_batches = tf.train.batch([Xarr, Yarr], batch_size=batch_size, enqueue_many=True, allow_smaller_final_batch=True)
## Random batch
#num_min = num_thread * dataset_size
#X_batches, Y_batches = tf.train.shuffle_batch([Xarr, Yarr], enqueue_many=True, batch_size=batch_size, capacity = (num_thread + 2) * num_min , min_after_dequeue=(num_min), allow_smaller_final_batch=True)


''' Variable for Dynamically change learning rate '''
#global_step = tf.Variable(0, trainable=False)
#my_learning_rate = tf.train.exponential_decay(my_initial_learning_rate, global_step, decay_steps * total_batch, decay_rate, staircase=True)

# 60% Get layer
print(" 60% Done")


''' Setting Layer '''
# list for tensor
wlist = list()
blist = list()
llist = list()

# set initial next input : X
next_input = X

### First layer - layer 1 ~ layer k - 1 - layer k
# get variable  : get initialize node values with xavier initializer (size is [layer size[i] , layer size[i + 1])
# varaible      : get constant variable
# relu function : Declining function (x > 0 ? x : 0.01x)
# dropout       : Probability that specific node is dropped in learning (Re_estimate for every study)
for i in range(0, total_layer - 2) : 
    next_input, W, B  = create_layer(next_input, layer_size[i], layer_size[i + 1], i, wlist, blist, llist)

_, W, B = create_layer(next_input, layer_size[total_layer - 2], layer_size[total_layer - 1], total_layer - 2, wlist, blist)


''' histogram_summay '''
whist, bhist, _ = summary_histogram(total_layer, wlist, blist, llist)


''' Your hypothesis (X => Layer => Hypothesis) '''
# set hypothesis
# hypothesis [0.9 0.1 0.0 0.0 ...] // O.9 might be an answer
hypothesis = tf.matmul(next_input, W) + B
#hypothesis = tf.nn.relu(tf.matmul(next_input, W) + B)
#hypothesis= tf.sigmoid(tf.matmul(next_input, W) + B)
#hypothesis= tf.nn.tanh(tf.matmul(next_input, W) + B)


''' Add hypothesis histogram '''
hyphist = tf.summary.histogram("hypothesis", hypothesis)


''' cost : Differences between hypothesis and Y '''
# Cost is difference between label & hypothesis(Use softmax for maximize difference
# [0.9 0.1 0.0 0.0 ...] ===(softmax)==> [1 0 0 0 ...] (Soft max is empthsize special answer)
# Square = {for all X in [X], init M = 0 , M += (WX - Y)^2 , finally M /= sizeof([X]) 
# label is true difference
# reduce mean is average of all matrix's elements
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels= Y))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y)) 
cost = tf.reduce_mean(tf.square(hypothesis - Y))


''' Record cost '''
tf.summary.scalar("cost", cost)


''' create output for tensorboard '''
merged = tf.summary.merge_all()


''' Regularization (If want) '''
# reqularization
l2reg = my_regularization_rate * tf.reduce_mean(tf.square(W))


''' Optimizer (Ways for training) '''
# define optimzer : To minimize cost / with learning rate / Use adam optimizer
# https://smist08.wordpress.com/tag/adam-optimizer/ For find more optimizer
# http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html
optimizer = tf.train.AdamOptimizer(learning_rate=my_learning_rate, beta1=0.9, beta2=0.9999, epsilon=1e-9).minimize((cost - l2reg))
#optimizer = tf.train.AdamOptimizer(learning_rate=my_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize((cost - l2reg), global_step=global_step)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=my_learning_rate).minimize(cost - l2reg)

# 80% Create Layer done
print(" 80% Done")


''' Restore Process '''
# saver
saver = tf.train.Saver()
writer = tf.summary.FileWriter("./logs/train_logs", sess.graph)


''' Graph information'''
if (printgraph is True) :
    Xgraph = list()
    Ygraph = list()

# Initialize variables if restore path is null
if restorepath == "NULL" :
    sess.run(tf.global_variables_initializer())
    print("Initialize variables")
# Restore
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


''' Prepare training '''
# coordinate
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

min_cost = float(4294967296)

# 100% : All ready
print("100% Done")


''' Train Model '''
for epoch in range(training_epochs):
    avg_cost = 0

    # Each epoch trains amount of total batch (num_input_data / num_batches) 
    for i in range(0, total_batch) :
        # add info into batch count
        X_batch, Y_batch = sess.run([X_batches, Y_batches])
        #X_batch, Y_batch = mnist.train.next_batch(batch_size) # for mnist

        feed_dict = {X: X_batch, Y: Y_batch, keep_prob: dropout_ratio}
        c, merge_result, _ = sess.run([cost, merged, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    if (avg_cost < min_cost) :
        min_cost = avg_cost

        if (snapshotmincost is True) and (snapshotmincostpath != "NULL") :
            saver.save(sess, snapshotmincostpath)

    # Print & Save cost
    if (epoch % print_interval) == 0 :
        print('Epoch' , '{:7d}'.format(epoch), 'done. Cost :', '{:.9f}'.format(avg_cost))

        if savepath != "NULL" :
            saver.save(sess, savepath)

        saver.save(sess, "tmp/tem_save")

    # Save variables for graph
    if (printgraph is True) and (epoch % graph_interval) == 0 :
        Xgraph.append(epoch)
        Ygraph.append(avg_cost)

    # for summary interval
    if (epoch % summary_interval) == 0 :
        writer.add_summary(merge_result, epoch)

coord.request_stop()
coord.join(threads)

# Learing end
print("Learning Done")


''' Save results if savepath exits '''
if savepath != "NULL" :
    saver.save(sess, savepath)
    print("save done")


''' Get test value '''
Xtest = list()
Ytest = list()

#Xtest, Ytest = get_raw_data_from_csv(Xtest, Ytest, "America_NASDAQ.csv", drop_yarr = True, skipfirstline = True)
Xtest, Ytest = get_raw_data_from_tsv(Xtest, Ytest, test_file, X_size = testdata_size,Y_size = 2, drop_yarr = False, skipfirstline = False)

print (len(Xtest), len(Xtest[0]), len(Ytest[0]), len(Ytest))

''' Test values '''
# Adjust correct prediction (set standards whether train result is same with expects)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
#correct_prediction = tf.equal(hypothesis, Y)
#correct_prediction = tf.square(hypothesis -  Y)

# calculate Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
#accuracy = tf.reduce_mean(correct_prediction)


''' Check result with train session feeded with test values '''
# sess.run([Output formats(hypothesis, Y)], feed_dictionary(see below)
print_accuracy, predict_val, _ = sess.run([accuracy, hypothesis, Y], feed_dict={X : Xtest, Y : Ytest, keep_prob: 1})
#print('Accuracy:', sess.run(accuracy, feed_dict={ X: mnist.test.images, Y: mnist.test.labels}))


''' Print result (TBD) ''' 
# Todo : Synchronize with output
#print(PRINTW.eval())


''' Create output '''
#print("Min value : " + str(min_cost) + " (Save : " + str(snapshotmincost) + ")")
print("Accuracy  : " + str(print_accuracy * 100.0) + "%")
print_result(predict_val, Ytest)
#print_data(predict_val, "test.csv")


''' Print Result '''
if (printalllayer is True) :
    origin_stdout, sys.stdout = sys.stdout, open(printalllayer_filename, "w")
    
    print(" --- W0 --- ")
    print(sess.run([wlist[0]]))

    if direct_bridge is False : 
        print(" --- B0 --- ")
        print(sess.run(blist[0]))
              
    for i in range(1, len(layer_size) - 1) :
        print("\n ====== NEW LAYER ======\n")

        print(" --- W" + str(i) + " --- ")
        print(sess.run(wlist[i]))

        print(" --- B" + str(i) + " --- ")
        if (direct_bridge is True) :
            print(sess.run(blist[i - 1]))   
        else :           
            print(sess.run(blist[i]))
    sys.stdout = origin_stdout


''' Print Cost '''
if (showcost is True) :
    print_cost(Xgraph, Ygraph, showcost_filename) 

    
''' Print Graph (Should be last) '''
if (printgraph is True) :
    plt.plot(Xgraph, Ygraph)
    plt.show()


''' End '''
