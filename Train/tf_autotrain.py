import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import datetime
import sys

from tf_functions import cost_predictor, get_data_with_float32, print_data, get_raw_data_from_csv, get_raw_data_from_tsv, print_result, print_cost

''' For Mnist Data '''
# Collect mnist data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


## print(Array.ndim) == print rank
## print(Array.shape) == print shape


''' User Input '''
######## NEEDS TO CONSIDER VARIABLES
# set random seed
tf.set_random_seed(665)

# parameter
#my_learning_rate = 1e-1
my_regularization_rate = 0
training_epochs = 500000
dataset_size = 1516
testdata_size = 121
batch_size = 50
print_interval = 10
graph_interval = 5

# input / output size
input_arraysize = 29
output_arraysize = 2

# dropout ratio
dropout_ratio = 1.0

# collecting cost list size
cost_list_size = 50
goal_descend_relation = int(cost_list_size / 2)

# snapshot min cost (not need in dropout)
snapshotmincost = False

# printgraph
printgraph = True

# show costs
showcost = True
showcost_filename = "showcost.txt"

# print all layer
printalllayer = False
# set stdout if you want to get results with standard output
printalllayer_filename = "alllayer.txt"

# variable learning rate
my_initial_learning_rate=1e-4
decay_steps = 100000
decay_rate = 3

''' Layers '''
#direct Bridge
direct_bridge = False

# Layer  input , layer '1' , layer '2'  ...  layer 'k' , output
if (direct_bridge is True) :
    layer_size=[input_arraysize, input_arraysize, 64, 16, 3, output_arraysize]
else : 
    layer_size=[input_arraysize, 64, 128, 256, 256, 64, 16, output_arraysize]

''' save & restore variables '''
# set "NULL" if don't have it
# Example : savepath='/tmp/model.ckpt' savepate='NULL'
# window " , linux '
#savepath="/tmp/model.ckpt"
savepath = restorepath="/tmp/model.ckpt"
#savepath = restorepath = "NULL"
snapshotmincostpath="/tmp/minmodel.ckpt"

######## END OF CONSIDER VARIABLES
######################################################
''' Check Validity '''
# check direct bridge
if (direct_bridge is True) and (input_arraysize != layer_size[1]) :
    print("Error : In direct bridge, first hidden layer size is same with input layer")
    print("Input : " + str(input_arraysize) + " / Hidden : " + str(layer_size[1]))
    exit
 
# dropout probability variable
keep_prob = tf.placeholder(tf.float64)

# automatical parts
total_layer= len(layer_size)

if total_layer <= 1 :
    print(total_layer + "Your layer is too small XD")
    exit

#total_batch = int(mnist.train.num_examples / batch_size)
total_batch = int(dataset_size / batch_size)

if (dataset_size % batch_size) != 0 :
    total_batch += 1

list_for_auto_control = list()

# get session
sess = tf.InteractiveSession()

# Set input, output placeholder
X = tf.placeholder(tf.float64, [None, layer_size[0]])
Y = tf.placeholder(tf.float64, [None, layer_size[total_layer - 1]])

''' Make & Get array for train data '''
# collect input data
Xarr = list()
Yarr = list()
#Xarr, Yarr = get_raw_data_from_csv(Xarr, Yarr, "America_NASDAQ.csv", drop_yarr = False, skipfirstline = True)
Xarr, Yarr = get_raw_data_from_tsv(Xarr, Yarr, "train.txt", X_size = dataset_size, Y_size = 2, drop_yarr = False, skipfirstline = False)

# From Input data, create Batch(Slice of train data)
# Todo make random
X_batches, Y_batches = tf.train.batch([Xarr, Yarr], batch_size=batch_size, enqueue_many=True, allow_smaller_final_batch=True)

''' Variable for learning rate '''
global_step = tf.Variable(0, trainable=False)
my_learning_rate = tf.train.exponential_decay(my_initial_learning_rate, global_step, decay_steps * total_batch, decay_rate, staircase=True)

''' Setting Layer '''
# list for tensor
wlist = list()
blist = list()

### First layer - layer 1 ~ layer k - 1 - layer k
# get variable  : get initialize node values with xavier initializer (size is [layer size[i] , layer size[i + 1])
# varaible      : get constant variable
# relu function : Declining function (x > 0 ? x : 0.01x)
# dropout       : Probability that specific node is dropped in learning (Re_estimate for every study)
for i in range(0, total_layer - 1) :
    ## Weight / Bias
    if (direct_bridge is False) or (i != 0) :
        W = tf.get_variable(('W'+str(i)), shape=[layer_size[i], layer_size[i + 1]],
                           initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        B = tf.Variable(tf.random_normal([layer_size[i + 1]], dtype=tf.float64), name=('B'+str(i)))
    else :
        W = tf.Variable(tf.convert_to_tensor(np.eye(layer_size[i], dtype=np.float64)), name='W0')

    ## Layer Result
    # First layer : Get input
    if (i == 0) :
        if (direct_bridge is False) :
            #L = tf.matmul(X, W) + B
            L = tf.nn.relu(tf.matmul(X, W) + B)
            L = tf.nn.dropout(L, keep_prob=dropout_ratio)
        else :
            L = tf.nn.relu(tf.matmul(X, W))
            #L = tf.matmul(X, W)
    # Else : Get previous hidden
    elif (i != total_layer - 2) :
        L = tf.nn.relu(tf.matmul(PREVL, W) + B)
        #L = tf.matmul(PREVL, W) + B
        L = tf.nn.dropout(L, keep_prob=dropout_ratio)

    wlist.append(W)
    if (direct_bridge is False) or (i != 0) :
        blist.append(B)

    PREVL = L

''' Your hypothesis (X => Layer => Hypothesis) '''
# set hypothesis
# hypothesis [0.9 0.1 0.0 0.0 ...] // O.9 might be an answer
#hypothesis = tf.matmul(L, W) + B
#hypothesis = tf.nn.relu(tf.matmul(L, W) + B)
#hypothesis= tf.sigmoid(tf.matmul(L, W) + B)
hypothesis= tf.nn.tanh(tf.matmul(L, W) + B)

''' cost : For adjust learning flow '''
# Cost is difference between label & hypothesis(Use softmax for maximize difference
# [0.9 0.1 0.0 0.0 ...] ===(softmax)==> [1 0 0 0 ...] (Soft max is empthsize special answer)
# Square = {for all X in [X], init M = 0 , M += (WX - Y)^2 , finally M /= sizeof([X]) 
# label is true difference
# reduce mean is average of all matrix's elements
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels= Y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y)) 
#cost = tf.reduce_mean(tf.square(hypothesis - Y))


''' Regularization (If want) '''
# reqularization
l2reg = my_regularization_rate * tf.reduce_mean(tf.square(W))

''' Optimizer (Calculate Gradient) '''
# define optimzer : To minimize cost / with learning rate / Use adam optimizer
# https://smist08.wordpress.com/tag/adam-optimizer/ For find more optimizer
# http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html
optimizer = tf.train.AdamOptimizer(learning_rate=my_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize((cost - l2reg), global_step=global_step)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=my_learning_rate).minimize(cost - l2reg)

''' Restore Process '''
# saver
saver = tf.train.Saver()

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

''' Setting for Tensorflow '''
# coordinate
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

min_cost = float(4294967296)

''' Train Model '''
for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(0, total_batch) :
        # add info into batch count
        X_batch, Y_batch = sess.run([X_batches, Y_batches])

        feed_dict = {X: X_batch, Y: Y_batch, keep_prob: dropout_ratio}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    if (avg_cost < min_cost) :
        min_cost = avg_cost

        if (snapshotmincost is True) and (snapshotmincostpath != "NULL") :
            saver.save(sess, snapshotmincostpath)

    if (epoch % print_interval) == 0 :
        print('Epoch' , '{:7d}'.format(epoch), 'done. Cost :', '{:.9f}'.format(avg_cost))
        #print(sess.run(my_learning_rate))
        #print(sess.run(global_step))
        #tf.Print(hypothesis, [hypothesis])

        if savepath != "NULL" :
            saver.save(sess, savepath)

    if (printgraph is True) and (epoch % graph_interval) == 0 :
        Xgraph.append(epoch)
        Ygraph.append(avg_cost)

coord.request_stop()
coord.join(threads)

print("Learning Done")

''' Save if savepath exits '''
if savepath != "NULL" :
    saver.save(sess, savepath)
    print("save done")


''' Get test value '''
Xtest = list()
Ytest = list()

# Ytest values are filled with dummy data (float(1.0)) 
#Xtest, Ytest = get_raw_data_from_csv(Xtest, Ytest, "America_NASDAQ.csv", drop_yarr = True, skipfirstline = True)
Xtest, Ytest = get_raw_data_from_tsv(Xtest, Ytest, "test.txt", X_size = testdata_size,Y_size = 2, drop_yarr = False, skipfirstline = False)

''' Test values '''
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
#correct_prediction = tf.equal(hypothesis, Y)
#correct_prediction = tf.square(hypothesis -  Y)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
#accuracy = tf.reduce_mean(correct_prediction)

''' Check result '''
# sess.run([Output formats(hypothesis, Y)], feed_dictionary(see below)
print_accuracy, predict_val, _ = sess.run([accuracy, hypothesis, Y], feed_dict={X : Xtest, Y : Ytest, keep_prob: 1})

''' Print result '''
# Todo : Synchronize with output
#print(PRINTW.eval())

''' Create output '''
print("Min value : " + str(min_cost) + " (Save : " + str(snapshotmincost) + ")")
#print("Accuracy  : " + str(print_accuracy * 100.0) + "%")
print("Accuracy  : " + str(print_accuracy))
#print_result(predict_val, Ytest)
#print_data(predict_val, "test.csv")

''' Print Result '''
if (printalllayer is True) :
    origin_stdout, sys.stdout = sys.stdout, open(printalllayer_filename, "w")
    
    print(" --- W0 --- ")
    print(sess.run(wlist[0]), 2)

    if direct_bridge is False : 
        print(" --- B0 --- ")
        print(sess.run(blist[0]))
              
    for i in range(1, len(layer_size) - 1) :
        print("\n ====== NEW LAYER ======\n")

        print(" --- W" + str(i) + " --- ")
        print(sess.run(wlist[i]))

        print(" --- B" + str(i) + " --- ")
        print(sess.run(blist[i]))

    sys.stdout = origin_stdout
    
''' Print Graph (Should be last) '''
if (printgraph is True) :
    plt.plot(Xgraph, Ygraph)
    plt.show()

if (showcost is True) :
    print_cost(Xgraph, Ygraph, showcost_filename) 
