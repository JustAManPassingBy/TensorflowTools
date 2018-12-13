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
my_learning_rate = 1e-7
my_regularization_rate = 0
training_epochs = 300000
dataset_size = 1332
testdata_size = 110
batch_size = 12
graph_interval = 10
print_interval = 100
break_in_best_sol = False

input_arraysize = 29
output_arraysize = 2

#direct Bridge
direct_bridge = True

# Layer  input , layer '1' , layer '2'  ...  layer 'k' , output
# layer size must not less than 2 (input / output)
layer_size=[input_arraysize, 29, 256, 256, 256, 256, output_arraysize]

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
        #W = tf.Variable(tf.random_normal([layer_size[i], layer_size[i + 1]]), name=('W'+str(i)))
        W = tf.get_variable(('W'+str(i)), shape=[layer_size[i], layer_size[i + 1]],
                           initializer=tf.contrib.layers.xavier_initializer())
       #B = tf.get_variable(('B'+str(i)), shape=[layer_size[i + 1]],
        #                       initializer=tf.contrib.layers.xavier_initializer())
        B = tf.Variable(tf.random_normal([layer_size[i + 1]]), name=('B'+str(i)))
    else :
        # Set Weight, Bias for direct bridge
        W = tf.eye(layer_size[i], name=('W0'))

        B = tf.constant(0.1, shape=[layer_size[i + 1]])

    # Layer Result
    # case of input node
    # Todo : how can i get save value in tf.nn
    if (i == 0) :
        L = tf.nn.relu(tf.matmul(X, W) + B)
        #L = tf.matmul(X, W) + B

        # First layer dropout must check direct bridge
        if (direct_bridge is False) :   
            L = tf.nn.dropout(L, keep_prob=dropout_ratio)
    # case of hidden nodes
    elif (i != total_layer - 2) :
        L = tf.nn.relu(tf.matmul(PREVL, W) + B)
        #L = tf.matmul(PREVL, W) + B

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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y)) 
#cost = tf.reduce_mean(tf.square(hypothesis - Y))

# reqularization
l2reg = my_regularization_rate * tf.reduce_mean(tf.square(W))

# define optimzer : To minimize cost / with learning rate / Use adam optimizer
# https://smist08.wordpress.com/tag/adam-optimizer/ For find more optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=my_learning_rate).minimize(cost - l2reg)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=my_learning_rate).minimize(cost - l2reg)

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
X_arr, Y_arr = get_raw_data_from_csv(Xarr, "America_NASDAQ.csv", Y_arr = Yarr, skipfirstline = True)
print (Xarr)
print (Yarr)


#newXarr = list()
#X_arr, _ = get_raw_data_from_csv(newXarr, "America_NASDAQ.csv", Y_arr = False, skipfirstline = True)
#print (Xarr)

cost_min = float(4294967296)

# graph variable
graph_x = list()
graph_y = list()

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
        #X_batch, Y_batch = mnist.train.next_batch(batch_size)
        #X_batch = list()
        #Y_batch = list()
        
        feed_dict = {X: X_batch, Y: Y_batch, keep_prob: dropout_ratio}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    if (cost_min > avg_cost) :
        cost_min = avg_cost;

    if (break_in_best_sol == True) and (round(avg_cost, 9) == float(0)) :
        graph_x.append(epoch)
        graph_y.append(avg_cost)
        break

    if (epoch % print_interval) == 0 :
        print('Epoch' , '{:6d}'.format(epoch), 'done. Cost :', '{:.9f}'.format(avg_cost))
        tf.Print(hypothesis, [hypothesis])

        if savepath != "NULL" :
            saver.save(sess, savepath)
        
    if (epoch % graph_interval) == 0 :
        #my_learning_rate /= cost_predictor(avg_cost, cost_list_size, my_learning_rate, list_for_auto_control, fix_num_descend=goal_descend_relation)

        graph_x.append(epoch)
        graph_y.append(avg_cost)

    
 
    #print(W.eval())
    
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

Xtest, Ytest = get_data_with_float32(testdata_size, "input_test.txt", input_arraysize, "output_test.txt", output_arraysize, Xtest, Ytest)

# Test model
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
#correct_prediction = tf.equal(hypothesis, Y)
#correct_prediction = tf.square(hypothesis -  Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#accuracy = tf.reduce_mean(correct_prediction)

print_accuracy, predict_val, answer_val = sess.run([accuracy, hypothesis, Y],
#                                                   feed_dict={X : mnist.test.images, Y : mnist.test.labels, keep_prob : 1})
                                                    feed_dict={X : Xtest, Y : Ytest, keep_prob: 1})
print_diff = math.sqrt(print_accuracy)


print("Min cost : " , cost_min)                                                  
print("Accuracy : " , print_accuracy)

print("Expect => Real")
print_data(predict_val, answer_val, output_arraysize, 0, 10)

# Print cost graph
plt.plot(graph_x, graph_y)
plt.show()

# Mnist
# Get one, show result
#r = random.randint(0, mnist.test.num_examples - 1)
#print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
#print("Prediction: ", sess.run(
#    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))

#plt.imshow(mnist.test.images[r:r + 1].
#            reshape(28, 28), cmap='Greys', interpolation='nearest')
#plt.show()
                    

