import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import datetime
import requests
from bs4 import BeautifulSoup

from tf_functions import print_data, print_cost, get_real_data_from_csv

''' Get data from investing.com '''
def get_data_from_web(web_address) :
    req_web = requests.get(web_address)
    req_html = req_web.text

    souped_web = BeautifulSoup(req_html, "html.parser")

    print(souped_web)

    return
    
''' User Input '''
######## NEEDS TO CONSIDER VARIABLES
# input / output size
input_arraysize = 116
output_arraysize = 2

''' Layers '''
#direct Bridge
direct_bridge = True

''' Notice ::: Same layer for things that you trained '''
# Layer  input , layer '1' , layer '2'  ...  layer 'k' , output
if (direct_bridge is True) :
    layer_size=[input_arraysize, input_arraysize, 86, 72, 32, 12, output_arraysize]
else : 
    layer_size=[input_arraysize, 434, 326, 257, 72, 24, output_arraysize]

''' save & restore variables '''
# set "NULL" if don't have it
# Example : savepath='/tmp/model.ckpt' savepate='NULL'
# window " , linux '
restorepath = "/tmp/model.ckpt"
#savepath = restorepath="/tmp/model.ckpt"
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

list_for_auto_control = list()

# get session
sess = tf.InteractiveSession()

# Set input, output placeholder
X = tf.placeholder(tf.float64, [None, layer_size[0]])
Y = tf.placeholder(tf.float64, [None, layer_size[total_layer - 1]])

''' Setting Layer '''
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

        # Get result for Direct Bridge
        # Todo : synchronize with liunx...
        W = tf.Print(W, [W], message="print")
        PRINTW = tf.add(W, W)

        B = tf.constant(float(0), shape=[layer_size[i + 1]], dtype=tf.float64)

    ## Layer Result
    # First layer : Get input
    if (i == 0) :
        if (direct_bridge is False) :
            L = tf.matmul(X, W) + B
            #L = tf.nn.relu(tf.matmul(X, W) + B)
        else :
            #L = tf.nn.relu(tf.matmul(X, W) + B)
            L = tf.matmul(X, W) + B
    # Else : Get previous hidden
    elif (i != total_layer - 2) :
        #L = tf.nn.relu(tf.matmul(PREVL, W) + B)
        L = tf.matmul(PREVL, W) + B

    PREVL = L
    
''' Your hypothesis (X => Layer => Hypothesis) '''
# set hypothesis
# hypothesis [0.9 0.1 0.0 0.0 ...] // O.9 might be an answer
hypothesis = tf.matmul(L, W) + B
#hypothesis = tf.nn.relu(tf.matmul(L, W) + B)
#hypothesis= tf.sigmoid(tf.matmul(L, W) + B)

''' Restore Process '''
# saver
saver = tf.train.Saver()

# Initialize variables if restore path is null
if restorepath == "NULL" :
    sess.run(tf.global_variables_initializer())
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
        isyes = input("type anything to exit : ")
        exit
    else :
        print ("restore done")

''' Get test value '''
Xtest = list()

#Xtest, Ytest = get_raw_data_from_csv(Xtest, Ytest, "America_NASDAQ.csv", drop_yarr = True, skipfirstline = True)
Xtest = get_real_data_from_csv(Xtest, "세계 주요 지수.csv")

''' Check result '''
# sess.run([Output formats(hypothesis, Y)], feed_dictionary(see below)
predict_val = sess.run([hypothesis], feed_dict={X : Xtest, keep_prob: 1})

''' Create output '''
#print("Accuracy  : " + str(print_accuracy * 100.0) + "%")
print("--- Expecting ---")
print(predict_val)

files = open("result.txt", "w")
files.write(str(predict_val))
files.close()
