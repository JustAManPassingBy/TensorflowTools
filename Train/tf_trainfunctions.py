import tensorflow as tf
import numpy as np
import random
import math
import datetime
import sys
import threading

# Function Cost predictor
# new_input   : input function
# list_input  : number of save variables
# num_descend : least number that has descend relationship 
def cost_predictor(new_input, list_input, cur_learning_rate, input_list, num_descend = -1, fix_num_descend = -1) :
    # add input
    input_list.append(new_input)

    # check input size
    if (len(input_list) < list_input) :
        return 1

    # check whether erase first list
    if (len(input_list) > list_input) :
        input_list.pop(0)

    # calculate checking number of descend
    if (num_descend < 0) :
        num_descend = 0
    elif (num_descend >= list_input) :
        num_descend = list_input - 1

    calculated_descend = 0

    # check list
    for i in range(0, list_input - 1) :
        # check descending
        if (input_list[i] > input_list[i + 1]) :
            calculated_descend += 1

    # print error if calculated_descend < num_descend
    if (calculated_descend < num_descend) :
        print(' Cost Predictor !! descend count calculate : {} / goal :  >= {} / list : {} '.format(calculated_descend, num_descend, (list_input - 1)))
        print('  - Might you need to adjust learning rate or nodes & layers')

        # clear all list
        input_list.clear()

    # divide learning rate 4 and take this value with new learning rate
    if (calculated_descend == fix_num_descend) :
        print("Decrease Learning rate Quarter")

        # clear all list
        input_list.clear()

        return 2
        
    return 1
# End of function

# Create CNN Layer
def create_cnn_layer() :
    return


# Create Layer
def create_layer(input_array, input_layersize_array, output_layersize_array, layer_index, wlist = False,
                 blist = False, llist = False, direct_bridge = False, dropout_ratio=1.0) :
    ## Weight / Bias
    if (direct_bridge is False) or (layer_index != 0) :
        W = tf.get_variable(('W'+str(layer_index)), shape=[input_layersize_array, output_layersize_array],
                           initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        B = tf.Variable(tf.random_normal([output_layersize_array], dtype=tf.float64), name=('B'+str(layer_index)))
    else :
        W = tf.Variable(tf.convert_to_tensor(np.eye(layer_size[i], dtype=np.float64)), name='W0')

    ## Layer Result
    if (direct_bridge is False) or (layer_index != 0) :
        #output_array = tf.matmul(input_array, W) + B
        output_array = tf.nn.relu(tf.matmul(input_array, W) + B)
        output_array = tf.nn.dropout(output_array, keep_prob=dropout_ratio)
    else :
        output_array = tf.nn.relu(tf.matmul(input_array, W))
        #output_array = tf.matmul(input_array, W)
    
    if (wlist is not False) :
        wlist.append(W)
    if (blist is not False) and ((direct_bridge is False) or (i != 0)) :
        blist.append(B)
    if (llist is not False) :
        llist.append(output_array)

    if (direct_bridge is True) :
        return output_array, W

    return output_array, W, B
    
def summary_histogram(total_layer, wlist, blist, llist, direct_bridge = False) :
    if (direct_bridge is False) :
        whist = tf.summary.histogram("weights" + "0", wlist[0])
        bhist = tf.summary.histogram("bias" + "0", blist[0])
    else :
        whist = tf.summary.histogram("weights" + "0", wlist[0])

    for i in range(1, total_layer - 1) :
        whist = tf.summary.histogram("weights" + str(i), wlist[i])

        if (direct_bridge is True) :
            bhist = tf.summary.histogram("bias" + str(i), blist[i - 1])
        else :
            bhist = tf.summary.histogram("bias" + str(i), blist[i])

    # Todo : Make lhist
    lhist = llist
    
    return whist, bhist, lhist
