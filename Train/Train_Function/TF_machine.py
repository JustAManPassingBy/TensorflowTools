import tensorflow as tf
import numpy as np
import threading
import matplotlib.pyplot as plt

from TF_get_data import get_data_with_float32, get_raw_data_from_csv, get_raw_data_from_tsv
from TF_print_result import print_cost, print_data, print_result, print_all_layer_function


class Tensorflow_Machine:
    def __init__(self,
                 sess,
                 name,
                 input_file=False,
                 layer_file=False,
                 mnist=False):
        self.sess = sess
        self.name = name
        self.mnist = mnist

        self._set_variables()

        if input_file is not False:
            self._get_variables(input_file)

        if layer_file is not False:
            self._get_all_layers(layer_file)
        else:
            self._set_conv2d_layer_size(self.input_arraysize)
            self._set_layer_size(self.reshaped_1d_layer, self.output_arraysize, self.direct_bridge)

        if mnist is not False:
            self.dataset_size = self.mnist.train.num_examples
            self.testdata_size = self.mnist.test.num_examples

        self._get_datas_and_create_batches()

        self._create_layers()

        self._restore_training_values()

    def _set_variables(self):
        self.my_learning_rate = 1e-5
        self.my_regularization_rate = 0
        self.dropout_ratio = 1.0
        self.training_epochs = 300000
        self.dataset_size = 1376
        self.testdata_size = 126
        self.batch_size = 50

        self.printgraph = True
        self.print_interval = 10
        self.graph_interval = 20
        self.summary_interval = 1000

        self.print_result_interval = 5

        self.input_arraysize = 48
        self.output_arraysize = 2
        self.cost_list_size = 50
        self.input_padding_zero = 0

        self.train_file_name = "train.txt"
        self.test_file_name = "test.txt"
        self.savepath = "/tmp/model.ckpt"
        self.restorepath = "/tmp/model.ckpt"

        self.showcost = True
        self.showcost_file_name = "showcost.txt"

        self.printalllayer = True
        self.printalllayer_file_name = "alllayer.txt"

        self.snapshotmincost = False
        self.snapshotmincostpath = "/tmp/minmodel.ckpt"

        self.input_dtype = tf.float64

        self.print_created_layer = False

        # For variable learning rate (TBU)
        self.my_initial_learning_rate = 1e-3
        self.decay_steps = 100000
        self.decay_rate = 0.98

        self.num_thread = 1

        self.direct_bridge = False

        ### Auto Calculated Options ###
        self.goal_descend_relation = int(self.cost_list_size / 2)
        self.total_batch = int(self.dataset_size / self.batch_size)

        return

    # Setting Pared Line
    # 1. Return "True" if object is acceptable
    # 2. Return "False" if 1 is false.
    def _setting_parsed_line(self,
                             object_name,
                             object_value):
        try:
            getattr(self, object_name)
        except AttributeError:
            return False

        # 1. String
        if ('"' in object_value) or ("'" in object_value):
            modified_object_value = str(object_value).replace("'", "").replace('"', '')
        # 2. Float
        elif (('e' in object_value) and ('-' in object_value)) or ('.' in object_value):
            modified_object_value = float(object_value)
        # 3. Boolean True
        elif object_value == "True":
            modified_object_value = True
        # 4. Boolean False
        elif object_value == "False":
            modified_object_value = False
        # 5. Else : integer
        else:
            modified_object_value = int(object_value)

        setattr(self, object_name, modified_object_value)

        return True

    def _get_variables(self, input_filename):
        with open(input_filename, 'r') as open_file:
            lines = open_file.readlines()

            for line in lines:
                parsed_line = line.replace(' ', '').replace('\r', '').replace('\n', '').split('=')

                # skip 1. Split value is not 2
                if len(parsed_line) != 2:
                    continue

                # skip 2. First letter is not an alphabet
                if parsed_line[0][0].isalpha() is False:
                    continue

                # Check item
                if self._setting_parsed_line(parsed_line[0], parsed_line[1]) is False:
                    print(" [_get_variables] Skipped Line : [" + str(line) + "]")

            open_file.close()

        self.input_arraysize += self.input_padding_zero

        return

    def _set_conv2d_layer_size(self,
                               input_arraysize):
        # reshaped layer (Batches(-1), Y, X, (channel = 1)
        self.reshape_input_layer = [-1, 8, 6, 1]

        self.filter_layers = []
        self.filter_strides = []

        self.max_pool_ksizes = []
        self.max_pool_strides = []

        self.padding_type = 'SAME'
        self.filter_stddev = 0.01

        # ! Filter Layer Shape
        # ! - [filter_Y, filter_X, (input_channel), (output_channel == num_of_filters)]
        # ! Filter / Max Pool Ksize & Stride Shape
        # ! - [*batches, Y, X, *num_channels]
        # !  * Values recommand to set 1, as all batches and channels should not be skipped

        ''' Your Layers '''
        # First (8 * 6 * 1 -> 8 * 6 * 10 -> 4 * 3 * 20)
        self.filter_layers.append([3, 3, 1, 20])
        self.filter_strides.append([1, 1, 1, 1])
        self.max_pool_ksizes.append([1, 2, 2, 1])
        self.max_pool_strides.append([1, 2, 2, 1])

        # Second (4 * 3 * 20 -> 4 * 3 * 48 -> 2 * 1 * 48)
        self.filter_layers.append([3, 3, 20, 48])
        self.filter_strides.append([1, 1, 1, 1])
        self.max_pool_ksizes.append([1, 2, 3, 1])
        self.max_pool_strides.append([1, 2, 3, 1])

        # Final 1d layer size (2 * 1 * 48)
        self.reshaped_1d_layer = (2 * 1 * 48)
        ''' End of Your Layers '''

        self.num_conv2d_layers = len(self.filter_layers)

        ''' Check validity '''
        if self.reshape_input_layer[1] * self.reshape_input_layer[2] != input_arraysize:
            print("Error : Your reshaped values makes wrond result")
            raise ValueError

        if ((len(self.filter_layers) != len(self.filter_strides)) or (
                len(self.filter_strides) != len(self.max_pool_ksizes))
                or (len(self.max_pool_ksizes) != len(self.max_pool_strides))):
            print("Error : Your conv2d layer's size is not same")
            raise ValueError

        return

    def _set_layer_size(self,
                        input_arraysize,
                        output_arraysize,
                        direct_bridge=False):
        # Layer  input , layer '1' , layer '2'  ...  layer 'k' , output
        # if (direct_bridge is True) :
        #    my_layer = [input_arraysize, input_arraysize, 86, 72, 32, 13, output_arraysize]
        # else :
        #    my_layer = [input_arraysize, 20, output_arraysize]

        # Layers definition using for
        my_layer = list()
        my_layer.append(input_arraysize)

        for i in range(input_arraysize - 8, output_arraysize, -12):
            my_layer.append(i)

        my_layer.append(output_arraysize)

        self.one_dim_layer = my_layer
        self.one_dim_layer_size = len(my_layer)

        # check validity
        if (direct_bridge is True) and (input_arraysize != layer_size[1]):
            print("Error : In direct bridge, first hidden layer size is same with input layer")
            print("Input : " + str(input_arraysize) + " / Hidden : " + str(layer_size[1]))
            raise ValueError

        if len(my_layer) <= 1:
            print(str(len(my_layer)) + " :: Your layer is too small XD")
            raise ValueError

        return

    def _record_cnn(self,
                    line):
        parse_item = line.replace("[", "").replace("]", "").split(",")

        # 1. conv
        if parse_item[0] == "conv":
            for i in range(1, len(parse_item)):
                self.reshape_input_layer.append(int(parse_item[i]))

        # 2. f_layer
        elif parse_item[0] == "f_layer":
            new_layer = list()
            for i in range(1, len(parse_item)):
                new_layer.append(int(parse_item[i]))

            self.filter_layers.append(new_layer)

        # 3. f_stride
        elif parse_item[0] == "f_stride":
            new_layer = list()
            for i in range(1, len(parse_item)):
                new_layer.append(int(parse_item[i]))

            self.filter_strides.append(new_layer)

        # 4. m_ksize
        elif parse_item[0] == "m_ksize":
            new_layer = list()
            for i in range(1, len(parse_item)):
                new_layer.append(int(parse_item[i]))

            self.max_pool_ksizes.append(new_layer)

        # 5. m_stride
        elif parse_item[0] == "m_stride":
            new_layer = list()
            for i in range(1, len(parse_item)):
                new_layer.append(int(parse_item[i]))

            self.max_pool_strides.append(new_layer)

            # After 5 is called, increase layersize 1
            self.num_conv2d_layers += 1

            # Check all layer's sizes are same
            if ((len(self.filter_layers) != len(self.filter_strides))
                    or (len(self.filter_strides) != len(self.max_pool_ksizes))
                    or (len(self.max_pool_ksizes) != len(self.max_pool_strides))):
                print("Error : Your conv2d layer's size is not same")
                raise ValueError

        # 6. 1d_size
        elif parse_item[0] == "1d_size":
            self.reshaped_1d_layer = int(parse_item[1])

        # 7. pad_type
        elif parse_item[0] == "pad_type":
            self.padding_type = str(parse_item[1])

        # 8. std_dev
        elif parse_item[0] == "std_dev":
            self.filter_stddev = float(parse_item[1])

        # Default
        else:
            print(" [_record_cnn] UNKNOWN Line : [" + str(line) + "]")

        return

    def _record_1d(self,
                   line):
        parse_item = line.replace("[", "").replace("]", "").split(",")

        # 1. item
        if parse_item[0] == "item":
            for i in range(1, len(parse_item)):
                self.one_dim_layer.append(int(parse_item[i]))

        # 2. for
        elif parse_item[0] == "for":
            for layer in range(int(parse_item[1]), int(parse_item[2]), int(parse_item[3])):
                self.one_dim_layer.append(layer)

        # Default
        else:
            print(" [_record_1d] UNKNOWN Line : [" + str(line) + "]")

        self.one_dim_layer_size = len(self.one_dim_layer)

        return

    def _init_all_layers(self):
        self.reshape_input_layer = []

        self.filter_layers = []
        self.filter_strides = []

        self.max_pool_ksizes = []
        self.max_pool_strides = []

        self.padding_type = 'SAME'
        self.filter_stddev = 0.01

        self.reshaped_1d_layer = -1

        self.num_conv2d_layers = 0

        self.one_dim_layer = list()

        self.one_dim_layer_size = 0

        return

    def _get_all_layers(self,
                        input_filename):
        with open(input_filename, 'r') as open_file:
            lines = open_file.readlines()

            # Mode : NULL, DL = 1d_layer, CNN = 2d_layer
            mode = "NULL"

            # Recording (True / False)
            recording = False

            self._init_all_layers()

            for line in lines:
                # Skip comment(#, /)
                if (line[0] == '#') or (line[0] == '/'):
                    continue

                # CNN
                if "CNN" in line:
                    mode = "CNN"
                # DL
                elif "DL" in line:
                    mode = "DL"

                # Catch { (Start Record)
                if "{" in line:
                    recording = True
                    continue

                # Catch } (End Record)
                elif "}" in line:
                    recording = False
                    continue

                if recording is True:
                    trimmed_line = line.replace(" ", "").replace("\r", "").replace("\n", "").replace("\t", "")
                    # CNN
                    if "CNN" == mode:
                        self._record_cnn(trimmed_line)

                    # DL
                    elif "DL" == mode:
                        self._record_1d(trimmed_line)

            open_file.close()

        return

    def _get_datas_and_create_batches(self):
        if self.mnist is False:
            self.Xtrain = list()
            self.Ytrain = list()

            # collect input data
            self.Xtrain, self.Ytrain = get_raw_data_from_tsv(self.Xtrain,
                                                             self.Ytrain,
                                                             self.train_file_name,
                                                             X_size=self.dataset_size,
                                                             Y_size=self.output_arraysize,
                                                             drop_yarr=False,
                                                             skipfirstline=False,
                                                             padding_zero=self.input_padding_zero)

            # create Batches(Slice of train data)
            ## Normal batch
            self.X_batches, self.Y_batches = tf.train.batch([self.Xtrain, self.Ytrain],
                                                            batch_size=self.batch_size,
                                                            enqueue_many=True,
                                                            allow_smaller_final_batch=True)
            ## Random batch
            # num_min = self.num_thread * self.dataset_size
            # self.X_batches, self.Y_batches = tf.train.shuffle_batch([self.Xtrain, self.Ytrain], enqueue_many=True, batch_size=self.batch_size,
            #                                                        capacity = (self.num_thread + 2) * num_min , min_after_dequeue=(num_min),
            #                                                        allow_smaller_final_batch=True)

            self.Xtest = list()
            self.Ytest = list()

            self.Xtest, self.Ytest = get_raw_data_from_tsv(self.Xtest,
                                                           self.Ytest,
                                                           self.test_file_name,
                                                           X_size=self.testdata_size,
                                                           Y_size=self.output_arraysize,
                                                           drop_yarr=False,
                                                           skipfirstline=False,
                                                           padding_zero=self.input_padding_zero)

        else:
            # For mnist
            self.Xtest = self.mnist.test.images
            self.Ytest = self.mnist.test.labels

        return

    # Create CNN(Conv2d + Max Pool)
    # 1. input_array = input items
    # (X). input_layersize_array =  [(batches == -1), Y, X, (num_channels)] => [Y, X, (num_channels)]
    # 3. filter_layersize_array = [filter_Y, filter_X, (input_channel), (output_channel == num_of_filters)]
    # 4. filter_stride_array = [(batches == 1 # no skips), Y, X, (num_channels == 1 # no skips)]
    # 5. padding_type = 'SAME' / 'VALID'
    # 6. max_pool_ksize = [(batches == 1 # no skips), Y, X, (num_channels == 1 # no skips)
    # 7. max_pool_stride_array = [(batches == 1 # no skips), Y, X, (num_channels == 1 # no skips)]
    # 8. final_reshape_to_1d = whether change 2dim * channels([Y, X, channels]) into 1dim 
    def _create_cnn_2d_layer(self,
                             input_array,
                             filter_layersize_array,
                             filter_stride_array,
                             max_pool_ksize,
                             max_pool_stride_array,
                             layer_index,
                             padding_type='SAME',
                             dropout_ratio=1.0,
                             input_dtype=tf.float64,
                             wlist=False,
                             final_reshape_to_1d=False,
                             stddev=0.01):
        if self.print_created_layer is True:
            print("CNN", filter_layersize_array, filter_stride_array, max_pool_ksize,
                  max_pool_stride_array, layer_index)

        W = tf.Variable(tf.random_normal(filter_layersize_array,
                                         stddev=stddev,
                                         dtype=tf.float64,
                                         name=('CNN_W' + str(layer_index))))

        L = tf.nn.conv2d(input_array, W, strides=filter_stride_array, padding=padding_type)
        L = tf.nn.relu(L)

        output_array = tf.nn.max_pool(L, ksize=max_pool_ksize, strides=max_pool_stride_array, padding=padding_type)
        output_array = tf.nn.dropout(output_array, keep_prob=dropout_ratio)

        if wlist is not False:
            wlist.append(W)

        if final_reshape_to_1d is True:
            output_layersize = self.reshaped_1d_layer
            output_array = tf.reshape(output_array, [-1, output_layersize])

            return output_array, output_layersize, W, L

        return output_array, W, L

    # Create 1 dimension Layer
    def _create_1d_layer(self,
                         input_array,
                         input_layersize_array,
                         output_layersize_array,
                         layer_index,
                         wlist=False,
                         blist=False,
                         llist=False,
                         direct_bridge=False,
                         dropout_ratio=1.0,
                         input_dtype=tf.float64):

        if self.print_created_layer is True:
            print("1Dim", input_layersize_array, output_layersize_array, layer_index)

        ## Weight / Bias
        if (direct_bridge is False) or (layer_index != 0):
            W = tf.get_variable(('W' + str(layer_index)), shape=[input_layersize_array, output_layersize_array],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=input_dtype)
            B = tf.Variable(tf.random_normal([output_layersize_array], dtype=input_dtype),
                            name=('B' + str(layer_index)))
        else:
            W = tf.Variable(tf.convert_to_tensor(np.eye(input_layersize_array, dtype=np.float64)), name='W0')

        ## Layer Result
        if (direct_bridge is False) or (layer_index != 0):
            # output_array = tf.matmul(input_array, W) + B
            output_array = tf.nn.relu(tf.matmul(input_array, W) + B)
            output_array = tf.nn.dropout(output_array, keep_prob=dropout_ratio)
        else:
            output_array = tf.nn.relu(tf.matmul(input_array, W))
            # output_array = tf.matmul(input_array, W)

        if wlist is not False:
            wlist.append(W)
        if (blist is not False) and ((direct_bridge is False) or (i != 0)):
            blist.append(B)
        if llist is not False:
            llist.append(output_array)

        if direct_bridge is True:
            return output_array, W

        return output_array, W, B

    def _summary_histogram(self,
                           total_layer,
                           wlist,
                           blist,
                           llist,
                           direct_bridge=False):
        if direct_bridge is False:
            whist = tf.summary.histogram("weights" + "0", wlist[0])
            bhist = tf.summary.histogram("bias" + "0", blist[0])
        else:
            whist = tf.summary.histogram("weights" + "0", wlist[0])

        for i in range(1, total_layer - 1):
            whist = tf.summary.histogram("weights" + str(i), wlist[i])

            if direct_bridge is True:
                bhist = tf.summary.histogram("bias" + str(i), blist[i - 1])
            else:
                bhist = tf.summary.histogram("bias" + str(i), blist[i])

        # Todo : Make lhist
        lhist = llist

        return whist, bhist, lhist

    def _create_layers(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(self.input_dtype, [None, self.input_arraysize])
            self.Y = tf.placeholder(self.input_dtype, [None, self.output_arraysize])

            self.keep_prob = tf.placeholder(self.input_dtype)

            # list for tensorboard
            self.wlist = list()
            self.blist = list()
            self.llist = list()

            if self.num_conv2d_layers is not 0:
                self.Ximage = tf.reshape(self.X, self.reshape_input_layer)
                next_input = self.Ximage
            else:
                next_input = self.X

            # 1. create conv2d
            for i in range(0, self.num_conv2d_layers - 1):
                next_input, W, B = self._create_cnn_2d_layer(next_input,
                                                             self.filter_layers[i],
                                                             self.filter_strides[i],
                                                             self.max_pool_ksizes[i],
                                                             self.max_pool_strides[i], i,
                                                             dropout_ratio=self.dropout_ratio,
                                                             input_dtype=self.input_dtype,
                                                             stddev=self.filter_stddev)

            # 2. last conv2d (change 2d * channel shape to 1d)
            if self.num_conv2d_layers is not 0:
                next_input, _, W, B = self._create_cnn_2d_layer(next_input,
                                                                self.filter_layers[self.num_conv2d_layers - 1],
                                                                self.filter_strides[self.num_conv2d_layers - 1],
                                                                self.max_pool_ksizes[self.num_conv2d_layers - 1],
                                                                self.max_pool_strides[self.num_conv2d_layers - 1],
                                                                self.num_conv2d_layers - 1,
                                                                dropout_ratio=self.dropout_ratio,
                                                                input_dtype=self.input_dtype,
                                                                stddev=self.filter_stddev,
                                                                final_reshape_to_1d=True)

            # 3. create 1d
            for i in range(0, self.one_dim_layer_size - 2):
                next_input, W, B = self._create_1d_layer(next_input,
                                                         self.one_dim_layer[i],
                                                         self.one_dim_layer[i + 1],
                                                         i,
                                                         self.wlist,
                                                         self.blist,
                                                         self.llist)

            # 4. last 1d
            _, self.W, self.B = self._create_1d_layer(next_input,
                                                      self.one_dim_layer[self.one_dim_layer_size - 2],
                                                      self.one_dim_layer[self.one_dim_layer_size - 1],
                                                      self.one_dim_layer_size - 2, self.wlist, self.blist)

            # 5. hypothesis
            # hypothesis [0.9 0.1 0.0 0.0 ...] // O.9 might be an answer
            self.hypothesis = tf.matmul(next_input, self.W) + self.B
            # hypothesis = tf.nn.relu(tf.matmul(next_input, W) + B)
            # hypothesis= tf.sigmoid(tf.matmul(next_input, W) + B)
            # hypothesis= tf.nn.tanh(tf.matmul(next_input, W) + B)

            # 6. cost
            # Cost is difference between label & hypothesis(Use softmax for maximize difference
            # [0.9 0.1 0.0 0.0 ...] ===(softmax)==> [1 0 0 0 ...] (Soft max is empthsize special answer)
            # Square = {for all X in [X], init M = 0 , M += (WX - Y)^2 , finally M /= sizeof([X]) 
            # label is true difference
            # reduce mean is average of all matrix's elements
            # cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels= Y))
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
            self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))

            # 7. Record all informations to tensorboard
            self.whist, self.bhist, _ = self._summary_histogram(self.one_dim_layer_size, self.wlist, self.blist,
                                                                self.llist)
            self.hyphist = tf.summary.histogram("hypothesis", self.hypothesis)
            tf.summary.scalar("cost", self.cost)

            self.merged = tf.summary.merge_all()

            # 8. Reqularization
            self.l2reg = self.my_regularization_rate * tf.reduce_mean(tf.square(W))

            # 9. Optimizer
            # define optimzer : To minimize cost / with learning rate / Use adam optimizer
            # https://smist08.wordpress.com/tag/adam-optimizer/ For find more optimizer
            # http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.my_learning_rate,
                                                    beta1=0.9,
                                                    beta2=0.999,
                                                    epsilon=1e-8).minimize((self.cost - self.l2reg))
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.my_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize((self.cost - self.l2reg), global_step=global_step)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.my_learning_rate).minimize(self.cost - self.l2reg)

            # 10. Accuracy
            # Adjust correct prediction (set standards whether train result is same with expects)
            self.correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
            # correct_prediction = tf.equal(self.hypothesis, self.Y)
            # correct_prediction = tf.square(self.hypothesis -  self.Y)

            # calculate Accuracy
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, self.input_dtype))
            # accuracy = tf.reduce_mean(self.correct_prediction)

        return

    def _restore_training_values(self):
        self.saver = tf.train.Saver()

        # Initialize variables if restore path is null
        if self.restorepath == "NULL":
            self.sess.run(tf.global_variables_initializer())
            print("Initialize variables")
        # Restore
        else:
            try:
                self.saver.restore(self.sess, self.restorepath)
            except ValueError:
                print("Invaild path : ", self.restorepath, " :: Initialize path")
                self.sess.run(tf.global_variables_initializer())
                print("Initialize variables")
            except:
                print("Fail to restore from previous checkpoint")
                print("It might be happened in shrinking or expanding layers")
                isyes = input("type [init] if you want to initilize all processes : ")
                if isyes.lower() == "init":
                    self.sess.run(tf.global_variables_initializer())
                    print("Initialize variables")
                else:
                    raise TypeError
            else:
                print("restore done")

        return

    def _get_next_batch(self):
        if self.mnist is False:
            X_batch, Y_batch = self.sess.run([self.X_batches, self.Y_batches])
        else:
            X_batch, Y_batch = self.mnist.train.next_batch(self.batch_size)  # for mnist

        return X_batch, Y_batch

    def setting_graph_for_training(self):
        if self.printgraph is not True:
            return

        self.Xgraph = list()
        self.Ygraph = list()

        return

    def training_model_once(self,
                            sess,
                            coord,
                            threads,
                            min_cost,
                            epoch):
        avg_cost = 0

        # Each epoch trains amount of total batch (num_input_data / num_batches) 
        for i in range(0, self.total_batch):
            X_batch, Y_batch = self._get_next_batch()

            feed_dict = {self.X: X_batch, self.Y: Y_batch, self.keep_prob: self.dropout_ratio}
            c, merge_result, _ = self.sess.run([self.cost, self.merged, self.optimizer], feed_dict=feed_dict)
            avg_cost += c / self.total_batch

        if avg_cost < min_cost:
            min_cost = avg_cost

        # Todo : Need to be removed
        if (self.snapshotmincost is True) and (self.snapshotmincostpath != "NULL"):
            saver.save(self.sess, self.snapshotmincostpath)

        # Print & Save cost
        if (epoch % self.print_interval) == 0:
            print('Epoch', '{:7d}'.format(epoch), 'done. Cost :', '{:.9f}'.format(avg_cost))

        # Temporary save path
        if self.savepath != "NULL":
            self.saver.save(sess, self.savepath)

        # self.saver.save(self.sess, "tmp/tem_save")

        # Save variables for graph
        if (self.printgraph is True) and (epoch % self.graph_interval) == 0:
            self.Xgraph.append(epoch)
            self.Ygraph.append(avg_cost)

        # for summary interval
        if (epoch % self.summary_interval) == 0:
            self.writer.add_summary(merge_result, epoch)

        return min_cost

    def training_model(self,
                       training_epochs=-1):
        if training_epochs < 1:
            training_epochs = self.training_epochs

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        min_cost = float(4294967296)

        self.setting_graph_for_training()

        self.writer = tf.summary.FileWriter("./logs/train_logs", self.sess.graph)

        ''' Train Model '''
        for epoch in range(training_epochs):
            self.training_model_once(self.sess, coord, threads, min_cost, epoch)

        coord.request_stop()
        coord.join(threads)

        # Learing end
        print("Learning Done")

        return min_cost

    def test_model(self):
        print_accuracy, predict_val, _ = self.sess.run([self.accuracy, self.hypothesis, self.Y],
                                                       feed_dict={self.X: self.Xtest, self.Y: self.Ytest,
                                                                  self.keep_prob: 1})

        ''' Print result (TBD) '''
        # Todo : Synchronize with output
        # print(PRINTW.eval())

        # print("Min value : " + str(min_cost) + " (Save : " + str(snapshotmincost) + ")")
        print("Accuracy  : " + str(print_accuracy * 100.0) + "%")
        # print_result(predict_val, self.Ytest, self.print_result_interval)
        # print_data(predict_val, "test.csv")

        if self.printalllayer is True:
            print_all_layer_function(self.sess, self.printalllayer_file_name, self.wlist, self.blist,
                                     self.one_dim_layer_size, self.direct_bridge)

        if self.showcost is True:
            print_cost(self.Xgraph, self.Ygraph, self.showcost_file_name)

        if self.printgraph is True:
            plt.plot(self.Xgraph, self.Ygraph)
            plt.show()

        return predict_val, self.Ytest

    def get_test_result(self):
        return self.Ytest

    def get_num_train_test_data(self):
        return self.dataset_size, self.testdata_size

    def destory_all(self):
        tf.reset_default_graph()
        self.sess.close()

        return

''' // Currently not use
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
'''
