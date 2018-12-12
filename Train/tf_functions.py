import math

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

# Function get_data
# num_of_data_set : data set
# X_filename  : input_filename
# X_size      : number of input with float 32
# Y_filename  : output_filename
# Y_size      : number of output with float 32
# Warning :: This function does not gurantee parameter is matched perfectly or num_of_data_set is matched with files
# V 1.01 -> Add case that each data's line includes space(" ") in end (ex : "1.0 1.0 ")
def get_data_with_float32(num_of_data_set, X_filename, X_size, Y_filename, Y_size, X_arr, Y_arr) :
    X_open = open(X_filename, 'r')
    Y_open = open(Y_filename, 'r')

    for i in range (0, num_of_data_set) :
        X_line = X_open.readline()
        # line's last is not space " "
        #X_floatlist = [float(x) for x in X_line.split(" ")]
        # line's last is space
        X_items = X_line.split(" ")
        X_items.pop()
        X_floatlist = [float(x) for x in X_items]
        X_arr.append(X_floatlist)
        
        Y_line = Y_open.readline()
        # line's last is not space " "
        #Y_floatlist = [float(y) for y in Y_line.split(" ")]
        # line's last is space
        Y_items = Y_line.split(" ")
        #if newline character included
        Y_items.pop() 
        Y_floatlist = [float(y) for y in Y_items]
        Y_arr.append(Y_floatlist)

    X_open.close()
    Y_open.close()
        
    return X_arr, Y_arr
# End of function

# Function print diff
# Get two parameters, And print Each information in same line
# X, Y : Array
# start_index, offset_index : print all start_index + (offset_index * i) if this value is less than array size
def print_data(X, Y, each_item_size, start_index, offset_index) :
    total_index = min(len(X), len(Y))

    if (start_index >= total_index) or (offset_index < 1) :
        print("Print data :: Value error")
        return

    for i in range (start_index, total_index, offset_index) :
        print(" Test : ", i)
        
        for j in range (0, each_item_size) :
            
            print(round(X[i][j], 2), " => ", round(Y[i][j], 2))
            
        print("-------------------------------------------")


        
