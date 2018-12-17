import math
import csv
import datetime

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

def print_cost(epoch_array, cost_array, filename) :
    showcostfile = open(filename, "w")

    for cnt in range (0, len(epoch_array)) :
        showcostfile.write('Epoch')
        showcostfile.write('{:7d}'.format(epoch_array[cnt]))
        showcostfile.write('done\tCost : ')
        showcostfile.write('{:.9f}'.format(cost_array[cnt]))
        showcostfile.write("\n")
    showcostfile.close()

    return

# Function print
# X        : Array that want to make for output file
# Filename : Output filename
def print_data(X, filename) :
    with open(filename, "w", encoding="utf-8", newline= '') as csvDataFile:
        writer = csv.writer(csvDataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        for item in X :
            writer.writerow(item)

    csvDataFile.close()

def print_result(X, Y) :
    row = max(len(X), len(Y))
    col_X = len(X[0])
    col_Y = len(Y[0])

    for cur_row in range(0, row) :
        print("Expect : ", end ='')

        for (cur_col) in range(0, col_X) :
            print(X[cur_row][cur_col], end ='')
            print(", ", end ='')

        print(" ==> Real: ", end ='')
        for (cur_col) in range(0, col_Y) :
            print(Y[cur_row][cur_col], end ='')
            print(", ", end ='')

        print("")

        

# clipping data with range between 0 ~ 1)
def clipping_all_data(data_arr) :
    num_row = len(data_arr)
    num_col = len(data_arr[0])

    divider = list()

    for col in range(0, num_col) :
        cur_max = -2000000000
        cur_min = 2000000000
        #cur_max_abs = 0 
        
        for row in range(0, num_row) :
            if (data_arr[row][col] > cur_max) :
                cur_max = data_arr[row][col]
            if (data_arr[row][col] < cur_min) :
                cur_min = data_arr[row][col]
            #if (cur_max_abs < abs(data_arr[row][col])) :
            #    cur_max_abs = abs(data_arr[row][col])
                    
        multipler = (1.0 / float(cur_max - cur_min))
        #multipler = (10.0 / float(cur_max_abs))

        for row in range(0, num_row) :
            data_arr[row][col] -= cur_min
            data_arr[row][col] *= multipler

    print("cliping data with row x col :: " + str(num_row) + " x " + str(num_col)) 

    return data_arr

# Get data from csv
# in train data, you have to write X_arr, Y_arr
# in test data, you have to skip Y_arr
# in csv, We assume first law is name of array
def get_raw_data_from_csv (X_arr, Y_arr, filename, drop_yarr = False, skipfirstline = True) :
    adjustdate = datetime.datetime.strptime("2018년 09월 01일", "%Y년 %m월 %d일")
    
    with open(filename, encoding="utf-8") as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for row in csv_reader :
            # skip first line
            if (skipfirstline) is True :
                skipfirstline = False
            else :
                row_items = list()

                data_count = 0
                
                # get each items
                for col in row :
                    data_count += 1
                    
                    # skip something number here
                    if (data_count <= 2) :
                        continue
                    
                    col = col.replace(",", "")
                    # check date
                    if ("-" in col) and (":" in col) :
                        tem_date = datetime.datetime.strptime(col, "%Y-%m-%d %h:%m:%s")
                        col_date = tem_date - adjustdate
                        col_item = float(col_date.day)
                    # check empty
                    elif (col == "") :
                        col_item = float(0)
                    # skip last item
                    elif (col[-1].isdigit()) is False :
                        col_item = float(col[:-1])
                    else :
                        col_item = float(col)
                    
                    row_items.append(col_item)

                # if Y_arr is exist
                # put last item onto Y array
                if (drop_yarr is False) :
                    tem_arr = list()
                    tem_arr.append(row_items[-1])
                    Y_arr.append(tem_arr)
                    
                    row_items.pop()
                else :
                    tem_arr = list()
                    tem_arr.append(float(1.0))
                    Y_arr.append(tem_arr)


                # add X_arr
                X_arr.append(row_items)
    
    # clipping data (only X array)
    X_arr = clipping_all_data(X_arr)

    return X_arr, Y_arr

# Get data from tsv
# in train data, you have to write X_arr, Y_arr
# in test data, you have to skip Y_arr
# in csv, We assume first law is name of array
def get_raw_data_from_tsv (X_arr, Y_arr, filename, Y_size = 1, drop_yarr = False, skipfirstline = True) :
    with open(filename, encoding="utf-8") as csvDataFile :
        tsv_reader = csv.reader(csvDataFile, delimiter='\t')
        
        for row in tsv_reader :
            # skip first line
            if (skipfirstline) is True :
                skipfirstline = False
            else :
                row_items = list()

                data_count = 0
                
                # get each items
                for col in row :
                    data_count += 1
                    
                    # skip something number here
                    if (data_count <= 0) :
                        continue
                    
                    col = col.replace(",", "")
                    # check date
                    if ("-" in col) and (":" in col) :
                        tem_date = datetime.datetime.strptime(col, "%Y-%m-%d %h:%m:%s")
                        col_date = tem_date - adjustdate
                        col_item = float(col_date.day)
                    # check empty
                    elif (col == "") :
                        col_item = float(0)
                    # skip last item
                    elif (col[-1].isdigit()) is False :
                        col_item = float(col[:-1])
                    else :
                        col_item = float(col)
                    
                    row_items.append(col_item)

                # pop last tab
                row_items.pop()
            
                # if Y_arr is exist
                # put last item onto Y array
                if (drop_yarr is False) :
                    tem_arr = list()
                    
                    for repeat in range (0, Y_size) :
                        tem_arr.append(row_items[-1])
                        row_items.pop()

                    Y_arr.append(tem_arr)    
                else :
                    tem_arr = list()
                    tem_arr.append(float(1.0))
                    Y_arr.append(tem_arr)


                # add X_arr
                X_arr.append(row_items)
    
    # clipping data (only X array)
    X_arr = clipping_all_data(X_arr)

    return X_arr, Y_arr
