import math
import csv
import datetime
import sys

# Function get_data
# num_of_data_set : data set
# X_filename  : input_filename
# X_size      : number of input with float 32
# Y_filename  : output_filename
# Y_size      : number of output with float 32
# X_arr, Y_arr: arrays that save given x, y data 
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

# clipping data with specific range
# data_arr : data array for clipping
# 1. Y = X * k (-t ~ t)
# 2. Y = (X - min_value) * k (0 ~ t)
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
        #multipler = (5.0 / float(cur_max_abs))

        for row in range(0, num_row) :
            data_arr[row][col] -= cur_min
            data_arr[row][col] *= multipler

    print("cliping data with row x col :: " + str(num_row) + " x " + str(num_col)) 

    return data_arr
# End of function


# clipping data that satisfy statistic values (average : 0.5 , std_dev : 0.5
# data_arr : data array for clipping in normalization
def clipping_all_data_with_normalization(data_arr) :
    num_row = len(data_arr)
    num_col = len(data_arr[0])

    # Data set : target mean : 0.5 // target std_dev 0.5
    target_mean=0
    target_std_dev=1

    for col in range(0, num_col) :
        mean = 0
        items = 0
        std_dev = 0
        
        for row in range(0, num_row) :
            mean += data_arr[row][col]
            items += 1

        mean = float(mean) / float(items)

        for row in range(0, num_row) :
            std_dev += pow(data_arr[row][col] - mean, 2)   
                    
        std_dev = math.sqrt(float(std_dev) / float(items))

        for row in range(0, num_row) :
            data_arr[row][col] = (target_std_dev * ((data_arr[row][col] - mean) / std_dev)) + target_mean

    print("cliping data in standard ::  row x col :: " + str(num_row) + " x " + str(num_col))
    print("mean : " + str(target_mean) + " // std_dev : " + str(target_std_dev))

    return data_arr
# End of function


# Get real data from csv
# You can use this function
# X_arr    : data will saved in this array
# filename : filename for reading data
def get_data_from_csv(arr, filename) :
    file = open(filename, 'r', encoding='utf-8')
    #csv_reader = csv.reader(file, qoutechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
    csv_reader = csv.reader(file)

    for row in csv_reader :
        arr.append(row)

    file.close()

    return arr
# End of function


# Get real data from csv
# You can use this function
# X_arr    : data will saved in this array
# filename : filename for reading data
def get_real_data_from_csv(X_arr, filename) :
    col_count = 8
    use_col = [2, 3, 4, 6]
    use_rowid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28, 29, 30, 34, 37, 38, 39, 42]

    variables = list()
    variables = get_data_from_csv(variables, "normalize.csv")

    print(len(variables))

    with open(filename, encoding="utf-8") as csvDataFile :
        csv_reader = csv.reader(csvDataFile)

        X_list = list()

        idx = 0
        rowid = 0 
        use_idx = 0
        use_rows = -1 # skip first line

        for row in csv_reader :
            for col in row :
                if (idx in use_col) and (rowid in use_rowid) :
                    # check empty
                    if (col == "") :
                        col_item = float(0)
                    # skip last item
                    elif (col[-1].isdigit()) is False :
                        col_item = float(col[:-1].replace(",", ""))
                    else :
                        col_item = float(col.replace(",", ""))
                    
                    col_replace = (col_item - float(variables[use_idx + (use_rows * len(use_col))][1]))* float(variables[use_idx + (use_rows * len(use_col))][0]) 
                    X_list.append(col_replace)

                    print(col_item, col_replace, variables[use_idx + (use_rows * len(use_col))][0], variables[use_idx + (use_rows * len(use_col))][1])

                    use_idx += 1

                if (idx == 0) and (rowid in use_rowid) :
                    print (col)

                idx += 1

                if (idx == col_count) :
                    use_idx = 0
                    idx = 0
                    rowid += 1

                    if (rowid in use_rowid) :
                        use_rows += 1

    X_arr.append(X_list)

    return X_arr
# End of function
                    

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
    #X_arr = clipping_all_data(X_arr)

    return X_arr, Y_arr
# End of function


# Get data from tsv
# in train data, you have to write X_arr, Y_arr
# in test data, you have to skip Y_arr
# in csv, We assume first law is name of array
# Data
# file line : [Input layers ...] [output Layers]
# Result    : [Input... , Padding Zero] / [Output]
def get_raw_data_from_tsv (X_arr, 
                           Y_arr, 
                           index_arr,
                           filename, 
                           X_size = -1,
                           Y_size = 0, 
                           drop_yarr = False, 
                           skipfirstline = True,
                           padding_zero=0,
                           delimiter='\t') :
    adjustdate = datetime.timedelta(days=0)

    

    with open(filename, encoding="utf-8") as csvDataFile :
        tsv_reader = csv.reader(csvDataFile, delimiter=delimiter)
        
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

                    if (data_count == 1) :
                        index_arr.append(col)
                    else :
                        row_items.append(col_item)

                # pop last tab(Removed with improvement of file reads)
                #row_items.pop()
            
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

                    for repeat in range (0, Y_size) :
                        tem_arr.append(float(1.0))
                        
                    Y_arr.append(tem_arr)

                # padding 0 in X_arr
                for i in range(padding_zero):
                    row_items.append(float(0))

                # add X_arr
                X_arr.append(row_items)

                X_size -= 1
                if (X_size is 0) : break

    
    # clipping data (only X array)
    #X_arr = clipping_all_data(X_arr)

    return X_arr, Y_arr, index_arr


