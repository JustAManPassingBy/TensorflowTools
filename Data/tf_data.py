import numpy as np
import datetime
import csv

def get_google_data_from_csv_file(prev_list, filename, original_item_num) :
    with open(filename, encoding="utf-8", newline="\n") as csvDataFile:
    #with open(filename) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)

        max = 0

        for row in csv_reader :            
            item_count = 0
            
            for col in row :
                item_count += 1

                # date info
                if (item_count == 1) :
                    if ("년" in col) and ("월" in col) and ("일" in col) :
                        my_date = datetime.datetime.strptime(col, "%Y년 %m월 %d일")
                    else :
                        my_date = datetime.datetime.strptime(col, "%Y-%m-%d %H:%M:%S")

                    skip_row = True
                        
                    for list_item in prev_list :
                        if (my_date in list_item) :
                            target_list = list_item
                            skip_row = False
                            break
                
                    if (skip_row is True) :
                        break

                else :
                    # else
                    col = col.replace(",","")

                     # check data type
                    if (len(col) < 1) :
                        col_item = float(0.0)
                    elif (col[-1].isdigit()) is False :
                        col_item = float(col[:-1])
                    else :
                        col_item = float(col)

                    # add list
                    target_list.append(col_item)

            if (max < item_count) :
                max = item_count

    print("restore " + str(max - 1) + " datas")
            
    return max + original_item_num - 1

def get_data_from_csv_file (prev_list, filename, isfirst = False) :
    skipfirstline = True
    
    with open(filename, encoding="utf-8") as csvDataFile:
        csv_reader = csv.reader(csvDataFile)

        for row in csv_reader :
            if (skipfirstline is True) :
                skipfirstline = False
                continue
            
            item_count = 0
            
            for col in row :
                item_count += 1

                # date info
                if (item_count == 1) :
                    my_date = datetime.datetime.strptime(col, "%Y년 %m월 %d일")

                    # first, create list
                    if (isfirst is True) :
                        target_list = list()
                        target_list.append(my_date)
                    # if not,vb find from prev_list
                    else :
                        skip_row = True
                        
                        for list_item in prev_list :
                            if (my_date in list_item) :
                                target_list = list_item
                                skip_row = False
                
                if (isfirst is False) and (skip_row is True) :
                    break

                # else if count >= 7
                #if (item_count != 1) and  (item_count != 3) and (item_count != 6) :
                if (item_count is 7) :
                    col = col.replace(",","")

                    # check data type
                    if (len(col) < 1) :
                        col_item = float(0.0)
                    elif (col[-1].isdigit()) is False :
                        col_item = float(col[:-1])
                    else :
                        col_item = float(col)

                    # add list
                    target_list.append(col_item)

                else :
                    col = col.replace(",","")


            if (isfirst is True) :
                prev_list.append(target_list)

    if (isfirst is True) :
        print("Len : " + str(len(prev_list)))
    
    return

def clipping_all_data(data_arr, idx, savename) :
    num_row = len(data_arr)
    num_col = idx

    divider = list()

    writer = open(savename, "w")
    
    # skip time
    for col in range(1, num_col) :
        #cur_max = -2000000000
        #cur_min = 2000000000
        #cur_max_abs = 0 

        #for row in range(0, num_row) :
        #    if (len(data_arr[row]) != idx) :
        #        continue
            
        #    if (data_arr[row][col] > cur_max) :
        #       cur_max = data_arr[row][col]
        #    if (data_arr[row][col] < cur_min) :
        #        cur_min = data_arr[row][col]
            #if (cur_max_abs < abs(data_arr[row][col])) :
            #    cur_max_abs = abs(data_arr[row][col])

        multipler = 1.0 / 100.0
        cur_min = 0
        #multipler = (5.0 / float(cur_max_abs))

        vitems = 0
        
        for row in range(0, num_row) :
            if (len(data_arr[row]) != idx) :
                continue
            
            data_arr[row][col] -= cur_min
            data_arr[row][col] *= multipler
            vitems += 1

        rowitems = list()
        rowitems.append(multipler)
        rowitems.append(cur_min)

        divider.append(rowitems)

        writer.write("col : " + str(col) + "// idx : " + str(divider.index(rowitems)) + " // multiple  : " + str(multipler) + " // cur_min : " + str(cur_min) + "\n")

    print("cliping data with row x col :: " + str(num_row) + " x " + str(num_col)) 
    print("valid : " + str(vitems))

    writer.close()

    ''' Make csv '''
    with open("normalize.csv", 'w', encoding='utf-8', newline='\n') as csv_file :
        csv_writer = csv.writer(csv_file, quotechar='"')

        for each_list in divider :
            csv_writer.writerow(each_list)    

    # clip : [col] matches with divider[col - 1]
    return data_arr, divider

def make_data (data_list, startdate, enddate, filename, output_count, num_data, multiple_data) :
    ''' Divide with tab '''
    total_item = 0
    valid_item = 0
    
    # index : erase date info
    total_index = (num_data)

    # will use TSV instead of CSV
    with open(filename, 'w', encoding='utf-8', newline='\n') as csv_file :
        csv_writer = csv.writer(csv_file, delimiter='\t')

        # Maybe write first line (if you want)
        
        for each_list in data_list :
            # check date
            if ((each_list[0] >= startdate) is False) or ((enddate >= each_list[0]) is False) :
                prev_list = each_list
                continue

            elif ((prev_list[0] >= startdate) is False) or ((enddate >= prev_list[0]) is False) :
                #prev_prev_list = prev_list
                prev_list = each_list
                continue

            #if (prev_prev_list[0] > startdate) is False or ((enddate > prev_prev_list[0]) is False):
                #prev_prev_list = prev_list
                #prev_list = each_list
                #continue

            # check omit data
            if (len(each_list) < total_index) :
                continue

            total_item += 1

            row_array=list()

            # input write
            for i in range (2, total_index) :
                #my_file.write(str(round(prev_list[i] - prev_prev_list[i], 2)) + "\t")
                row_array.append(prev_list[i] / 10)
                #if (prev_list[i] >= 0) :
                #    my_file.write("1.0\t")
                #else :
                #    my_file.write("0.0\t")

            # output write
            for i in range(1, 2) :
                # idx 104 matches with multiple_data[103], 0 multipler, 1 cur_min
                # restore = (X / multipler) + cur_min
                if ((each_list[i]) > 0) :
                    row_array.append(float(1.0))
                    row_array.append(float(0.0))
                else :
                    row_array.append(float(0.0))
                    row_array.append(float(1.0))
                #row_array.append(round(each_list[i], 2))

            # add newline
            csv_writer.writerow(row_array)

            valid_item += 1

            prev_prev_list = prev_list
            prev_list = each_list
        

    print(":: Make File info ::")
    print("File : " + filename)
    print("items : ", valid_item, " / ", total_item, " index(input) : ", total_index - 1 - output_count, " index(output) : ", output_count)

    
    return


datalist = list()

# num_datas
prev_num_datas = 2

muldata = 100

output_loop = 1

get_data_from_csv_file(datalist, "코스피지수 내역.csv", isfirst = True) # 1
num_datas = get_google_data_from_csv_file(datalist, "trend_data.csv", prev_num_datas)

'''
# multiple(row item)
multiple = 4

# 0 : date
# (n - 1) * 4 + 1 ~ (n) * 4

get_data_from_csv_file(datalist, "S&P 500 내역.csv")
get_data_from_csv_file(datalist, "나스닥 내역.csv")
get_data_from_csv_file(datalist, "Russell 2000 내역.csv")
get_data_from_csv_file(datalist, "CBOE Volatility Index 내역.csv") # 5

get_data_from_csv_file(datalist, "_캐나다 S&P_TSX 내역.csv") # 6
get_data_from_csv_file(datalist, "브라질 보베스파 내역.csv") # 7
get_data_from_csv_file(datalist, "S&P_BMV IPC 내역.csv") # 8
get_data_from_csv_file(datalist, "DAX 내역.csv") # 9
get_data_from_csv_file(datalist, "영국 FTSE 내역.csv") # 10

print ("1/3")

get_data_from_csv_file(datalist, "프랑스 CAC 내역.csv") # 11
get_data_from_csv_file(datalist, "네덜란드 AEX 내역.csv") # 13
get_data_from_csv_file(datalist, "스페인 IBEX 내역.csv") # 14
get_data_from_csv_file(datalist, "ITALY.csv") # 15
get_data_from_csv_file(datalist, "스위스 SMI 내역.csv") # 16

get_data_from_csv_file(datalist, "벨기에 BEL 내역.csv") # 18
get_data_from_csv_file(datalist, "스웨덴 OMXS 내역.csv") # 20
get_data_from_csv_file(datalist, "러시아 MOEX Russia 내역.csv") # 21
get_data_from_csv_file(datalist, "RTSI 지수 내역.csv") # 22
get_data_from_csv_file(datalist, "폴란드 WIG 20 내역.csv") # 23

print("2/3")

get_data_from_csv_file(datalist, "TA 35 내역.csv") # 26
get_data_from_csv_file(datalist, "닛케이 내역.csv") # 28
get_data_from_csv_file(datalist, "호주 S&P_ASX 내역.csv") # 29
get_data_from_csv_file(datalist, "상하이종합 내역.csv") # 30
get_data_from_csv_file(datalist, "항셍 내역.csv") # 34

get_data_from_csv_file(datalist, "코스피지수 내역.csv") # 37
get_data_from_csv_file(datalist, "인도네시아 IDX 내역.csv") # 38
get_data_from_csv_file(datalist, "Nifty 50 내역.csv") # 39
get_data_from_csv_file(datalist, "CSE All-Share 내역.csv") # 42

num_datas += 29 * multiple
print("read 1.9 done")

datalist, muldata = clipping_all_data(datalist, num_datas, "item data.txt")


# 1.1
get_data_from_csv_file(datalist, "ITALY.csv", isfirst = True)
get_data_from_csv_file(datalist, "TA 35 내역.csv")
num_datas += 2 * multiple
print("read 1.1 done")


# 1.2
get_data_from_csv_file(datalist, "코스피지수 내역.csv")
get_data_from_csv_file(datalist, "_캐나다 S&P_TSX 내역.csv")
get_data_from_csv_file(datalist, "브라질 보베스파 내역.csv")
get_data_from_csv_file(datalist, "S&P_BMV IPC 내역.csv")
get_data_from_csv_file(datalist, "DAX 내역.csv")
get_data_from_csv_file(datalist, "영국 FTSE 내역.csv")
get_data_from_csv_file(datalist, "프랑스 CAC 내역.csv")
get_data_from_csv_file(datalist, "네덜란드 AEX 내역.csv")
get_data_from_csv_file(datalist, "스페인 IBEX 내역.csv")
get_data_from_csv_file(datalist, "벨기에 BEL 내역.csv")
get_data_from_csv_file(datalist, "스웨덴 OMXS 내역.csv")
get_data_from_csv_file(datalist, "폴란드 WIG 20 내역.csv")
get_data_from_csv_file(datalist, "호주 S&P_ASX 내역.csv")
get_data_from_csv_file(datalist, "항셍 내역.csv")
get_data_from_csv_file(datalist, "인도네시아 IDX 내역.csv")
get_data_from_csv_file(datalist, "Nifty 50 내역.csv")
get_data_from_csv_file(datalist, "CSE All-Share 내역.csv")
num_datas += 17 * multiple
print("read 1.2 done")

# 1.3
get_data_from_csv_file(datalist, "다우존스 내역.csv")
get_data_from_csv_file(datalist, "S&P 500 내역.csv")
get_data_from_csv_file(datalist, "나스닥 내역.csv")
get_data_from_csv_file(datalist, "Russell 2000 내역.csv")
get_data_from_csv_file(datalist, "CBOE Volatility Index 내역.csv")
get_data_from_csv_file(datalist, "스위스 SMI 내역.csv")
num_datas += 6 * multiple
print("read 1.3 done")

# 1.4
get_data_from_csv_file(datalist, "닛케이 내역.csv")
get_data_from_csv_file(datalist, "상하이종합 내역.csv")
num_datas += 2 * multiple

# 1.9
get_data_from_csv_file(datalist, "러시아 MOEX Russia 내역.csv")
get_data_from_csv_file(datalist, "RTSI 지수 내역.csv")
num_datas += 2 * multiple
print("read 1.9 done")
'''

my_start_date = datetime.datetime.strptime("2007년 01월 01일", "%Y년 %m월 %d일")
my_end_date = datetime.datetime.strptime("2017년 12월 31일", "%Y년 %m월 %d일")

make_data(datalist, my_start_date, my_end_date, "train.txt", output_loop, num_datas, muldata)
print("train data make done")

my_test_start_date = datetime.datetime.strptime("2018년 01월 01일", "%Y년 %m월 %d일")
my_test_end_date = datetime.datetime.strptime("2018년 12월 6일", "%Y년 %m월 %d일")

make_data(datalist, my_test_start_date, my_test_end_date, "test.txt", output_loop, num_datas, muldata)
print("test data make done")
