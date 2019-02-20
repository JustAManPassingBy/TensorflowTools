import numpy as np
import datetime
import csv

def get_google_trend_csv_data(prev_list, filename, isfirst = False) :
    skiptwoline = 4
    
    with open(filename, encoding="utf-8") as csvDataFile:
        csv_reader = csv.reader(csvDataFile)

        for row in csv_reader :
            if (skiptwoline > 0) :
                skiptwoline -= 1
                continue
            
            # 1. date
            date = datetime.datetime.strptime(row[0], "%Y-%m-%d")

            if (isfirst is True) :
                colitems = list()
                colitems.append(date)
                
            else :
                isempty = True
                
                for list_item in prev_list :
                    if (date in list_item) :
                        colitems = list_item
                        isempty = False

            if (isfirst is False) and (isempty is True) :
                continue
            

            # value
            colitems.append(float(row[1]))

            if (isfirst is True) :
                prev_list.append(colitems)


def get_data_from_csv_file (prev_list, filename, isfirst = False) :
    skipfirstline = True

    my_std_date = datetime.datetime.strptime("2018년 1월 1일", "%Y년 %m월 %d일")
    
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

                    if (my_date < my_std_date) :
                        break

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
                if (item_count == 2) :
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


            if (isfirst is True) and (item_count > 1) :
                prev_list.append(target_list)
            
    return

def insert_data_in_trend(dest_addr, src_addr) :
    end_date = datetime.datetime.strptime("2018년 1월 14일", "%Y년 %m월 %d일")
    boundary_date = datetime.datetime.strptime("2018년 12월 31일", "%Y년 %m월 %d일")

    increase_date = datetime.timedelta(days = 7)
    decrease_date = datetime.timedelta(days = 6)

    one_date = datetime.timedelta(days = 1)

    while (end_date <= boundary_date) :
        first_date = end_date - decrease_date
        last_date = end_date

        for dest_items in dest_addr :
            if (dest_items[0] == end_date) :
                append_dest = dest_items
                break

        for items in reversed(src_addr) :
            if (items[0] >= first_date) :
                first_date = items[0]
                first_index = src_addr.index(items) 
                break

        for items in src_addr :
            if (items[0] <= last_date) :
                last_date = items[0]
                last_index = src_addr.index(items)
                break

        # calculate result
        diff = round(float((src_addr[last_index][1] - src_addr[first_index][1]) / src_addr[first_index][1] * 100), 2)

        # insert item
        append_dest.append(diff)

        #print(append_dest, diff)
        #print(first_date, last_date)
        #print(first_index, last_index)

        end_date += increase_date
    

def clipping_all_data(data_arr, idx, savename) :
    num_row = len(data_arr)
    num_col = idx

    divider = list()

    writer = open(savename, "w")
    
    # skip time
    for col in range(1, num_col) :
        cur_max = -2000000000
        cur_min = 2000000000
        #cur_max_abs = 0 

        for row in range(0, num_row) :
            if (len(data_arr[row]) != idx) :
                continue
            
            if (data_arr[row][col] > cur_max) :
               cur_max = data_arr[row][col]
            if (data_arr[row][col] < cur_min) :
                cur_min = data_arr[row][col]
            #if (cur_max_abs < abs(data_arr[row][col])) :
            #    cur_max_abs = abs(data_arr[row][col])

        multipler = (1.0 / float(cur_max - cur_min))
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
    total_index = num_data

    # will use TSV instead of CSV
    with open(filename, 'w', encoding='utf-8', newline='\n') as csv_file :
        csv_writer = csv.writer(csv_file, delimiter='\t')

        # Maybe write first line (if you want)
        for each_list in data_list :
            '''
            # check omit data
            if (len(each_list) < total_index) : continue


            # check date
            if ((each_list[0] > startdate) is False) or ((enddate > each_list[0]) is False) :
                prev_list = each_list
                continue

            if ((prev_list[0] > startdate) is False) or ((enddate > prev_list[0]) is False) :
                #prev_prev_list = prev_list
                prev_list = each_list
                continue

            #if (prev_prev_list[0] > startdate) is False or ((enddate > prev_prev_list[0]) is False):
                #prev_prev_list = prev_list
                #prev_list = each_list
                #continue
            '''

            total_item += 1

            row_array=list()

            # input write
            for i in range (1, total_index) :
                #my_file.write(str(round(prev_list[i] - prev_prev_list[i], 2)) + "\t")
                row_array.append(each_list[i])
                #if (prev_list[i] >= 0) :
                #    my_file.write("1.0\t")
                #else :
                #    my_file.write("0.0\t")

            # output write
            for i in range(total_index, total_index + output_count) :
                row_array.append(each_list[i])
                '''
                # idx 104 matches with multiple_data[103], 0 multipler, 1 cur_min
                # restore = (X / multipler) + cur_min
                if (((each_list[i] / multiple_data[103][0]) + multiple_data[103][1]) > 0) :
                    row_array.append(float(1.0))
                    row_array.append(float(0.0))
                else :
                    row_array.append(float(0.0))
                    row_array.append(float(1.0))
                #row_array.append(round(each_list[i], 2))
                '''

            # add newline
            csv_writer.writerow(row_array)

            valid_item += 1

            #prev_prev_list = prev_list
            #prev_list = each_list
        

    print(":: Make File info ::")
    print("File : " + filename)
    print("items : ", valid_item, " / ", total_item, " index(input) : ", total_index - 1, " index(output) : ", output_count)

    
    return


datalist = list()
newlist = list()
newlist_two = list()

# date info
num_datas = 1

# multiple(row item)
multiple = 4

get_google_trend_csv_data(datalist, "1.csv", isfirst = True)
get_google_trend_csv_data(datalist, "2.csv")
get_google_trend_csv_data(datalist, "3.csv")
get_google_trend_csv_data(datalist, "4.csv")
get_google_trend_csv_data(datalist, "5.csv")
get_google_trend_csv_data(datalist, "7.csv")
get_google_trend_csv_data(datalist, "8.csv")
get_google_trend_csv_data(datalist, "9.csv")
get_google_trend_csv_data(datalist, "10.csv")

get_google_trend_csv_data(datalist, "11.csv")
get_google_trend_csv_data(datalist, "12.csv")
get_google_trend_csv_data(datalist, "13.csv")
get_google_trend_csv_data(datalist, "14.csv")
get_google_trend_csv_data(datalist, "15.csv")
get_google_trend_csv_data(datalist, "16.csv")
get_google_trend_csv_data(datalist, "18.csv")
get_google_trend_csv_data(datalist, "19.csv")
get_google_trend_csv_data(datalist, "20.csv")

get_google_trend_csv_data(datalist, "21.csv")
get_google_trend_csv_data(datalist, "22.csv")
get_google_trend_csv_data(datalist, "23.csv")
get_google_trend_csv_data(datalist, "24.csv")
get_google_trend_csv_data(datalist, "25.csv")
get_google_trend_csv_data(datalist, "26.csv")
get_google_trend_csv_data(datalist, "27.csv")
get_google_trend_csv_data(datalist, "28.csv")
get_google_trend_csv_data(datalist, "29.csv")
get_google_trend_csv_data(datalist, "30.csv")

get_data_from_csv_file(newlist, "다우존스 내역2.csv", isfirst = True)
get_data_from_csv_file(newlist_two, "나스닥종합지수 내역2.csv", isfirst = True)

insert_data_in_trend(datalist, newlist)
insert_data_in_trend(datalist, newlist_two)

my_start_date = datetime.datetime.strptime("2018년 01월 01일", "%Y년 %m월 %d일")
my_end_date = datetime.datetime.strptime("2019년 1월 7일", "%Y년 %m월 %d일")

make_data(datalist, my_start_date, my_end_date, "trend.txt", 2, len(datalist[0]) - 2, 0)

'''
my_start_date = datetime.datetime.strptime("2007년 01월 01일", "%Y년 %m월 %d일")
my_end_date = datetime.datetime.strptime("2017년 12월 31일", "%Y년 %m월 %d일")

make_data(datalist, my_start_date, my_end_date, "train.txt", 1, num_datas, muldata)
print("train data make done")

make_data(datalist, my_test_start_date, my_test_end_date, "test.txt", 1, num_datas, muldata)
print("test data make done")
'''
