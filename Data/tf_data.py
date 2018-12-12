import numpy as np
import datetime

def get_data_from_file (prev_list, filename, adjust_date, isfirst = False) :
    my_file = open(filename, "r",encoding='UTF8')

    first = False

    # get all line
    while True :
        my_line = my_file.readline()

        # break if line is empty
        if not my_line :
            break

        my_items = my_line.split('"')

        # skip opening ment
        if my_items[1][0] is not "2" :
            continue

        # get info
        if (isfirst) is True :
            isfirst = False
            first = True
            
            index = 0
            
            for my_item in my_items :
                print(str(index) + " => " + str(my_item))
                index += 1

        adjustdate = datetime.timedelta(days = adjust_date)
        
        my_date = datetime.datetime.strptime(my_items[1], "%Y년 %m월 %d일") + adjustdate

        find_index = 0
        
        # check items in list
        for list_item in prev_list :
            if my_date in list_item :
                find_index = 1

                list_item.append(float(my_items[13].replace("%", "")))
                
                #list_item.append(float(my_items[3].replace(",", "")))
                #list_item.append(float(my_items[7].replace(",", "")))
                #list_item.append(float(my_items[9].replace(",", "")))

        # if item is not in list
        if find_index is 0 :
            new_item = list();

            new_item.append(my_date)
            new_item.append(float(my_items[13].replace("%", "")))
            #new_item.append(float(my_items[3].replace(",", "")))
            #new_item.append(float(my_items[7].replace(",", "")))
            #new_item.append(float(my_items[9].replace(",", "")))

            prev_list.append(new_item)          
        
    my_file.close()

    return

def make_data (data_list, startdate, enddate, input_filename, output_filename, num_data) :
    total_item = 0
    valid_item = 0
    # index : erase date info
    total_index = (num_data) - 1
    
    my_file = open(input_filename, "w")
    my_file2 = open(output_filename, "w")
    #my_file3 = open("test.txt", "w")
    
    for each_list in data_list :
        # check omit data
        if (len(each_list) < total_index + 1) : continue

        #my_file3.write("[ " + str(len(each_list) - 1) + " ] ")
        #for items in each_list :
        #    my_file3.write("%9s"%(str(items) + " "))
        #my_file3.write("\n")

        # check date validate
        #if (total_item is not 0) :
        #    if (prev_list[0] > each_list[0]) is False :
        #        print("Data Warning : ", prev_list[0], " vs ", each_list[0])

        # check date
        if ((each_list[0] > startdate) is False) or ((enddate > each_list[0]) is False) :
            prev_list = each_list
            continue

        if ((prev_list[0] > startdate) is False) or ((enddate > prev_list[0]) is False) :
            prev_prev_list = prev_list
            prev_list = each_list
            continue

        if (prev_prev_list[0] > startdate) is False or ((enddate > prev_prev_list[0]) is False):
            prev_prev_list = prev_list
            prev_list = each_list
            continue

        total_item += 1

        # input write
        for i in range (0, total_index) :
            #if abs(prev_list[i + 1] - prev_prev_list[i + 1]) > 10000 :
            #    print(str(each_list[0]))
            #    print(str(prev_list[0]))
            #    print(str(prev_prev_list[0]))
            #    print(str(i) + " :: " + str(prev_list[i + 1]) + (" / ") + str(prev_prev_list[i + 1]))
                
            my_file.write(str(round(prev_list[i + 1] - prev_prev_list[i + 1], 2)) + " ")

        # output write
        for i in range (2, 3) :
            #my_file2.write(str(round(each_list[i * 3 + 1] - prev_list[i * 3 + 1], 2)) + " ")
            if ((each_list[i * 3 + 1] - prev_list[i * 3 + 1]) > 0) :
                my_file2.write("1.0 0.0 ")
            else :
                my_file2.write("0.0 1.0 ")

        # add newline
        my_file.write("\n")
        my_file2.write("\n")

        valid_item += 1

        prev_prev_list = prev_list
        prev_list = each_list
        

    print(":: Make File info ::")
    print("Input file : ", input_filename, "    Output file : ", output_filename)
    print("items : ", valid_item, " / ", total_item, " index(input) : ", total_index, " index(output) : ", int(total_index / 3))

    my_file.close()
    my_file2.close()
    #my_file3.close()
    
    return




datalist = list()

# date info
num_datas = 1

# 1.1
get_data_from_file(datalist, "ITALY.csv", 0, isfirst = True)
get_data_from_file(datalist, "TA 35 내역.csv", 0)
num_datas += 2
print("read 1.1 done")

# 1.2
get_data_from_file(datalist, "코스피지수 내역.csv", 0)
get_data_from_file(datalist, "_캐나다 S&P_TSX 내역.csv", 0)
get_data_from_file(datalist, "브라질 보베스파 내역.csv", 0)
get_data_from_file(datalist, "S&P_BMV IPC 내역.csv", 0)
get_data_from_file(datalist, "DAX 내역.csv", 0)
get_data_from_file(datalist, "영국 FTSE 내역.csv", 0)
get_data_from_file(datalist, "프랑스 CAC 내역.csv", 0)
get_data_from_file(datalist, "네덜란드 AEX 내역.csv", 0)
get_data_from_file(datalist, "스페인 IBEX 내역.csv", 0)
get_data_from_file(datalist, "벨기에 BEL 내역.csv", 0)
get_data_from_file(datalist, "스웨덴 OMXS 내역.csv", 0)
get_data_from_file(datalist, "폴란드 WIG 20 내역.csv", 0)
get_data_from_file(datalist, "호주 S&P_ASX 내역.csv", 0)
get_data_from_file(datalist, "항셍 내역.csv", 0)
get_data_from_file(datalist, "인도네시아 IDX 내역.csv", 0)
get_data_from_file(datalist, "Nifty 50 내역.csv", 0)
get_data_from_file(datalist, "CSE All-Share 내역.csv", 0)
num_datas += 17
print("read 1.2 done")

# 1.3
get_data_from_file(datalist, "다우존스 내역.csv", 0)
get_data_from_file(datalist, "S&P 500 내역.csv", 0)
get_data_from_file(datalist, "나스닥 내역.csv", 0)
get_data_from_file(datalist, "Russell 2000 내역.csv", 0)
get_data_from_file(datalist, "CBOE Volatility Index 내역.csv", 0)
get_data_from_file(datalist, "스위스 SMI 내역.csv", 0)
num_datas += 6
print("read 1.3 done")

# 1.4
get_data_from_file(datalist, "닛케이 내역.csv", 0)
get_data_from_file(datalist, "상하이종합 내역.csv", 0)
num_datas += 2

# 1.9
get_data_from_file(datalist, "러시아 MOEX Russia 내역.csv", 0)
get_data_from_file(datalist, "RTSI 지수 내역.csv", 0)
num_datas += 2
print("read 1.9 done")

my_start_date = datetime.datetime.strptime("2007년 01월 01일", "%Y년 %m월 %d일")
my_end_date = datetime.datetime.strptime("2017년 12월 31일", "%Y년 %m월 %d일")

make_data(datalist, my_start_date, my_end_date, "input.txt", "output.txt", num_datas)
print("train data make done")

my_test_start_date = datetime.datetime.strptime("2018년 01월 01일", "%Y년 %m월 %d일")
my_test_end_date = datetime.datetime.strptime("2018년 12월 6일", "%Y년 %m월 %d일")

make_data(datalist, my_test_start_date, my_test_end_date, "input_test.txt", "output_test.txt", num_datas)
print("test data make done")
