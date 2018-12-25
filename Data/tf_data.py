import numpy as np
import datetime
import csv

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
                    # if not, find from prev_list
                    else :
                        for list_item in prev_list :
                            if (my_date in list_item) :
                                target_list = list_item
                        
                # else if count >= 7
                elif (item_count >= 7) :
                    col = col.replace(",","")

                    # check data type
                    if (col == "") :
                        col_item = float(0)
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
            
    return

def make_data (data_list, startdate, enddate, filename, output_count, num_data) :
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

            total_item += 1

            row_array=list()

            # input write
            for i in range (1, total_index) :
                #my_file.write(str(round(prev_list[i] - prev_prev_list[i], 2)) + "\t")
                row_array.append(round(prev_list[i] , 2))
                #if (prev_list[i] >= 0) :
                #    my_file.write("1.0\t")
                #else :
                #    my_file.write("0.0\t")

            # output write
            for i in range(3, 4) :
                if ((each_list[i]) > 0) :
                    row_array.append(float(1.0))
                    row_array.append(float(0.0))
                else :
                    row_array.append(float(0.0))
                    row_array.append(float(1.0))

            # add newline
            csv_writer.writerow(row_array)

            valid_item += 1

            prev_prev_list = prev_list
            prev_list = each_list
        

    print(":: Make File info ::")
    print("File : " + filename)
    print("items : ", valid_item, " / ", total_item, " index(input) : ", total_index - 1, " index(output) : ", output_count)


    
    return


datalist = list()

# date info
num_datas = 1

# 1.1
get_data_from_csv_file(datalist, "ITALY.csv", isfirst = True)
get_data_from_csv_file(datalist, "TA 35 내역.csv")
num_datas += 2
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
num_datas += 17
print("read 1.2 done")

# 1.3
get_data_from_csv_file(datalist, "다우존스 내역.csv")
get_data_from_csv_file(datalist, "S&P 500 내역.csv")
get_data_from_csv_file(datalist, "나스닥 내역.csv")
get_data_from_csv_file(datalist, "Russell 2000 내역.csv")
get_data_from_csv_file(datalist, "CBOE Volatility Index 내역.csv")
get_data_from_csv_file(datalist, "스위스 SMI 내역.csv")
num_datas += 6
print("read 1.3 done")

# 1.4
get_data_from_csv_file(datalist, "닛케이 내역.csv")
get_data_from_csv_file(datalist, "상하이종합 내역.csv")
num_datas += 2

# 1.9
get_data_from_csv_file(datalist, "러시아 MOEX Russia 내역.csv")
get_data_from_csv_file(datalist, "RTSI 지수 내역.csv")
num_datas += 2
print("read 1.9 done")

my_start_date = datetime.datetime.strptime("2007년 01월 01일", "%Y년 %m월 %d일")
my_end_date = datetime.datetime.strptime("2017년 12월 31일", "%Y년 %m월 %d일")

make_data(datalist, my_start_date, my_end_date, "train.txt", 2, num_datas)
print("train data make done")

my_test_start_date = datetime.datetime.strptime("2018년 01월 01일", "%Y년 %m월 %d일")
my_test_end_date = datetime.datetime.strptime("2018년 12월 6일", "%Y년 %m월 %d일")

make_data(datalist, my_test_start_date, my_test_end_date, "test.txt", 2, num_datas)
print("test data make done")
