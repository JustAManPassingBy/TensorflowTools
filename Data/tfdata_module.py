import numpy as np
import pandas as pd
import datetime
import csv

class Data_Manager:
    def __init__(self,
                 initialize_type="NUMBER"):
        
        self._initialize_list(initialize_type)
        
        return

    
    def _initialize_list(self,
                         initialize_type="NUMBER"): # NUMBER, DATE ...        
        if initialize_type is "NUMBER" :
            self.data_type = "NUMBER"
        elif initialize_type is "DATE" :
            self.data_type = "DATE"
        else :
            print("initialize_list : Unknown initialiize type : " + str(initialize_type))
            raise(ValueError)

        self.data_is_first = True

        return

    def append_csv_list(self,
                        filename, # We only get csv type data
                        skiprows=False, # Use (1), ([0, 1, 3, ....])
                        usecols=False, # Use (1), ([0, 1, 3, ....])
                        index_col=False) : # False means there are no index 
        
        csv_result = pd.read_csv(filename,  
                                 skiprows=skiprows,
                                 index_col=index_col,
                                 usecols=usecols)

        csv_result = _extract_valid_columns(csv_result, self.data_is_first)

        if (self.data_is_first is True):
            self.pandas_list=csv_result
            self.data_is_first = False
        else :
            self.pandas_list = pd.concat([self.pandas_list, csv_result], axis=1)     

        return

    def _make_data(self,
                   data_list, 
                   filename, 
                   output_count) :

    total_index = len(data_list[0])
    valid_item = 0
    total_item = len(data_list)

    # will use TSV instead of CSV
    with open(filename, 'w', encoding='utf-8', newline='\n') as csv_file :
        csv_writer = csv.writer(csv_file, delimiter='\t')

        for each_list in data_list :
            row_array=list()

            # input / output write
            for i in range (0, total_index) :
                if (i == 1) : continue

                row_array.append(each_list[i])
            
            # add newline
            csv_writer.writerow(row_array)

            valid_item += 1
        

    print(":: Make File info ::")
    print("File : " + filename)
    print("items : ", valid_item, " / ", total_item, " index(input) : ", total_index - output_count, " index(output) : ", output_count)

    return

    def get_sample_from_csv(self,
                            train_file,
                            test_file,
                            output_count):
        # sort by 2nd[1] index
        sort_pandas = self.pandas_list.sort_values(by=self.pandas_list.columns[1])

        train_pandas = sort_pandas.loc[ sort_pandas[sort_pandas.columns[1]] == 0, : ]
        test_pandas = sort_pandas.loc[ sort_pandas[sort_pandas.columns[1]] == 0, : ]

        # numpy
        train_np = train_pandas.values()
        test_np = test_pandas

        self._make_data(train_np, train_file, output_count)
        self._make_data(test_np, test_file, output_count)
    
    '''
    def _restore_trend_list_from_file(self,
                                      filename):
        
        return

    def restore_trend_data(self,
                           filename):
        read_list = list()
        read_list = _restore_trend_list_from_file(filename)



    # each list : ["keyword", suggestion_id]
    # Return    : ["Keyword", suggestion_id, restore_success(T/F)]
    def _restore_trend_list(self,
                            read_list):
        scan_trend_list = list()
        
        # scan all items

        for read_item in read_list:
            if read_item in scan_trend_list:
                # restore process

            else :
                read_item.append(False) 

        return read_list

    def _request_google_trend_data_once(self,
                                        startdate,
                                        enddate,
                                        kw_list,
                                        catinfo,
                                        suggestion_ids):
        #pytrends = TrendReq(hl='en-US',tz=360)
        pytrends = TrendReq(hl='ko', tz=540)
        #pytrends = TrendReq(hl='en-US', tz=360, proxies = {'https': 'https://34.203.233.13:80'})

        if (suggestion_id != -1) :
            # Get suggestion keywords from original keyword if required (suggestion_id is not -1)
            keywords_suggestions = pytrends.suggestions(keyword=keyword[0])

            if (len(keywords_suggestions) is 0) :
                # Case could not find keyword suggestion
                keywords_list = keyword
            else :
                # Case find keyword suggestion, with using suggestion id
                keywords_list = list()
                keywords_list.append(keywords_suggestions[0].get("mid"))
        else :
            # If suggestion is not required (suggestion_id is -1)
            keywords_list = keyword
            
        #print("Keyword :: " + str(keywords_list))
    
        ## options for build_payload
        # cat        : category (number (0 = all, ...))
        #               SEE categories https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
        # geo        : Conuntry (United state = 'US'), defaults to world all
        # tz         : timezone offset (??)
        # timeframe  : format should be YYYY-MM-DD YYYY-MM-DD"
        # gprop      : google property (images, news, youtubes)
        pytrends.build_payload(kw_list=keywords_list, cat=catinfo, timeframe=str(startdate + " " + enddate), geo='KR', gprop='')

        getdatainfo = pytrends.interest_over_time()

        # Delete 'isPartial' column
        #del getdatainfo['isPartial']

        print(getdatainfo)

        data_list = np.array(getdatainfo)

        # reverse order
        data_list = np.flipud(data_list)

        end_date = datetime.datetime.strptime(enddate, "%Y-%m-%d")
        decrease_date = datetime.timedelta(days = 1)

        return initlist, len(data_list), data_list

    def _request_google_trend_data(self,
                                   pd_list,
                                   startdate
                                   read_list,
                                   catinfo=0):
         # (each repeat count gets 6 month datas)
        repeat_count = 2 * 12
    
        start_year = 2018

        # even = YYYY-06-01 ~ YYYY-12-31
        # odd  = YYYY-01-01 ~ YYYY-05-31
        startdate_even = '-06-01'
        enddate_even = '-12-31'

        startdate_odd = '-01-01'
        enddate_odd = '-05-31'

        print("Keyword : " + str(keyword))

        items = 0
        totallist = list()

        for i in range(0, repeat_count) :
            #print("Act : " + str(i))
            cur_year = int(start_year - (i / 2))

            if (i % 2 == 0) :
                # Even
                initlist, item_cnt, curlist = self._request_google_trend_data_once(initlist, str(str(cur_year) + startdate_even), str(str(cur_year) + enddate_even),
                                        keyword, catinfo, suggestion_id, isfirst = original_isfirst)
            else :
                # Odd
                initlist, item_cnt, curlist = _request_google_trend_data_once(initlist, str(str(cur_year) + startdate_odd),  str(str(cur_year) + enddate_odd),
                                        keyword, catinfo, suggestion_id, isfirst = original_isfirst)
            items += item_cnt
        
        for curitem in curlist:
            totallist.append(curitem)

        # We need time interval between getting pytrend info, so that avoid blocking from GOOGLE.
        time.sleep(60)

    print("Done !! item count : " + str(items))
    
    return initlist, (call_count + 1)


    def _get_google_trend_data(self,
                               read_item,
                               pending_items,
                               collect_lists,
                               is_last=False):
        if (is_last is False):
            pending_items.append(read_item[0:-2])
            collect_lists += 1

        if (is_last is True) or (collect_lists == 5):
            self._request_google_trend_data(pending_items)
            collect_lists = 0
            pending_items = list()

        return collect_lists, pending_items

    def _get_google_trend_list(self,
                               read_list):

        current_trend_need_items = 0
        pending_read_items=list()

        for read_item in read_list:
            if read_item[2] is False :
                current_trend_need_items, pendinig_read_items = self._get_google_trend_data(read_item,
                                                                                            pending_read_items,
                                                                                            current_trend_need_items)
        if (current_trend_need_items != 0) :
            _ , _ = self._get_google_trend_data(False,
                                            pending_read_items,
                                            current_trend_need_items,
                                            is_last=True)

        return

    def get_google_trends_read_list(self,
                                    filename):
        self.trend_read_list = pd.read_csv(filename).to_numpy()

        self.trend_read_list = self._restore_trend_list(self.trend_read_list)

        self._get_google_trend_list(self.trend_read_list)

        




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

def _make_data_with_date (data_list, startdate, enddate, filename, output_count, num_data, multiple_data) :
    # Divide with tab 
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
    '''

dm = Data_Manager()
dm.append_csv_list(filename, False, False, False)
dm.get_sample_from_csv("train.txt", "test.txt", 3)