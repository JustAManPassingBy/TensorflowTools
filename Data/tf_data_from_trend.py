from pytrends.request import TrendReq
import pandas as pd
import numpy as np
import datetime
import time
import csv
import random

### Keywords Example
# keywords_list = ['pizza'](["pizza"])
# keywords_list = ['pizza'. 'italian', 'spaghetti']

def append_data_into_list(initlist, appendlist, end_date, decrease_date, isfirst = False) :
    for data in appendlist :
        if (isfirst is True) :
            add_list = list()

            add_list.append(end_date)
            add_list.append(data[0])

            initlist.append(add_list)
        else :
            isfind = False
            
            for eachlist in initlist :
                if (end_date in eachlist) :
                    append_list = eachlist
                    isfind = True
                    break

            if (isfind is True) :
                append_list.append(data[0])

        end_date = end_date - decrease_date

    return initlist

def classify_col_item(col , isint = True) :
    col = col.replace(",", "")
    # check date
    if ("-" in col) and (":" in col) :
        col_item = datetime.datetime.strptime(col, "%Y-%m-%d %H:%M:%S")
    elif ("년" in col) and ("월" in col) and ("일" in col) :
        col_item = datetime.datetime.strptime(col, "%Y년 %m월 %d일")
    # check empty space
    elif (col == "") :
        col_item = float(0)
        if (isint is True) :
            col_item = int(col_item)
    # skip last item (%, M, G, etc...)
    elif (col[-1].isdigit()) is False :
        col_item = float(col[:-1])
        if (isint is True) :
            col_item = int(col_item)
    else :
        col_item = float(col)
        if (isint is True) :
            col_item = int(col_item)

    return col_item


def get_pytrend_info(initlist, startdate, enddate, keyword, catinfo, suggestion_id = -1, isfirst = False) :
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

    initlist = append_data_into_list(initlist, data_list, end_date, decrease_date, isfirst)

    return initlist, len(data_list), data_list

def write_all_list_in_csv(writelist, filename, call_count) :
    with open(filename, 'w', encoding='utf-8', newline='\n') as csv_file :
        csv_writer = csv.writer(csv_file)

        if (call_count != -1) :
            call_count_item = list()
            call_count_item.append(call_count)

            csv_writer.writerow(call_count_item)

        for item in writelist :
            csv_writer.writerow(item)


        print("Write Done, call count : ", str(call_count))

    return

def write_backup_list_in_csv(writelist, filename, write_count, suggestion_id) :
    with open(filename, 'w', encoding='utf-8', newline='\n') as file :
        file.write(str(write_count) + "\n")
        file.write(str(suggestion_id) + "\n")

        for writeitem in writelist:
            file.write(str(writeitem[0]) + "\n")

    return

def restore_all_list_from_csv(prevlist, filename) :

    try :
        with open(filename, 'r', encoding='utf-8', newline='\n') as csv_file :
            new_call_count = int(csv_file.readline())
            csv_reader = csv.reader(csv_file)


            for row in csv_reader :
                row_items = list()

                
                for col in row :
                    col_item = classify_col_item(col)
                    
                    row_items.append(col_item)      

            
                prevlist.append(row_items)


            print("Resore Done, call count : ", str(new_call_count))
    except :
        print("Restore Fail")
        new_call_count = -1

    return prevlist, new_call_count

def get_all_pytrend_infos(initlist, keyword, catinfo, suggestion_id, call_count, restore_call_count, original_isfirst = False) :
    if(restore_call_count >= call_count) :
        print("skip keyword : ", str(keyword))
        return initlist, (call_count + 1)
    
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
            initlist, item_cnt, curlist = get_pytrend_info(initlist, str(str(cur_year) + startdate_even), str(str(cur_year) + enddate_even),
                                        keyword, catinfo, suggestion_id, isfirst = original_isfirst)
        else :
            # Odd
            initlist, item_cnt, curlist = get_pytrend_info(initlist, str(str(cur_year) + startdate_odd),  str(str(cur_year) + enddate_odd),
                                        keyword, catinfo, suggestion_id, isfirst = original_isfirst)
        #print("Act : " + str(i) + " Done!")

        items += item_cnt
        
        for curitem in curlist:
            totallist.append(curitem)

        # We need time interval between getting pytrend info, so that avoid blocking from GOOGLE.
        time.sleep(60)

    print("Done !! item count : " + str(items))

    write_all_list_in_csv(initlist, "backup.csv", call_count)
    
    savedirname = "pytrends_data/" + str(keyword[0]) + ".tmp"
    write_backup_list_in_csv(totallist, savedirname, items, suggestion_id)

    return initlist, (call_count + 1)

def show_list_of_keyword(keyword) :
    #pytrends = TrendReq(hl='en-US',tz=360)
    pytrends = TrendReq(hl='ko', tz=540)
    #pytrends = TrendReq(hl='en-US', tz=360, proxies = {'https': 'https://34.203.233.13:80'})

    if (len(keyword) > 0) :
        keywords_list = keyword[0]
    else :
        keywords_list = keyword

    # print related suggestion
    print(pytrends.suggestions(keyword=keywords_list))

    return
    


newlist = list()
call = 0
newlist, restore_call = restore_all_list_from_csv(newlist, "backup.csv")

# 1
newlist, call = get_all_pytrend_infos(newlist, ["주가", "주식", "상승", "하락", "전망"], 0, 0, call, restore_call, original_isfirst = True)
'''
newlist, call = get_all_pytrend_infos(newlist, ["주식"], 0, 0,  call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["상승"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["하락"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["전망"], 0, -1, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["예측"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["보고서"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["낙관"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["비관"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["긍정"], 0, -1, call, restore_call)

# 11
newlist, call = get_all_pytrend_infos(newlist, ["부정"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["미래"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["기대"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["실망"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["걱정"], 0, 0, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["우려"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["충격"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["금융시장"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["경제 전망"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["이자율"], 0, 0, call, restore_call)

# 21
newlist, call = get_all_pytrend_infos(newlist, ["거래"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["수출"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["수입"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["원화가치"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["전쟁"], 0, 0, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["무역전쟁"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["관세"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["불안"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["위축"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["공포"], 0, 0, call, restore_call)

# 31
newlist, call = get_all_pytrend_infos(newlist, ["경기 후퇴"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["성장"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["활력"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["투자"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["소비"], 0, 0, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["소비심리"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["생산"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["소비세"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["지수"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["유가"], 0, -1, call, restore_call)

# 41
newlist, call = get_all_pytrend_infos(newlist, ["유출"], 0, 2, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["유입"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["은행"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["정부"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["국민"], 0, 0, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["자영업"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["4차 산업혁명"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["세금"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["연말정산"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["미래 기대"], 0, -1, call, restore_call)

# 51
newlist, call = get_all_pytrend_infos(newlist, ["호재"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["이자율 하락"], 0, -1,  call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["이자율 상승"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["수출 감소"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["수입 증가"], 0, -1, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["수입 감소"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["수출 증가"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["원화가치 하락"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["원화가치 상승"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["평화"], 0, 0, call, restore_call)

# 61
newlist, call = get_all_pytrend_infos(newlist, ["안정"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["관세 하락"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["관세 상승"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["규제 감소"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["규제 증가"], 0, -1, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["팽창"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["성장 둔화"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["호황"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["조건 완화"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["조건 강화"], 0, -1, call, restore_call)

# 71
newlist, call = get_all_pytrend_infos(newlist, ["빚 증가"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["빚 감소"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["성장률 증가"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["투자 증가"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["투자 감소"], 0, -1, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["소비 증가"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["소비 감소"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["생산 증가"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["생산 감소"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["세금 감면"], 0, -1, call, restore_call)

# 81
newlist, call = get_all_pytrend_infos(newlist, ["증세"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["유가 하락"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["유가 상승"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["자금 유입"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["자금 유출"], 0, -1, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["유동성 증가"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["유동성 감소"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["낙천적"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["투자 증가"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["일자리 증가"], 0, -1, call, restore_call)

# 91
newlist, call = get_all_pytrend_infos(newlist, ["투자 감소"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["일자리 감소"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["활력"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["부동산"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["금"], 0, 0, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["광물"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["안전자산"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["부도"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["고위험"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["고수익"], 0, -1, call, restore_call)
'''
'''        
# 1
newlist, call = get_all_pytrend_infos(newlist, ["주가"], 0, 0, call, restore_call, original_isfirst = True)
newlist, call = get_all_pytrend_infos(newlist, ["주식"], 0, 0,  call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["상승"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["하락"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["전망"], 0, -1, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["예측"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["보고서"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["낙관"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["비관"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["긍정"], 0, -1, call, restore_call)

# 11
newlist, call = get_all_pytrend_infos(newlist, ["부정"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["미래"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["기대"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["실망"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["걱정"], 0, 0, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["우려"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["충격"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["금융시장"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["경제 전망"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["이자율"], 0, 0, call, restore_call)

# 21
newlist, call = get_all_pytrend_infos(newlist, ["거래"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["수출"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["수입"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["원화가치"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["전쟁"], 0, 0, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["무역전쟁"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["관세"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["불안"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["위축"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["공포"], 0, 0, call, restore_call)

# 31
newlist, call = get_all_pytrend_infos(newlist, ["경기 후퇴"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["성장"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["활력"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["투자"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["소비"], 0, 0, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["소비심리"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["생산"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["소비세"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["지수"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["유가"], 0, -1, call, restore_call)

# 41
newlist, call = get_all_pytrend_infos(newlist, ["유출"], 0, 2, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["유입"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["은행"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["정부"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["국민"], 0, 0, call, restore_call)

newlist, call = get_all_pytrend_infos(newlist, ["자영업"], 0, -1, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["4차 산업혁명"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["세금"], 0, 0, call, restore_call)
newlist, call = get_all_pytrend_infos(newlist, ["연말정산"], 0, 0, call, restore_call)
# 49


# stock, rise, down, view, report
newlist = get_all_pytrend_infos(newlist, ["stock"], 0, 0, original_isfirst = True)
newlist = get_all_pytrend_infos(newlist, ["rise"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["down"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["view"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["report"], 0, 0)

# optimist
newlist = get_all_pytrend_infos(newlist, ["optimist"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["positive"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["negative"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["future"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["expect"], 0, 0)

newlist = get_all_pytrend_infos(newlist, ["worry"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["shock"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["interset rate"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["trade"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["war"], 0, 0)

newlist = get_all_pytrend_infos(newlist, ["uncertainty"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["boom"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["depress"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["freeze"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["recession"], 0, 0)

newlist = get_all_pytrend_infos(newlist, ["growth"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["anxiety"], 0, 0)
#newlist = get_all_pytrend_infos(newlist, ["rise"], 0)
newlist = get_all_pytrend_infos(newlist, ["slump"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["jump"], 0, 0)

newlist = get_all_pytrend_infos(newlist, ["announce"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["invest"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["consume"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["consumption tax"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["index"], 0, 0)

newlist = get_all_pytrend_infos(newlist, ["government"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["frb"], 0, 0)
'''

write_all_list_in_csv(newlist, "trend_data.csv", -1)
