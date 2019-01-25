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
            keywords_list = keywords_suggestions[suggestion_id]
    else :
        # If suggestion is not required (suggestion_id is -1)
        keywords_list = keyword
            
    print("Keyword :: " + str(keywords_list))
    
    ## options for build_payload
    # cat        : category (number (0 = all, ...))
    #               SEE categories https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
    # geo        : Conuntry (United state = 'US'), defaults to world all
    # tz         : timezone offset (??)
    # timeframe  : format should be YYYY-MM-DD YYYY-MM-DD"
    # gprop      : google property (images, news, youtubes)
    pytrends.build_payload(keywords_list, cat=catinfo, timeframe=str(startdate + " " + enddate), geo='KR', gprop='')

    getdatainfo = pytrends.interest_over_time()

    # Delete 'isPartial' column
    #del getdatainfo['isPartial']

    # change data info to numpy array
    data_list = np.array(getdatainfo)

    # reverse order
    data_list = np.flipud(data_list)

    end_date = datetime.datetime.strptime(enddate, "%Y-%m-%d")
    decrease_date =datetime.timedelta(days = 1)

    for data in data_list :
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

def write_all_list_in_csv(writelist, filename) :
    with open(filename, 'w', encoding='utf-8', newline='\n') as csv_file :
        csv_writer = csv.writer(csv_file)

        for item in writelist :
            # Change datetime struct to Korean date display method
            item[0] = str(item[0])
            
            csv_writer.writerow(item)

    csv_file.close()
    
    return

def restore_all_list_from_csv(prevlist, filename, isfirst = False) :
    if (len(prevlist) != 0) or (isfirst is True) :
        return

    with open(filename, 'w', encoding='utf-8', newline='\n') as csv_file :
        csv_reader = csv.reader(csv_file)

        for row in csv_reader :
            row_items = list()
                
            for col in row :
                col = col.replace(",", "")
                # check date
                if ("-" in col) and (":" in col) :
                    col_item = datetime.datetime.strptime(col, "%Y년 %m월 %d일")
                # check empty
                elif (col == "") :
                    col_item = float(0)
                # skip last item
                elif (col[-1].isdigit()) is False :
                    col_item = float(col[:-1])
                else :
                    col_item = float(col)
                    
                row_items.append(col_item)      
            
            prevlist.append(row_items)

    return

def get_all_pytrend_infos(initlist, keyword, catinfo, suggestion_id, original_isfirst = False) :
    #restore_all_list_from_csv(initlist, "backup.csv", original_isfirst)
    
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
    
    for i in range(0, repeat_count) :
        print("Act : " + str(i))
        cur_year = int(start_year - (i / 2))
        
        if (i % 2 == 0) :
            # Even
            initlist = get_pytrend_info(initlist, str(str(cur_year) + startdate_even), str(str(cur_year) + enddate_even),
                                        keyword, catinfo, suggestion_id, isfirst = original_isfirst)
        else :
            # Odd
            initlist = get_pytrend_info(initlist, str(str(cur_year) + startdate_odd),  str(str(cur_year) + enddate_odd),
                                        keyword, catinfo, suggestion_id, isfirst = original_isfirst)
        print("Act : " + str(i) + " Done!")

        # We need time interval between getting pytrend info, so that avoid blocking from GOOGLE.
        time.sleep(random.randrange(30, 60))

    write_all_list_in_csv(initlist, "backup_" + str(len(initlist[0]) - 1) + ".csv")

    return initlist

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

# 1
newlist = get_all_pytrend_infos(newlist, ["주가"], 0, 0, original_isfirst = True)
newlist = get_all_pytrend_infos(newlist, ["주식"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["상승"], 0, -1)
newlist = get_all_pytrend_infos(newlist, ["하락"], 0, -1)
newlist = get_all_pytrend_infos(newlist, ["전망"], 0, -1)

newlist = get_all_pytrend_infos(newlist, ["예측"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["보고서"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["낙관"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["비관"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["긍정"], 0, -1)

# 11
newlist = get_all_pytrend_infos(newlist, ["부정"], 0, -1)
newlist = get_all_pytrend_infos(newlist, ["미래"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["기대"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["실망"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["걱정"], 0, 0)

newlist = get_all_pytrend_infos(newlist, ["우려"], 0, -1)
newlist = get_all_pytrend_infos(newlist, ["충격"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["금융시장"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["경제 전망"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["이자율"], 0, 0)

# 21
newlist = get_all_pytrend_infos(newlist, ["거래"], 0, -1)
newlist = get_all_pytrend_infos(newlist, ["수출"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["수입"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["원화가치"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["전쟁"], 0, 0)

newlist = get_all_pytrend_infos(newlist, ["무역전쟁"], 0, -1)
newlist = get_all_pytrend_infos(newlist, ["관세"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["불안"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["위축"], 0, -1)
newlist = get_all_pytrend_infos(newlist, ["공포"], 0, 0)

# 31
newlist = get_all_pytrend_infos(newlist, ["경기 후퇴"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["성장"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["활력"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["투자"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["소비"], 0, 0)

newlist = get_all_pytrend_infos(newlist, ["소비심리"], 0, -1)
newlist = get_all_pytrend_infos(newlist, ["생산"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["소비세"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["지수"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["유가"], 0, -1)

# 41
newlist = get_all_pytrend_infos(newlist, ["유출"], 0, 2)
newlist = get_all_pytrend_infos(newlist, ["유입"], 0, -1)
newlist = get_all_pytrend_infos(newlist, ["은행"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["정부"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["국민"], 0, 0)

newlist = get_all_pytrend_infos(newlist, ["자영업"], 0, -1)
newlist = get_all_pytrend_infos(newlist, ["4차 산업혁명"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["세금"], 0, 0)
newlist = get_all_pytrend_infos(newlist, ["연말정산"], 0, 0)
# 49


'''
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

write_all_list_in_csv(newlist, "trend_data.csv")
