from pytrends.request import TrendReq
import pandas as pd
import numpy as np
import datetime
import time
import csv

# date : YYYY-MM-DD
def get_pytrend_info(initlist, startdate, enddate, keyword, catinfo, isfirst = False) :
    pytrends = TrendReq(hl='en-US',tz=360)
    #pytrends = TrendReq()
    #pytrends = TrendReq(hl='en-US', tz=360, proxies = {'https': 'https://34.203.233.13:80'})

    # Keywords
    ### Example
    # keywords_list = ['pizza'](["pizza"])
    # keywords_list = ['pizza'. 'italian', 'spaghetti']
    keywords_list = keyword

    # options
    # cat = category (number (0 = all, 
    # see categories https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
    # geo = Conuntry (United state = 'US'), defaults to world all
    # tz  = timezone offset (??)
    # timeframe  "YYYY-MM-DD YYYY-MM-DD
    # gpropg : google property (images, news, youtubes)
    pytrends.build_payload(keywords_list, cat=catinfo, timeframe=str(startdate + " " + enddate), geo='US', gprop='')

    # Get data info
    getdatainfo = pytrends.interest_over_time()

    # panda data management
    # Delete 'ispartial' column
    del getdatainfo['isPartial']

    # change data info to numpy array
    data_list = np.array(getdatainfo)

    # reverse order
    # high date comes in 
    data_list = np.flipud(data_list)

    # create start date_array, increase date_array
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


    # save all items in initlist
    return initlist

def get_all_pytrend_infos(initlist, keyword, catinfo, original_isfirst = False) :
    # Get_data time (each repeat gets 6 month datas)
    repeat_time = 2 * 12

    # start_year
    start_year = 2018

    # may you can touch
    startdate_even = '-06-01'
    enddate_even = '-12-31'

    startdate_odd = '-01-01'
    enddate_odd = '-05-31'

    print("Keyword : " + str(keyword))
    
    for i in range(0, repeat_time) :
        print("Act : " + str(i))
        cur_year = int(start_year - (i / 2))
        
        if (i % 2 == 0) :
            initlist = get_pytrend_info(initlist, str(str(cur_year) + startdate_even),
                                        str(str(cur_year) + enddate_even), keyword, catinfo, isfirst = original_isfirst)
        else :
            initlist = get_pytrend_info(initlist, str(str(cur_year) + startdate_odd),
                                        str(str(cur_year) + enddate_odd), keyword, catinfo, isfirst = original_isfirst)
        print("Act : " + str(i) + " Done!")
    
        time.sleep(5)

    return initlist

def write_all_list_in_csv(writelist, filename) :

    with open(filename, 'w', encoding='utf-8', newline='\n') as csv_file :
        csv_writer = csv.writer(csv_file)

        for item in writelist :
            item[0] = item[0].strftime("%Y년 %m월 %d일")
            
            csv_writer.writerow(item)

    return

newlist = list()

## Todo (검색어 -> 주제)
# stock, rise, down, view, report
newlist = get_all_pytrend_infos(newlist, ["stock"], 0, original_isfirst = True)
newlist = get_all_pytrend_infos(newlist, ["rise"], 0)
newlist = get_all_pytrend_infos(newlist, ["down"], 0)
newlist = get_all_pytrend_infos(newlist, ["view"], 0)
newlist = get_all_pytrend_infos(newlist, ["report"], 0)

# optimist
newlist = get_all_pytrend_infos(newlist, ["optimist"], 0)
newlist = get_all_pytrend_infos(newlist, ["positive"], 0)
newlist = get_all_pytrend_infos(newlist, ["negative"], 0)
newlist = get_all_pytrend_infos(newlist, ["future"], 0)
newlist = get_all_pytrend_infos(newlist, ["expect"], 0)

newlist = get_all_pytrend_infos(newlist, ["worry"], 0)
newlist = get_all_pytrend_infos(newlist, ["shock"], 0)
newlist = get_all_pytrend_infos(newlist, ["interset rate"], 0)
newlist = get_all_pytrend_infos(newlist, ["trade"], 0)
newlist = get_all_pytrend_infos(newlist, ["war"], 0)

newlist = get_all_pytrend_infos(newlist, ["uncertainty"], 0)
newlist = get_all_pytrend_infos(newlist, ["boom"], 0)
newlist = get_all_pytrend_infos(newlist, ["depress"], 0)
newlist = get_all_pytrend_infos(newlist, ["freeze"], 0)
newlist = get_all_pytrend_infos(newlist, ["recession"], 0)

newlist = get_all_pytrend_infos(newlist, ["growth"], 0)
newlist = get_all_pytrend_infos(newlist, ["anxiety"], 0)
#newlist = get_all_pytrend_infos(newlist, ["rise"], 0)
newlist = get_all_pytrend_infos(newlist, ["slump"], 0)
newlist = get_all_pytrend_infos(newlist, ["jump"], 0)

newlist = get_all_pytrend_infos(newlist, ["announce"], 0)
newlist = get_all_pytrend_infos(newlist, ["invest"], 0)
newlist = get_all_pytrend_infos(newlist, ["consume"], 0)
newlist = get_all_pytrend_infos(newlist, ["consumption tax"], 0)
newlist = get_all_pytrend_infos(newlist, ["index"], 0)

newlist = get_all_pytrend_infos(newlist, ["government"], 0)
newlist = get_all_pytrend_infos(newlist, ["frb"], 0)


write_all_list_in_csv(newlist, "trend_data.csv")
