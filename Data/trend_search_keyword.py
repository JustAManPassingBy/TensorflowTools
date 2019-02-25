from pytrends.request import TrendReq

def search(keyword) :
    #pytrends = TrendReq(hl='en-US',tz=360)
    pytrends = TrendReq(hl='ko', tz=540)
    #pytrends = TrendReq(hl='en-US', tz=360, proxies = {'https': 'https://34.203.233.13:80'})

    if keyword is not list :
        tem_keyword = list()
        tem_keyword.append(keyword)
        keyword = tem_keyword
    if (len(keyword) > 0) :
        keywords_list = keyword[0]
    else :
        keywords_list = keyword

    # print related suggestion
    print(pytrends.suggestions(keyword=keywords_list))

    return

