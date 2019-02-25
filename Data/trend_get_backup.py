import datetime
import os

def append_data_into_list(initlist, appendlist, end_date, decrease_date, isfirst=False):
    for data in appendlist:
        if (isfirst is True):
            add_list = list()

            add_list.append(end_date)
            add_list.append(int(data))

            initlist.append(add_list)
        else:
            isfind = False

            for eachlist in initlist:
                if (end_date in eachlist):
                    append_list = eachlist
                    isfind = True
                    break

            if (isfind is True):
                append_list.append(int(data))

        end_date = end_date - decrease_date

    return initlist, False


def classify_col_item(col, isint = True):
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

def restore_all_item_from_backup(prevlist, filename, isfirst):
    startdate = datetime.datetime.strptime("2018-12-31", "%Y-%m-%d")
    datediff = datetime.timedelta(days=1)

    with open(filename, 'r', encoding='utf-8', newline='\n') as file:
        itemcnt = int(file.readline())
        opt = int(file.readline())
        lines = file.readlines()

        if (itemcnt < 4383) or (opt < -1):
            return prevlist, isfirst

        if filename is "trend_data.csv":
            print("skip trend data")
            return prevlist, isfirst

        nextlist, isfirst = append_data_into_list(prevlist, lines, startdate, datediff, isfirst=isfirst)

        print("restore : " + filename)

        file.close()

    return nextlist, isfirst


def write_all_list_in_csv(writelist, filename, call_count=-1):
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

newlist = list()
isfirst = True

file_list = os.listdir("./")

for file_name in file_list:
    newlist, isfirst = restore_all_item_from_backup(newlist, file_name, isfirst)

write_all_list_in_csv(newlist, "trend_data.csv", -1)