import csv
import sys

# Print costs
# Get arrays of training epochs and each step's cost
# Save it onto file name
# epoch_array, cost_array : array for epoch, cost [Notice : function only consider epoch's array length]
# filename                : location that two relations will be saved
def print_cost(epoch_array, cost_array, filename) :
    showcostfile = open(filename, "w")

    for cnt in range (0, len(epoch_array)) :
        showcostfile.write('Epoch')
        showcostfile.write('{:7d}'.format(epoch_array[cnt]))
        showcostfile.write('done\tCost : ')
        showcostfile.write('{:.9f}'.format(cost_array[cnt]))
        showcostfile.write("\n")
    showcostfile.close()

    return
# End of function


# Function print
# X        : Array that want to make for output file
# Filename : Output filename
def print_data(X, filename) :
    with open(filename, "w", encoding="utf-8", newline= '') as csvDataFile:
        writer = csv.writer(csvDataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        for item in X :
            writer.writerow(item)

    csvDataFile.close()
# End of function


# Print result function
# Print relation between expected value(X) and real value(Y) in stdout
# X : expected value's array
# Y : real value's array
def print_result(X, Y, interval=1) :
    row = max(len(X), len(Y))
    col_X = len(X[0])
    col_Y = len(Y[0])

    item = 0
    correct = 0

    for cur_row in range(0, row, interval) :
        print("Expect : ", end ='')

        for (cur_col) in range(0, col_X) :
            print(X[cur_row][cur_col], end ='')
            print(", ", end ='')

        print(" ==> Real: ", end ='')
        for (cur_col) in range(0, col_Y) :
            print(Y[cur_row][cur_col], end ='')
            print(", ", end ='')

        #check
        item += 1

        if (X[cur_row][cur_col] * Y[cur_row][cur_col] >= 0) :
            correct += 1

        print("")


    print("Accuracy with p/m : " + str(float(correct) / float(item) * 100.0))
# End of function

# Print All layer Function
def print_all_layer_function(sess, filename, wlist, blist, total_layer, direct_bridge=False):
    origin_stdout, sys.stdout = sys.stdout, open(filename, "w")

    print(" --- W0 --- ")
    print(sess.run([wlist[0]]))

    if direct_bridge is False:
        print(" --- B0 --- ")
        print(sess.run(blist[0]))

    for i in range(1, total_layer - 1):
        print("\n ====== NEW LAYER ======\n")

        print(" --- W" + str(i) + " --- ")
        print(sess.run(wlist[i]))

        print(" --- B" + str(i) + " --- ")
        if (direct_bridge is True):
            print(sess.run(blist[i - 1]))
        else:
            print(sess.run(blist[i]))
    sys.stdout = origin_stdout

    return
# End of function