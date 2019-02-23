import pandas as pd
import numpy as np

'''
-  Note : We consider given prediction's feature with :
  1. size_of_prediction is 1
     -> Program thinks prediction is results of specific amount
         - Ex : increase of market price.
  2. else
     -> Program thinks prediction is results of solving classification problem 
  
  Check option "MINIMUM_NUMBER_FOR_SETTING_CLASSIFICATION"

'''

class CP_Machine :
    def __init__(self,
                 answer, # or correct prediction
                 list_of_prediction,
                 size_of_prediction,
                 prediction_list_size):
        self.correct_answer = answer
        self.size_of_prediction = size_of_prediction
        self.prediction_list_size = prediction_list_size

        self.accuracy = 0

        if self.size_of_prediction >= 2: # MINIMUM_NUMBER_FOR_SETTING_CLASSIFICATION = 2
            self.classification = True
        else :
            self.classification = False

        if self.classification is True:
            temporary_list = list()
            temporary_list.append(self.correct_answer)
            self.correct_answer = self._classify_predict_lists(temporary_list)[0]

        self.predict_answer, self.df_predictions = self._extract_major_answer(list_of_prediction,
                                                                              self.size_of_prediction,
                                                                              self.classification)

        return

    def _classify_predict_lists(self,
                                lists):
        newlists = list()

        for each_list in lists:
            newlist = list()

            for item in each_list:
                # item : classification problem's array(hold each index's probability)
                max_value = np.max(item)
                max_index = np.where(item == max_value)

                newlist.append(int(max_index[0]))

            newlists.append(newlist)

        return newlists

    def _extract_major_answer(self,
                              lists,
                              each_list_size,
                              classification=False):
        # Classification
        # [1, 0.1, 0.2, 0.3, ...] => int 0
        # [0, 0, 1, 0, ..., 0 ]   => int 2
        if (classification is True) and (each_list_size is not 1):
            lists = self._classify_predict_lists(lists)

        # create pandas structure
        # (idx)       LIST1            LIST2            LIST3      ...
        #   1   (each_list_size) (each_list_size) (each_list_size)
        #   2          ...              ...             ...
        #  ...
        df_prediction = pd.DataFrame(lists).T

        if classification is False:
            # Get average of each list's answer
            # save it with pandas
            # (idx)       Average
            #   1    (each_list_size)
            #   2          ...
            #  ...
            df_answer = df_prediction.mean(axis=1,
                                           skipna=False)
        else:
            # Get major value from list's answers
            # save it with pandas
            # (idx)      Max_Frequency
            #   1      (each_list_size)
            #   2            ...
            #   ...
            df_answer = df_prediction.mode(axis=1)[0]

        new_answer = df_answer.values

        return new_answer, df_prediction

    def result(self):
        if len(self.predict_answer) != len(self.correct_answer):
            print(len(self.predict_answer), len(self.correct_answer))
            print("CP_Machine -> Result : Item size mismatch")
            raise ValueError

        if len(self.predict_answer) <= 0:
            print("CP_Machine -> Result : Item size is 0")
            raise ZeroDivisionError

        answer_count = 0
        compare_count = len(self.correct_answer)

        for i in range(0, len(self.predict_answer)):
            if self.predict_answer[i] == self.correct_answer[i]:
                answer_count += 1

        self.accuracy = float(answer_count) / float(compare_count)

        print("Accuracy : " + str(round(self.accuracy * 100, 4)) + "%")

        return

    def report(self):
        print("------------ Predictions -----------")
        print(self.df_predictions)

        for _ in range(0, 5):
            print("")

        print("-------- Predict results VS Answer ---------")
        print_list = list()
        print_list.append(self.predict_answer)
        print_list.append(self.correct_answer)

        df_compare = pd.DataFrame(print_list).T
        print(df_compare)

        for _ in range(0, 5):
            print("")

        print("-------- ACCURACY --------")
        self.result()

        return


