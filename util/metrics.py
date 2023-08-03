import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

def computeMetrics_manyTops(Y_test, Y_pred_list):
    ## True positive (TP): detects a next participant which happened
    ## False positive (FP): does not detect a next participant which happened
    ## True negative (TN): does not detect a next participant which did not happen
    ## False negative (FN): detects a next participant which did not happen

    if(len(Y_test)>0 and len(Y_pred_list)>0):
        if(len(Y_test)==len(Y_pred_list)):
            truePos = 0.
            falsePos = 0.
            trueNeg = 0.
            falseNeg = 0.
            for i in range(0,len(Y_test)):
                happened = Y_test[i]
                top1, top2 = Y_pred_list[i]

                if happened == top1 or happened == top2:
                    truePos += 1
                else:
                    falsePos += 1

            acc = truePos / (truePos + falsePos)
            print("accuracy_manyTops")
            print( acc )


def computeMetrics(Y_test, Y_pred):
    if(len(Y_test)>0 and len(Y_pred)>0):
        if(len(Y_test)==len(Y_pred)):
            print("Computing evaluation metrics...")
            try:
                print("accuracy_score")
                print(accuracy_score(Y_test, Y_pred))
            except:
                print ("got exception on accuracy_score")
                print (sys.exc_info())
            # try:
            #     print("classification_report")
            #     print(classification_report(Y_test, Y_pred))
            # except:
            #     print ("got exception on classification_report")
            #     print (sys.exc_info())
            try:
                print("confusion_matrix")
                print(confusion_matrix(Y_test, Y_pred))
            except:
                print ("got exception on confusion_matrix")
                print (sys.exc_info())
            # try:
            #     print("mean_squared_error")
            #     print(mean_squared_error(Y_test, Y_pred))
            # except:
            #     print ("got exception on mean_squared_error")
            #     print (sys.exc_info())
            # try:
            #     print("f1_score")
            #     print(f1_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred)))
            # except:
            #     print ("got exception on f1_score")
            #     print (sys.exc_info())
        else:
            print("Could not compute evaluation metrics. Lists have different sizes.")
    else:
        print("Could not compute evaluation metrics. Empty list.")


def computeProbsPercent(probs_filepath, value):


    if(probs_filepath==""):
        print("\n [computeProbsPercent]Please, specify the probs file path to load predictor. \n")
        sys.exit(0)

    with open(probs_filepath) as fp:
        line = fp.readline()
        min = None
        max = None

        while line:

            fields = line.split("\t");
            prob =  float(fields[1])

            if min is None:
                min = prob
            else:
                if prob<min:
                    min = prob

            if max is None:
                max = prob
            else:
                if prob>max:
                    max = prob

            line = fp.readline()

        threshold = ((max-min)*float(value))+min
        return threshold
