import copy
import math
import pickle
import os
import time
import sys
import string

from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim import corpora, models
import numpy as np
from matplotlib import pyplot
from nltk import cluster
from nltk.cluster import KMeansClusterer
from nltk.cluster.gaac import GAAClusterer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy import linalg
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

from util import finch, metrics

sys.stdout.flush()

def test(file_path, percentage, lb, wordvecPath, kmeansPath, probs_filepath):
    print("\n **************** Started *************************** \n")

    print("\n")
    print("Loading gensim.Word2Vec model from file ["+wordvecPath+"]... ", flush=True)

    wordvecFile = open(wordvecPath, 'rb')
    model = pickle.load(wordvecFile)

    vocab_words = list(model.wv.index_to_key)
    print("len(list(model.wv.index_to_key)):", str(len(vocab_words)))

    print("\n")
    print("Loading nltk.KMeansClusterer model from file ["+kmeansPath+"]... ", flush=True)

    kmeansFile = open(kmeansPath, 'rb')
    kclusterer = pickle.load(kmeansFile)

    lines = []
    with open(file_path, 'rb') as f:
        lines = [x.decode('utf8').strip() for x in f.readlines()]

    print("\n")
    print("Tokenizing sentences and vectorizing senders... ")

    prevStates_test, test_senders, lastPrevStatesIndex = finch.parseToTest(lines, percentage, model, kclusterer, lb)

    print("\n")
    print("Testing data size:")
    print("len(prevStates_test): ", str(len(prevStates_test)))
    print("len(test_senders): ", str(len(test_senders)))

    print("\n")
    print("Loading transitions model from file ["+ probs_filepath + "]...")
    print("\n")

    lines = []
    with open(probs_filepath, 'rb') as f:
        lines = [x.decode('utf8').strip() for x in f.readlines()]

    i = 0
    nextStates_pred = []
    nextStates_pred_list = []
    nextStates_test = []
    print("\n")
    print("Predicting transitions for sentence...")
    for binStrVec, binSenderStrVec in prevStates_test:
        foundNextSenders = {}
        for line in lines:
            print(line)
            transitionInfo = line.split("\t")
            clusterBinStr = transitionInfo[0]
            senderBinStr = transitionInfo[1]
            prob = transitionInfo[2]
            nextSenderBinStr = transitionInfo[3]

            if(binStrVec==clusterBinStr and binSenderStrVec==senderBinStr):
                if nextSenderBinStr not in foundNextSenders:
                    foundNextSenders[nextSenderBinStr] = prob
                else:
                    lastProb = foundNextSenders[nextSenderBinStr]
                    if prob > lastProb:
                        foundNextSenders[nextSenderBinStr] = prob

        topSender, topVal, topSenderBinStr = finch.getTopNext(foundNextSenders)
        foundNextSendersWithoutTop = copy.deepcopy(foundNextSenders)
        if topSenderBinStr in foundNextSendersWithoutTop:
            del foundNextSendersWithoutTop[topSenderBinStr]
            secTopSender, secTopVal, secTopSenderBinStr = finch.getTopNext(foundNextSendersWithoutTop)
        else:
            secTopSender, secTopVal, secTopSenderBinStr = topSender, topVal, topSenderBinStr

        if ( i>1 and finch.shouldAdd(i,lastPrevStatesIndex, lb) and i < (len(test_senders)-1)):
        #if ( i>1 and i not in lastPrevStatesIndex and i < (len(test_senders)-1)):
        #if (i < (len(prevStates_test)-1) and (i not in lastPrevStatesIndex)):
            nextStates_pred.append(int(topSenderBinStr,2))
            nextStates_pred_list.append( (int(topSenderBinStr,2), int(secTopSenderBinStr,2) ) )
            nextState_test = finch.getSenderBinStrVec(test_senders[i+1])
            nextStates_test.append(int(nextState_test,2))

            if i <= 3:
                print("\n")
                print("binStrVec:", binStrVec)
                print("binSenderStrVec:", binSenderStrVec)
                print("foundNextSenders:", foundNextSenders)
                print("topSender:", topSender)
                print("secTopSender:",secTopSender)
                print("nextSender:", finch.getSenderName(nextState_test))
                for nextSenderBinStr in foundNextSenders:
                    print(finch.getSenderName(nextSenderBinStr) + ":" + str(foundNextSenders[nextSenderBinStr]))
                print("\n")
                print("nextStates_pred: ", nextStates_pred)
                print("nextStates_test: ", nextStates_test)

        i += 1

    print("\n")

    nextStates_test = np.array(nextStates_test)
    nextStates_pred = np.array(nextStates_pred)

    # nextStates_test -> nextStates observed in the testing data
    # nextStates_pred -> nextStates predicted in the training data for each previousState in the testing data
    print("nextStates_test: ", str(len(nextStates_test)))
    print("nextStates_pred: ", str(len(nextStates_pred)))

    print("\n")

    metrics.computeMetrics(nextStates_test, nextStates_pred)
    metrics.computeMetrics_manyTops(nextStates_test, nextStates_pred_list)
    print("**************************** Finished ******************************")

if __name__ == "__main__":
    percentage = 0.7
    lb = 1

    print("\n **************** Started *************************** \n")

    file_path="data/finch/finch_multi_dialogue.txt"
    wordvecPath = "data/finch/exec_1691001610269.7148/finch_multi_dialogue.txt-wordvec.pkl"
    kmeansPath = "data/finch/exec_1691001610269.7148/finch_multi_dialogue.txt-kmeans.pkl"
    probs_filepath = "data/finch/exec_1691001610269.7148/finch_multi_dialogue-probs.dat"
    test(file_path, percentage, lb, wordvecPath, kmeansPath, probs_filepath)
