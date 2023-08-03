import math
import os
import string
import sys
import time

from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim import corpora, models

from matplotlib import pyplot

import nltk
from nltk.cluster import KMeansClusterer
from nltk.cluster.gaac import GAAClusterer
from nltk import cluster
from nltk.stem.snowball import SnowballStemmer

import numpy as np

import pandas
import pickle

from scipy import linalg
from scipy.cluster.vq import vq, kmeans, whiten
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

from util import finch

nltk.download('stopwords')

def getLines(file_path):
    print("Loading file: ", file_path)
    lines = []
    with open(file_path, 'rb') as f:
        lines = [x.decode('utf8').strip() for x in f.readlines()]

    print("Corpus size:" + str(len(lines)))
    return lines

def secElem(elem):
    return elem[1]

def train(file_path, percentage, outputDir, lb):

    NUM_CLUSTERS=5

    ys_train_vecs = []
    train_sendersBinVecs = []
    probTransitionsMap = {}
    probs_filepath = ""
    vecmodel_filepath = ""

    outputWordVecFilePath = outputDir + '/' + fileName.replace(".csv","") + '-wordvec.pkl'
    outputKMeansFilePath = outputDir + '/' +fileName.replace(".csv","") + '-kmeans.pkl'

    print("wordvec-file ", outputWordVecFilePath)
    print("kmeans-file ", outputKMeansFilePath)

    outputWordVec = open(outputWordVecFilePath, 'wb')
    outputKMeans = open(outputKMeansFilePath, 'wb')

    vecmodel_filepath = outputDir + "/" + fileName.replace(".csv", "-ys_train_sendervecs.dat")
    print("vecmodel_filepath ", vecmodel_filepath)

    probs_filepath = outputDir + "/" + fileName.replace(".csv", "-probs.dat")
    probs_filepath = outputDir + "/" + fileName.replace(".txt", "-probs.dat")
    print("probs_filepath ", probs_filepath)

    lines = getLines(file_path)

    print("\n")
    print("Tokenizing sentences and vectorizing senders... ")

    corpus = finch.parseToTrain(lines, percentage)

    train_sentences = corpus["train_sentences"]
    train_rawSentences = corpus["train_rawSentences"]
    train_senders = corpus["train_senders"]
    test_sentences = corpus["test_sentences"]
    lastRoomsTrainIndex = corpus["lastRoomsTrainIndex"]

    ldamodel, train_sentences_not_toocommon = finch.doLDA(train_sentences, train_rawSentences, outputDir)

    all_sentences = []
    all_sentences.extend(train_sentences)
    all_sentences.extend(test_sentences) #needed because of wordvec dictionary

    print("\n")
    print("Vectorizing words from all_sentences with gensim.Word2Vec... ")
    model = Word2Vec(min_count=1, hs=1, negative=0)
    model.build_vocab(all_sentences)
    model.train(all_sentences, total_examples=model.corpus_count, epochs=100) #model.iter)

    # Pickle the list using the highest protocol available.
    pickle.dump(model, outputWordVec, -1)

    vocab_words = list(model.wv.index_to_key)
    print("len(list(model.wv.index_to_key)):", str(len(vocab_words)))
    words_vecs = model.wv
    print("Size of word vector: ", str(len(model.wv['sou'])))

    print("\n")
    print("Vectorizing sentences by computing the mean of word vectors... ")

    trainSentenceVecs = []
    for sentence in train_sentences:
        lastSent = sentence
        trainSentenceVec = np.ones(len(model.wv['sou']))
        for word in sentence:
            wordVec = np.array(model.wv[word])
            trainSentenceVec = np.add(trainSentenceVec, wordVec)

        trainSentenceVec = np.divide(trainSentenceVec,len(sentence))
        trainSentenceVecs.append(trainSentenceVec)

    print("len(trainSentenceVecs)", len(trainSentenceVecs))

    print("\n")
    print("Clustering with nltk.KMeansClusterer... ")
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=cluster.util.cosine_distance, repeats=25, avoid_empty_clusters=True)
    #ys_train= assigned_clusters
    ys_train= kclusterer.cluster(trainSentenceVecs, assign_clusters=True)

    pickle.dump(kclusterer, outputKMeans, -1)

    clusterBOW = {}
    topClusterBOW = {}
    for c in range(0,NUM_CLUSTERS):
        clusterBOW["Cluster " + str(c)] = {}
        topClusterBOW["Cluster " + str(c)] = {}

    totalWords = 0
    j = 0
    colors = []
    for clusterId in ys_train:
        if (clusterId==0):
            colors.append("lightskyblue")
        if (clusterId==1):
            colors.append("mediumblue")
        if (clusterId==2):
            colors.append("lightpink")
        if (clusterId==3):
            colors.append("deeppink")
        if (clusterId==4):
            colors.append("orangered")
        if (clusterId==5):
            colors.append("peachpuff")
        if (clusterId==6):
            colors.append("lime")
        if (clusterId==7):
            colors.append("lightgreen")

        words = train_sentences_not_toocommon[j]
        j+=1
        totalWords += len(words)
        for stemmedWord in words:
            if stemmedWord not in clusterBOW["Cluster " + str(clusterId)]:
                clusterBOW["Cluster " + str(clusterId)][stemmedWord] =  1
            else:
                clusterBOW["Cluster " + str(clusterId)][stemmedWord] +=  1

    #replace words with one occurrence by unique value
    j = 0
    for clusterId in ys_train:
        clusterBOW["Cluster " + str(clusterId)]["_UNIQUE_"] = 0
        words = train_sentences_not_toocommon[j]
        j+=1
        for stemmedWord in words:
            if (clusterBOW["Cluster " + str(clusterId)][stemmedWord] ==  1):
                clusterBOW["Cluster " + str(clusterId)]["_UNIQUE_"] += 1
                del clusterBOW["Cluster " + str(clusterId)][stemmedWord]
        if(clusterBOW["Cluster " + str(clusterId)]["_UNIQUE_"] == 0):
            del clusterBOW["Cluster " + str(clusterId)]["_UNIQUE_"]


    #compute log frequencies
    for clusterId in range(0,NUM_CLUSTERS):
        wordList = []
        for stemmedWord in clusterBOW["Cluster " + str(clusterId)]:
            freq = math.log10((clusterBOW["Cluster " + str(clusterId)][stemmedWord]*1.)/totalWords)
            clusterBOW["Cluster " + str(clusterId)][stemmedWord] =  freq
            wordList.append( (stemmedWord, freq) )

        #keep top words only
        topWordList = []
        if(len(wordList)>10):
            wordList.sort(key=secElem, reverse=True)

            h = 20
            if(len(wordList)<20):
                h = len(wordList)<20
            for w in range(0,h):
                topWordList.append(wordList[w])
        else:
            topWordList = wordList

        topWordDict = dict(topWordList)

        for stemmedWord in clusterBOW["Cluster " + str(clusterId)]:
            if stemmedWord in topWordDict:
                topClusterBOW["Cluster " + str(clusterId)][stemmedWord] = clusterBOW["Cluster " + str(clusterId)][stemmedWord]

    print("topClusterBOW: ")
    print(topClusterBOW)
    print("\n")

    #add collumn for chart
    topClusterBOW["Top Words"] = {}
    for clusterId in range(0,NUM_CLUSTERS):
        for stemmedWord in topClusterBOW["Cluster " + str(clusterId)]:
            if stemmedWord not in topClusterBOW["Top Words"]:
                topClusterBOW["Top Words"][stemmedWord] = stemmedWord


    df = pandas.DataFrame(topClusterBOW)
    df = df.set_index('Top Words')

    g = sns.clustermap(df, metric="jaccard", cmap="mako", robust=True, figsize=(18, 10))
    ms = time.time()*1000.0
    g.fig.savefig(outputDir + "/finch-cluster-bow-map-"+str(ms)+".png")

    # fit a 2d PCA model to the vectors
    pca = PCA(n_components=2)
    result = pca.fit_transform(trainSentenceVecs)
    # create a scatter plot of the projection
    pyplot.clf()
    pyplot.scatter(result[:, 0], result[:, 1], c=colors)
    pyplot.legend()

    ms = time.time()*1000.0
    pyplot.savefig(outputDir + '/finch_KMeansClusterer-PCA-'+str(NUM_CLUSTERS)+"-"+str(ms)+'.png', dpi=200)

    print("\n")
    print("Creating binary cluster vectors for training data...")

    ys_train_sendervecsFile = open(vecmodel_filepath, "a")
    i = 0
    for y in ys_train:
        binStrVec = finch.getClusterBinVector(y)

        ys_train_vecs.append(binStrVec)

        binSenderStrVec = ""
        for sender in train_senders[i]:
            binSenderStrVec += str(sender)
        train_sendersBinVecs.append(binSenderStrVec)
        ys_train_sendervecsFile.write(binStrVec + "\t" + binSenderStrVec + "\n")
        i += 1

        if(len(ys_train_vecs)==1):
            print(ys_train_vecs)


    ys_train_sendervecsFile.close()

    if(len(ys_train_vecs)==0 and vecmodel_filepath!=""):
        print("Loading file: ", vecmodel_filepath)
        p = 0
        with open(vecmodel_filepath, 'rb') as f:
            lines = [x.decode('utf8').strip() for x in f.readlines()]
            for line in lines:
                line = line.replace("\n", "")
                strLine = line.split("\t")
                ys_train_vecs.append(strLine[0])
                train_sendersBinVecs.append(strLine[1])

    if(len(ys_train_vecs)>0):
        print("\n")
        print("Learning the transitions... ")
        numSenders = 0
        statesMap = {}
        transitionsMap = {}
        states = []
        transitions = []
        for i in range(0,len(ys_train_vecs)-1):

            state = ( ys_train_vecs[i], train_sendersBinVecs[i] )
            numSenders = len(train_sendersBinVecs[i])

            if(state not in statesMap):
                statesMap[state] = 1
            else:
                statesMap[state] += 1
            states.append(state)

            if ( i>1 and finch.shouldAdd(i, lastRoomsTrainIndex, lb)):
                prevStateSenderVec = []
                prevStateClusterVec = []
                for w in range(1,lb+1):
                    if (i-w) >= 0:
                        y, sv = states[i-w]
                        if i<=5:
                            print("states[i-"+str(w)+"]:", states[i-w])
                        prevStateClusterVec.extend( finch.getSenderBinVec(y))
                        prevStateSenderVec.extend( finch.getSenderBinVec(sv))
                prevStateSenderVec = np.array(prevStateSenderVec)
                prevStateClusterVec = np.array(prevStateClusterVec)
                y_lb = finch.getBinStrVec(prevStateClusterVec)
                y_lb, svmin1 =  states[i-1] # ignoring content from lb > 1
                sv_lb = finch.getSenderBinStrVec(prevStateSenderVec)
                prevState = (y_lb, sv_lb)
                if i<=5:
                   print("prevStateSenderVec:", prevStateSenderVec)
                transition = ( prevState,  train_sendersBinVecs[i] )
                if i<=5:
                    print(transition)
                if(transition not in transitionsMap):
                    transitionsMap[transition] = 1
                else:
                    transitionsMap[transition] += 1
                transitions.append(transition)

        print("len(states):" + str(len(states)))
        print("len(transitions):" + str(len(transitions)))
        print("len(statesMap):", str(len(statesMap)))
        print("len(transitionsMap):", str(len(transitionsMap)))

        print("\n")
        print("Computing probabilities... ")

        probsFile = open(probs_filepath, "a")
        clusterTransMapDic = {}
        clusterTransMapDic["Transition"] = {}
        minProb = 0.
        prevProb = 0.
        maxProb = 0.

        for transition in transitionsMap:
            prevState,nextState = transition

            prob =  (transitionsMap[transition]+1)/(len(transitions)+len(statesMap))
            if(prob>maxProb):
                maxProb = prob
            if(prob<prevProb):
                minProb = prob
            prevProb = prob

            #transition = (prevState,nextState,prob)
            probTransitionsMap[(prevState,nextState,prob)] = transitionsMap[transition]

            strCluster,sender = prevState

            strSender = finch.getSenderBinStrVec(sender)
            if nextState != "None":
                strSender2 = finch.getSenderBinStrVec(nextState)
            else:
                strSender2 = "None"
            # observed probabilities
            # previous state: (clusterid,  sender ) -> (prob) next state ( = sender)
            probsFile.write(strCluster + "\t" + strSender + "\t" + str(prob) + "\t" + strSender2 + "\n")

            clusterName = finch.getClusterName(strCluster)
            tid = strSender + "->" + strSender2
            if tid not in clusterTransMapDic["Transition"]:
                clusterTransMapDic["Transition"][tid] = finch.getSenderName(strSender) + "->" + finch.getSenderName(strSender2)

            if (clusterName not in clusterTransMapDic):
                clusterTransMapDic[clusterName] = {}
            clusterTransMapDic[clusterName][tid] = prob

        probsFile.close()

        df = pandas.DataFrame(clusterTransMapDic)
        df = df.set_index('Transition')

        g = sns.clustermap(df, metric="jaccard", cmap="mako", robust=True, figsize=(18, 10))
        ms = time.time()*1000.0
        g.fig.savefig(outputDir + "/finch-transitions-map-"+str(ms)+".png")

        print("maxProb: ", str(maxProb))
        print("minProb: ", str(minProb))
        print("\n")

    outputWordVec.close()
    outputKMeans.close()

    print("**************************** Finished Training ******************************")


if __name__ == "__main__":

    print("\n **************** Started *************************** \n")

    percentage = 0.5
    lb = 1
    file_path="data/finch/finch_multi_dialogue.txt"

    filePathDirs = file_path.split("/")
    fileName = filePathDirs[len(filePathDirs)-1]
    filePathDir = file_path.replace(fileName, "")
    ms = time.time()*1000.0
    outputDir = filePathDir + "models_" + str(ms)

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        print("Created output directory [" +outputDir + "], all model files will be saved there.")

    train(file_path, percentage, outputDir, lb)
