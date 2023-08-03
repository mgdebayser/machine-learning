import sys
import os
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from gensim import corpora, models
import pandas
import seaborn as sns
import time
import numpy as np
import random

stemmer = SnowballStemmer("portuguese")
stopWordsFile = os.path.join(os.path.dirname(__file__), 'stopwords-pt.txt')

def getSenderVec(sender, mapping):
    ret = [0 for i in range(len(mapping))]
    ret[mapping[sender]] = 1
    return ret

def getSendersMapping(lines):
    mapping = {}
    idx = 0
    for line in lines:
        lineInfo = line.split("|")
        sender = lineInfo[3]
        if(sender not in mapping):
            mapping[sender] = idx
            idx+=1
    return mapping

def getSenderName(strSenderBin):

    if(strSenderBin=="None"):
        return "None"
    names = []
    for i in range(0,len(strSenderBin)):
        if i == 0 and strSenderBin[i] == "1":
            names.append("investimentoGuru")
        if i == 1 and strSenderBin[i] == "1":
            names.append("poupancaGuru")
        if i == 2 and strSenderBin[i] == "1":
            names.append("cdbGuru")
        if i == 3 and strSenderBin[i] == "1":
            names.append("tesouroGuru")
        if i == 4 and strSenderBin[i] == "1":
            names.append("finch-user")

    if len(names) == 0:
        names.append("Unkown")
    ret = ""
    for i in range(0,len(names)):
        if i != len(names)-1:
            ret += names[i] + ","
        else:
            ret += names[i]
    return ret

def getSenderBinVec(sender):
    vec = []
    for n in sender:
        vec.append(int(n))
    return vec

def getBinVec(strBin):
    vec = []
    for n in strBin:
        vec.append(int(n))
    return vec

def getSenderBinStrVec(senderVec):
    binSenderStrVec = ""
    for sender in senderVec:
        binSenderStrVec += str(sender)
    return binSenderStrVec

def getBinStrVec(vec):
    binStrVec = ""
    for c in vec:
        binStrVec += str(c)
    return binStrVec

def getClusterBinVector(clusterId):
    if(str(clusterId)=="0"):
        return "10000"
    if(str(clusterId)=="1"):
        return "01000"
    if(str(clusterId)=="2"):
        return "00100"
    if(str(clusterId)=="3"):
        return "00010"
    if(str(clusterId)=="4"):
        return "00001"

def getClusterName(strClusterBin):
    if(strClusterBin=="10000"):
        return "Cluster 0"
    if(strClusterBin=="01000"):
        return "Cluster 1"
    if(strClusterBin=="00100"):
        return "Cluster 2"
    if(strClusterBin=="00010"):
        return "Cluster 3"
    if(strClusterBin=="00001"):
        return "Cluster 4"

def doLDA(train_sentences, train_rawSentences, outputDir):

    numTopics = 10

    stopwordsList = []
    with open(stopWordsFile, "r") as data:
        for line in data:
            stopwordsList = line.split(",")

    stopwordsList.extend(stopwords.words('portuguese'))

    stopwordsList.extend(["é", "eh", "e", "de", "a", "o", "nossa", "para"])

    toocommon = set(stopwordsList)
    train_sentences_not_toocommon = []
    for sentence in train_sentences:
        non_toocommon = [i for i in sentence if not i in toocommon]
        stemmed_sentence_non_toocommon = []
        for word in non_toocommon:
            stemmed_word = stemmer.stem(word)
            if stemmed_word not in stemmed_sentence_non_toocommon:
                stemmed_sentence_non_toocommon.append(stemmed_word)
        train_sentences_not_toocommon.append(stemmed_sentence_non_toocommon)

    dictionary = corpora.Dictionary(train_sentences_not_toocommon)
    corpus = [dictionary.doc2bow(sentence) for sentence in train_sentences_not_toocommon]
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=numTopics, id2word=dictionary)
    print(ldamodel.print_topics(num_topics=numTopics, num_words=4))

    wordTopicMapDic = {}
    wordTopicMapDic["Topic"] = {}
    for topicId in range(0,numTopics):
        print("topic: " + str(topicId))

        wordTopicMapDic["Topic"][topicId] = "Topic " + str(topicId)
        wordPairs = ldamodel.get_topic_terms(topicId)
        for k,v in wordPairs:
                word = ldamodel.id2word[k]
                print(word+"-"+str(v))
                if (word not in wordTopicMapDic):
                    wordTopicMapDic[word] = {}
                wordTopicMapDic[word][topicId] = v

        print("\n")
    print(wordTopicMapDic)

    df = pandas.DataFrame(wordTopicMapDic)
    df = df.set_index('Topic')
    #del df.index.name

    g = sns.clustermap(df, metric="jaccard")
    ms = time.time()*1000.0
    g.fig.savefig(outputDir + "/finch-lda-topics-map-"+str(numTopics)+"-"+str(ms)+".png")

    return ldamodel, train_sentences_not_toocommon

def getAllWords(utterance):
    allWordsInText = []
    if(utterance!=""):
        allWordsInText = utterance.lower().split()
        #remove punctuations
        allWordsInText = [''.join(c for c in s if c not in string.punctuation) for s in allWordsInText]
        allWordsInText = [s for s in allWordsInText if s]
    return allWordsInText

def shouldAdd(i, uttLastindexList, lb):
    for j in range(0,lb-1):
        if (i-j) in uttLastindexList:
            return False
    return True

def getCorpus(lines):
    corpus = []
    i = 0
    lastSender = ""
    lastTuple = None
    lastRoomsIndex = []
    lid = ""
    lroom = ""
    ids = {}
    for line in lines:
        lineInfo = line.split("|")
        id = lineInfo[0]
        room = lineInfo[2]
        sender = lineInfo[3]
        utterance = lineInfo[4]
        allWordsInText = getAllWords(utterance)

        if utterance.strip() != "" and len(allWordsInText) > 0:
            if lastTuple != None:
                li, lid, lroom, lsender, lutt, lall = lastTuple
                if (lroom != room):
                    lastRoomsIndex.append(i-1)

            #same sender and not first utterance in a room
            if(lastSender == sender and (i-1) not in lastRoomsIndex):
                utterance = utterance + " - " + lutt
            elif (lastTuple != None):
                    corpus.append( lastTuple )

            i += 1
            lastSender = sender
            lastTuple = (i, id, room, sender, utterance, allWordsInText)

    if (lastTuple != None):
        corpus.append( lastTuple )

    print("getCorpus - len(corpus): ", len(corpus))
    print("diff after merge: ", str(len(lines) - len(corpus)))
    return corpus

def getParticipantCorpus(lines, participant):
    corpus = []
    i = 0
    for line in lines:
        lineInfo = line.split("|")
        id = lineInfo[0]
        room = lineInfo[2]
        sender = lineInfo[3]
        utterance = lineInfo[4]

        if(sender == participant):
            allWordsInText = getAllWords(utterance)
            corpus.append( (i, id, room, sender, utterance, allWordsInText) )
            i += 1

    return corpus

def parseToTrain(lines, percentage):
    MAX = len(lines) * percentage
    train_sentences = []
    train_senders = []
    train_rawSentences = []
    test_rawSentences = []
    test_sentences = []
    test_senders = []
    lastRoomsTrainIndex = []
    lastRoomsTestIndex = []
    lastRoom = ""
    corpus = getCorpus(lines)
    utt = ""
    count = 0
    mapping = getSendersMapping(lines)
    for i, id, room, sender, utterance, allWordsInText in corpus:
        try:
            utt = utterance
            if(i<MAX):
                if (lastRoom != room and lastRoom != ""):
                    lastRoomsTrainIndex.append(count-1)

                train_rawSentences.append(utterance)
                train_sentences.append(allWordsInText)
                train_senders.append(getSenderVec(sender, mapping))

            else:
                if (lastRoom != room and lastRoom != ""):
                    lastRoomsTestIndex.append(count-1)

                test_rawSentences.append(utterance)
                test_sentences.append(allWordsInText)
                test_senders.append(getSenderVec(sender, mapping))
            lastId = id
            lastRoom = room
            count += 1
        except:
             print(utt)
             print( "Unexpected error on encoding line["+utt+ "]. \nError: "  )
             print (sys.exc_info())
             sys.exit(-1)
    corpus = {}
    corpus["train_sentences"] = train_sentences
    corpus["train_senders"] = train_senders
    corpus["train_rawSentences"] = train_rawSentences
    corpus["test_sentences"] = test_sentences
    corpus["test_rawSentences"] = test_rawSentences
    corpus["test_senders"] = test_senders
    corpus["lastRoomsTrainIndex"] = lastRoomsTrainIndex
    corpus["lastRoomsTestIndex"] = lastRoomsTestIndex
    return corpus

def getPrevStatesTestMLE(test_sentences, test_senders, model, kclusterer, lb=0):
    prevStates_test = []
    if(lb==0):
        for i in range(0,len(test_sentences)):
            sentence = test_sentences[i]
            senderVec = test_senders[i]

            binSenderStrVec = ""
            for sender in senderVec:
                binSenderStrVec += str(sender)

            sentenceVec = np.ones(len(model.wv['investir']))
            for word in sentence:
                try:
                    wordVec = np.array(model.wv[word])
                    sentenceVec = np.add(sentenceVec, wordVec)
                except:
                    print("Word ["+word+"] not in dictionary, it will be skipped.")


            sentenceVec = np.divide(sentenceVec,len(sentence))
            clusterId = kclusterer.classify_vectorspace(sentenceVec)

            binStrVec = getClusterBinVector(clusterId)
            prevStates_test.append( (binStrVec, binSenderStrVec) )
    else:
        for i in range(0,len(test_senders)):
            sentence = test_sentences[i]
            sentenceVec = np.ones(len(model.wv['investir']))
            for word in sentence:
                try:
                    wordVec = np.array(model.wv[word])
                    sentenceVec = np.add(sentenceVec, wordVec)
                except:
                    print("Word ["+word+"] not in dictionary, it will be skipped.")

            sentenceVec = np.divide(sentenceVec,len(sentence))
            clusterId = kclusterer.classify_vectorspace(sentenceVec)

            binStrVec = getClusterBinVector(clusterId)

            prevStateSenderVec = []
            for w in range(0,lb):
                if (i-w) >= 0:
                    prevStateSenderVec.extend( test_senders[i-w] )
            binSenderStrVec = ""
            for sender in prevStateSenderVec:
                binSenderStrVec += str(sender)
            binStrVec = getClusterBinVector(clusterId)# ignoring content from lb > 1

            prevStates_test.append( (binStrVec, binSenderStrVec) )
    return prevStates_test

def getPrevStatesTestBasicMLE(test_senders, lb=0):
    prevStates_test = []
    if lb == 0:
        for i in range(0,len(test_senders)):
            senderVec = test_senders[i]
            binSenderStrVec = ""
            for sender in senderVec:
                binSenderStrVec += str(sender)
            prevStates_test.append( binSenderStrVec )
    else:
        for i in range(0,len(test_senders)):
            prevState = []#np.array(test_senders[i])
            for w in range(0,lb):
                if (i-w) >= 0:
                    #prevState = np.add(prevState, np.array(test_senders[i-w]))
                    prevState.extend(test_senders[i-w])
            binSenderStrVec = ""
            for sender in prevState:
                if int(sender) > 1:
                    sender = "1"
                binSenderStrVec += str(sender)

            prevStates_test.append( binSenderStrVec )
    return prevStates_test

def parseToTest(lines, percentage, model, kclusterer, lb=0):
    corpus = parseToTrain(lines, percentage)
    prevStates_test = getPrevStatesTestMLE(corpus["test_sentences"], corpus["test_senders"], model, kclusterer, lb)
    return prevStates_test, corpus["test_senders"], corpus["lastRoomsTestIndex"]

def parseToTestBasicMLE(lines, percentage, lb=0):
    corpus = parseToTrain(lines, percentage)
    prevStates_test = getPrevStatesTestBasicMLE(corpus["test_senders"], lb)
    return prevStates_test, corpus["test_senders"], corpus["lastRoomsTestIndex"]

def getTestSentenceVecs(corpus, model, lb=0):
    test_sentences = []
    for i in range(1,len(corpus["test_sentences"])):
        allWordsInText = corpus["test_sentences"][i]

        if(lb==0 or lb==1):
            binSenderVec = corpus["test_senders"][i]
        else:
            binSenderVec = []
            for w in range(0,lb):
                if (i-w) >= 0:
                    binSenderVec.extend(corpus["test_senders"][i-w])
                    if i<=5:
                        print("test_senders["+str(i)+"-"+str(w)+"]:", corpus["test_senders"][i-w])
            if i<=5:
                print("binSenderVec:", binSenderVec)
                print("\n")

        sentenceVec = np.ones(len(model.wv['investir']))
        for word in allWordsInText:
            try:
                wordVec = np.array(model.wv[word])
                sentenceVec = np.add(sentenceVec, wordVec)
            except:
                print("Word ["+word+"] not in dictionary, it will be skipped.")

        sentenceVec = np.divide(sentenceVec,len(allWordsInText))
        test_sentences.append( (sentenceVec, binSenderVec) )

    corpus["test_sentences"] = test_sentences
    return corpus

def parseToTestSVM(lines, percentage, model, lb=0):
    corpus = parseToTrain(lines, percentage)
    return getTestSentenceVecs(corpus, model, lb)

def getTopNext(nextProbs):
    topVal = 0.
    topPartName = "None"
    topPartId = "0000000"
    for nextProbId in nextProbs:
        #print(getSenderName(nextProbId) + ":" + str(nextProbs[nextProbId]))
        if float(nextProbs[nextProbId]) > topVal:
            topVal = float(nextProbs[nextProbId])
            topPartName = getSenderName(nextProbId)
            topPartId = nextProbId
    return topPartName, topVal, topPartId

def getRandomNext(nextProbs):
    i = random.randint(0, len(nextProbs)-1)
    j = 0
    for key in nextProbs:
        nextProbId = key
        topVal = nextProbs[key]
        if i == j:
            break

    topVal = float(nextProbs[nextProbId])
    topPartName = getSenderName(nextProbId)
    topPartId = nextProbId
    return topPartName, topVal, topPartId
