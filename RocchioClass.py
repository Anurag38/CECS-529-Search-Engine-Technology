from documents import DocumentCorpus, Document
from indexing import Index
from text.advancedtokenprocessor import AdvancedTokenProcessor
import numpy as np
from text import EnglishTokenStream




class RocchioClassifier:


    def __init__(self, index : Index, corpus : DocumentCorpus, jayCorpus : DocumentCorpus, 
                       hamCorpus : DocumentCorpus, madCorpus : DocumentCorpus, dispCorpus : DocumentCorpus):
        
        self.JAY = "JAY"
        self.HAMILTON = "HAMILTON"
        self.MADISON = "MADISON"
        self.PAPER_ID = "paper_52"
        self.index = index
        self.corpus = corpus
        self.centroids = {}
        self.jayDocs = []
        self.hamDocs = []
        self.madDocs = []
        it = jayCorpus.documents()
        for doc in it:
            self.jayDocs.append(doc.getTitle)
    
        it = hamCorpus.documents()
        for doc in it:
            self.hamDocs.append(doc.getTitle)

        it = madCorpus.documents()
        for doc in it:
            self.madDocs.append(doc.getTitle)
       
        print()
        self.centroids[self.JAY] = findCentroid(self.corpus, self.index, self.jayDocs)
        self.centroids[self.HAMILTON] = findCentroid(self.corpus, self.index, self.hamDocs)
        self.centroids[self.MADISON] = findCentroid(self.corpus, self.index, self.madDocs)

        self.dispCorpus = dispCorpus


    def printCentroids(self):
        print("first 10 components for Jay: ")
        for i in range(10):
            print(self.centroids[self.JAY][i])

        print()
        print("first 10 components for Hamilton: ")
        for i in range(10):
            print(self.centroids[self.HAMILTON][i])

        print()
        print("first 10 components for Madison: ")
        for i in range(10):
            print(self.centroids[self.MADISON][i])



    def classify(self) -> dict():
        dispDocOwners = {}
        it = self.dispCorpus.documents()

        for doc in it:
            # docVector = self.getVector(doc)
            docVector = getVector(self.index, doc, self.PAPER_ID)
            jayEucDist = calEuclidianDistance(docVector, self.centroids[self.JAY])
            hamEucDist = calEuclidianDistance(docVector, self.centroids[self.HAMILTON])
            madEucDist = calEuclidianDistance(docVector, self.centroids[self.MADISON])

            # print("Document: ", doc.getTitle)
            print("JAY: ", jayEucDist)
            print("Hamilton: ", hamEucDist)
            print("Madison: ", madEucDist)
            print()

            if (jayEucDist < hamEucDist and jayEucDist < madEucDist):
                dispDocOwners[doc.getTitle] = self.JAY

            elif (hamEucDist < madEucDist and hamEucDist < jayEucDist):
                dispDocOwners[doc.getTitle] = self.HAMILTON
            
            else:
                dispDocOwners[doc.getTitle] = self.MADISON

        return dispDocOwners


def getVector(index : Index, doc : Document, PAPER_ID : str):
    wdt = {}
    tkn_processor = AdvancedTokenProcessor()
    resVector = []
    docWeights = 0
    vocabList = index.getVocabulary()
    weight = 0
    eng = EnglishTokenStream(doc.getContent())
    for token in eng:
        tokenList = tkn_processor.process_token(token)
        for newToken in tokenList:
            if newToken in wdt:
                wdt[newToken] += 1
            else:
                wdt[newToken] = 1

    for key, value in wdt.items():
        weight = 1 + np.log(value)
        wdt[key] = weight
        docWeights += np.power(weight, 2) 

    docWeights = np.sqrt(docWeights)

    for term in vocabList:
        resVector.append(wdt.get(term, 0) / docWeights)

    
    print("Title: ", doc.getTitle)
    if (doc.getTitle == PAPER_ID):
        
        print("Vector Components = ")
        for i in range(30):
            print(resVector[i])
        print()

    return resVector



def findCentroid(corpus : DocumentCorpus, index : Index, inList : Document):
    it = corpus.documents()

    docVectorList = []
    vocabList = index.getVocabulary()

    for doc in it:
        if doc.getTitle in inList: # list of docs
            wdt = getWdtMap(doc)
            ld = getLd(wdt)
            docVector = []
            for term in vocabList:
                docVector.append(wdt.get(term, 0)/ld)

            docVectorList.append(docVector)
    # print("Length of docVectorList: ", len(docVectorList))
    return calculateCentroid(docVectorList)



def calEuclidianDistance(vec1 : list, vec2 : list):
    euclidianDist = 0
    for i in range(len(vec1)):
        euclidianDist += np.power(vec1[i] - vec2[i], 2)

    euclidianDist = np.sqrt(euclidianDist)
    
    return euclidianDist


def getWdtMap(doc : Document) -> dict:
    tkn_processor = AdvancedTokenProcessor()
    wdt_map = {}  #wdt
    eng = EnglishTokenStream(doc.getContent())
    for token in eng:
        tokenList = tkn_processor.process_token(token)
        for newToken in tokenList:
            if newToken in wdt_map:
                wdt_map[newToken] += 1
            else:
                wdt_map[newToken] = 1

    for key, value in wdt_map.items():
        weight = 1 + np.log(value)
        wdt_map[key] = weight

    return wdt_map


def getLd(wdt):
    docWeights = 0
    for key, value in wdt.items():
        docWeights += np.power(value, 2)
    ld = np.sqrt(docWeights)

    return ld


def calculateCentroid(docVectorList: list):
    length = len(docVectorList[0])
    centroid = []
    for i in range(length):
        sum = 0
        for j in range(len(docVectorList)):
            sum += docVectorList[j][i]

        sum = sum / len(docVectorList)
        centroid.append(sum)

    return centroid