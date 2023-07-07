from indexing import InvertedIndex, SoundexIndex, DiskIndexWriter, DiskPositionalIndex
from variants.defaultvariant import DefaultVariant
from variants.tfidfvariant import TfidfVariant
from variants.okapivariant import OkapiVariant
from variants.wackyvariant import WackyVariant
from text.advancedtokenprocessor import AdvancedTokenProcessor
from text.soundextokenprocessor import SoundexTokenProcessor
from documents import DocumentCorpus, DirectoryCorpus
from queries import BooleanQueryParser
from text import EnglishTokenStream
from porter2stemmer import Porter2Stemmer
import numpy as np
import heapq as hq
import time
import os
import math

from RocchioClass import RocchioClassifier


def load_directory(path):
    if os.listdir(path)[0].endswith('.json'):
        dd = DirectoryCorpus.load_json_directory(path, ".json")
        fTyp = 1
    else:
        dd = DirectoryCorpus.load_text_directory(path, ".txt")
        fTyp = 0
    return dd, fTyp


def index_corpus(corpus : DocumentCorpus, corpus_path: str):
    print("Indexing started....")
    start = time.time()
    tkn_processor = AdvancedTokenProcessor()
    # soundex_processor = SoundexTokenProcessor()
    inverted_index = InvertedIndex()
    # soundex_index = SoundexIndex()
    # diw = DiskIndexWriter()
    # docWeights_dict = {}
    # docLengthA = 0
    # totalDocs = 0

    for d in corpus:
        # doc_data = []  # docWeights, docLenD, byteSize, avg(tftd) in order
        stream = EnglishTokenStream(d.getContent())
        # byteSize = d.getByteSize()
        # wdt_sum = 0
        # this_doc_hash = {}
        number_of_tokens = 0

        # Adding tokens to index
        for position, s in enumerate(stream):
            processed_token_list = tkn_processor.process_token(s)
            inverted_index.add_term(processed_token_list, d.id, position + 1)
            number_of_tokens += 1

    elapsed = time.time() - start
    print("Finished Indexing. Elapsed time = " + time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:11],
                                                               time.gmtime(elapsed)))

    return inverted_index

if __name__ == "__main__":

    # filePath = ""
    while(True):



        # filePath = input("Enter the path for corpus: ")    
        # filePath = input()
        # filePath = Path
        filePath = "C:/Users/anura/Desktop/CSULB/Sem 1/CECS_529_SET/Homework/final_milestone/MySearchEngine-main/Data/federalist-papers"
        allPath = filePath + "\\ALL"
        jayFilePath = filePath + "\\JAY"
        hamFilePath = filePath + "\\HAMILTON"
        madFilePath = filePath + "\\MADISON"
        dispFilePath = filePath + "\\DISPUTED"
        

        # corpus = load_directory(corpus_path)

        corpus = DirectoryCorpus.load_text_directory(allPath, ".txt")
        jayCorpus = DirectoryCorpus.load_text_directory(jayFilePath, ".txt")
        hamCcorpus = DirectoryCorpus.load_text_directory(hamFilePath, ".txt")
        madCorpus = DirectoryCorpus.load_text_directory(madFilePath, ".txt")
        dispCorpus = DirectoryCorpus.load_text_directory(dispFilePath, ".txt")

        index = index_corpus(corpus, allPath)
        
        rocchioClassifier = RocchioClassifier(index, corpus, jayCorpus, hamCcorpus, madCorpus, dispCorpus)

        res = rocchioClassifier.classify()
        print("Vocabulary: ")
        print(index.getVocabulary())
        
        rocchioClassifier.printCentroids()

        print()
        for key, value in res.items():
            print(key, " Author: ", value)

        print("Do you want to continue: ")
        choice = input()
        if (choice == 'y'):
            continue
        else:
            break