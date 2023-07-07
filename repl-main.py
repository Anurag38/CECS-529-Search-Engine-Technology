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


def print_10docs_with_scores(doc_dict, dd):
    heap = [(-value, key) for key, value in doc_dict.items()]
    largest = hq.nsmallest(10, heap)
    largest = [(key, -value) for value, key in largest]
    for tup in largest:
        print(dd.get_document(int(tup[0])).getTitle, end="")
        print(" - " + str(tup[1]))


def print_variant_names():
    print("1. Default method")
    print("2. tf-idf method")
    print("3. Okapi BM25")
    print("4. Wacky")


def print_docs_with_docId(doc_id, dd):
    cont = EnglishTokenStream(dd.get_document(int(doc_id)).getContent())
    count = 0
    for ss in cont:
        if count == 20:
            count = 0
            print()
        print(ss, end=" ")
        count += 1


def load_directory(path):
    if os.listdir(path)[0].endswith('.json'):
        dd = DirectoryCorpus.load_json_directory(path, ".json")
        fTyp = 1
    else:
        dd = DirectoryCorpus.load_text_directory(path, ".txt")
        fTyp = 0
    return dd, fTyp


def index_corpus(corpus: DocumentCorpus, typ: int, corpus_path: str):
    # Typ 0 - .txt
    # Typ 1 - .json
    print("Indexing started....")
    start = time.time()
    tkn_processor = AdvancedTokenProcessor()
    soundex_processor = SoundexTokenProcessor()
    inverted_index = InvertedIndex()
    soundex_index = SoundexIndex()
    diw = DiskIndexWriter()
    docWeights_dict = {}
    docLengthA = 0
    totalDocs = 0

    for d in corpus:
        doc_data = []  # docWeights, docLenD, byteSize, avg(tftd) in order
        stream = EnglishTokenStream(d.getContent())
        byteSize = d.getByteSize()
        wdt_sum = 0
        this_doc_hash = {}
        number_of_tokens = 0

        # Adding tokens to index
        for position, s in enumerate(stream):
            processed_token_list = tkn_processor.process_token(s)
            inverted_index.add_term(processed_token_list, d.id, position + 1)
            number_of_tokens += 1

            # calculating tftd
            if processed_token_list[-1] in this_doc_hash.keys():
                this_doc_hash[processed_token_list[-1]] += 1
            else:
                this_doc_hash[processed_token_list[-1]] = 1

        # Calculate (wdt)^2 sum
        for key, value in this_doc_hash.items():
            # Get wdt for every term from tftd using this_doc_hash
            wdt = 1 + np.log(value)
            wdt_sum += wdt * wdt

        # Calculate ld and insert into ld dict for this document
        doc_data.append(math.sqrt(wdt_sum))
        # docWeights_dict[d.id] = math.sqrt(wdt_sum)

        # Insert No. of Tokens in the dictionary
        doc_data.append(number_of_tokens)
        # docLengthD_dict[d.id] = number_of_tokens

        # Insert bytesize
        doc_data.append(byteSize)

        # Insert Avg(tftd) = (total terms / total unique terms)
        if len(this_doc_hash) != 0:
            doc_data.append(number_of_tokens / len(this_doc_hash))
        else:
            doc_data.append(0)

        # Update docLengthA
        docLengthA += 1

        # increment doc count
        totalDocs += 1

        # add docdata to docW dict
        docWeights_dict[d.id] = doc_data

        # getting authors for soundex
        if typ == 1:
            auth = EnglishTokenStream(d.getAuthor())
            for ss in auth:
                soundex_index.add_term(soundex_processor.process_token(ss), d.id)

    # docLengthA = avg number of tokens in all documents in corpus. i.e. number of tokens of corpus/total no. of docs
    docLengthA = len(inverted_index.getEntireVocab().keys())/docLengthA

    # write inverted index to disk
    diw.writeIndex(inverted_index, corpus_path, docWeights_dict, docLengthA, totalDocs)
    # write soundex index to disk
    diw.writeSoundexIndex(soundex_index, corpus_path)

    elapsed = time.time() - start
    print("Finished Indexing. Elapsed time = " + time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:11],
                                                               time.gmtime(elapsed)))


def boolean_mode(dpIndex, dd, path):
    while True:
        pList = []
        query = input("Enter query: ")
        if query[0] == ":":
            if query[1:5] == "stem":
                # print(Porter2Stemmer().stem(query[5:]))
                continue
            elif query[1:7] == "author":
                postings = dpIndex.getPostingsSoundex(SoundexTokenProcessor().process_token(query[8:]))
                for p in postings:
                    auth = EnglishTokenStream(dd.get_document(p.doc_id).getAuthor())
                    print(f"Title: {dd.get_document(p.doc_id).getTitle}")
                    print("Author:", end=" ")
                    for ss in auth:
                        print(ss, end=" ")
                    print()
                print(f"Total documents with author name '{query[8:]}' in it: {len(postings)}")
                continue
            elif query[1:6] == "index":
                path = "C:/Users/anura/Desktop/CSULB/Sem 1/CECS_529_SET/Homework/final_milestone/MySearchEngine-main/Data/" + query[7:]
                dd, f_type = load_directory(path)
                # Build the index over this directory.
                index_corpus(dd, f_type, path)
                continue
            elif query[1:6] == "vocab":
                vocab = dpIndex.getVocabulary()
                if len(vocab) > 1000:
                    for i in range(1000):
                        print(vocab[i])
                else:
                    for i in range(len(vocab)):
                        print(vocab[i])
                print(f"Length of Vocabulary: {len(vocab)}")
                continue
            elif query[1] == "q":
                break
            else:
                print("Invalid special query")
                continue

        # Processing the query as terms
        postings = booleanQueryParser.parse_query(query).get_postings(dpIndex)

        print(f"The query '{query}' is found in documents: ")
        doc_ids = []
        for posting in postings:
            print(d.get_document(posting.get_document_id()).getTitle, end="")
            print(" (DOCID " + str(posting.get_document_id()) + ")")
            doc_ids.append(posting.get_document_id())
        print(f"Length of Documents: {len(postings)}")

        while True:
            user_choice = input("Would you like to view any document from the list?(y/n)")
            if user_choice == 'y' or user_choice == 'Y':
                doc_choice = input("Please choose the doc_id of the document: ")

                if doc_choice.isnumeric() and int(doc_choice) in doc_ids:
                    print_docs_with_docId(doc_choice, dd)
                    print()
                else:
                    print("Not a valid option")
            elif user_choice == 'n' or user_choice == 'N':
                break
            else:
                print("Not a valid input")


def ranked_mode(dp_index, dd, path, tkn_processor):
    while True:
        print_variant_names()
        choice = input()
        choice_dict = {'1': DefaultVariant,
                       '2': TfidfVariant,
                       '3': OkapiVariant,
                       '4': WackyVariant
                       }
        query = input("Enter query: ")
        if query == ":q":
            return

        accumulator_dict = choice_dict.get(choice).get_accumulator_dict(choice_dict.get(choice), query, path, dp_index, token_processor)

        print_10docs_with_scores(accumulator_dict, dd)


if __name__ == "__main__":
    corpus_path = input("Enter the path for corpus: ")
    booleanQueryParser = BooleanQueryParser()
    token_processor = AdvancedTokenProcessor()
    d, fType = load_directory(corpus_path)

    print("1. Build Corpus")
    print("2. Query Corpus")
    query_build_inp = input()

    if query_build_inp == '1':
        # Build the index over this directory.
        index_corpus(d, fType, corpus_path)

    elif query_build_inp == '2':
        disk_positional_index = DiskPositionalIndex(corpus_path)
        query = ""
        print("1. Boolean query Mode")
        print("2. Ranked query Mode")
        mode = input()
        d.documents()

        if mode == '1':
            boolean_mode(disk_positional_index, d, corpus_path)
        elif mode == '2':
            # Ranked query mode
            ranked_mode(disk_positional_index, d, corpus_path, token_processor)
        else:
            print("Invalid Input")
    else:
        print("Invalid Input")

# test Path: /Users/aatishdhami/IdeaProjects/CECS529Python/SearchEngine/Data/test
# Moby Path: /Users/aatishdhami/IdeaProjects/CECS529Python/SearchEngine/Data/MobyDick10Chapters
# Npss Path: /Users/aatishdhami/IdeaProjects/CECS529Python/SearchEngine/Data/all-nps-sites-extracted
# tsta PATH: /Users/aatishdhami/IdeaProjects/CECS529Python/SearchEngine/Data/testauth
# 4000 Path: /Users/aatishdhami/IdeaProjects/CECS529Python/SearchEngine/Data/mlb-articles-4000
