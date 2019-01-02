import json
import math

import pandas as pd


def get_document_length(dl_file):
    document_length = {}
    f = open(dl_file)
    lines = f.readlines()
    for line in lines:
        doc = line[:9]
        length = line.split(" : [")[1][:-2]
        document_length[doc] = length
    return document_length


def get_inverted_index(index_file):
    f = open(index_file)
    index = json.load(f)
    inverted_index = {}
    for term, postings in index.items():
        inverted_index[term] = {}
        for posting in postings:
            inverted_index[term][posting[0]] = posting[1]
    return inverted_index


def get_cacm_queries(queries_file):
    f = open(queries_file)
    lines = f.readlines()
    queries = [x.rstrip('\n') for x in lines]
    return queries


invertedIndex = get_inverted_index("stop_inverted_index.txt")
docLens = get_document_length("stop_doc_length")
Queries = get_cacm_queries("cacm.query.stopped.txt")
totalDocs = len(docLens)
columnNames = ["query_id", "Q0", "doc_id", "rank", "tfidf_score", "system_name"]
rankDF = pd.DataFrame(list(range(1, totalDocs + 1)), columns=["rank"])
resultFile = open("tfidf_stop.txt", "w")

for query, query_id in zip(Queries, range(len(Queries))):
    resultDF = pd.DataFrame(columns=columnNames)
    # iterate through each document
    for docID, length in docLens.items():
        docScore = 0
        for word in query.split():
            tf = 0
            idf = 0
            if word in invertedIndex:
                unigramSection = invertedIndex[word]
                idf = math.log(totalDocs / len(unigramSection))

                if docID in unigramSection:
                    tf = float(unigramSection[docID]) / float(length)

            score = tf * idf
            docScore += score

        resultDF.loc[len(resultDF)] = [query_id + 1, "Q0", docID, 0, docScore, "tfidf"]
    resultDF = resultDF.sort_values('tfidf_score', ascending=False)
    resultDF['rank'] = rankDF['rank'].values
    resultFile.write(str(query_id + 1) + ") " + query + "\n")
    print("Processing Query " + str(query_id + 1))
    for index, row in resultDF[:100].iterrows():
        resultFile.write(str(row["query_id"]) + " Q0 " + row["doc_id"] + " " + str(row["rank"]) + " " + str(
            row["tfidf_score"]) + " " + row["system_name"] + "\n")
# np.savetxt('tfidf_results/'+"query_"+str(query_id)+".txt", resultDF.values[:100], fmt='%s')
resultFile.close()
