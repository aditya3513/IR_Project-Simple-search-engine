import json
import math
import operator

import numpy as np


def perform_sqlm(Qterms, doc, document_length, inverted_index):
    l = 0.35
    score = 0.0
    V = len(inverted_index)

    for term in Qterms:
        tf = 0
        df = 0

        if term in inverted_index:
            tf = np.sum(list(inverted_index[term].values()))

            if doc in inverted_index[term]:
                df = inverted_index[term][doc]

            score += math.log10((1 - l) * df / float(document_length[doc]) + l * tf / float(V))

    return score


# Read inverted index from file and store it
def get_inverted_index(index_file):
    f = open(index_file)
    index = json.load(f)
    inverted_index = {}
    for term, postings in index.items():
        inverted_index[term] = {}
        for posting in postings:
            inverted_index[term][posting[0]] = posting[1]
    return inverted_index


# Read document length from file and store it
def get_document_length(dl_file):
    document_length = {}
    f = open(dl_file)
    lines = f.readlines()
    for line in lines:
        doc = line[:9]
        length = line.split(" : [")[1][:-2]
        document_length[doc] = length
    return document_length


# Read queries from file and store it
def get_cacm_queries(queries_file):
    f = open(queries_file)
    lines = f.readlines()
    queries = [x.rstrip('\n') for x in lines]
    return queries


def write_output_file(f_write, query, query_count, sqlm_scores_sorted, system):
    f_write.write(str(query_count) + ') query : ' + query + '\n\n')
    for rank in range(100):
        f_write.write(str(query_count) + " Q0 " + sqlm_scores_sorted[rank][0] + " " + str(rank + 1) + " " + str(
            sqlm_scores_sorted[rank][1]) + " " + system + "\n")
    f_write.write('\n')


def get_sqlm_output(index_file, dl_file, queries_file, file_name, system):
    inverted_index = get_inverted_index(index_file)
    document_length = get_document_length(dl_file)
    queries = get_cacm_queries(queries_file)

    query_count = 0
    f_write = open(file_name, "w+")
    for query in queries:

        print(query_count, ') query : ', query, '\n')
        query_terms = query.split()
        query_count += 1

        sqlm_scores = {}
        for doc in list(document_length.keys()):
            sqlm_scores[doc] = perform_sqlm(query_terms, doc, document_length, inverted_index)

        sqlm_scores_sorted = sorted(sqlm_scores.items(), key=operator.itemgetter(1), reverse=True)
        write_output_file(f_write, query, query_count, sqlm_scores_sorted, system)

    f_write.close()


get_sqlm_output("inverted_index.json", "doc_length.txt", "cacm_queries.txt", "JMsqlm.txt", "SQLM")
