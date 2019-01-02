import json
import math
import operator
from collections import Counter


def calculate_avdl(document_length):
    sum = 0
    for doc in document_length:
        sum += int(document_length[doc])
    return sum / len(document_length)


def get_query_freq(query, term):
    query_freq = Counter()
    query_freq.update(query.split())
    return query_freq[term]


def perform_BM25(Qterms, doc, document_length, inverted_index, query):
    score = 0.0
    N = len(document_length)
    avdl = calculate_avdl(document_length)

    for term in Qterms:
        f = 0
        dl = 0
        if term in inverted_index:
            n = len(list(inverted_index[term]))
            qf = get_query_freq(query, term)

            if doc in inverted_index[term]:
                f = inverted_index[term][doc]
                dl = int(document_length[doc])
            score += calculate_score_BM25(n, f, qf, N, dl, avdl)

    return score


def calculate_score_BM25(n, f, qf, N, dl, avdl):
    k1 = 1.2
    b = 0.75
    k2 = 100
    K = calculateK(k1, b, dl, avdl)
    t1 = math.log((N - n + 0.5) / (n + 0.5))
    t2 = ((k1 + 1) * f) / (K + f)
    t3 = ((k2 + 1) * qf) / float(k2 + qf)
    return t1 * t2 * t3


def calculateK(k1, b, dl, avdl):
    return k1 * ((1 - b) + b * (float(dl) / float(avdl)))


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


def write_output_file(f_write, query, query_count, BM25_scores_sorted, system):
    f_write.write(str(query_count) + ') query : ' + query + '\n\n')
    for rank in range(100):
        f_write.write(str(query_count) + " Q0 " + BM25_scores_sorted[rank][0] + " " + str(rank + 1) + " " + str(
            BM25_scores_sorted[rank][1]) + " " + system + "\n")
    f_write.write('\n')


def proximity(docID, terms, inverted_index):
    totalScore = 0

    docs_term1 = [inverted_index[terms[0]][docID]]
    docs_term2 = [inverted_index[terms[1]][docID]]
    i = 0
    for index in docs_term1:
        termScore = 0
        for i in range(5):
            if index + i in docs_term2:
                termScore += (1.0 - (0.5 * i))
        totalScore += termScore
    return totalScore


def getProximityScore(BM25_scores, query, query_count, inverted_index, document_length, searchType, N):
    docProximityScore = {}
    termProximityScore = BM25_scores
    queryTerms = query.split()

    if searchType == 1:
        for i in range(len(queryTerms) - N):
            # this will check proximity for every continous terms, the more the terms occcur closer, better score is
            proximityTerms = queryTerms

            if proximityTerms[i] in inverted_index:
                for docID in inverted_index[proximityTerms[i]]:
                    score = 0
                    if docID in inverted_index[proximityTerms[i + 1]]:
                        score = proximity(docID, proximityTerms, inverted_index, N)

                    if docID in docProximityScore:
                        docProximityScore[docID] += score
                    else:
                        docProximityScore[docID] = score
    else:

        for i in range(len(queryTerms) - N):
            # this will check proximity for  N continous terms
            proximityTerms = queryTerms[i:i + N]

            for ptIndex in range(len(proximityTerms) - 1):

                if proximityTerms[ptIndex] in inverted_index and proximityTerms[ptIndex + 1] in inverted_index:
                    for docID in inverted_index[proximityTerms[ptIndex]]:
                        score = 0
                        if docID in inverted_index[proximityTerms[ptIndex + 1]]:
                            score = proximity(docID, proximityTerms, inverted_index)

                        if docID in docProximityScore:
                            docProximityScore[docID] += score
                        else:
                            docProximityScore[docID] = score

    for docID, score in docProximityScore.items():
        docLen = document_length[docID]
        lmbd = 0.2
        termProximityScore[docID] = (1 - lmbd) * BM25_scores[docID] + lmbd * score
    return termProximityScore


def get_BM25_output(index_file, dl_file, queries_file, file_name, system):
    inverted_index = get_inverted_index(index_file)
    document_length = get_document_length(dl_file)
    queries = get_cacm_queries(queries_file)
    print("input the type of search you want:")
    print("1 -> exact")
    print("2 -> best match")
    print("3 -> Ordered best match within proximity N")
    searchType = input("your input = ")
    if searchType == 3:
        N = 5
    elif searchType == 2:
        N = 2
    else:
        N = 1
    query_count = 0
    f_write = open(file_name, "w+")
    for query in queries:
        print(query_count, ') query : ', query, '\n')
        query_terms = query.split()
        query_count += 1

        BM25_scores = {}
        for doc in list(document_length.keys()):
            BM25_scores[doc] = perform_BM25(query_terms, doc, document_length, inverted_index, query)

        BM25_proximity_scores = getProximityScore(BM25_scores, query, query_count, inverted_index, document_length,
                                                  searchType, N)
        Scores_sorted = sorted(BM25_scores.items(), key=operator.itemgetter(1), reverse=True)
        write_output_file(f_write, query, query_count, Scores_sorted, system)

    f_write.close()


get_BM25_output("inverted_index.json", "doc_length", "cacm.query.parsed.txt", "BM25_proximity.txt", "BM25_proximity")
