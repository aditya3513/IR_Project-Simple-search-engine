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


def perform_BM25_PRF(Qterms, queryTerms_index, doc, document_length, inverted_index, query, relevant_docs):
    score = 0.0
    N = len(document_length)
    avdl = calculate_avdl(document_length)

    for term in Qterms:
        f = 0
        dl = 0
        r = 0
        R = 0

        if term in inverted_index:
            n = len(list(inverted_index[term]))
            qf = get_query_freq(query, term)

            if doc in inverted_index[term]:
                f = inverted_index[term][doc]
                dl = int(document_length[doc])
                for doc in relevant_docs:
                    if queryTerms_index.get(term) != None:
                        term_indexlist = queryTerms_index[term]
                        if term_indexlist.get(doc) != None:
                            r += 1
                if r == 0:
                    R = 0
                else:
                    R = 50
            score += calculate_score_BM25(n, f, qf, N, dl, avdl, r, R)

    return score


def calculate_score_BM25(n, f, qf, N, dl, avdl, r, R):
    k1 = 1.2
    b = 0.75
    k2 = 100
    K = k1 * ((1 - b) + b * (float(dl) / float(avdl)))
    t1 = math.log(((r + 0.5) / (R - r + 0.5)) / ((n - r + 0.5) / (N - n - R + r + 0.5)))
    t2 = ((k1 + 1) * f) / (K + f)
    t3 = ((k2 + 1) * qf) / float(k2 + qf)
    return t1 * t2 * t3


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


# using rochio algorithm
def get_terms_for_expansion(query_terms, relevant_docs, inverted_index):
    alpha = 8
    beta = 16
    gama = 4

    queryTerms_freq = dict(Counter(query_terms))
    term_list = list(set(list(inverted_index.keys()) + list(queryTerms_freq.keys())))

    relevance_score = {key: 0.0 for key in term_list}
    non_relevance_score = {key: 0.0 for key in term_list}
    term_score = {key: 0.0 for key in term_list}

    for term in term_list:
        if queryTerms_freq.get(term) == None:
            queryTerms_freq[term] = 0.0
        if inverted_index.get(term) != None:
            for doc, count in inverted_index[term].items():
                if doc in relevant_docs:
                    relevance_score[term] += count
                else:
                    non_relevance_score[term] += count

    Rel = sum(list(relevance_score.values()))
    NonRel = sum(list(non_relevance_score.values()))

    for term in term_list:
        term_score[term] = (alpha * queryTerms_freq[term]) + (beta * (1 / Rel) * relevance_score[term]) \
                           - (gama * (1 / NonRel) * non_relevance_score[term])

    term_score = dict(sorted(term_score.items(), key=lambda x: x[1], reverse=True))
    return term_score


def get_BM25_relevant_docs(index_file, dl_file, queries_file, file_name, system, stop_words):
    inverted_index = get_inverted_index(index_file)
    document_length = get_document_length(dl_file)
    queries = get_cacm_queries(queries_file)

    query_count = 0
    f_write = open(file_name, "w+")

    for query in queries:

        print(query_count, ') query : ', query, '\n')
        query_terms = query.split()
        query_count += 1
        length_query = len(query_terms)

        Qterms = [term for term in query_terms if term in list(inverted_index.keys())]
        queryTerms_index = {query_term: inverted_index[query_term] for query_term in set(Qterms)}

        docs = []
        for term in list(queryTerms_index.keys()):
            docs += list(queryTerms_index[term].keys())
        query_docIDs = list(set(docs))

        relevant_docs = []
        BM25_PRF_scores = {}

        for docID in query_docIDs:
            BM25_PRF_scores[docID] = perform_BM25_PRF(query_terms, queryTerms_index, docID, document_length,
                                                      inverted_index, query, relevant_docs)

        relevant_docs = list(BM25_PRF_scores.keys())[:50]
        term_scores = get_terms_for_expansion(query_terms, relevant_docs, inverted_index)
        stopwords = open(stop_words).read().split()
        top_terms = []
        count = 0
        for term in list(term_scores.keys()):
            if count > (length_query + 15):
                break
            if term.isdigit() == False and term not in stopwords:
                count += 1
                top_terms.append(term)

        final_docScores = {}
        for doc in list(document_length.keys()):
            final_docScores[doc] = perform_BM25_PRF(top_terms, queryTerms_index, doc, document_length, inverted_index,
                                                    query, relevant_docs)
        print(top_terms)

        BM25_PRF_scores_sorted = sorted(final_docScores.items(), key=operator.itemgetter(1), reverse=True)
        write_output_file(f_write, query, query_count, BM25_PRF_scores_sorted, system)

    f_write.close()


get_BM25_relevant_docs("inverted_index.json", "doc_length", "cacm.query.parsed.txt",
                       "BM25_pseudo_relevance_feedback.txt", "BM25_PRF", "common_words")
