import collections

import matplotlib.pyplot as plt
from numpy import interp


# Function to retrieve relavant documents provided in task to evaluate
def gold_relevance_docs(file):
    relevance = {}
    with open(file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            query = line[0]
            if relevance.get(query) == None:
                relevance[query] = [line[-2]]
            else:
                relevance[query].append(line[-2])
    return relevance


# Function to get 100 relavant documents for all queries
def model_relevance_docs(file):
    model_relevance = {}
    temp = {}
    with open(file, 'r+') as f:
        flag = ""
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if len(line) == 6:
                query = line[0]
                if temp.get(query) == None:
                    temp[query] = [line[-4]]
                else:
                    temp[query].append(line[-4])

    for k, v in temp.items():
        if ')' not in k:
            if model_relevance.get(k) == None:
                model_relevance[k] = v

    return (model_relevance)


# Function to find Precision, Recall, P@5, P@20, MAP, MPP

'''
Input - Gold Label Documents, Model Predicted Documents

Output - 
query_pr : Query_iD as key, [Doc, Precision, recall] as value
p5 = P@5
p20 = P@20
MRR = Mean Reciprocal Rank
MAP = Mean Average Precision
'''


def evaluate(relevance_score, doc_score):
    precision_sum = 0.0
    reciprocal_rank_sum = 0.0
    query_pr = {}
    p5 = {}
    p20 = {}

    for query_id in doc_score:

        overall_precision = 0.0
        rr = 0.0  # reciprocal rank
        relevant = 0.0
        retrieved = 0.0
        retrieved_relevant = 0.0
        rank = 0  # rank count
        pr = []  # List to store Doc_ID, Precision, Recall

        if query_id in relevance_score:

            for doc in doc_score[query_id]:
                rank = rank + 1  # Rank of document

                if doc in relevance_score[query_id]:
                    retrieved = retrieved + 1  # Adding Retrieved Document

                    if retrieved == 1:
                        rr = float(1 / rank)  # Returning Reciprocal Rank

                    retrieved_relevant = retrieved_relevant + 1
                    precision = float(retrieved_relevant / retrieved)  # Finding Precision
                    overall_precision = overall_precision + precision
                    recall = float(retrieved_relevant / len(relevance_score[query_id]))  # Finding Recall
                    pr.append([doc, precision, recall])  # Appending Precision, Recall for Doc_Id
                else:
                    retrieved = retrieved + 1
                    precision = float(retrieved_relevant / retrieved)
                    recall = float(retrieved_relevant / len(relevance_score[query_id]))
                    pr.append([doc, precision, recall])

                if rank == 5:  # for precision at 5
                    p5[query_id] = precision

                if rank == 20:  # for precision at 20
                    p20[query_id] = precision

                query_pr[query_id] = pr
        else:
            continue

        if retrieved != 0:
            precision_sum = precision_sum + overall_precision / retrieved
            reciprocal_rank_sum = reciprocal_rank_sum + rr

    MAP = precision_sum / len(relevance_score)  # MAP
    MRR = reciprocal_rank_sum / len(relevance_score)  # MRR

    return query_pr, p5, p20, MAP, MRR


# Function to write output to a file and return Precision for plotting PR Curve
def write_precision_recall_table(relevance_score, model_scores, outputfile):
    precision_recall, p_5, p_20, MAP, MRR = evaluate(relevance_score, model_scores)

    pr = {}  # Dictionary to store list of Precision for Recall(Key)
    for key, val in precision_recall.items():
        for i in val:
            pre = round(i[1], 2)  # Precision
            rec = round(i[2], 2)  # Recall
            if pr.get(rec) == None:
                pr[rec] = [pre]
            else:
                pr[rec].append(pre)

    pre_call = {}
    # Sorting By Recall Values
    pre_call = collections.OrderedDict(sorted(pr.items()))
    # getting Mean Precision for Recall values
    for k, v in pre_call.items():
        pre_call[k] = round((sum(v) / len(v)), 2)

    # interpolation for pr-curve
    interpolated_precision = interp([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], list(pre_call.keys()),
                                    list(pre_call.values()))

    # writing PR table to a file
    with open(outputfile, 'w') as f:
        f.write("Mean Average Precision (MAP) : " + str(MAP)[:5] + "\n")
        f.write("Mean Reciprocal Rank (MRR) : " + str(MRR)[:5] + "\n\n")
        for k, v in precision_recall.items():
            f.write("Query ID: " + str(k) + '\n\n')
            f.write("P@5 : " + str(p_5[k]) + '\n')
            f.write("P@10 : " + str(p_20[k]) + '\n\n')

            f.write('{:<5}\t {:<5}\t {:<5}\t {:<5}\n'.format("Rank", "Document_Id", "Precision", "Recall") + "\n")
            rank = 0
            for i in v:
                rank += 1
                doc = str(i[0])
                precision = i[1]
                recall = i[2]
                doc_precision_recall = '{:<5}\t {:<5}\t {:<10}\t {:<5}\n'.format(rank, doc, round(precision, 3),
                                                                                 round(recall, 3))
                f.write(doc_precision_recall + '\n')

    return interpolated_precision


# Getting Relavant documents for comparison
relavance_file = '/Users/romilrathi/Downloads/test-collection/cacm.rel.txt'
gold_relevance = gold_relevance_docs(relavance_file)

# Lucene
Lucene_file = '/Users/romilrathi/Desktop/Lucene.txt'
lucene_rel = model_relevance_docs(Lucene_file)
lucene_precision = write_precision_recall_table(gold_relevance, lucene_rel, "Lucene_PR_Table")

# Query Relavance
query_file = '/Users/romilrathi/Desktop/BM25_pseudo_relevance_feedback.txt'
query_rel = model_relevance_docs(query_file)
write_precision_recall_table(gold_relevance, query_rel, 'Query_PR_Table')

# TF-IDF
tfidf_file = '/Users/romilrathi/Downloads/tfidf_result.txt'
tfidf_rel = model_relevance_docs(tfidf_file)
tf_idf_precision = write_precision_recall_table(gold_relevance, tfidf_rel, "TFIDF_PR_Table")

stop_tfidf_file = '/Users/romilrathi/Desktop/tfidf_stop.txt'
stop_tfidf_rel = model_relevance_docs(stop_tfidf_file)
stop_tfidf_precision = write_precision_recall_table(gold_relevance, stop_tfidf_rel, "Stop_TFIDF_PR_Table")

# BM-25 Model
bm_25 = '/Users/romilrathi/Downloads/BM25.txt'
bm25_rel = model_relevance_docs(bm_25)
bm25_precision = write_precision_recall_table(gold_relevance, bm25_rel, "BM25_PR_Table")

bm_25_stop = '/Users/romilrathi/Downloads/BM25_stop.txt'
bm25_stop_rel = model_relevance_docs(bm_25_stop)
bm25_stop_precision = write_precision_recall_table(gold_relevance, bm25_stop_rel, "Stop_BM25_PR_Table")

# For JM- Smoothing Likelihood Model
jmsqlm_file = '/Users/romilrathi/Desktop/Project/SQLM/SQLM.txt'
jsqlm_relevance = model_relevance_docs(jmsqlm_file)
jsqlm_precision = write_precision_recall_table(gold_relevance, jsqlm_relevance, "JmSQLM_PR_Table")

stop_jmsqlm_file = '/Users/romilrathi/Desktop/Project/Stop/SQLM_Stop.txt'
stop_jsqlm_relevance = model_relevance_docs(stop_jmsqlm_file)
stop_jsqlm_precision = write_precision_recall_table(gold_relevance, stop_jsqlm_relevance, "Stop_JmSQLM_PR_Table")

# Plotting PR Curve
recall_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

fig, ax = plt.subplots()
ax.plot(recall_levels, lucene_precision, label='Lucene')
ax.plot(recall_levels, lucene_precision, label='Query-Enrichment')
ax.plot(recall_levels, tf_idf_precision, label='TF-IDF')
ax.plot(recall_levels, stop_tfidf_precision, label='TF-IDF-STOP')
ax.plot(recall_levels, bm25_precision, label='BM25')
ax.plot(recall_levels, bm25_stop_precision, label='BM25_STOP')
ax.plot(recall_levels, jsqlm_precision, label='JMSQLM')
ax.plot(recall_levels, stop_jsqlm_precision, label='JMSQLM_STOP')
plt.legend(['Lucene', 'Query-Enrichment', 'TF-IDF', 'TF-IDF-STOP', 'BM25', 'BM25_STOP', 'JMSQLM', 'JMSQLM_STOP'])
# plt.legend(['Lucene', 'TF-IDF', 'TF-IDF-STOP', 'BM25', 'BM25_STOP', 'JMSQLM','JMSQLM_STOP'])
plt.savefig('precision_recall_plot.png', dpi=200)
plt.show()
