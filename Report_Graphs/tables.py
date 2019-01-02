import matplotlib.pyplot as plt


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
def write_precision_recall_table(relevance_score, model_scores, outputfile, key):
    precision_recall, p_5, p_20, MAP, MRR = evaluate(relevance_score, model_scores)

    prec = precision_recall[key]

    precision = []
    recall = []

    for i in prec:
        precision.append(i[1])
        recall.append(i[2])

    return precision, recall


# Getting Relavant documents for comparison
relavance_file = '/Users/romilrathi/Downloads/test-collection/cacm.rel.txt'
gold_relevance = gold_relevance_docs(relavance_file)

# File with Relavance Output
Lucene_file = '/Users/romilrathi/Desktop/Lucene.txt'
query_file = '/Users/romilrathi/Desktop/BM25_pseudo_relevance_feedback.txt'
tfidf_file = '/Users/romilrathi/Downloads/tfidf_result.txt'
stop_tfidf_file = '/Users/romilrathi/Desktop/tfidf_stop.txt'
bm_25 = '/Users/romilrathi/Downloads/BM25.txt'
bm_25_stop = '/Users/romilrathi/Downloads/BM25_stop.txt'
jmsqlm_file = '/Users/romilrathi/Desktop/Project/SQLM/SQLM.txt'
stop_jmsqlm_file = '/Users/romilrathi/Desktop/Project/Stop/SQLM_Stop.txt'

rank = []
for i in range(100):
    rank.append(i + 1)


###----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Function to plot Precision and recall Curves

def plot_curves(Lucene_file, query_file, tfidf_file, stop_tfidf_file, bm_25, bm_25_stop, jmsqlm_file, stop_jmsqlm_file,
                query_id):
    lucene_rel = model_relevance_docs(Lucene_file)
    lucene_precision, lucene_recall = write_precision_recall_table(gold_relevance, lucene_rel, "Lucene_PR_Table",
                                                                   query_id)

    # Query Relavance
    query_rel = model_relevance_docs(query_file)
    p, r = write_precision_recall_table(gold_relevance, query_rel, 'Query_PR_Table', query_id)

    # TF-IDF
    tfidf_rel = model_relevance_docs(tfidf_file)
    tf_idf_precision, tf_idf_recall = write_precision_recall_table(gold_relevance, tfidf_rel, "TFIDF_PR_Table",
                                                                   query_id)

    stop_tfidf_rel = model_relevance_docs(stop_tfidf_file)
    stop_tfidf_precision, stop_tfidf_recall = write_precision_recall_table(gold_relevance, stop_tfidf_rel,
                                                                           "Stop_TFIDF_PR_Table", query_id)

    # BM-25 Model
    bm25_rel = model_relevance_docs(bm_25)
    bm25_precision, bm25_recall = write_precision_recall_table(gold_relevance, bm25_rel, "BM25_PR_Table", query_id)

    bm25_stop_rel = model_relevance_docs(bm_25_stop)
    bm25_stop_precision, bm25_stop_recall = write_precision_recall_table(gold_relevance, bm25_stop_rel,
                                                                         "Stop_BM25_PR_Table", query_id)

    # For JM- Smoothing Likelihood Model
    jsqlm_relevance = model_relevance_docs(jmsqlm_file)
    jsqlm_precision, jsqlm_recall = write_precision_recall_table(gold_relevance, jsqlm_relevance, "JmSQLM_PR_Table",
                                                                 query_id)

    stop_jsqlm_relevance = model_relevance_docs(stop_jmsqlm_file)
    stop_jsqlm_precision, stop_jsqlm_recall = write_precision_recall_table(gold_relevance, stop_jsqlm_relevance,
                                                                           "Stop_JmSQLM_PR_Table", query_id)

    # Plotting PR Curve

    fig, ax = plt.subplots()
    ax.plot(rank, lucene_precision, label='Lucene')
    ax.plot(rank, lucene_precision, label='Query-Enrichment')
    ax.plot(rank, tf_idf_precision, label='TF-IDF')
    ax.plot(rank, stop_tfidf_precision, label='TF-IDF-STOP')
    ax.plot(rank, bm25_precision, label='BM25')
    ax.plot(rank, bm25_stop_precision, label='BM25_STOP')
    ax.plot(rank, jsqlm_precision, label='JMSQLM')
    ax.plot(rank, stop_jsqlm_precision, label='JMSQLM_STOP')
    plt.legend(['Lucene', 'Query-Enrichment', 'TF-IDF', 'TF-IDF-STOP', 'BM25', 'BM25_STOP', 'JMSQLM', 'JMSQLM_STOP'])
    plt.ylabel('Precision')
    plt.title('Precision Query-' + query_id)
    plt.savefig(query_id + '_precision.png', dpi=200)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(rank, lucene_recall, label='Lucene')
    ax.plot(rank, r, label='Query-Enrichment')
    ax.plot(rank, tf_idf_recall, label='TF-IDF')
    ax.plot(rank, stop_tfidf_recall, label='TF-IDF-STOP')
    ax.plot(rank, bm25_recall, label='BM25')
    ax.plot(rank, bm25_stop_recall, label='BM25_STOP')
    ax.plot(rank, jsqlm_recall, label='JMSQLM')
    ax.plot(rank, stop_jsqlm_recall, label='JMSQLM_STOP')
    plt.legend(['Lucene', 'Query-Enrichment', 'TF-IDF', 'TF-IDF-STOP', 'BM25', 'BM25_STOP', 'JMSQLM', 'JMSQLM_STOP'])
    plt.ylabel('Recall')
    plt.title('Recall Query-' + query_id)
    plt.savefig(query_id + '_recall.png', dpi=200)
    plt.show()


# ###----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
plot_curves(Lucene_file, query_file, tfidf_file, stop_tfidf_file, bm_25, bm_25_stop, jmsqlm_file, stop_jmsqlm_file,
            '14')
plot_curves(Lucene_file, query_file, tfidf_file, stop_tfidf_file, bm_25, bm_25_stop, jmsqlm_file, stop_jmsqlm_file,
            '25')
plot_curves(Lucene_file, query_file, tfidf_file, stop_tfidf_file, bm_25, bm_25_stop, jmsqlm_file, stop_jmsqlm_file,
            '59')
plot_curves(Lucene_file, query_file, tfidf_file, stop_tfidf_file, bm_25, bm_25_stop, jmsqlm_file, stop_jmsqlm_file,
            '36')
