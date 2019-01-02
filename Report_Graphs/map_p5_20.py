import matplotlib.pyplot as plt
import pandas as pd

# Getting Relavant documents for comparison
relavance_file = '/Users/romilrathi/Downloads/test-collection/cacm.rel.txt'

# File with Relavance Output
Lucene_file = '/Users/romilrathi/Desktop/Lucene.txt'
query_file = '/Users/romilrathi/Desktop/BM25_pseudo_relevance_feedback.txt'
tfidf_file = '/Users/romilrathi/Downloads/tfidf_result.txt'
stop_tfidf_file = '/Users/romilrathi/Desktop/tfidf_stop.txt'
bm_25 = '/Users/romilrathi/Downloads/BM25.txt'
bm_25_stop = '/Users/romilrathi/Downloads/BM25_stop.txt'
jmsqlm_file = '/Users/romilrathi/Desktop/Project/SQLM/SQLM.txt'
stop_jmsqlm_file = '/Users/romilrathi/Desktop/Project/Stop/SQLM_Stop.txt'


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


def write_p5_p20(relevance_score, model_scores):
    precision_recall, p_5, p_20, MAP, MRR = evaluate(relevance_score, model_scores)
    p_5 = sum(p_5.values()) / len(p_5)
    p_20 = sum(p_20.values()) / len(p_20)
    return p_5, p_20


###----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
# Function to plot Precision and recall Curves

def plot_p5_p20(Lucene_file, query_file, tfidf_file, stop_tfidf_file, bm_25, bm_25_stop, jmsqlm_file, stop_jmsqlm_file):
    ## Query ID - 36

    lucene_rel = model_relevance_docs(Lucene_file)
    lucene_p5, lucene_p20 = write_p5_p20(gold_relevance, lucene_rel)

    # Query Relavance
    query_rel = model_relevance_docs(query_file)
    p5, p20 = write_p5_p20(gold_relevance, query_rel)

    # TF-IDF
    tfidf_rel = model_relevance_docs(tfidf_file)
    tf_idf_p5, tf_idf_p20 = write_p5_p20(gold_relevance, tfidf_rel)

    stop_tfidf_rel = model_relevance_docs(stop_tfidf_file)
    stop_tfidf_p5, stop_tfidf_p20 = write_p5_p20(gold_relevance, stop_tfidf_rel)

    # BM-25 Model
    bm25_rel = model_relevance_docs(bm_25)
    bm25_p5, bm25_p20 = write_p5_p20(gold_relevance, bm25_rel)

    bm25_stop_rel = model_relevance_docs(bm_25_stop)
    stop_bm25_p5, bm25_stop_p20 = write_p5_p20(gold_relevance, bm25_stop_rel)

    #   For JM- Smoothing Likelihood Model
    jsqlm_relevance = model_relevance_docs(jmsqlm_file)
    jsqlm_p5, jsqlm_p20 = write_p5_p20(gold_relevance, jsqlm_relevance)

    stop_jsqlm_relevance = model_relevance_docs(stop_jmsqlm_file)
    stop_jsqlm_p5, stop_jsqlm_p20 = write_p5_p20(gold_relevance, stop_jsqlm_relevance)

    df1 = pd.DataFrame(
        {'query_enrichment': [p5], 'TF-IDF': [tf_idf_p5], 'TF-IDF-Stop': [stop_tfidf_p5], 'BM25': [bm25_p5],
         'BM25-Stop': [stop_bm25_p5], 'JMSQLM-STOP': [stop_jsqlm_p5], 'lucene': [lucene_p5], 'JMSQLM': [jsqlm_p5]})
    f = plt.figure()
    df1.sort_values(by='lucene').plot(width=0.5, kind='bar', ax=f.gca())
    plt.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
    plt.title("P@5 For all models")
    plt.savefig('p@5.png', dpi=500)

    df2 = pd.DataFrame(
        {'query_enrichment': [p20], 'TF-IDF': [tf_idf_p20], 'TF-IDF-Stop': [stop_tfidf_p20], 'BM25': [bm25_p20],
         'BM25-Stop': [bm25_stop_p20], 'JMSQLM-STOP': [stop_jsqlm_p20], 'lucene': [lucene_p20], 'JMSQLM': [jsqlm_p20]})
    df2.sort_values(by='lucene').plot(width=0.5, kind='bar', ax=f.gca())
    plt.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
    plt.title("P@20 For all models")
    plt.savefig('p@20.png', dpi=200)


###----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
def write_map_mrr(relevance_score, model_scores):
    precision_recall, p_5, p_20, MAP, MRR = evaluate(relevance_score, model_scores)
    return (MAP, MRR)


def plot_map_mrr(Lucene_file, query_file, tfidf_file, stop_tfidf_file, bm_25, bm_25_stop, jmsqlm_file,
                 stop_jmsqlm_file):
    # Lucene
    lucene_rel = model_relevance_docs(Lucene_file)
    lucene_map, lucene_mrr = write_map_mrr(gold_relevance, lucene_rel)

    # Query Relavance

    query_rel = model_relevance_docs(query_file)
    map_, mrr = write_map_mrr(gold_relevance, query_rel)

    # TF-IDF
    tfidf_rel = model_relevance_docs(tfidf_file)
    tf_idf_map, tf_idf_mrr = write_map_mrr(gold_relevance, tfidf_rel)

    stop_tfidf_rel = model_relevance_docs(stop_tfidf_file)
    stop_tfidf_map, stop_tfidf_mrr = write_map_mrr(gold_relevance, stop_tfidf_rel)

    # BM-25 Model
    bm25_rel = model_relevance_docs(bm_25)
    bm25_map, bm25_mrr = write_map_mrr(gold_relevance, bm25_rel)

    bm25_stop_rel = model_relevance_docs(bm_25_stop)
    stop_bm25_map, bm25_stop_mrr = write_map_mrr(gold_relevance, bm25_stop_rel)

    # For JM- Smoothing Likelihood Model
    jsqlm_relevance = model_relevance_docs(jmsqlm_file)
    jsqlm_map, jsqlm_mrr = write_map_mrr(gold_relevance, jsqlm_relevance)

    stop_jsqlm_relevance = model_relevance_docs(stop_jmsqlm_file)
    stop_jsqlm_map, stop_jsqlm_mrr = write_map_mrr(gold_relevance, stop_jsqlm_relevance)

    map_df = pd.DataFrame(
        {'query_enrichment': [map_], 'TF-IDF': [tf_idf_map], 'TF-IDF-Stop': [stop_tfidf_map], 'BM25': [bm25_map],
         'BM25-Stop': [stop_bm25_map], 'JMSQLM-STOP': [stop_jsqlm_map], 'lucene': [lucene_map], 'JMSQLM': [jsqlm_map]})

    f = plt.figure()
    map_df.sort_values(by='lucene').plot(width=0.5, kind='bar', ax=f.gca(), grid=None)
    plt.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
    plt.title("MAP For all models")
    plt.savefig('MAP.png', dpi=200)

    mrr_df = pd.DataFrame(
        {'query_enrichment': [mrr], 'TF-IDF': [tf_idf_mrr], 'TF-IDF-Stop': [stop_tfidf_mrr], 'BM25': [bm25_mrr],
         'BM25-Stop': [bm25_stop_mrr], 'JMSQLM-STOP': [stop_jsqlm_mrr], 'lucene': [lucene_mrr], 'JMSQLM': [jsqlm_mrr]})
    f = plt.figure()
    mrr_df.sort_values(by='lucene').plot(width=0.5, kind='bar', ax=f.gca(), grid=None)
    plt.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
    plt.title("MRR For all models")
    plt.savefig('MRR.png', dpi=200)


# ###----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
gold_relevance = gold_relevance_docs(relavance_file)
plot_p5_p20(Lucene_file, query_file, tfidf_file, stop_tfidf_file, bm_25, bm_25_stop, jmsqlm_file, stop_jmsqlm_file)
plot_map_mrr(Lucene_file, query_file, tfidf_file, stop_tfidf_file, bm_25, bm_25_stop, jmsqlm_file, stop_jmsqlm_file)
