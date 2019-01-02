# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
import glob
import json
from collections import Counter

from nltk import ngrams

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
files = glob.glob('/Users/romilrathi/Desktop/Corpus/*')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fetch_stop_words(file):
    stop_words = []
    f = open(file)
    words = f.readlines()
    f.close()
    for word in words:
        stop_words.append(word[:-1])
    return stop_words


def term_freq(file, n):
    with open(file, 'r+') as f:
        term_freq_count = Counter()
        lines = f.read()
        line = lines.split()
        ngram = ngrams(line, n)
        term_freq_count.update(ngram)

    return (dict(term_freq_count))


def filter(term_freq_count):
    stop_words_list = fetch_stop_words('/Users/romilrathi/Downloads/test-collection/common_words')
    index = {}
    for k, v in term_freq_count.items():
        # print(k)
        if str(k[0]) not in stop_words_list:
            if index.get(k) == None:
                index[str(k[0])] = v
    return (index)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''
Function to generate inverted index and write it to a file
Input : files - path of all the files
		n - n-gram
		ngram_term_freq - dict to store number of n grams for a doc_id
		ngram_inv_list - dictionary to store inverted index
		filename - name of output file generated
Output : ngram_inv_list - dictionary containing inverted index
		 file_name.txt - writes inverted index to a file
		 file_name_term.txt - writing the freq corresponding to each doc_id
'''


def inverted_index(files, n, ngram_term_freq, ngram_inv_list):
    count = 1
    doc_length = {}
    for file in files:
        ngram_temp = term_freq(file, n)

        ngram = filter(ngram_temp)

        file = str(file.split("/")[-1])

        ngram_term_freq.setdefault(file, []).append(len(list(ngram.keys())))

        doc_length.setdefault(file, []).append(sum(ngram.values()))

        # appending (document ID , term frequency) of every n-gram in the file to the final inverted index
        for term in list(ngram.keys()):
            ngram_inv_list.setdefault(term, []).append([file, ngram[term]])

    file = open('stop_inverted_index.json', 'w', encoding='utf-8')
    file.write(json.dumps(ngram_inv_list))
    file.close()

    with open("stop_doc_length", 'w') as f:
        for doc, length in doc_length.items():
            f.write(str(doc) + " : " + str(length) + "\n")

    return ngram_inv_list


# -----------------------------------------------------------------------Unigram Inverted Index---------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Initializing variables and Data Structures
unigram = {}
unigram_term_count = {}

unigram = inverted_index(files, 1, unigram_term_count, unigram)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
