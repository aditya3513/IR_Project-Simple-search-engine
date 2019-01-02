# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
import glob
import json
from collections import Counter

from nltk import ngrams

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
files = glob.glob('/Users/romilrathi/Desktop/T/C*')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''
Function to generate n-grams and store their count in a dictionary
Input : file, n(Number of words for n-gram)
Output : Dictionary with n-gram as key and their count as value
'''


def term_freq(file, n):
    with open(file, 'r+') as f:
        term_freq_count = Counter()
        lines = f.read()
        line = lines.split()
        ngram = ngrams(line, n)
        term_freq_count.update(ngram)
    return (dict(term_freq_count))


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


def inverted_index(files, n, ngram_term_freq, ngram_inv_list, file_name):
    count = 1
    doc_length = {}
    for file in files:
        ngram = term_freq(file, n)

        file = str(file.split("/")[-1])

        ngram_term_freq.setdefault(file, []).append(len(list(ngram.keys())))

        doc_length.setdefault(file, []).append(sum(ngram.values()))

        # appending (document ID , term frequency) of every n-gram in the file to the final inverted index
        for term in list(ngram.keys()):
            ngram_inv_list.setdefault(term, []).append([file, ngram[term]])

    for k, v in ngram_inv_list.items():
        k = str(k[0])

    index = {}
    for key, value in ngram_inv_list.items():
        if index.get(key) == None:
            index[str(key[0])] = value

    file = open('stem_inverted_index.txt', 'w', encoding='utf-8')
    file.write(json.dumps(index))
    file.close()

    # Writing doc length for each doc_id
    file_name = file_name[:-18] + "term.txt"
    with open(file_name, "w", encoding="utf-8") as f:
        for doc, count in ngram_term_freq.items():
            f.write(str(doc) + " : " + str(count) + "\n")

    with open("stem_doc_length", 'w') as f:
        for doc, length in doc_length.items():
            f.write(str(doc) + " : " + str(length) + "\n")

    return ngram_inv_list


# -----------------------------------------------------------------------Unigram Inverted Index---------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Initializing variables and Data Structures
unigram = {}
unigram_term_count = {}
file1 = "stem_unigram_Inverted_index.txt"

unigram = inverted_index(files, 1, unigram_term_count, unigram, file1)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''
length of the inverted list with each term
output: unigram_inv_length - Length of Inverted list for Unigrams
'''
with open("stem_unigram_inv_length.txt", 'w') as f:
    for k, v in unigram.items():
        key = k
        value = len(v)
        f.write(str(key)[1:-2] + " : " + str(value))
        f.write("\n")
