stem_file = '/Users/romilrathi/Downloads/test-collection/cacm_stem.txt'

file = open(stem_file).read()


def get_documents(file):
    start_symbol = file.find('#')
    data = {}

    while start_symbol != -1:
        doc_id = file[(start_symbol + 2): file.find("\n", (start_symbol + 2))]
        end_symbol = file.find('#', file.find("\n", (start_symbol + 2)))

        if len(doc_id) == 1:
            did = "CACM-000" + doc_id
        if len(doc_id) == 2:
            did = "CACM-00" + doc_id
        if len(doc_id) == 3:
            did = "CACM-0" + doc_id
        if len(doc_id) == 4:
            did = "CACM-" + doc_id

        data[did] = []

        contents = file[start_symbol + 2: end_symbol]

        data[did] = contents[len(doc_id):]

        start_symbol = file.find('#', end_symbol)

    return data


def output_documents(file_name, document):
    with open(file_name, 'w') as f:
        f.write(document)


stemmed_documents = get_documents(file)

for k, v in stemmed_documents.items():
    tokens = v.split()
    if 'pm' in tokens:
        text = ''
        index = tokens.index('pm')
        word = tokens[:(index + 1)]
        for term in word:
            text += term + " "
        stemmed_documents[k] = text
    elif 'am' in tokens:
        text = ''
        index = tokens.index('am')
        word = tokens[:(index + 1)]
        for term in word:
            text += term + " "
        stemmed_documents[k] = text

for title, document in stemmed_documents.items():
    output_documents(title, document)
