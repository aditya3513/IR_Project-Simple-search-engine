import re


def remove_punctuations(raw_text):
    processed_text = re.sub(r'(?<!\d)[.,;:#!?$*&+](?!\d)', "", raw_text)
    processed_text = re.sub('"', "", processed_text)
    processed_text = re.sub(
        r'\(*\)|\(*\(|\(*\#|\(*\,|\(*\.\(*\ |\(*\%|\(*\/|\(*\'|\(*\ \(*\&|\(*\ \(*\-|\(*\ \(*\+|\(*\-\(*\ ', "",
        processed_text)
    processed_text = re.sub(r'\(*\.\(*\ |\(*\ \(*\;\(*\ |\(*\ \(*\#\(*\ ', " ", processed_text)

    return processed_text


file = open("cacm.query.txt")
lines = file.readlines()
i = -1
queries = {}
for line in lines:
    if line == "</DOC>\n" or line == "<DOC>\n":
        continue
    elif line.startswith("<DOCNO>"):
        i += 1
        queries[i] = ""
    else:
        if line != "\n":
            line = line.replace("\n", "")
            queries[i] = queries[i] + " " + line.lower()
file.close()

file2 = open("cacm.query.parsed.txt", "w+")
for query in queries.values():
    file2.write(remove_punctuations(query)[2:] + "\n")
file2.close()
