import glob
import re

from bs4 import BeautifulSoup

# Path of file
file_path = input("Enter Path for all files: ") or "/Users/romilrathi/Downloads/test-collection/cacm/*"
files = glob.glob(file_path)

# Pre-Processing Requirements
punc = input("Handle Punctuation(Y/N): ") or "Y"
case = input("CaseFolding(Y/N): ") or "Y"

# List containing raw text from each HTML Doc
corpus = []
# List containing filename for each HTML Doc
file_names = []
# List containing pre-processed corpus for each CACM file
processed_text = []


# Function to pre-process text
def process_data(raw_text, punc, case):
    # Square Brackets
    processed_text = re.sub(r'\[.+?\]\s*', "", raw_text)

    # Curly Brackets
    processed_text = re.sub(r'\{.*\}\s*', "", processed_text)

    # removing Extra white spaces
    processed_text = " ".join(processed_text.split())

    # removing punctuations
    if (punc == "Y"):
        processed_text = re.sub(r'(?<!\d)[.,;:#!?$*&+](?!\d)', "", processed_text)
        processed_text = re.sub('"', "", processed_text)
        processed_text = re.sub(
            r'\(*\)|\(*\(|\(*\#|\(*\,|\(*\.\(*\ |\(*\%|\(*\/|\(*\'|\(*\ \(*\&|\(*\ \(*\-|\(*\ \(*\+|\(*\-\(*\ ', "",
            processed_text)
        processed_text = re.sub(r'\(*\.\(*\ |\(*\ \(*\;\(*\ |\(*\ \(*\#\(*\ ', " ", processed_text)

    # Non Ascii characters
    processed_text = re.sub('[^\x00-\x7F]+', "", processed_text)

    # removing  extra space chararcters
    processed_text = re.sub(" +", " ", processed_text)

    # case folding
    if (case == "Y"):
        processed_text = processed_text.lower()

    # removing normal brackets and their content
    processed_text = re.sub(r'\([^)]*\)', "", processed_text)

    return (processed_text)


# Generating Raw Corpus
for file in files:
    with open(file, 'r+', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
        data = ""

        for tags in soup.findAll("pre"):
            for lines in tags.findAll(text=True):
                data = data + lines.strip() + " "
                corpus.append(data)
    # Getting the filename
    file_names.append(str(file.split("/")[-1])[:-5])

# cleaning Data
for data in corpus:
    raw_text = " "
    for text in data:
        raw_text = raw_text + text
    clean_text = ""
    clean_text = process_data(raw_text, punc, case)
    processed_text.append(clean_text)

# Writing processed corpus
for i in range(len(file_names)):
    file = file_names[i]
    data = processed_text[i]
    with open(file, 'w') as f:
        f.write(data)
