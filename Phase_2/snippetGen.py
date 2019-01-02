from bs4 import BeautifulSoup


def get_cacm_queries(queriesFile):
    f = open(queriesFile)
    lines = f.readlines()
    queries = [x.rstrip('\n') for x in lines]
    return queries


def getTopDocIds(qid):
    path = "Lucene_top_docs_result/"
    topDocFile = open(path + str(qid) + "_lucene_docs.txt", "r")
    docIds = []
    for data in topDocFile.readlines():
        row = data.split()
        docIds.append(row[2])
    return list(set(docIds))


def generateSnippetNgram(queryTerms, doc, ngramSize):
    lookAhead = 40
    postTail = 50
    htmlContent = BeautifulSoup(doc, "html.parser").find('pre').get_text()

    for i in range(len(queryTerms) - (ngramSize - 1)):
        if ngramSize == 3:
            queryTerm = queryTerms[i] + " " + queryTerms[i + 1] + " " + queryTerms[i + 2]
        elif ngramSize == 2:
            queryTerm = queryTerms[i] + " " + queryTerms[i + 1]
        else:
            queryTerm = queryTerms[i]
        termLocation = htmlContent.find(queryTerm)
        if termLocation != -1:
            startIndex = termLocation - lookAhead
            if startIndex <= 0:
                startIndex = 0
            else:
                while startIndex > 0:
                    if htmlContent[startIndex - 1:startIndex] not in [" ", "\n"]:
                        startIndex -= 1
                    else:
                        break
            endIndex = htmlContent.index(queryTerm) + len(queryTerm) + postTail
            if endIndex > len(htmlContent):
                endIndex = len(htmlContent)
            while endIndex < len(htmlContent):
                if htmlContent[endIndex:startIndex - 1] not in [" ", "\n"]:
                    endIndex += 1
                else:
                    break
            first = htmlContent[startIndex: htmlContent.index(queryTerm)]
            second = htmlContent[htmlContent.index(queryTerm): htmlContent.index(queryTerm) + len(queryTerm)]
            third = htmlContent[htmlContent.index(queryTerm) + len(queryTerm): endIndex]
            return first, second, third

    return False, False, False


def getSnippets(query, doc, docId, qid):
    queryTerms = query.split()
    found = False
    for i in reversed(range(1, 4)):
        if found:
            break
        for j in reversed(range(1, i + 1)):
            first, second, third = generateSnippetNgram(queryTerms, doc, j)
            if first != False:
                print("\t" + docId)
                print(first + " " + "\033[44;33m" + second + "\033[m" + " " + third)

                outputFile.write("<h3>\t" + docId + "</h3><br/>")
                snipContent = first + " <strong style='color:blue;''>" + second + "</strong> " + third
                outputFile.write("<p>\t" + snipContent + "</p><br/>")
                found = True
                break


Queries = get_cacm_queries("cacm.query.parsed.txt")
# print(RawFiles)
outputFile = open("snippets.html", "w")
outputFile.write("<html><body>")
for query, qid in zip(Queries, range(1, len(Queries) + 1)):
    print("qid", qid)
    docIds = getTopDocIds(qid)
    outputFile.write("<h1>" + str(qid) + ") " + query + "</h2><br/>")
    for docId in docIds:
        rawDoc = open("cacm/" + docId).read()
        snippet = getSnippets(query, rawDoc, docId, qid)

outputFile.write("</body></html>")
outputFile.close()
