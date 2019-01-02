def fetch_stop_words(file):
    stop_words = []
    f = open(file)
    words = f.readlines()
    f.close()
    for word in words:
        stop_words.append(word[:-1])
    return stop_words


def write_processed_query(stop_words, file):
    query_file = open(file)
    queries = query_file.readlines()
    query_file.close()

    query_stopped = open("cacm.query.stopped.txt", "w")

    for query in queries:
        query = query[:-1]
        query_words = query.split(" ")
        stopped_query = ""
        for query_word in query_words:
            if query_word not in stop_words:
                if stopped_query == "":
                    stopped_query = query_word
                else:
                    stopped_query = stopped_query + " " + query_word
        query_stopped.write(stopped_query + "\n")
    query_stopped.close()


stop_words = fetch_stop_words('common_words.txt')
query_file = "cacm.query.parsed.txt"
write_processed_query(stop_words, query_file)
