*                                                                              *
*  Information Retrieval Fall 2018                                             *
*                                                                              *
*  Project - Aayushi Maheshwari                                                *
*            Aditya Sharma                                                     *
*            Romil Rathi                                                       *
*                                                                              *
********************************************************************************

IMPORTANT: This is the readme for the project and extra credit.

This project is split into various phases and tasks.

The Lucene part of this project is implemented in Java. All other parts of the project are completely implemented in Python.

External libraries
------------------

    * BeautifulSoup : This library is used to parse CACM documents
    * Numpy         : This library is used to perform interpolation
    * Pandas        : This library is used to create dataframe
    * nltk 	    : This library is used to pre_process the CACM data
    * Matplotlib    : This library is used to plot the curves
    

How to setup
------------

  Java
  ----

    * Java version     : Java 8 (output of "javac -version" : "javac 1.8.0_151")
    * Lucene version   : 4.7.2
    * Operating System : The project was developed in Linux and Windows simultaneously. Though the project can be executed in either 
			 operating systems, Linux is preferred and also a shell script to run all parts of the implementation is 
			 provided.

    Lucene 4.7.2's jar files are provided in this package itself (refer to the directory structure section of this readme). 
    Therefore, the user of this package does not have to download any package to run Lucene program.

  Python
  ------

    * Python version 3.6
    * Operating System : The assigment was done in Linux (Ubuntu). Though the project can be executed in either operating
                         systems, Linux is preferred and also a shell script to run all parts of the implementation is provided.

    Having the external libraries specified should be enough to run the programs.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Folders : 
---------------------------------------------------------------------------------------------------------------------------------------
Phase 1:
Contains Source code and Output for Tasks in phase 1.

	- Folder Task_1_A : Contains source code and output for pre-processing CACM files and Generate Inverted Index.
			Files:
			(i) Pre_processing.py
			(ii)inverted_index.py
			(iii) inverted_index.json - Index Generated
			(iv) doc_length

	-Folder Task_1_B : Contains source code for 
			(i) TF-IDF Model - tf_idf.py
			(ii) Lucene Model - Lucene.java
			(iii)BM-25 	- BM25.py
			(iv) JM Smoothing Query likelihood model - JMsqlm.py
			
	-Folder Task_1_Output : Contains output for task_1
			(i)BM25.txt	
			(ii)JMsqlm.txt	
			(iii)Lucene.txt		
			(iv)tfidf.txt
	
	- Folder Task_2 : Contains source code for Query Enrichment Technique
			(i) BM25_PRF.py : contains source code to perform task 2
			(ii) Files cacm.rel and 
	
	
	- Folder Task_2_output : Contains output for task_2
			(i) BM25_pseudo_relevance_feedback.txt
	
	
	
	-Folder Task_3_A : Contains source code to perform query_processing and Stopping to generate a different index and fit three models implemented above.
			
			(i) BM25_stop.py
			(ii)JMsqlm_stop.py
			(iii)tf_idf_stop.py
			(iv)stop_index.py
			(v)stop_inverted_index.txt
			(vi)stop_doc_length
			(viii) q_process.py - contains code for processing query for stopping
			(vii) cacm.query.stopped.txt - Contains Query after stopping
	
	
	
	-Folder Task_3_B : Contains source code to perform Task 1 and Task 2 on stemmed corpus.
			(i) stem_processor.py - Code to process stemmed corpus and write to a file
			(ii)stem_indexer.py - Code to generate index for stemmed corpus
			(iii)BM25_stem.py
			(v)JMsqlm_stem.py
			(vi)tf_idf_stem.py
			(vii)cacm_stem.query.txt
			(viii)cacm_stem.txt
	
	
	
	-Folder Task_3_output : contains output for task 3
			(i)BM25_stem.txt
			(ii)BM25_stop.txt
			(iii)JMsqlm_stem.txt
			(iv)JMsqlm_stop.txt	
			(v)tfidf_stem.txt
			(vi)tfidf_stop.txt

	
	
	- query_processing.py : contains source code to parse and process query into cacm.query.parsed.txt
	
	
	- cacm.query.parsed.txt : contains processed queries.
			


Phase 2:
Contains source code and Output for Sinppet generation task in Phase 2

	(i)snippetGen.py - Source code for Phase 2
	(ii)cacm.query.parsed.txt - Query
	(iii)snippets.html - Output




Phase 3:
Contains Source code and Output for evaluation in phase 3.
	
	-Folder Phase_3_Outputs: contains output for phase 3
		(i) Lucene_PR_Table
		(ii) BM25_PR_Table
		(iii)JmSQLM_PR_Table
		(iv)Stop_JmSQLM_PR_Table
		(v)Stop_BM25_PR_Table
		(vi)TFIDF_PR_Table
		(vii) Stop_TFIDF_PR_Table
		(viii) Stop_BM25_PR_Table
		(ix)precision_recall_plot.png
		

	- phase3.py : contains source code and generates output for Evaluation of Information Retrieval Systems.
	
	


extra_credit :
Contains Source code and Output for Extra Credit Problem
	
	- proximity.py - Contains source code for extra credit problem
	- BM25_proximity.txt - Contains output by proximity file
	
	
Report Graph:
Contains Source code and Output for Plots used in Project Report.

	- Folder Output: Contains Output graphs for evaluation metrics
		(i) MAP.png
	 	(ii)MRR.png
	 	(iii)P@5.png
	 	(iv)P@20.png
		(v) Query_ID_precision.png
		(vi)Query_ID_recall.png
	 
	 - map_p5_20.py - Generates MAP, MRR, P@5 and P@20
	 - tables.py - contains source code to generate separate precision and recall curve for given query	





---------------------------------------------------------------------------------------------------------------------------------------
Phase 1: INDEXING AND RETRIEVAL

TASK 1:

INSTRUCTIONS ON COMPILING AND RUNNING CODE TO GENERATE INDEX:

Import the below mentioned libraries:
-- import glob
-- import re
-- from bs4 import BeautifulSoup

To generate a clean corpus from raw files, execute the "pre-processing.py" file. It provides the user options for case-folding and punctuation handling.

Please provide the path of the raw files when prompted.
Specify Y or N for Handling punctuation or Casefolding, and depending on the user input the corpus is cleaned.

When the file is executed, the user gets the prompt to enter Y or N.
Handle Punctuation(Y/N): Y
CaseFolding(Y/N): Y

We have generated a clean corpus by incorporating both punctuation and casefolding for this project.

OUTPUT of "pre-processing.py":
The output will be separate files per article, which contain the plain textual content of the article. A folder is created:
Preprocessed -- contain the clean files with casefolding and punctuation removed both

DESIGN CHOICE:
Since, the inverted index is the main data structure of any information retrieval system, it should be very efficient. Hence, We have used a Hashtable (in python known as dictionary) to store the inverted index in memory which can perform lots of lookups (one for every term in the document), and can add lots of keys (every term is a key) and its values. Since Hashtables have average O(1) lookup time and amortized O(1) insertion time, they are very suitable for this purpose.

Import the below mentioned libraries:
-- from collections import Counter
-- import glob
-- from nltk import ngrams
-- import json

To generate the inverted index, execute the "inverted_index.py" file. It takes the absolute(full) path of the folder to read the parsed files. By default it is performed on unigram value.

OUTPUT of "inverted_index.py":
It creates two files:
-- first named, "inverted_index.json" which has the inverted index stored in the format 
{"term":"[[doc_id, term_frequency], [doc_id, term_frequency],...], ..."}
-- second named, "doc_length" file which has the number of terms in each document in the corpus in the format 
doc_id : [token_count]\n doc_id : [token_count]\n...


INSTRUCTIONS ON COMPILING AND RUNNING CODE TO SCORE DOCUMENTS:

LUCENE:
To score the documents, execute the "Lucene.java" file. When prompted,

1 -- please enter the path where the index will be created; provide the absolute(full) path along with file name.
2 -- please enter the path of the folder to read the parsed files; provide the absolute(full) path. If you want to add files from more folders, enter the path again or press "q" to proceed to next steps.
3 -- please enter the path of search query file; provide the absolute(full) path along with file name.

OUTPUT:
For each query,
1. it prints the total number of matches found.
2. for each term in the query, it prints the term frequency and the document freqeuncy.
3. a file is generated which has the 100 top ranked documents in the default TREC format: query_id Q0 doc_id rank Lucene_score system_name

BM25:
Import the below mentioned libraries:
-- import operator
-- import json
-- import math
-- from collections import Counter

To score the documents, execute the "BM25.py" file. The get_BM25_output() takes in 5 parameters as below:
1. "inverted_index.json"
2. "doc_length.txt"
3. "cacm.query.parsed.txt"
4. "BM25.txt"
5. "BM25"

OUTPUT:
A file "BM25.txt" is generated which has the 100 top ranked documents for each query in the default TREC format: query_id Q0 doc_id rank BM25_score system_name

tfidf:
Import the below mentioned libraries:
-- import numpy as np
-- import pandas as pd
-- import json
-- import math

To score the documents, execute the "tf_idf.py" file. The resulting files are present in tfidf_top_docs folder.

OUTPUT:
For each query,
-- a file "query_id" is generated which has the 100 top ranked documents in the default TREC format: query_id Q0 doc_id rank tfidf_score system_name

JMsqlm:
Import the below mentioned libraries:
-- import operator
-- import json
-- import math
-- import numpy as np

To score the documents, execute the "JMsqlm.py" file. The get_sqlm_output() takes in 5 parameters as below:
1. "inverted_index.json"
2. "doc_length.txt"
3. "cacm.query.parsed.txt"
4. "JMsqlm.txt"
5. "SQLM"

OUTPUT:
A file "JMsqlm.txt" is generated which has the 100 top ranked documents for each query in the default TREC format: query_id Q0 doc_id rank JMsqlm_score system_name


---------------------------------------------------------------------------------------------------------------------------------------

TASK 2:

INSTRUCTIONS ON COMPILING AND RUNNING CODE FOR QUERY ENRICHMENT:

We have selected BM25 out of the four baseline runs to perform query enrichment.
 
Import the below mentioned libraries:
-- import json
-- import math
-- import operator
-- from collections import Counter

To score the documents based on pseudo relevance feedback, execute the "BM25_PRF.py" file. The get_BM25_relevant_docs() takes in 6 parameters as below:
1. "inverted_index.json"
2. "doc_length"
3. "cacm.query.parsed.txt"
4. "BM25_pseudo_relevance_feedback.txt"
5. "BM25_PRF"
6. "common_words"

OUTPUT:
A file "BM25_pseudo_relevance_feedback.txt" is generated which has the 100 top ranked pseudo relevant documents for each query in the default TREC format: query_id Q0 doc_id rank BM25_score BM25_PRF


---------------------------------------------------------------------------------------------------------------------------------------

TASK 3:

INSTRUCTIONS ON COMPILING AND RUNNING CODE TO GENERATE STOPPED_INDEX:

Import the below mentioned libraries:
-- from collections import Counter
-- import glob
-- from nltk import ngrams
-- import json

To generate the stopped inverted index, execute the "stop_index.py" file. It takes the absolute(full) path of the folder to read the parsed files and the "common_words" file provided to us in the "test-collection" folder. By default it is performed on unigram value.

OUTPUT of "stop_index.py":
It creates two files:
-- first named, "stop_inverted_index.txt" which has the stopped inverted index stored in the format 
{"term":"[[doc_id, term_frequency], [doc_id, term_frequency],...], ..."}
-- second named, "stop_doc_length" file which has the number of terms in each stopped document in the corpus in the format 
doc_id : [token_count]\n doc_id : [token_count]\n...


INSTRUCTIONS ON COMPILING AND RUNNING CODE TO GENERATE STEMMED_INDEX:

Import the below mentioned libraries:
-- from collections import Counter
-- import glob
-- from nltk import ngrams
-- import json

To generate the stemmed inverted index, execute the "stem_index.py" file. It takes the absolute(full) path of the folder to read the parsed files. By default it is performed on unigram value.

OUTPUT of "stem_index.py":
It creates two files:
-- first named, "stem_inverted_index.json" which has the stemmed inverted index stored in the format 
{"term":"[[doc_id, term_frequency], [doc_id, term_frequency],...], ..."}
-- second named, "stem_doc_length" file which has the number of terms in each stemmed document in the corpus in the format 
doc_id : [token_count]\n doc_id : [token_count]\n...


INSTRUCTIONS ON COMPILING AND RUNNING CODE TO SCORE DOCUMENTS:

We have selected BM25, tfidf and JMsqlm as the 3 baseline runs.

We are using the same code for the 3 models and providing respective stopped: inverted index, document length and queries as inputs. 
BM25:
To score the documents, execute the "BM25_stop.py" and "BM25_stem.py" files. The get_BM25_output() takes in 5 parameters as below:
1. "stop_inverted_index.txt" OR "stem_inverted_index.json"
2. "stop_doc_length" OR "stem_doc_length"
3. "cacm.query.stopped.txt" OR "cacm_stem.query.txt"
4. "BM25_stop.txt" OR "BM25_stem.txt"
5. "BM25"

OUTPUT:
A file "BM25_stop.txt" is generated which has the 100 top ranked stopped documents for each stopped query in the default TREC format: query_id Q0 doc_id rank BM25_score BM25
A file "BM25_stem.txt" is generated which has the 100 top ranked stemmed documents for each stemmed query in the default TREC format: query_id Q0 doc_id rank BM25_score BM25

tfidf:
To score the documents, execute the "tf_idf_stop.py" and "tf_idf_stem.py" files. The resulting files are present in "tfidf_stop_top_docs" and "tfidf_stem_top_docs" folders respectively.

OUTPUT:
For each stopped query,
-- a file "query_id" is generated which has the 100 top ranked stopped documents in the default TREC format: query_id Q0 doc_id rank tfidf_score tfidf
For each stemmed query,
-- a file "query_id" is generated which has the 100 top ranked stemmed documents in the default TREC format: query_id Q0 doc_id rank tfidf_score tfidf

JMsqlm:
To score the documents, execute the "JMsqlm_stop.py" and "JMsqlm_stem.py" files. The get_sqlm_output() takes in 5 parameters as below:
1. "stop_inverted_index.txt" OR "stem_inverted_index.json"
2. "stop_doc_length" OR "stem_doc_length"
3. "cacm.query.stopped.txt" OR "cacm_stem.query.txt"
4. "JMsqlm_stop.txt" OR "JMsqlm_stem.txt"
5. "SQLM"

OUTPUT:
A file "JMsqlm_stop.txt" is generated which has the 100 top ranked stopped documents for each stopped query in the default TREC format: query_id Q0 doc_id rank JMsqlm_score SQLM
A file "JMsqlm_stem.txt" is generated which has the 100 top ranked stemmed documents for each stemmed query in the default TREC format: query_id Q0 doc_id rank JMsqlm_score SQLM


---------------------------------------------------------------------------------------------------------------------------------------


PHASE 2: DISPLAYING RESULTS

INSTRUCTIONS ON COMPILING AND RUNNING CODE TO GENERATE SNIPPETS:

Import the below mentioned libraries:
-- from bs4 import BeautifulSoup

To generate the snippets , execute the "snippetGen.py: file in Phase2 filter, it has all the revenant files need for it to run. These files are cacm.query.parsed file, top documents from lucent for each query in a folder and cacm corpus.

OUTPUT:
1) it shows the results on terminal by showing the snippets for each query like:

 Query_id) Query
	 Document_Name
 snippet text where the query terms are highlighted by blue color.

2) It also generates a "snippets.html" file which follows a format similar to adobe but query terms are set to bold.

---------------------------------------------------------------------------------------------------------------------------------------

PHASE 3:EVALUATION

INSTRUCTIONS ON COMPILING AND RUNNING CODE TO EVALUATE ALL THE MODELS:

Import the below mentioned libraries:

-- from bs4 import BeautifulSoup
-- import matplotlib.pyplot as plt
-- import pandas as pd
-- import collections
-- from numpy import interp

Open phase3.py and set path for the output of relevance rank generated by models as:

relavance_file   - path to cacm.rel
Lucene_file	 - path to output of Lucene Model
query_file	 - path to output of query enrichment model
tfidf_file	 - path to output of tf-idf model
stop_tfidf_file	 - path to output of tf-idf model with stopping
bm_25		 - path to output of bm25 model
bm_25_stop	 - path to output of bm25 model with stopping
jmsqlm_file	 - path to output of JM SQLM model
stop_jmsqlm_file - path to output of JM SQLM model with smoothing

Execute phase3.py to evaluate all the models

OUTPUT:

Lucene_PR_Table - Contains Query ID and 
			Rank, Dod_id, Precision, Recall of 100 retrieved documents for each query for Lucene Model
		- Contains MAP, MRR for Lucene model
		- Contains P@5 and P@20 for each query for Lucene Model
		
Query_PR_Table - Contains Query ID and 
			Rank, Dod_id, Precision, Recall of 100 retrieved documents for each query for Query Enrichment Task
		- Contains MAP, MRR for Query Enrichment Task
		- Contains P@5 and P@20 for each query for Query Enrichment Task
		
TFIDF_PR_Table - Contains Query ID and 
			Rank, Dod_id, Precision, Recall of 100 retrieved documents for each query for TFIDF Model
		- Contains MAP, MRR for Query Enrichment Task
		- Contains P@5 and P@20 for each query for TFIDF model
		
Stop_TFIDF_PR_Table - Contains Query ID and 
			   Rank, Dod_id, Precision, Recall of 100 retrieved documents for each query for Stopping TFIDF model
		    - Contains MAP, MRR for Stopping TFIDF model
		    - Contains P@5 and P@20 for each query for Stopping TFIDF model
		
BM25_PR_Table.      - Contains Query ID and 
			Rank, Dod_id, Precision, Recall of 100 retrieved documents for each query for BM25 Model
		- Contains MAP, MRR for BM25 Model
		- Contains P@5 and P@20 for each query for BM25 Model
		
Stop_BM25_PR_Table   -	Contains Query ID and 
		   Rank, Dod_id, Precision, Recall of 100 retrieved documents for each query for BM25 with stopping Model
		- Contains MAP, MRR for BM25 with Stopping Model
		- Contains P@5 and P@20 for each query for BM25 with Stopping Model
		
JmSQLM_PR_Table	     -	Contains Query ID and 
			Rank, Dod_id, Precision, Recall of 100 retrieved documents for each query for JM SQL Model
		- Contains MAP, MRR for JM SQL Model
		- Contains P@5 and P@20 for each query for JM SQL Model
		
Stop_JmSQLM_PR_Table -	Contains Query ID and 
			Rank, Dod_id, Precision, Recall of 100 retrieved documents for each query for JM SQL Model with stopping
		     - Contains MAP, MRR for JM SQL Model with Stopping
		     - Contains P@5 and P@20 for each query for JM SQL Model with Stopping
		     
precision_recall_plot.png - Contains Precision-Recall Curve for all 8 models in one plot.

Design Choice - 

Function gold_relevance_docs(file) takes path of 'cacm.rel' to get the gold standard relevant files for evaluation

Function model_relevance_doc(file) takes path of file generated by the model we want to evaluate. It gives output as 100 relavant docs for each query

Function evaluate() takes the argument as output of above two functions to evaluate our model.
It returns following entities:

query_pr - {Query_ID : [DOC_ID, Precision, Recall]} i.e. For each document-Precision and Recall values
p5 	 - Contains P@5 for each query i.e. {Query_id : p@5}
p20 	 - Contains P@20 for each query i.e. {Query_id : p@20}
MAP	 - Mean Average Precision for all queries in the model
MRR	 - Mean Reciprocal Rank for all queries in the model

Function write_precision_recall_table() takes the argument as relevance_score(Gold Relevant), model_scores(Model Relevance), outputfile(Name of Model for Precision_Recall table).

It generates Precision-Recall table for each query for a the model provided as argument and returns interpolated precision.

Format : 
MAP - 
MRR - 

Query ID - 

P@5 -
P@10 -

Rank  Doc_ID Precision Recall

Corresponding to each recall value - Precision Value are stored in a list for all queries in a model. Precision values are interpolated for 11 values of recall = [0.0,0.1,0.2,...,1.0]

Now, average of interpolated value is taken to plot Precision_recall curve. here, x-axis is recall level and y-axis is average precision for that recall across all queries and the curve is generated.	


---------------------------------------------------------------------------------------------------------------------------------------


Extra Credit: Proximity Search

INSTRUCTIONS ON COMPILING AND RUNNING CODE TO RUN PROXIMITY SEARCH:

Import the below mentioned libraries:
-- import json
-- import math
-- import operator
-- from collections import Counter
-- from bs4 import BeautifulSoup

To generate the proximity results , execute the "proximity.py" file in extra_credit folder, it needs these files tu run: cacm.query.parsed file, inverted_index.json, doc_length.txt.

OUTPUT:
It also generates a "BM25_proximity.txt" generated which has the 100 top ranked documents for each query in the default TREC format: query_id Q0 doc_id rank proximity BM25_proximity

---------------------------------------------------------------------------------------------------------------------------------------

Report Graphs : Contains Source code for generating graphs used in Project Report

Import the below mentioned libraries:

-- from bs4 import BeautifulSoup
-- import matplotlib.pyplot as plt
-- import pandas as pd
-- import collections
-- from numpy import interp

tables.py : Generates two graphs for selected query:
	  	(i) Precision Curve
		(ii) Recall Curve
To execute tables.py - Open the file and add the path of results of all models in variable declared inside.

Output : [Query_id]_precision.png
	 [Query_id]_recall.png
	 
map_p5_20.py : Generates graph for MAP, MRR, P@5 and P@20 values

To execute map_p5_20.py - Open the file and add the path of results of all models in variable declared inside.

Output : MAP.png
	 MRR.png
	 P@5.png
	 P@20.png
	 
Note :  These files calculates Precision, Recall, MAP, MRR, K@5, K@20 as explained above in Phase 3 and then uses those values to generate the plots.

