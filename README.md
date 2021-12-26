# Vector Space Based Ranked Retrieval System

Assignment done as part of my <u>INFORMATION RETRIEVAL course</u> at BITS Pilani.

### Problem Statement Part 1:
Implementation of a vector space-based information retrieval system with the following characteristics:
1. The vector space model should be used for computing the score between document and query.
2. Use the lnc.ltc scoring scheme (based on SMART notation).
3. The queries should be free-text queries.
4. Retention of stop words, removal of punctuations and stemming/lemmatization and normalization shouldn't be done.

Detailed Problem Statement can be found [here](https://github.com/lakshya0904/RankedRetrievalSystem/blob/main/Assignment_IR_RS-1.pdf)
<hr>

### Problem Statement Part 2:
Improve the retrieval and ranking for the documents. Further, answer following questions for each proposed improvement:
1. What are the issues with the vector space model built in part 1?
2. What improvement are you proposing?
3. How will the proposed improvement address that issue?
4. A corner case (if any) where this improvement might not work or can have an adverse effect.
5. Demonstrate the actual impact of the improvement. Give three queries, where the improvement yields better results compared to the Part 1 implementation.

Detailed Problem Statement can be found [here](https://github.com/lakshya0904/RankedRetrievalSystem/blob/main/Assignment_IR_RS-1.pdf)
<hr>

### The following packages need to be installed

- nltk
- bs4
- collections
- pyspellchecker
- pickle
- numpy
- requests

### Structure

There are 2 functionalities:

1. index_builder: This file creates an inverted index for documents in the folder `wikis` and stores it in a pickle file for further processing.
2. search: It takes as an input a query and gives as output the top K documents. This file never reads the text corpus.

### User Interface

Enter the query: `tv`

Enter your choice (either 1, 2, 3, 4 or 5): `1`

Enter No. of documents to retrieve: `10`
['tv']
Time:  0.0003894120000040857

Number of retrieved documents: 10

```
score = 0.1500, document_id = 26399, title = rms laconia
score = 0.1132, document_id = 27344, title = telecommunications in slovenia
score = 0.0915, document_id = 17056, title = kiyoshi atsumi
score = 0.0891, document_id = 12114, title = telecommunications in greece
score = 0.0878, document_id = 13039, title = gordon michael woolvett
score = 0.0871, document_id = 27846, title = sorious samura
score = 0.0862, document_id = 23875, title = phoenix (tv series)
score = 0.0823, document_id = 12073, title = telecommunications in ghana
score = 0.0803, document_id = 21813, title = naked news
score = 0.0783, document_id = 20659, title = mv blue marlin
```

Enter 1 to go back to main menu else 0 to quit: 0

### Implementation Details

Weighting scheme for ranked retrieval is lnc.ltc:

1. Before asking any queries the system pre-calculates term-document weights, using the formula `1 + log10(term_frequency)` and normalizes it by document vector's length (for cosine similarity). Results are stored in a Dictionary for fast future accesses. Also, inverse document frequency (idf) is computed for all terms.

2. When free-text query is typed, the system computes term-query weights using formula `(1 + log10(term_frequency_in_query)) * idf(term)` and normalizes them. It requires linear time depending on the query length.

3. To efficiently calculate document scores, term-at-a-time approach (bag of words) is used for query terms:
```
	for term, query_weight in term_query_weights.items():
		for doc_id, doc_weight in term_doc_weights[term].items():
			doc_id_score[doc_id] += query_weight * doc_weight
```
Time complexity will linearly depend on the number of term-document pairs for query terms.

4. Documents are sorted by their scores in `(O(N log N)`, where N is the number of documents, containing query terms) to show top relevant.

### Improvements (Problem Statement Part 2)

#### 1. Synonyms
The IR System built matches the query term exactly to the term present in the document. It does not take any morphological analysis or the root word of the term into consideration. Synonyms are useful when the query words in the document are acronyms or less commonly used words such as tv, occult, phantom. In this case, getting a list of synonyms first such as television for tv, magical for occult greatly increases the relevant queries as the queries search become larger by incorporating more common terms. We are proposing to consider relevant words during search using nltk corpus.

#### 2. Champion list
In vector space model time consumed will be more because it will consider all documents some of which are less relevant. The effect gets highlighted as the size of the corpus increases. We tried to increase the speed of execution by precomputing a list of 200 documents which have the highest weight for a particular term per their term frequency. Through this, we aim to increase the speed of execution by avoiding the computation of all document rankings at query time. The speed of execution increases by roughly 3 times even though we are using a relatively small corpus of 9610 documents.

#### 3. Zone indexing
In basic model impleneted in part 1 titles of documents are not given any importance, but it is quite intuitive that while dealing with Wikipedia Corpus titles have more importance. Improved proposed is that assigned score for a (query, document) pair will be linear combination of zone score. Two zones ‘content’ and ‘title’ have weights of 0.8 and 0.2, respectively. The improved score can be seen in examples like - jaguar, lunar eclipse, and Jaffrey Dahmer.

<hr>
Finally, we use a combination of synonyms, zone indexing and champion list simultaneously. This is to indicate the scalability of our system that all the changes can work independently as well as in combination with other improvements in the system.


### Guidelines to run the assignments

1. Build the index (Needs to be done only once and it will create `index.pickle` and `modified_index.pickle` files which would store inverted_index and other relevant information)

```
python3 build_index.py
```

2. To search any query:

```
python3 test_queries.py
```


### Group Members

- Lakshya Agarwal 2017B5A70904P
- Anuj Hydrabadi 2017A8PS0420P
- Samarth Gupta 2017B4A70467P
- Raj Shree Singh 2017B4A70808P
- Aditya Vishwakarma 2017B5A70954P
