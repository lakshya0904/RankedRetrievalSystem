
#   Wikipedia files used: AA - wiki_10 to wiki_29 (20 files)
# Group 3
# Lakshya Agarwal			2017B5A70904P
# Anuj Hydrabadi 			2017A8PS0420P
# Samarth Gupta 			2017B4A70467P
# Raj Shree Singh 			2017B4A70808P
# Aditya Vishwakarma			 2017B5A70954P


import pandas as pd
import numpy as np
import re
import pickle
import nltk
import os
import timeit
import string
import math
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from collections import Counter
from collections import defaultdict
from nltk.corpus import wordnet

IDX_FILE = 'index.pickle'
MODIFIED_IDX_FILE = 'modified_index.pickle'

# One time download
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

_trm_doc_wgt = None
_idf = None
_doc_id_ttl_map = None
_chmp_lst = None
_trm_ttl_wgt = None
_title_idf = None

query = None

_modified_trm_doc_wgt = None
_modified_idf = None
_modified_doc_id_ttl_map = None
_modified_chmp_lst = None


def load_index():
    """
    Loads both normal index and modified index as saved by build_index.py 
    and assigns them as per requirements.
    """
    global _trm_doc_wgt
    global _idf
    global _trm_ttl_wgt
    global _chmp_lst
    global _doc_id_ttl_map
    global _title_idf
    global _title_trm_freq
    global _trm_freq

    global _modified_trm_doc_wgt
    global _modified_doc_id_ttl_map
    global _modified_idf
    global _modified_chmp_lst

    with open(IDX_FILE, 'rb') as f:
        ldd_obj = pickle.load(f)
        _idf = ldd_obj['idf']
        _title_idf = ldd_obj['title_idf']
        _chmp_lst = ldd_obj['chmp_lst']
        _trm_doc_wgt = ldd_obj['trm_doc_wgt']
        _doc_id_ttl_map = ldd_obj['doc_id_ttl_map']
        _title_trm_freq=ldd_obj['title_trm_freq']
        _trm_freq=ldd_obj['trm_freq']
        _trm_ttl_wgt = ldd_obj['trm_ttl_wgt']

    with open(MODIFIED_IDX_FILE, 'rb') as f:
        ldd_obj = pickle.load(f)
        _modified_idf = ldd_obj['idf']
        _modified_chmp_lst = ldd_obj['chmp_lst']
        _modified_trm_doc_wgt = ldd_obj['trm_doc_wgt']
        _modified_doc_id_ttl_map = ldd_obj['doc_id_ttl_map']
        _title_idf = ldd_obj['title_idf']
        _title_trm_freq=ldd_obj['title_trm_freq']
        _trm_ttl_wgt = ldd_obj['trm_ttl_wgt']


def get_term_weights_for_query(query, idf, qry_typ = 0):
    """
    This function tokenizes the query considering removal of the punctuations. 
    For queries we are using ltc model. ‘l’ represents term frequency is 
    considered using a logarithmic model that is the value of 1 plus logarithm 
    of frequency of occurrence of  the term in the query. ‘t’ represents 
    the inverse document frequency which is already created in 
    get_term_document_weights() function. ‘c’ is cosine normalisation. 
    """
    raw_query_tokens = [token for token in word_tokenize(query) if token not in string.punctuation]
    query_terms = [token.lower() for token in raw_query_tokens] #as a list for multiple occurence of same word
    
    print(query_terms)
    if (qry_typ == 1 or qry_typ == 4):
        synList=[]
        for q in query_terms:
            synsets = wordnet.synsets(q)
            for synset in synsets:
                for syn in synset.lemma_names():
                    syn=syn.replace("_"," ")
                    synList.append(syn)
        # print(synList)
        # query_terms=synList
        trm_qry_trm=[]
        for w in synList:
            if w in _trm_freq and w not in trm_qry_trm:
                trm_qry_trm.append(w)
        for q in query_terms:
            if q in _trm_freq and q not in trm_qry_trm:
                trm_qry_trm.append(q)
        query_terms=trm_qry_trm
        print("Query terms after including synonyms: ")
        print(query_terms)
    
    # print(query_terms)

    result = defaultdict(int)
    for term in query_terms:
        result[term] += 1

    for term in result:
        result[term] = (1 + math.log10(result[term])) * (idf.get(term, 0))#tf_idf

    vctr_len = 0
    for weight in result.values():
        vctr_len += weight ** 2
    vctr_len = math.sqrt(vctr_len)#for normalisation of query

    if vctr_len == 0:
      vctr_len = 1     # To avoid Divide by Zero errors

    # normalization
    for term in result:
        result[term] /= vctr_len

    return query_terms, result

def search(qry_typ = 0):
    """
    Calculates cosine similarity score on the basis of query types entered by 
    the user.
    """
    k = int(input("\nEnter No. of documents to retrieve: "))   


    if (qry_typ == 0 or qry_typ == 1 or qry_typ == 3):
        idf = _idf
        trm_doc_wgt = _trm_doc_wgt
        chmp_lst = _chmp_lst
        doc_id_ttl_map = _doc_id_ttl_map
        trm_ttl_wgt = _trm_ttl_wgt

    elif (qry_typ == 2 or qry_typ == 4):
        idf = _modified_idf
        chmp_lst = _modified_chmp_lst
        trm_doc_wgt = _modified_trm_doc_wgt
        doc_id_ttl_map = _modified_doc_id_ttl_map
        trm_freq = _trm_freq
        trm_ttl_wgt = _trm_ttl_wgt

    query_terms, trm_qry_wgts = get_term_weights_for_query(query, idf, qry_typ)
    doc_id_score = defaultdict(int)

    if (qry_typ == 2 or qry_typ == 4):
        chmp_dcs = set()
        for word in query_terms:
            if word in _trm_freq:
                chmp_dcs = chmp_dcs.union(set(chmp_lst[word]))

    start1 = timeit.default_timer()
    for term, qry_wgt in trm_qry_wgts.items():
        for doc_id, doc_weight in trm_doc_wgt[term].items():
            if (qry_typ == 2 or qry_typ == 4) and doc_id not in chmp_dcs: continue
            doc_id_score[doc_id] += qry_wgt * doc_weight
    stop1 = timeit.default_timer()

    start2=0
    stop2=0
    if (qry_typ == 3 or qry_typ == 4):
   # Zone indexing
        for doc_id, val in doc_id_score.items():
            doc_id_score[doc_id] *=0.8

        start2 = timeit.default_timer()
        for term, qry_wgt in trm_qry_wgts.items():
            for doc_id, title_weight in trm_ttl_wgt[term].items():
                if (qry_typ == 2 or qry_typ == 4) and doc_id not in chmp_dcs: continue
                doc_id_score[doc_id] += qry_wgt * title_weight * 0.2
        stop2 = timeit.default_timer()
    print('Time: ', stop1 - start1 + stop2 - start2) 
           


    doc_scr_pr = [(doc_id, score) for doc_id, score in doc_id_score.items()]

    doc_scr_pr.sort(key=lambda pair: pair[1], reverse=True)

    k = min(k, len(doc_scr_pr))

    print("\nNumber of retrieved documents:",min(k,len(doc_scr_pr)))
    print("\n")

    #df=pd.DataFrame(columns = ['Query', 'Top k documents', 'Score', 'Is the document relevant to the query?'])
    for pair in doc_scr_pr[:k]:
        doc_id = int(pair[0])
        title = doc_id_ttl_map[str(doc_id)]
        score = pair[1]
        print("score = {0:.4f}, document_id = {1}, title = {2}".format(score, doc_id, title))
        #df2 = {'Query': query, 'Top k documents': title, 'Score' : score}
        #df = df.append(df2, ignore_index = True)
        #df.to_csv('/content/drive/MyDrive/IRProject/'+query[:5]+'.xlsx', index=False, encoding='utf-8-sig')



def run_repl():
    """
    Facilitates Main menu functioning. Takes input query from the user in 
    global variable ‘query’ and a choice from available options.
    """
    global query
    print('\nOptions:')
    print('1. Normal search based on lnc.ltc scoring scheme')
    print('2. Improvement using synonyms only')
    print('3. Improvement using champion list')
    print('4. Improvement using zone indexing')
    print('5. Improvement using synonyms, championship list and zone indexing')
    while True:
        print('\nMain Menu\n')
        query = input('Enter the query: ')

        qry_typ = int(input('\nEnter your choice (either 1, 2, 3, 4 or 5): '))
        if qry_typ>5 or qry_typ<1:
            print('\nWrong choice, terminating... :-|')
            return
        qry_typ = qry_typ - 1
        search(qry_typ)
        again = int(input('\nEnter 1 to go back to main menu else 0 to quit: '))
        if again!=1:
            print('\nTerminating...')
            return


def main():
    choice =int(input('\nLoad index file from default location enter 0 else enter 1: '))
    if(choice==1):
        IDX_FILE = input('Enter path to basic index file: ')
        MODIFIED_IDX_FILE = input('Enter path to modified index file: ')
    print('\nLoading indices....')
    load_index()
    run_repl()

if __name__ == '__main__':
    main()