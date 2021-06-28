
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

CHMP_LST_COUNT = 200
WIKIS_FOLDER = './wikis'
PICKLE_FILE = 'index.pickle'
MODIFIED_PICKLE_FILE = 'modified_index.pickle'

# One time download
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Instead of only one wiki, added 10 wikis for better results
response = ""

entries = os.listdir(WIKIS_FOLDER)
for entry in entries:
  f = open(os.path.join(WIKIS_FOLDER, entry), "r", encoding='utf8')
  response = response + f.read().lower()
#   print(len(response))

doc_dicts = []
doc_id_ttl_map = {}

idf = dict()            # A dictionary is a collection which is unordered, changeable and indexed
trm_freq = {}     # Creating dictionary of term frequencies
invtd_idx = {}     # Inverted index containing document-wise frequency

chmp_dcs = set()
chmp_lst = {}

# qry_typ = 1      
# qry_typ = 0 => No improvement, part-1
# qry_typ = 1 =>  improvements

"""## Parsing the text"""

def preprocess(qry_typ = 0):
  """
  The first step is to create the index of words. First, we parse the content 
  using BeautifulSoup library and tokenize the raw text. 
  """
  soup = BeautifulSoup(response, 'html.parser')
  TAG_RE = re.compile(r'<[^>]+>')

  all_docs = soup.find_all('doc')
#   print(all_docs[0])

  for doc in all_docs:
    doc_cnt = TAG_RE.sub('', ''.join(map(lambda x: str(x), doc.contents)))
    # print(doc_cnt)
    # break
    doc_cnt = doc_cnt.translate(str.maketrans('', '', string.punctuation))  # Remove punctuations from the doc_cnt
    # print(doc_cnt)
    # break
    doc_cnt = doc_cnt.replace("\n", " ")                                    # Remove unnecessary newlines
    doc_cnt = ''.join(i for i in doc_cnt if ord(i)<128)
    doc_cnt = " ".join(doc_cnt.split())
    # print(doc_cnt)
    # break
    
    if qry_typ == 1:
      doc_cnt = word_tokenize(doc_cnt)
      doc_cnt = ' '.join(doc_cnt)

    doc_dict = {
      'id': doc['id'],
      'title': doc['title'],
      'url': doc['url'],
      'content': doc_cnt
    }

    doc_id_ttl_map[doc['id']] = doc['title']
    doc_dicts.append(doc_dict)
  print(len(doc_dicts))

"""## Build the Index - Inverted Index Construction"""

def build_index(qry_typ = 0):
  """
  Dictionary data structure is used  because of its constant look-up time. 
  For each document parsed, the tokens are populated in the dictionary 
  ‘trm_freq’ and parallely necessary changes are made in the posting list 
  to create inverted index, ‘invtd_idx’. The dictionary consists of the term
  as the key and the value as another dictionary with document ids where 
  the term appears as the key along with the frequency of the  term in that 
  document as the value.
  """
  for doc_dict in doc_dicts:
    print("Building_index for doc_id: {0}".format(doc_dict['id']))
    for word in word_tokenize(doc_dict['content']):    
      if word in trm_freq:
        trm_freq[word] = trm_freq[word] + 1
      else:
        trm_freq[word] = 1

      if word in invtd_idx:
        pstg_lst = invtd_idx[word]
        if doc_dict['id'] in pstg_lst:
          pstg_lst[doc_dict['id']] = pstg_lst[doc_dict['id']] + 1
        else:
          pstg_lst[doc_dict['id']] = 1
      else:
        invtd_idx[word] = {doc_dict['id']:1}
    
    for word in word_tokenize(doc_dict['title']):    
      if word in title_trm_freq:
        title_trm_freq[word] = title_trm_freq[word] + 1
      else:
        title_trm_freq[word] = 1

      if word in title_invtd_idx:
        ttl_pstg_lst = title_invtd_idx[word]
        if doc_dict['id'] in ttl_pstg_lst:
          ttl_pstg_lst[doc_dict['id']] = ttl_pstg_lst[doc_dict['id']] + 1
        else:
          ttl_pstg_lst[doc_dict['id']] = 1
      else:
        title_invtd_idx[word] = {doc_dict['id']:1}


def get_term_document_weights(invtd_idx):
  """
  This function uses lnc model for calculating term weights for documents. 
  ‘l’ represents term frequency is considered using a logarithmic model 
  that is the value of 1 plus logarithm of frequency of occurrence of the 
  term in the document. ‘n’ represents document frequency is not considered.
  ‘c’ represents normalisation is done using cosine normalization. idf is 
  also created here which will be used for querylater in 
  get_term_weights_for_query().
  """
  documents_count = len(doc_dicts)
  # print(documents_count)
  document_length = defaultdict(int)  # Used as normalization factor (Cosine Similarity)
  # Defaultdict is a container like dictionaries present in the module collections. Defaultdict is a sub-class of the dict class that returns a dictionary-like object. The functionality of both dictionaries and defualtdict are almost same except for the fact that defualtdict never raises a KeyError
  trm_doc_wgt = defaultdict(dict)

  for term, pstg_lst in invtd_idx.items():
      idf[term] = math.log10(documents_count / len(pstg_lst))
      for doc_id, tf in pstg_lst.items():
          weight = 1 + math.log10(tf)
          trm_doc_wgt[term][doc_id] = weight
          document_length[doc_id] += weight ** 2

  # Use sqrt of weighted square distance for cosine normalization
  for doc_id in document_length:
      document_length[doc_id] = math.sqrt(document_length[doc_id])

  # normalization
  for term in trm_doc_wgt:
      for doc_id in trm_doc_wgt[term]:
          trm_doc_wgt[term][doc_id] /= document_length[doc_id]

  return trm_doc_wgt

def get_trm_ttl_wgt(title_invtd_idx):
  """
  Similar to get_term_document_weights
  """
  documents_count = len(doc_dicts)
  title_length = defaultdict(int)  # Used as normalization factor (Cosine Similarity)
    # Defaultdict is a container like dictionaries present in the module collections. Defaultdict is a sub-class of the dict class that returns a dictionary-like object. The functionality of both dictionaries and defualtdict are almost same except for the fact that defualtdict never raises a KeyError
  trm_ttl_wgt = defaultdict(dict)

  for term, pstg_lst in title_invtd_idx.items():
      title_idf[term] = math.log10(documents_count / len(pstg_lst))
      for doc_id, tf in pstg_lst.items():
          weight = 1 + math.log10(tf)
          trm_ttl_wgt[term][doc_id] = weight
          title_length[doc_id] += weight ** 2

  # Use sqrt of weighted square distance for cosine normalization
  for doc_id in title_length:
      title_length[doc_id] = math.sqrt(title_length[doc_id])

  # normalization
  for term in trm_ttl_wgt:
      for doc_id in trm_ttl_wgt[term]:
          trm_ttl_wgt[term][doc_id] /= title_length[doc_id]

  return trm_ttl_wgt


def create_chmp_lst():
    """
    Creates champion list for each word in the corpus. 
    Gets the posting list corresponding to each word and get a minimum of 
    most common top 200 documents or length posting list after sorting them 
    in reverse order of number of occurrences of word in each doc.
    """
    for word in trm_freq:
        pstg_lst = invtd_idx[word]
        c = Counter(pstg_lst)#Counter is a sub-class which is used to count hashable objects. It implicitly creates a hash table of an iterable when invoked
        mc = c.most_common(min(CHMP_LST_COUNT, len(pstg_lst)))
        most_common_docs = [i[0] for i in mc]
        chmp_lst[word] = most_common_docs

def main():
    global invtd_idx
    global idf
    global title_idf
    global trm_freq
    global doc_id_ttl_map
    global title_invtd_idx
    global chmp_lst
    global title_trm_freq

    idf = dict()
    invtd_idx = {}
    trm_freq = {}    
    title_trm_freq={} 
    doc_id_ttl_map = {}
    chmp_lst = {}
    title_invtd_idx={}
    title_idf = dict()

    qry_typ = 0
    print("\nBuilding_index for normal search")

    preprocess(qry_typ)#giving list of dictionaries
    build_index(qry_typ)#term freq and std inverted index
    trm_doc_wgt = get_term_document_weights(invtd_idx)
    trm_ttl_wgt = get_trm_ttl_wgt(title_invtd_idx)
    create_chmp_lst()
    pkld_obj = {}
    pkld_obj['trm_freq'] = trm_freq
    pkld_obj['invtd_idx'] = invtd_idx
    pkld_obj['trm_doc_wgt'] = trm_doc_wgt
    pkld_obj['title_idf'] = title_idf
    pkld_obj['doc_id_ttl_map'] = doc_id_ttl_map
    pkld_obj['chmp_lst'] = chmp_lst
    pkld_obj['idf'] = idf
    pkld_obj['title_invtd_idx'] = title_invtd_idx
    pkld_obj['title_trm_freq'] = title_trm_freq
    pkld_obj['trm_ttl_wgt'] = trm_ttl_wgt
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(pkld_obj, f)


    idf = dict()
    invtd_idx = {}
    trm_freq = {}     
    chmp_lst = {}
    doc_id_ttl_map = {}

    qry_typ = 1
    print("\nBuilding_index for improved search")

    preprocess(qry_typ)
    build_index(qry_typ)
    trm_doc_wgt = get_term_document_weights(invtd_idx)
    create_chmp_lst()
    pkld_obj_modified = {}
    pkld_obj_modified['trm_doc_wgt'] = trm_doc_wgt
    pkld_obj_modified['idf'] = idf
    pkld_obj_modified['chmp_lst'] = chmp_lst
    pkld_obj_modified['doc_id_ttl_map'] = doc_id_ttl_map
    pkld_obj_modified['title_idf'] = title_idf
    pkld_obj_modified['title_invtd_idx'] = title_invtd_idx
    pkld_obj_modified['title_trm_freq'] = title_trm_freq
    pkld_obj_modified['trm_ttl_wgt'] = trm_ttl_wgt
    with open(MODIFIED_PICKLE_FILE, 'wb') as f:
        pickle.dump(pkld_obj_modified, f)

    print('Index created')

if __name__ == '__main__':
    main()
