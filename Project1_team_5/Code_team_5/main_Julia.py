#### IR project BM25 part ###

from bm25 import BM25  as git_bm25
from gensim import corpora, models
from gensim.summarization import bm25
import numpy as np
import json
from collections import defaultdict
from numpy.linalg import inv
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import spacy
from scipy.spatial.distance import cosine

# prepare english model for spacy model
nlp = spacy.load('en_default')

# build a dictionary to store qrels, queries files
qrels = defaultdict(lambda : defaultdict(str))
with open("qrels-v2.txt") as f :
    for line in f.readlines():
        line_split = line.strip().split("\t")
        qrels[line_split[0]][line_split[2]] = line_split[3]

queries_dict = {}
ID_list = []
with open("queries-v2.txt") as f :
    
    lines = f.readlines()
    for line in lines:
        split_line = line.strip().split("\t")
        queries_dict[split_line[0]] = split_line[1]
        ID_list.append(split_line[0])
    
print(queries_dict)
print()
print(ID_list[:19])

# format the json file of DBdoc and store it into a dictionary with 
with open('DBdoc.json') as data_file:    
    DBdoc = json.load(data_file)
DBdoc_dict = {}
for i in range(len(DBdoc)):
    DBdoc_dict[DBdoc[i]["abstract"]] =DBdoc[i]["entity"] 
    

# stop words 
stop = stopwords.words('english') + list(string.punctuation) + ["``","''"]

# Create corpus for training and entity list
corpus = []
entity_list = []
for abstract,entity in zip(list(DBdoc_dict.keys()),list(DBdoc_dict.values())):
    if abstract != None:
        corpus.append( [i for i in word_tokenize(abstract.lower()) if i not in stop])
        entity_list.append(entity)


### use BM25 model in gensim package ###
bm25Model = bm25.BM25(corpus)

dictionary = corpora.Dictionary(corpus)
print (len(dictionary))

doc_vectors = [dictionary.doc2bow(text) for text in corpus]

#compute idf term
average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())


# begin to testing
query_id =[]
dbpedia_entity = []
ranking = []
score = []
top = 100
for instance_id in ID_list:
    query_str = queries_dict[instance_id]
    query = []
    for word in query_str.strip().split():
        query.append(word)
    print(query)
    # get ranking score
    scores = bm25Model.get_scores(query,average_idf)
    document_rank = np.argsort(scores)[::-1]
    
    # recored the result in well format
    query_id += [instance_id for _ in range(top)]
    dbpedia_entity+=  ["<dbpedia:"+entity_list[idx]+">" for idx in document_rank[:top]]
    ranking += [i for i in range(1,top+1)]
    score += sorted(scores, reverse=True)[:top]


# build list to write column for pandas dataframe
Q0 = ["Q0" for _ in range(len(score))]
Standard = ["STANDARD" for _ in range(len(score))]

output_df = pd.DataFrame(np.vstack((query_id,Q0,dbpedia_entity,ranking,score,Standard))).T

# output rank
output_df.to_csv("trec_eval-master/rank_output", sep="\t", index= None, header= None)


### use BM25 implementation modify from https://github.com/fanta-mnix/python-bm25
git_bm25_ = git_bm25(corpus, PARAM_K1 = 2,PARAM_B = 0.75)
query_id =[]
dbpedia_entity = []
ranking = []
score_list = []
top = 100

# begin to testing
for instance_id in ID_list:
    query = word_tokenize(queries_dict[instance_id])
    print(query)
    # sort the document by score
    scores = [(index, score) for index, score in enumerate(git_bm25_._get_scores(query))]
    scores.sort(key=lambda x: x[1], reverse=True)

    # get the top 100 documents
    index_score_tup = scores[:top]
    query_id += [instance_id for _ in range(top)]
    dbpedia_entity+=  ["<dbpedia:"+entity_list[idx]+">" for idx, _ in index_score_tup]
    ranking += [i for i in range(1,top+1)]
    score_list += [sc for _ ,sc in index_score_tup]

# format the string for output file
Q0 = ["Q0" for _ in range(len(score_list))]
Standard = ["STANDARD" for _ in range(len(score_list))]

output_df = pd.DataFrame(np.vstack((query_id,Q0,dbpedia_entity,ranking,score_list,Standard))).T

output_df.to_csv("trec_eval-master/rank_output_BM25", sep="\t", index= None, header= None)



# w2v (directly compare the vetor of words in query and each entity)

def clean_string(string, entity= False):
    ### If the string you would like to clean is "Entity", set it True.
    if entity == True:
        string = string.replace(".","").replace("'","").replace(",","")
        return " ".join([w for w in word_tokenize(" ".join((string).split("_"))) if w  not in stop ] )
    else :
        string = string.replace(".","").replace("'","").replace(",","")
        return " ".join([w for w in word_tokenize(string) if w  not in stop ] )

def w2v_sim(query,top=100):
    ### input query return sorted entity list with score (base on cosine similarity of w2v)
    result_list = []
    print(clean_string(query))
    q = nlp(clean_string(query))
    # compute cosine similarity between query and each entity
    for ent in entity_list:
        a = nlp(clean_string(ent, entity=True))
        if np.count_nonzero(a.vector) == 300:
            a.vector = np.random.rand(300)
        sim_score = 1- cosine(q.vector,a.vector)
        result_list.append((ent,sim_score))
    result_list.sort(key=lambda x: x[1])
    return result_list[::-1][:top]


query_id =[]
dbpedia_entity = []
ranking = []
score_list = []
top = 100
for instance_id in ID_list:
    res = w2v_sim(queries_dict[instance_id],top=100)
    query_id += [instance_id for _ in range(top)]
    dbpedia_entity+=  ["<dbpedia:"+ent+">" for ent,_ in res]
    ranking += [i for i in range(1,top+1)]
    score_list += [sc for _, sc in res]



Q0 = ["Q0" for _ in range(len(score_list))]
Standard = ["STANDARD" for _ in range(len(score_list))]

output_df = pd.DataFrame(np.vstack((query_id,Q0,dbpedia_entity,ranking,score_list,Standard))).T

output_df.to_csv("trec_eval-master/rank_output", sep="\t", index= None, header= None)

