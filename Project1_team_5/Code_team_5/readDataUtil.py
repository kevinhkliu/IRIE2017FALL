
import numpy as np
import json
import re
# nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

#nltk.download('popular')
#nltk.download('punkt')

#stemmer = SnowballStemmer('english') 

#porter: [in]: having  [out]: hav 
stemmer = PorterStemmer()
#lemmatizer: [in]: having  [out]: have
wnl = WordNetLemmatizer()

tokenizer = RegexpTokenizer(r'\w+')

def tokenize(text):
    lower_tokens = text.lower()
    tokens = tokenizer.tokenize(lower_tokens)
    # remove stop words from tokens
    non_stop_tokens = [w for w in tokens if w not in stopwords.words('english')]
    # remove numbers
    non_number_tokens = [re.sub(r'[\d]', ' ', w) for w in non_stop_tokens]
    non_number_tokens = ' '.join(non_number_tokens).split()
    return non_number_tokens

def tokenization(text):
    tokens = tokenize(text)
    stemmed_tokens = [stemmer.stem(w) for w in tokens]
    # remove empty
    stems = [w for w in stemmed_tokens if len(w) > 1]
    return stems

def lemmatokenization(text):
    tokens = tokenize(text)
    stemmed_tokens = [wnl.lemmatize(w) for w in tokens]
    # remove empty
    stems = [w for w in stemmed_tokens if len(w) > 1]
    return stems

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def apk(actual, predicted, k):
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k): 
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

#from collections import defaultdict

def read_query_ans():
    qrels = {}
    with open("data/qrels-v2.txt", encoding='utf-8') as f :
        for line in f.readlines():
            line_split = line.strip().split("\t")
            if line_split[3] != '0':
                if line_split[0] in qrels:
                    qrels[line_split[0]].append(line_split[2])
                else:
                    qrels[line_split[0]] = [line_split[2]]
    return qrels

#load data set
def loadData():
    queries_dict = {}
    queryID_list = []
    with open("data/queries-v2.txt", encoding='utf-8') as f : 
        lines = f.readlines()
        for line in lines:
            split_line = line.strip().split("\t")
            queries_dict[split_line[0]] = split_line[1]
            queryID_list.append(split_line[0])

    with open('data/DBdoc.json') as data_file:    
        DBdoc = json.load(data_file)
    DBdoc_dict = {}
    
    for i in range(len(DBdoc)):
        DBdoc_dict[DBdoc[i]["abstract"]] =DBdoc[i]["entity"] 
    abstract_list = []
    entity_list = []
    
    for abstract,entity in zip(list(DBdoc_dict.keys()),list(DBdoc_dict.values())):
        if abstract != None:
            entity_list.append(entity)
            abstract_list.append(abstract.lower())
    return queryID_list, queries_dict, entity_list, abstract_list