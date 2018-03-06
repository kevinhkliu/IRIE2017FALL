import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import readDataUtil as rd
import gensim
import time

#tokenization = rd.tokenization
# tokenize words by using lemmatizer
tokenization = rd.lemmatokenization

# generate a query's weighted word2vector by using the tfidf value 
# of each word (term) in the query as the weight
# termid_word_list: a list of words (terms) in document corpus
# w2vModel.wv[w] : word2vector of a word w

def gen_query_tfidfw2vec(query_tfidf, termid_word_list, w2vModel, w2vector_size):
    query_w2vec = np.zeros(w2vector_size)
    # query: only one row 0; [1]: column; find non-zero term index 
    terms_index = query_tfidf[0,:].nonzero()[1] 
    for id in terms_index:
        if termid_word_list[id] in w2vModel.wv.vocab:
            query_w2vec = query_w2vec +  w2vModel.wv[termid_word_list[id]] * query_tfidf[0,id]
    return query_w2vec.reshape(1, -1) # convert an array with a simple sample to a 2-d array

# For each document: generate a doc's weighted word2vector by 
# using the tfidf value of each word (term) in the doc as the weight

def gen_doc_tfidfw2vecs(docs_tfidf, termid_word_list, w2vModel, w2vector_size):
    docs_w2vecs = []
    for doc_indx in range(docs_tfidf.shape[0]):
        doc_w2vec = np.zeros(w2vector_size)
        # find non-zero term index for doc: doc_indx
        terms_index = docs_tfidf[doc_indx,:].nonzero()[1] # 1: column; find non-zero term index 
        for id in terms_index:
            if termid_word_list[id] in w2vModel.wv.vocab:
                doc_w2vec = doc_w2vec +  w2vModel.wv[termid_word_list[id]] * docs_tfidf[doc_indx,id]
        docs_w2vecs.append(doc_w2vec)
    return docs_w2vecs

#load data set and query
queryID_list, queries_dict, entity_list, doc_abstract_list = rd.loadData()

doc_set = doc_abstract_list
n_features = 20000    
w2vector_size = 300
topk = 100

print("Extracting tf-idf features ...")
start_time = time.time()
# vectorize documents by using tfidf vectorizer  //term 出現高於90%文章濾掉，term出現小於2篇文章也濾掉，再取排名前20000個term
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenization, max_features=n_features,  max_df=0.9, min_df=2)
docs_tfidf = tfidf_vectorizer.fit_transform(doc_set)
termid_word_list = tfidf_vectorizer.get_feature_names() # word = termid_word_list[indx] 

print("Loading the glove word2vec model...")
## use the pretrained Glove word2vec
w2vModel = gensim.models.KeyedVectors.load_word2vec_format('w2vfile/glove.6B.300d.word2vec')
    
#generate word2vector for each document based on w2vector and tfidf weight of the terms in the document
docs_w2vecs = gen_doc_tfidfw2vecs(docs_tfidf, termid_word_list, w2vModel, w2vector_size)
docs_w2vecs = np.array(docs_w2vecs)

for qIndex in range(0, len(queryID_list)):
    print(str(qIndex) + "/" + str(len(queryID_list)))
    query_str = queries_dict[queryID_list[qIndex]]
    query = [query_str]
    # generate tfidf vector for the query
    query_tfidf = tfidf_vectorizer.transform(query)
    # generate word2vector for the query based on tfidf weight
    query_w2vec = gen_query_tfidfw2vec(query_tfidf, termid_word_list, w2vModel, w2vector_size)

    # calculate ranking scores of docs using cosine similarity between query and each doc
    # based on non-weighted word2vector (word2vec　model)       
    vsm_scores = cosine_similarity(docs_tfidf, query_tfidf).flatten()
    div = vsm_scores.max(0) - vsm_scores.min(0)
    if div == 0: div = 1
    # normalized the vsm scores 
    vsm_scores_normed = (vsm_scores - vsm_scores.min(0)) / div
    related_docs_indices = vsm_scores.argsort()[::-1][:topk]
    count=0
    vsmPred_entityList = []
    with open("result/qResult_VSM.txt",'a+') as f:
        for index in related_docs_indices:
            line_new = queryID_list[qIndex].encode("utf8").decode("cp950", "ignore") + "\t" + "Q0" + "\t" + "<dbpedia:" + entity_list[index].encode("utf8").decode("cp950", "ignore") + ">" + "\t" + str(count+1) + "\t" + str(vsm_scores[related_docs_indices][count]) + "\t" +"STANDARD"
            f.writelines(line_new +'\n')
            count = count + 1 
    f.close

    # calculate ranking scores of docs using cosine similarity between query and each doc
    # based on weighted (tfidf weight) word2vector (word2vec　model)           
    w2v_scores = cosine_similarity(docs_w2vecs, query_w2vec).flatten()
    div = w2v_scores.max(0) - w2v_scores.min(0)
    if div == 0: div = 1
    # normalized the w2v scores 
    w2v_scores_normed = (w2v_scores - w2v_scores.min(0)) / div
    related_docs_indices = w2v_scores.argsort()[::-1][:topk]
    count=0
    w2vPred_entityList = []
    with open("result/qResult_tfidfW2V.txt",'a+') as f:
        for index in related_docs_indices:
            line_new = queryID_list[qIndex].encode("utf8").decode("cp950", "ignore") + "\t" + "Q0" + "\t" + "<dbpedia:" + entity_list[index].encode("utf8").decode("cp950", "ignore") + ">" + "\t" + str(count+1) + "\t" + str(w2v_scores[related_docs_indices][count]) + "\t" +"STANDARD"
            f.writelines(line_new +'\n')
            count = count + 1 
    f.close
    
    # combine the ranking scores of VSM model and Word2Vec model by using diffrent weight
    # from 0.1 ~ 0.9     
    for i in range(1, 10):
        vsm_weight = i/10
        vsmw2v_scores =  vsm_weight * vsm_scores_normed + (1-vsm_weight) * w2v_scores_normed
        related_docs_indices = vsmw2v_scores.argsort()[::-1][:topk]
        count=0
        vsmw2vPred_entityList = []
        outfile = "result/qResult_VSMW2V" + "_w" + str(i) + ".txt"
        with open(outfile,'a+') as f:
            for index in related_docs_indices:
                line_new = queryID_list[qIndex].encode("utf8").decode("cp950", "ignore") + "\t" + "Q0" + "\t" + "<dbpedia:" + entity_list[index].encode("utf8").decode("cp950", "ignore") + ">" + "\t" + str(count+1) + "\t" + str(vsmw2v_scores[related_docs_indices][count]) + "\t" +"STANDARD"
                f.writelines(line_new +'\n')
                count = count + 1 
        f.close

end_time = time.time()
print('processed in %ds' % (end_time - start_time))       