
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import readDataUtil as rd
import time

#tokenization = rd.tokenization
tokenization = rd.lemmatokenization

#load data set and query
queryID_list, queries_dict, entity_list, doc_abstract_list = rd.loadData()

doc_set = doc_abstract_list

n_features = 20000 # number of feature terms
n_factors = 2000 # number of latent factors for NMF model
num_iter = 600
topk = 100

print("Extracting tf-idf features ...")
start_time = time.time()
# vectorize documents by using tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenization, max_features=n_features,  max_df=0.9, min_df=2)
docs_tfidf = tfidf_vectorizer.fit_transform(doc_set)
termid_word_list = tfidf_vectorizer.get_feature_names() # word = termid_word_list[indx] 

print("Fitting the NMF model...")
# solver: coordinate descent; learning rate: alpha = 0.1; 
#l1_ratio 0: L2 regularization, NO L1 regularization
nmf_model = NMF(n_components=n_factors, random_state=1, solver='cd', alpha=.1, l1_ratio=.0)
# generate latent factors for documents based on NMF model
docs_lf = nmf_model.fit_transform(docs_tfidf)
    
for qIndex in range(0, len(queryID_list)):
#for qIndex in range(0, 2):
    print(str(qIndex) + "/" + str(len(queryID_list)))
    query_str = queries_dict[queryID_list[qIndex]]
    query = [query_str]
    # generate tfidf vector for the query
    query_tfidf = tfidf_vectorizer.transform(query)
    # generate latent factor for the query based on NMF model
    query_lf = nmf_model.transform(query_tfidf)
    
    vsm_scores = cosine_similarity(docs_tfidf, query_tfidf).flatten()
    div = vsm_scores.max(0) - vsm_scores.min(0)
    if div == 0: div = 1
    # normalize the vsm scores 
    vsm_scores_normed = (vsm_scores - vsm_scores.min(0)) / div
    related_docs_indices = vsm_scores.argsort()[::-1][:topk]
    count=0
    vsmPred_entityList = []
    with open("NMFresult/qResult_VSM.txt",'a+') as f:
        for index in related_docs_indices:
            line_new = queryID_list[qIndex].encode("utf8").decode("cp950", "ignore") + "\t" + "Q0" + "\t" + "<dbpedia:" + entity_list[index].encode("utf8").decode("cp950", "ignore") + ">" + "\t" + str(count+1) + "\t" + str(vsm_scores[related_docs_indices][count]) + "\t" +"STANDARD"
            f.writelines(line_new +'\n')
            count = count + 1 
    f.close
     
    # normalize the NMF scores 
    nmf_scores = cosine_similarity(docs_lf, query_lf).flatten()
    div = nmf_scores.max(0) - nmf_scores.min(0)
    if div == 0: div = 1
    nmf_scores_normed = (nmf_scores - nmf_scores.min(0)) / div
    related_docs_indices = nmf_scores.argsort()[::-1][:topk]
    count=0
    nmfPred_entityList = []
    with open("NMFresult/qResult_NMF.txt",'a+') as f:
        for index in related_docs_indices:
            line_new = queryID_list[qIndex].encode("utf8").decode("cp950", "ignore") + "\t" + "Q0" + "\t" + "<dbpedia:" + entity_list[index].encode("utf8").decode("cp950", "ignore") + ">" + "\t" + str(count+1) + "\t" + str(nmf_scores[related_docs_indices][count]) + "\t" +"STANDARD"
            f.writelines(line_new +'\n')
            count = count + 1 
    f.close
    
    # combine the ranking scores of VSM model and NMF model by using diffrent weight
    # from 0.1 ~ 0.9     
    for i in range(1, 10):
        vsm_weight = i/10
        vsmnmf_scores =  vsm_weight * vsm_scores_normed + (1-vsm_weight) * nmf_scores_normed
        related_docs_indices = vsmnmf_scores.argsort()[::-1][:topk]
        count=0
        vsmnmfPred_entityList = []
        outfile = "NMFresult/qResult_VSMNMF" + "_w" + str(i) + ".txt"
        with open(outfile,'a+') as f:
            for index in related_docs_indices:
                line_new = queryID_list[qIndex].encode("utf8").decode("cp950", "ignore") + "\t" + "Q0" + "\t" + "<dbpedia:" + entity_list[index].encode("utf8").decode("cp950", "ignore") + ">" + "\t" + str(count+1) + "\t" + str(vsmnmf_scores[related_docs_indices][count]) + "\t" +"STANDARD"
                f.writelines(line_new +'\n')
                count = count + 1 
        f.close

end_time = time.time()
print('processed in %ds' % (end_time - start_time))