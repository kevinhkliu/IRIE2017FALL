import numpy as np
import re
from collections import deque
import gensim
from sklearn.preprocessing import LabelEncoder
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora

'''=====================Read data======================================================='''
with open('data/stopwords_zh.txt',encoding='utf-8') as f:
    stopwords = [s.strip() for s in f.readlines()]
corpus = []
all_entity = []
with open("data/Dream_of_the_Red_Chamber_seg.txt", "r", encoding='UTF-8') as f:
    context = f.read().strip()
    context = re.sub("[a-zA-Z]+","",context).replace("_","")
    context_ = re.sub("[1-9]+","",context).replace("_","")
    context = context_.split("。")
with open("data/train.txt","r", encoding='UTF-8') as f:
    train = [line.strip().split("\t") for line in f.readlines()[1:]]
    for line in train:
        all_entity.append(line[1])
        all_entity.append(line[2])
    train = [(line[1][-2:],line[2][-2:],line[3]) for line in train]
    
with open("data/test.txt","r", encoding='UTF-8') as f:
    test = [line.strip().split("\t") for line in f.readlines()[1:]]
    for line in test:
        all_entity.append(line[1])
        all_entity.append(line[2])
    test = [(line[1][-2:],line[2][-2:],line[3]) for line in test]
'''=====================check name entity======================================================='''    
'''
# 檢查是不是所有名字都在 context 裡
#print(all_entity)
for i in all_entity:
    if i not in context_:
        print(i)
 '''       
'''=====================get extract sentence======================================================='''    
def get_extract_sentence(list_):  
    extract_sent = []
    condition_list = []
    for ent in list_:
        three_sent = deque(maxlen=3)
        condition = 3
        first_appear_0 = 0
        first_appear_1 = 0
        for sent in context:
            three_sent.append(sent)
            three_context = " ".join(three_sent)
            if ent[0] in sent and ent[1] in sent:
                extract_sent.append(sent)
                condition = 1
                condition_list.append(condition)
                break
            elif ent[0] in three_context and ent[1] in three_context:
                extract_sent.append(three_context)
                condition = 2
                condition_list.append(condition)
                break
            if ent[0] in sent and first_appear_0 == 0:
                first_sent_0 = sent
                first_appear_0 = 1
            if ent[1] in sent and first_appear_1 == 0:
                first_sent_1 = sent
                first_appear_1 = 1
        if condition == 3:
            extract_sent.append(first_sent_0+first_sent_1)
            condition_list.append(condition)
    return extract_sent
#print("看condition的分布")
#print(np.unique(condition_list, return_counts=True))

'''=====================train word2vec======================================================='''    
train_corpus = []
for sent in context:
    train_corpus.append(sent.strip().split(" "))
embedding_size = 300    # Word vector dimensionality                      
min_word_count = 4   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
emb_context = 5          # Context window    size                                                                                    
load = 0
if load == 0:
    print("train gensim word2vector")
    w2v_model = gensim.models.Word2Vec(train_corpus, size=embedding_size, window=emb_context,
                                  min_count=min_word_count, workers=num_workers)
    w2v_model.save('word2vec/gensimword2vec')
else:
    print("load gensim word2vector")
    w2v_model = gensim.models.Word2Vec.load('word2vec/gensimword2vec')
#w2v_model.most_similar("賈政")

'''=====================tf-idf======================================================='''   
word_dict = corpora.Dictionary(train_corpus)
sentences_vectors = [word_dict.doc2bow(sent) for sent in train_corpus]
tfidf = TfidfModel(sentences_vectors, dictionary=word_dict)
sentences_tfidf = tfidf[sentences_vectors]

def gen_sent_tfidfwvec(sent, sent_tfidf, word_dict, w2v_model, embedding_size):
    word_list = []
    word_weight = []
    sent_vec = np.zeros(embedding_size)
    sent_words = [w for w in sent if w in w2v_model.wv.vocab]
    if len(sent_words) != 0:
        for id, term_weight in sent_tfidf:
            if word_dict[id] in w2v_model.wv.vocab:
                word_list.append(word_dict[id])
                word_weight.append(term_weight)
                sent_vec = sent_vec + w2v_model.wv[word_dict[id]] * term_weight 
    return sent_vec, word_list, word_weight

'''=====================train sentence embedding======================================================='''    
print("處理embedding")
extract_sent = get_extract_sentence(train)
extract_sent_vec = []
for sent in extract_sent:
    single_sent_vec = []
    for word in sent:
        try:
            single_sent_vec.append(w2v_model[word])
        except:
            pass
    extract_sent_vec.append(np.average(single_sent_vec,axis=0))

'''=====================get train======================================================='''   

train_X = np.array(extract_sent_vec)
# 把 關係 變成 1 2 3 4...
encoder = LabelEncoder()
train_y = [line[2] for line in train]
train_y = encoder.fit_transform(train_y)
print("==save train data==")
np.save('data/train_X.npy', train_X)
np.save('data/train_y.npy', train_y)

'''=====================build rule based======================================================='''   
word_relate_list = []
word_relate_weight_list = []
extract_sent_vec = []
for sent in extract_sent:
    sent = sent.strip().split(" ")
    sent_tfidf = tfidf[word_dict.doc2bow(sent)]
    sent_vec, word_list, word_weight = gen_sent_tfidfwvec(sent, sent_tfidf, word_dict, w2v_model, embedding_size)
    word_relate = []
    word_relate_weight = []
    for idx in range(len(word_list)):
        if word_weight[idx] > 0.01  and word_list[idx] not in stopwords:
            word_relate.append(word_list[idx])
            word_relate_weight.append(word_weight[idx])
    
    word_relate_list.append(word_relate)
    word_relate_weight_list.append(word_relate_weight)


relation_word =  [[] for i in range(12)]
relation_word_weight = [[] for i in range(12)]

for idx in range(len(train_y)):
    for idx2 in range(len(word_relate_list[idx])):
        relation_word[train_y[idx]].append(word_relate_list[idx][idx2])
        relation_word_weight[train_y[idx]].append(word_relate_weight_list[idx][idx2])
        
'''=====================test sentence embedding=======================================================''' 
print("test set 抽資料")
extract_sent_t = get_extract_sentence(test)
extract_sent_vec_t = []
for sent in extract_sent_t:
    single_sent_vec = []
    for word in sent:
        try:
            single_sent_vec.append(w2v_model[word])
        except:
            pass
    extract_sent_vec_t.append(np.average(single_sent_vec,axis=0))
    
'''=====================get test=======================================================''' 
test_X = np.array(extract_sent_vec_t)
test_y = [line[2] for line in test]
test_y = encoder.transform(test_y)

'''=====================rule based approach======================================================='''        
from sklearn.metrics import accuracy_score
print("------Rule base----")
score_list = [0]*12
extract_sent_vec_t = []
y_label = []
for sent in extract_sent_t:
    sent_tfidf = tfidf[word_dict.doc2bow(sent.strip().split(" "))]
    sent_vec, word_list, word_weight = gen_sent_tfidfwvec(sent, sent_tfidf, word_dict, w2v_model, embedding_size)
    score_list = [0]*12
    for word_idx in range(len(word_list)):
        if word_weight[word_idx] > 0.01  and word_list[word_idx] not in stopwords : 
            for idx in range(len(relation_word)):
                if word_list[word_idx] in relation_word[idx]:
                    get_weight_idx = relation_word[idx].index(word_list[word_idx])
                    score_list[idx] = score_list[idx] + relation_word_weight[idx][get_weight_idx]
    score_array = np.asarray(score_list)
    y_label.append(np.argmax(score_array))
    
print("Rule base accuracy:", accuracy_score(test_y,y_label))
'''=====================Rf approach======================================================='''   
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
print("------RandomForest----")
rf_train = 1
if rf_train == 1:
    RF = RandomForestClassifier(n_estimators=100, max_depth=8, max_features='sqrt')
    joblib.dump(RF, 'rfmodel.pkl')
else:
    RF = joblib.load('model/rfmodel.pkl')

RF.fit(train_X, train_y)
print("RF accuracy:",accuracy_score(test_y, RF.predict(test_X)))

'''=============================XGBoost taining====================================='''
from xgboost import XGBClassifier

print("------XGBoost----")
xgb_train = 1
if xgb_train == 1:
    XGB = XGBClassifier(learning_rate= 0.6, max_depth=7, n_estimators=10) # 0482
    joblib.dump(XGB, 'XGBmodel.pkl')
else:
    XGB = joblib.load('model/XGBmodel.pkl')

XGB.fit(train_X, train_y, eval_metric="auc")
print("XGBoost accuracy:",accuracy_score(test_y, XGB.predict(test_X)))

'''=============================Ensemble XGBoots+rule based====================================='''
predictionXGB = XGB.predict_proba(test_X)
answers=[]
pred_prob=[]

for idx in range(len(predictionXGB)):    
    maxIdx = predictionXGB[idx].argmax()
    if predictionXGB[idx][maxIdx] > 0.3:
        answers.append(maxIdx)
    else:
        answers.append(y_label[idx])
print("Ensemble accuracy:",accuracy_score(test_y,answers))









