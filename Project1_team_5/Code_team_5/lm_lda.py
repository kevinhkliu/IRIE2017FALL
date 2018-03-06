#################################################
#LDA+LM Model using scikit-learn 0.19.1
#kpchen@nlg.csie.ntu.edu.tw
#本程式執行過程約需 10 分鐘
#################################################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
import numpy as np
import json

#################################################
#決定 topic 總數量
#################################################
no_topics = 200

#################################################
#程式目錄輸出 result.txt 
#################################################
def output_result(query_id, entitys, scores):
    fw = open("result.txt", "w", encoding="utf-8")
    num_result = 1000
    top_index = [i for i in scores.argsort()[:-num_result-1:-1]]
    for i in range(num_result):
        #print("{}\t{}\t<dbpedia:{}>\t{}\t{:.4f}\tSTANDARD".format(query_id,top_index[i],entitys[top_index[i]],i+1,scores[top_index[i]]))
        fw.write(query_id+"\tQ0\t<dbpedia:"+entitys[top_index[i]]+">\t"+str(i+1)+"\t"+str(scores[top_index[i]])+"\tSTANDARD\n")
        i += 1
    fw.close()

#################################################
#讀檔 DBdoc.json 並轉換英文大寫字元為小寫字元 
#################################################
f = open("DBdoc.json", 'r')
data = json.load(f)

no_documents = len(data)
documents, entitys = [], []
for row in range(len(data)):
    document = data[row]['abstract']
    if (document is not None):
        entitys.append(data[row]['entity'])
        documents.append(document.lower())
    else:
        no_documents -= 1

#################################################
#計算求得 tf = document-term matrix
#排除出現頻率在所有 document 占50%以上或3篇以下的字元
#求得 Cf = 每個 term 在所有 document 加總數量
#求得 dF = 每篇 document 內含的 term 總數量
#求得 CF = 所有 document 內含的 term 加總數量
#################################################
tf_vectorizer = CountVectorizer(max_df=0.5, min_df=3, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
Cf = np.sum(tf, axis=0)
dF = np.sum(tf, axis=1).transpose()
CF = np.sum(Cf)
Cf_list = Cf.tolist()[0]
dF_list = dF.tolist()[0]
tf_feature_names = tf_vectorizer.get_feature_names()
tf_csr = tf.tocsr()

#################################################
#訓練 lda 模型 fit document-term matrix
#求得 lda_W = document-topic matrix
#求得 lda_H = topic-term matrix
#求得 lda_WH = document-term matrix
#################################################
lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=10, learning_method='batch', random_state=0).fit(tf)
lda_W = lda_model.transform(tf)
lda_H = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
lda_WH = lda_W.dot(lda_H)

#################################################
#建立用 term 可查詢對應欄位索引的字典，查無返回 -1
#################################################
wordIdx = defaultdict(lambda:-1)
for i in range(len(tf_feature_names)):
    wordIdx[tf_feature_names[i]] = i

#################################################
#讀檔 queries-v2.txt 並預測 Relevant entity
#################################################
fr = open("queries-v2.txt", 'r')
row = fr.readline().strip('\n')
count = 0
while len(row):
    query_id = row.split('\t')[0]
    if query_id.startswith("SemSearch_ES"): #依照 Query categories 調整評分公式參數
        lambda1, lambda2 = 0.99, 0.01
    elif query_id.startswith("INEX-LD"):
        lambda1, lambda2 = 0.01, 0.01
    elif query_id.startswith("QALD2"):
        lambda1, lambda2 = 0.01, 0.01
    elif query_id.startswith("ListSearch"):
        lambda1, lambda2 = 0.01, 0.99
    else:
        lambda1, lambda2 = 0.5, 0.5
    query = row.split('\t')[1].split(' ')
    query_terms = []
    for x in query: #排除不重要字元並將 Query 轉換英文大寫字元為小寫字元 
        x_strip = x.strip('(').strip(')').strip('.')        
        query_terms.append(x_strip.lower())
    p1scores, p2scores, p3scores = [], [], [] #模型包含三個評分項目 p1:考慮單一文檔詞頻 p2:考慮全文檔詞頻 p3:考慮 Topic 相關性
    for d in range(no_documents):
        p1score, p2score, p3score = 1, 1, 1
        for term in query_terms:
            position = wordIdx[term]
            if position >= 0:
                if tf_csr[d,position] > 0:
                    p1score *= tf_csr[d,position] / dF_list[d]
                else:
                    p1score *= 1e-10 #若單一文檔詞頻為 0 改為 1e-10
                p2score *= Cf_list[position] / CF
                p3score *= lda_WH[d][position]
        p1scores.append(p1score)
        p2scores.append(p2score)
        p3scores.append(p3score)
    scores = [lambda1*(lambda2*p1 + (1-lambda2)*p2) + (1-lambda1)*p3 for p1, p2, p3 in zip(p1scores, p2scores, p3scores)] # 依照公式進行評分
    output_result(query_id, entitys, np.asarray(scores))
    row = f.readline().strip('\n')
    count += 1
fr.close()