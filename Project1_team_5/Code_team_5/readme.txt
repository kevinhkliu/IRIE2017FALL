-------- BM25 model -------------
main_Julia.py主執行檔，包含bm25和query2entity的w2v model 的實踐
需要import bm25.py內的模組

-------- LDA + LM model -------------
程式目錄要放 DBdoc.json, queries-v2.txt當作輸入檔，執行完畢會在同目錄產生排序結果檔 result.txt 。


--------VSM, NMF, w2v model ----------
1. VSMNMF.py program process in 12429s
   run VSM, NMF, hybrid of VSM+NMF
   須創data資料夾：包含DBdoc.json、qrels-v2、queries-v2
   須創result資料夾：產生outfile

2. VSMword2vec.py program process in 2234s
   run VSM, w2v, hybrid of VSM+w2v
   須創data資料夾：包含DBdoc.json、qrels-v2、queries-v2
   須創w2vfile資料夾：glove.6B.300d.word2vec download in https://nlp.stanford.edu/projects/glove/
   須創result資料夾：產生outfile	

3. readDataUtil.py
   loadData function for VSMNMF.py and VSMword2vec.py

