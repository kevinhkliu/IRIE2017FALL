-------- BM25 model -------------
main_Julia.py�D�����ɡA�]�tbm25�Mquery2entity��w2v model �����
�ݭnimport bm25.py�����Ҳ�

-------- LDA + LM model -------------
�{���ؿ��n�� DBdoc.json, queries-v2.txt��@��J�ɡA���槹���|�b�P�ؿ����ͱƧǵ��G�� result.txt �C


--------VSM, NMF, w2v model ----------
1. VSMNMF.py program process in 12429s
   run VSM, NMF, hybrid of VSM+NMF
   ����data��Ƨ��G�]�tDBdoc.json�Bqrels-v2�Bqueries-v2
   ����result��Ƨ��G����outfile

2. VSMword2vec.py program process in 2234s
   run VSM, w2v, hybrid of VSM+w2v
   ����data��Ƨ��G�]�tDBdoc.json�Bqrels-v2�Bqueries-v2
   ����w2vfile��Ƨ��Gglove.6B.300d.word2vec download in https://nlp.stanford.edu/projects/glove/
   ����result��Ƨ��G����outfile	

3. readDataUtil.py
   loadData function for VSMNMF.py and VSMword2vec.py

