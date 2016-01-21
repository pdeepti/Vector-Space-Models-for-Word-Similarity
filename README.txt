This code is used to compute semantic similarity between words using vector space models. The experiments were done on the English language. The data used were wikipedia articles from 2001 to 2014 and the Europarl Corpus.

The approach was as follows : First, a suffix array was built of the corpus for optimal access of words. Then a context vector was computed. The scores used in the context vector were raw count (number of occurences on left or right of the word) and tfidf. 

The file pysuffix_rawcount.py is the file that builds the context vector with raw counts and pysuffix_tfidf.py builds the context vector with tfidf scores. Both codes write their output to stdout.

Evaluation was done on the Simlex-999 dataset. The file pscore.py computes both Pearson and Spearman correlation between our computed scores for the word pairs in Simlex-999 and the original Simlex-999 scores which are stored in the file simle_score.txt. To get the correlation the code can be run as follows :

>> ./pysuffix_tfidf.py > pysuffix_tifidf_scores.txt
>> ./pscore.py -i pysuffix_tfidf_scores.txt

Another approach is to build the vector using Google's word2vec. The file my_word2vec.py uses gensim library to build the word2vec model on the data. It can be run in the same way as before.

The Europarl corpus was used to test temporal features. The file temporal.py computes the context vector on the Europarl data. It can also be run same as before.

The folder "Manaal" has files that were used with the website http://wordvectors.org/ which has a tool, built by Manaal Faruqui for computing the correlation between our vectors and different gold standard data. The instructions for using it are on the website. For very large datasets use  https://github.com/mfaruqui/eval-word-vectors instead.

pysuffix.py and pysuffixtest.py are files that were used mainly for testing and were included because they might be useful for reference.

temporal_en is the Europarl Corpus and archive.es.en has the wikipedia articles.
