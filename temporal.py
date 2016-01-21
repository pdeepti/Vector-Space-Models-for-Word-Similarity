#!/usr/bin/env python
import optparse
import sys
import os
import os.path
import operator
import bisect
import math
from operator import itemgetter
from string import *
from StringIO import StringIO
#from tools import *
#from suffix_array import Suffix_array
from sklearn.metrics.pairwise import cosine_similarity

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="input_directory", default="Temporal_en", help="data to build context vectors")
optparser.add_option("-w", "--words", dest="word_pairs", default="simlex_pairs.txt", help="word pairs used to test the model")
optparser.add_option("-n", "--n", dest="n", default=5, type="int", help="Number of nearest neighbors to be found")
# optparser.add_option("-s", "--tm2", dest="tm2", default=-0.5, type="float", help="Lexical translation model p_lex(f|e) weight")
(opts, _) = optparser.parse_args()

corpus = list()
indexed_corpus = list()
test_words = set()
sorted_corpus = list()
context_vector = {}
suffix_array = list()
term_frequency = {}
document_frequency = {}
tf_idf = {}
dates_vec = {} #For temporal features
num_documents = 0
count = {}    # FOR DEBUGGING
words = list()

for line in open(opts.word_pairs):        #MAINTAINING CONTEXT VECTORS ONLY FOR THESE WORDS
    test_words.add(line.strip().split()[0])
    test_words.add(line.strip().split()[1])

sys.stderr.write("number of words: " + str(len(test_words)) + "\n\n")

def get_all_files(directory):
    all_files = []
    for root,dirs,files in os.walk(directory):
        for f in files:
            if f!= ".DS_Store":
                all_files.append(os.path.join(root,f))
    return all_files



data_files = get_all_files(opts.input_directory)


context_words = set()
num_documents = len(data_files)
k = len(data_files)
for word in test_words:
    dates_vec[word] = [0]*k #len(data_files)

#sys.stderr.write("\n\n\nessay: " + str(dates_vec['essay']) + "\n\n\n")

for i, input_file in enumerate(data_files):               #BUILDING OUR CORPUS
    for line in open(input_file, 'r'):
        for word in line.split():
            if word in test_words:
                    if word not in corpus:
                        corpus.append(word)
                    dates_vec[word][i] += 1

#sys.stderr.write('CORPUS: ' + str(corpus) + '\n\n')
sys.stderr.write('FINISHED BUILDING CORPUS\n\n')   

sys.stderr.write("number of test words in corpus: " + str(len(corpus)))   
def get_knn(n):
    zero_sim = []
    non_occuring = 0
    cosine_similarity_list = {}
    for i, word1 in enumerate(test_words):
        cosine_similarity_list[word1] = []
        cs_list = []
        for j, word2 in enumerate(test_words):
            if word1 is not word2:
                sys.stderr.write("word1: " + word1 + "\nword2: " + word2 + "\n\n")

                #sys.stderr.write(str(dates_vec[word1]))
                cs = float(cosine_similarity(dates_vec[word1], dates_vec[word2]))
                cs_list.append( (word2, cs) )
                cosine_similarity_list[word1] = cs_list

        sorted_similarities = sorted(cosine_similarity_list[word1], key=operator.itemgetter(1), reverse = True)

        #sys.stderr.write("sorted_similarities is: " + str(sorted_similarities) + "\n\n")

        #sys.stderr.write("sorted_similarities has " + str(len(sorted_similarities)) + "\n")

        top_matches = [(tup[0],) for tup in sorted_similarities[:n]];
        sys.stdout.write("Top Matches for " + word1 + ": " + str(top_matches) + "\n")
    #sys.stdout.write(str(word1) + '\t' + str(word2) + '\t'+ str(cosine_similarity_list[i]) + '\n')


def build_temporal_vector_file():
    for word in test_words:
    	sys.stdout.write(word + " ")
    	for val in dates_vec[word]:
        	sys.stdout.write(str(val)+ " ")
        sys.stdout.write("x\n")

#get_knn(opts.n)
build_temporal_vector_file()


