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
optparser.add_option("-d", "--data", dest="input_directory", default="SelectedData", help="data to build context vectors")
optparser.add_option("-w", "--words", dest="word_pairs", default="simlex_pairs.txt", help="word pairs used to test the model")
optparser.add_option("-n", "--n", dest="n", default=3, type="int", help="Number of nearest neighbors to be found")
# optparser.add_option("-s", "--tm2", dest="tm2", default=-0.5, type="float", help="Lexical translation model p_lex(f|e) weight")
(opts, _) = optparser.parse_args()

corpus = list()
indexed_corpus = list()
word_pairs = list()
sorted_corpus = list()
context_vector = {}
suffix_array = list()
count = {}    # FOR DEBUGGING

for line in open(opts.word_pairs):        #MAINTAINING CONTEXT VECTORS ONLY FOR THESE WORDS
    word_pairs.append(line.strip().split())

#word_pairs = ['the','cat','ate', 'mouse']

#sys.stderr.write('WORD PAIRS: ' + str(word_pairs) + '\n\n')

def roundup(x, n):
    return int(math.ceil(x / float(n))) * n

def get_all_files(directory):
    all_files = []
    for root,dirs,files in os.walk(directory):
        for f in files:
            if f!= ".DS_Store":
                all_files.append(os.path.join(root,f))
    return all_files

data_files = get_all_files(opts.input_directory)


for input_file in data_files:               #BUILDING OUR CORPUS
    for line in open(input_file, 'r'):
        corpus.extend(line.split())  
#sys.stderr.write('CORPUS: ' + str(corpus) + '\n\n')
sys.stderr.write('FINISHED BUILDING CORPUS\n\n')   
       

#corpus = ['the','cat','ate','the','cat','mouse']

for i, word in enumerate(corpus):
    indexed_corpus.append((word, i))
sys.stderr.write('FINISHED BUILDING INDEXED CORPUS\n\n')
#sys.stderr.write('INDEXED CORPUS: ' + str(indexed_corpus) + '\n\n')

sorted_indexed_corpus = sorted(indexed_corpus, key = itemgetter(0))
sys.stderr.write('FINISHED SORTING INDEXED CORPUS' + '\n\n')
        #TODO : NEED TO BE ABLE TO DETECT END OF A FILE IN CORPUS SO WE DON'T CONSIDER LAST WORD OF ONE FILE AND FIRST WORD OF ANOTHER AS CONTEXTS


# def build_suffix_array(corpus, sorted_corpus):
#     suffix_array = list()
#     for i,word in sorted(corpus):
#         if word not in sorted_corpus:
#             indices = [i for i,x in enumerate(corpus) if x == word]
#             for index in indices:
#                 sorted_corpus.append(word)
#                 suffix_array.append(index)
#     sys.stderr.write('FINISHED BUILDING SUFFIX ARRAY\n\n')

def build_suffix_array(corpus, sorted_indexed_corpus):
    for word, index in sorted_indexed_corpus:
        sorted_corpus.append(word)
        suffix_array.append(index)
    #sys.stderr.write('SORTED CORPUS: ' + str(sorted_corpus) + '\n' + 'SUFFIX ARRAY: ' + str(suffix_array) + '\n\n')
    sys.stderr.write('FINISHED BUILDING SUFFIX ARRAY\n\n')


def get_first_index(a, x):
    #Locate the leftmost value exactly equal to x
    i = bisect.bisect_left(a, x)
    #sys.stdout.write('first index: ' + str(i) + '\n')
    if i != len(a) and a[i] == x:
        return i
    return -1

def get_last_index(a,x):
    #Locate the rightmost value exactly equal to x
    i = bisect.bisect_right(a, x)
    #sys.stdout.write('last index: ' + str(i) + '\n')
    if a[i - 1] == x:
        return i
    return -1

def build_context_vector(corpus):
    context_vector_left = {}
    context_vector_right = {}
    context_words = set()
    for word in corpus:
        context_words.add(word)
        #context_vector_left[word] = []
        #context_vector_right[word] = []
    
    context_words = list(context_words)       #CONVERT SET TO LIST FOR INDEXING
    #context_words.sort()                      #SORT CONTEXT WORDS FOR EASY ACCESS
    
    #sys.stderr.write('CONTEXT WORDS: ' + str(context_words) + '\n\n')
    vector_length = len(context_words)
    corpus_length = len(corpus)

    word_pairs_set = set()
    for word in word_pairs:
        word_pairs_set.add(word[0])
        word_pairs_set.add(word[1])

    for i, word in enumerate(word_pairs_set):                #INITIALIZING CONTEXT VECTOR
        context_vector_left[word] = [0] * vector_length
        context_vector_right[word] = [0] * vector_length
        count[word] = 0

    #sys.stderr.write(str(vector_length) + '\n\n')
    #sys.stderr.write('\n\n number of words: ' + str(len(word_pairs_set)) + '\n\n')
    #for word in word_pairs_set:
     #   sys.stderr.write('\n' + str(word) + '\n')
    # for i, context_word in enumerate(context_words):
    #     for j, corpus_word in enumerate(corpus):
          #   if corpus_word is context_word:
             #    if j + 1 != vector_length and corpus[j+1] in word_pairs:
                #     context_vector_left[corpus[j + 1]][i] += 1
             #    if j - 1 >= 0 and corpus[j - 1] in word_pairs:
                #     context_vector_right[corpus[j - 1]][i] += 1
    
    #sys.stderr.write('word pairs set: ' + str(word_pairs_set) + '\n\n')
    non_occuring = 0
    non_occuring_words = set()
    for i, word in enumerate(word_pairs_set):
        sys.stderr.write('CURRENT WORD ' + str(i) + ' : ' + str(word) + '\n\n')
        start_index = int(get_first_index(sorted_corpus, word))
        #stop_index = int(get_last_index(sorted_corpus, word))
        stop_index = start_index
        #sys.stderr.write('START INDEX OF ' + str(word) + ' is: ' + str(start_index) + '\n')
        #sys.stderr.write('STOP INDEX OF ' + str(word) + ' is: ' + str(stop_index) + '\n')
        if start_index >= 0:                                                                        #WORD WAS FOUND
            for i in range(start_index + 1, corpus_length):
                while(sorted_corpus[i] == word):
                    stop_index = i
                    i += 1 
                #sys.stderr.write('EXITED WHILE\n\n')
                break
            for i in range(start_index, stop_index + 1):
                if suffix_array[i] - 1 >= 0:
                    context_left_index = context_words.index(corpus[suffix_array[i] - 1])               #GET INDEX OF LEFT CONTEXT
                    context_vector_left[word][context_left_index] += 1
                    #sys.stderr.write('LEFT CONTEXT OF ' + str(word) + ' is: ' + str(context_left_index) + '\n')
                if suffix_array[i] + 1 < corpus_length:
                    context_right_index = context_words.index(corpus[suffix_array[i] + 1])              #GET INDEX OF RIGHT CONTEXT
                    context_vector_right[word][context_right_index] += 1
                    #sys.stderr.write('RIGHT CONTEXT OF ' + str(word) + ' is: ' + str(context_right_index) + '\n')
        else: 
            non_occuring += 1
            non_occuring_words.add(word)
    # for i, corpus_word in enumerate(corpus):               #BUILDING CONTEXT VECTOR
    #     if corpus_word in word_pairs and i != 0:
    #         context_index = context_words.index(corpus[i - 1])
    #         context_vector_left[corpus_word][context_index] += 1
    #         #count[corpus_word] += 1
    #         #sys.stderr.write('DIDNT FIND LEFT CONTEXT \n\n')
    #     if corpus_word in word_pairs and i != vector_length - 1:
    #         context_index = context_words.index(corpus[i + 1])
    #         context_vector_right[corpus_word][context_index] += 1
            #count[corpus_word] += 1
            #sys.stderr.write('FOUND RIGHT CONTEXT \n\n')
            #sys.stderr.write('DIDNT FIND RIGHT CONTEXT \n\n')
  
    #for context_word in word_pairs:                       #DEBUGGING
        #sys.stderr.write('NUMBER OF CONTEXTS FOUND FOR ' + str(context_word) + ': \n\n' + str(count[context_word]) + '\n\n')
    sys.stderr.write('NON OCCURING WORDS ARE: \n' + str(non_occuring_words) + '\n')
    sys.stderr.write('NUMBER OF NON OCCURING WORDS: ' + str(non_occuring) + '\n')
    for word in word_pairs_set:
        combined_context = []
        #sys.stderr.write('CONTEXT VECTOR RIGHT : \n\n' + str(context_vector_right[word]) + '\n\n' + 'CONTEXT VECTOR LEFT : \n\n' + str(context_vector_left[word]) + '\n\n')
        combined_context.extend(context_vector_left[word])
        combined_context.extend(context_vector_right[word])
        #sys.stderr.write('COMBINED LIST : \n\n' + str(combined_context) + '\n\n')

        context_vector[word] = combined_context
        for i, score in enumerate(context_vector[word]):
            if score < 10:
                context_vector[word][i] = 5
            if score < 50:
                context_vector[word][i] = roundup(score, 5)
            if score > 50 and score < 500:
                context_vector[word][i] = roundup(score, 10)
            if score > 500:
                context_vector[word][i] = roundup(int(score)/2, 50)
    sys.stderr.write('FINISHED BUILDING CONTEXT VECTOR' + '\n\n')

def get_knn(n):
    zero_sim = []
    non_occuring = 0
    cosine_similarity_list = []
    for i, [word1, word2] in enumerate(word_pairs):
        #sys.stderr.write('context vector of ' + str(word1) + ' :' + str(context_vector[word1]) + '\n\n')
        cosine_similarity_list.append(float(cosine_similarity(context_vector[word1], context_vector[word2])))
        #if cosine_similarity_list[i] == 0.0:
            #non_zero = 0
            #sys.stderr.write(str(word1) + '  ' + str(word2) + '\n')
            # for i,num in enumerate(context_vector[word1]):
            #     if num != 0.0:
            #         sys.stderr.write('score of word1: ' + str(num) + ' score of word2: ' + str(context_vector[word2][i]) + '\n')
            # #sys.stderr.write('\nNUMBER OF NON ZERO CONTEXTS OF WORD1 ' + str(word1) + ' : ' + str(non_zero) + '\n')
            # #non_zero = 0
            # for i,num in enumerate(context_vector[word2]):
            #     if num != 0.0:
            #         sys.stderr.write('score of word2: ' + str(num) + ' score of word1: ' + str(context_vector[word1][i]) + '\n')
            #sys.stderr.write('\nNUMBER OF NON ZERO CONTEXTS OF WORD2 ' + str(word2) + ' : ' + str(non_zero) + '\n') 
            #non_occuring += 1
        #sys.stderr.write('Cosine similarity for ' + str(word1) + ' and ' + str(word2) + ' are: ' + str(cosine_similarity_list[word2]) + '\n')
        #sorted_similarities = sorted(cosine_similarity_list.items(), key=operator.itemgetter(1), reverse = True)
        #top_matches = [(tup[0],) for tup in sorted_similarities[:n]]
        sys.stdout.write(str(word1) + '\t' + str(word2) + '\t'+ str(cosine_similarity_list[i]) + '\n')
    #sys.stderr.write('NUMBER OF 0 COSINE SIMILARITY PAIRS: ' + str(non_occuring) + '\n')
    #for i, num in enumerate(sorted(cosine_similarity_list)):
     #   sys.stderr.write(str(i) + '. ' + str(num) + '\n')
        #sys.stderr.write('Cosine similarities for ' + str(word1) + ' are: ' + str(sorted_similarities) + '\n')
    #sys.stderr.write('Cosine similairty between [0,1,0] and [0,0,1] is ' + str(cosine_similarity([0,2,0,0,0,0,0,0],[0,0,1,0,0,0,0,0])) + '\n')

build_suffix_array(corpus, sorted_indexed_corpus)
build_context_vector(corpus)
get_knn(opts.n)
#sys.stderr.write('CONTEXT VECTOR OF NOON: \n\n' + str(context_vector['noon']) + '\n\n END OF CONTEXT VECTOR')