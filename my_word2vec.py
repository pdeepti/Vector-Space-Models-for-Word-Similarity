#!/usr/bin/env python
import gensim, logging
import optparse
import sys
import os
import os.path
import operator
import bisect
import math
import re
from operator import itemgetter
from string import *
from StringIO import StringIO
from nltk.stem.porter import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="input_directory", default="temporal_en", help="data to build context vectors")
optparser.add_option("-w", "--words", dest="word_pairs", default="simlex_pairs.txt", help="word pairs used to test the model")
(opts, _) = optparser.parse_args()

corpus = list()
word_pairs = list()
words = list()

def get_words():
    for word1, word2 in word_pairs:
        if word1 not in words:
            words.append(word1)
        if word2 not in words:
            words.append(word2)

def get_all_files(directory):
    all_files = []
    for root,dirs,files in os.walk(directory):
        for f in files:
            if f!= ".DS_Store":
                all_files.append(os.path.join(root,f))
    return all_files

class sentence_generator:
    
    def __init__(self, data_files):
        self.data_files = data_files
    
    def __iter__(self):
    	for input_file in self.data_files:               #BUILDING OUR CORPUS
    	    text = [line.rstrip() for line in open(input_file, 'r')]
    	    text_str = ''.join(text)
    	    sentences = re.split(r'[?!.]', text_str)
    	    for sentence in sentences:
                #sys.stderr.write(str(sentence) + '\n')
    	        yield sentence.split()

#sys.stderr.write('Built Corpus\n')

def build_model(model, data_files):
    for input_file in data_files:
        text = [line.rstrip() for line in open(input_file)]
        text_str = ''.join(text)
        sentences = re.split(r'[?!.]', text_str)
        model.train(sentences)
    return model

def get_similarity(model):
    #sys.stderr.write(model.similarity('is', 'sentence') + '\n')
    for word1, word2 in word_pairs:
    	if word1 in model.vocab and word2 in model.vocab:
        	similarity =  model.similarity(word1, word2)
        	#sys.stderr.write(str(word1) + '\t' + str(word2) + '\t'+ str(similarity) + '\n')
        	sys.stdout.write(str(word1) + '\t' + str(word2) + '\t'+ str(similarity) + '\n')
        else:
        	similarity =  0
        	#sys.stderr.write(str(word1) + '\t' + str(word2) + '\t'+ str(similarity) + '\n')
        	sys.stdout.write(str(word1) + '\t' + str(word2) + '\t'+ str(similarity) + '\n')

def build_vectors_file(model):
    for word in words:
        if word in model.vocab:
            sys.stdout.write(word + " " + str(model[word]))
            sys.stdout.write("newline\n")

def debugging():
	sys.stderr.write(str(sentences) + '\n')
	sys.stderr.write('Built Model\n')

for line in open(opts.word_pairs):        #MAINTAINING CONTEXT VECTORS ONLY FOR THESE WORDS
    word_pairs.append(line.strip().split())

sys.stderr.write('Building Model\n')
data_files = get_all_files(opts.input_directory)

# ONLY USE NEXT 3 LINES WHEN BUILDING A MODEL FOR THE FIRST TIME. AFTER THAT LOAD IT FROM DISK.
sentences = sentence_generator(data_files)
model = gensim.models.Word2Vec(sentences, min_count = 1, workers = 4)

model.save('word2vec_model_temporal')

#new_model = gensim.models.Word2Vec.load('word2vec_model_temporal') UNCOMMENT TO USE A MODEL ALREADY BUILT

#sys.stderr.write(str(sentences) + 'END OF SENTENCE\n')
sys.stderr.write('Built Model\n')
#get_similarity(new_model)
get_words()
#build_vectors_file(new_model)
sys.stdout.write(str(words))

#debugging()
       