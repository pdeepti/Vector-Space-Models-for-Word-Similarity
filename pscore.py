#!/usr/bin/env python
import optparse
import sys
import os
import os.path
import operator
from operator import itemgetter
from string import *
import numpy
import scipy
from scipy import stats

optparser = optparse.OptionParser()
#optparser.add_option("-d", "--data", dest="input_directory", default="SelectedData", help="data to build context vectors")
optparser.add_option("-i", "--input_file", dest="input_file", default="pysuffix_test_results.txt", help="word pairs used to test the model")
optparser.add_option("-s", "--simlex", dest="simlex", default="simlex_score.txt", help="file with simlex scores")
# optparser.add_option("-s", "--tm2", dest="tm2", default=-0.5, type="float", help="Lexical translation model p_lex(f|e) weight")
(opts, _) = optparser.parse_args()

simlex_vec = []
cos_sim_vec = []

for line in open(opts.input_file):
	simlex_vec.append(float(line.split('\t')[2].split('\n')[0]))

for line in open(opts.simlex):
	cos_sim_vec.append(float(line.split('\t')[2].split('\n')[0]))
#sys.stderr.write(str(simlex_vec) + '\n\n' + str(cos_sim_vec) + '\n\n')
sys.stderr.write('Pearson Correlation: ' + str(numpy.corrcoef(simlex_vec, cos_sim_vec)[0,1]) + '\n')
sys.stderr.write('Spearman Correlation: ' + str(scipy.stats.spearmanr(simlex_vec, cos_sim_vec)) + '\n')