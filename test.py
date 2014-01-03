###! /usr/bin/python

import sys
import math
import random
import re
from collections import OrderedDict
import os
import pickle
import time
import nltk


## unpickling the index
index = {}
train_data = {}
model_params = {}

index = pickle.load(open("index","rb"))
train_data = pickle.load(open("train_data","rb"))
model_params = pickle.load(open("model_params","rb"))

def classify(doc):

    porter = nltk.PorterStemmer()
    score = {}
    terms = re.findall(r'\w+', doc,flags = re.UNICODE | re.LOCALE)    ## extract words from the text
    for cat in ['0','1']:
        other_cat = str(int(cat) ^ 1)
        log_prior = float(math.log(model_params['prior'][cat],2))
        score[cat] = log_prior
            
        for i in range(0,len(terms)):           ## for each word
            term = terms[i]
            negatives = set()
            conj = set()
            prev3_terms = set()
            succ4_terms = set()
            
            negatives = set(['no','not','isnt'])    ##,'without','nothing','nor','although'])       ## using a list of negatives to handle cases like 'not interesting'
            conj = set(['but','though'])                                                            ## using this list to handle cases like 'was sharp but.. '
            
            prev3_terms = set(terms[i-3:i])
            succ4_terms = set(terms[i+1:i+5])
            
            negatives_list = list(negatives.intersection(prev3_terms))
            conj_list = list(conj.intersection(succ4_terms))
            
            term = term.lower().strip()                 ## lower-case the word
            if term in model_params.keys():
                if len(negatives_list) != 0: ## or len(conj_list) != 0:               
                    log_term = float(math.log(model_params[term][other_cat],2))
                else:
                    log_term = float(math.log(model_params[term][cat],2))
            else:
                log_term = 0                   

            score[cat] = score[cat] + log_term
    max_score = float("-inf")
    category = ''
    for cat in ['0','1']:
        if score[cat] > max_score:
            max_score = score[cat]
            category = cat
    return category       

def test(test_file):
    start = time.clock()
    header = test_file.readline()
    print 'Column1',",",header.strip()
    for doc in test_file:
        category = classify(doc) 
        print category,",",doc.strip()
    elapsed = (time.clock() - start)
##    print elapsed
    

if __name__ == "__main__":
        if len(sys.argv)!=2: # Expect exactly one argument: the test data file
                sys.exit(2)
                
        input1 = file(sys.argv[1],"r")
        test(input1)
