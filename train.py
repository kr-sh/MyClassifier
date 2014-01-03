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

porter = nltk.PorterStemmer()

## This function performs feature selection using the MI method
def feat_select(train_data,index):
##    print "calculating MI counts for feature selection.."
    MI = {}
    n = len(train_data)                                         ## total number of documents
    for word in index.keys():
        for cat in ['0','1']:
            n00=0
            n01=0
            n10=0
            n11=0
            for docid in train_data.keys():
                if index[word].__contains__(docid):             ## if doc contains the term
                    if train_data[docid] == cat:                    ## if doc is in class c
                        n11 += 1
                    else:                                           ## if doc is not in class c
                        n10 += 1
                else:                                           ## if doc does not contain the term
                    if train_data[docid] == cat:                    ## if doc is in class c
                        n01 += 1
                    else:                                           ## if doc is not in class c
                        n00 += 1
                        
            n00=float(n00)
            n01=float(n01)
            n10=float(n10)
            n11=float(n11)
            if n00==0:
                n00=.1
            if n01==0:
                n01=.1
            if n10==0:
                n10=.1
            if n11==0:
                n11=.1
                    
            n1x = n10 + n11
            nx1 = n01 + n11
            n0x = n00 + n01
            nx0 = n00 + n10

            MI[(word,cat)] = 0
            MI[(word,cat)] = (((n11/n) * (math.log(n*n11,2)-math.log(n1x*nx1,2))) + ((n01/n) * (math.log(n*n01,2)-math.log(n0x*nx1,2)))+
                                ((n10/n) * (math.log(n*n10,2)-math.log(n1x*nx0,2))) + ((n00/n) * (math.log(n*n00,2)-math.log(n0x*nx0,2))))

    
##    print "finished calculating MI counts"
    return MI

## This function extracts the most relevant i.e. top k features with highest MI values
def feat_extract(MI,most_freq_adv_adj):

    features = set()
    for cat in ['0','1']:

##        print "processing features for category %s" %cat
        cnt = 1
        k=5000
        for word_cat,val in sorted(MI.iteritems(), key=lambda item: item[1],reverse = True):       ## iterating through the sorted key tuple/ value pairs
            word = word_cat[0]
            category = word_cat[1]
            if cat == category and cnt<=k:
                features.add(word) 
                cnt += 1
        
    for adv_adj in most_freq_adv_adj:           ## The list of most frequent adjectives and adverbs are also used as features along with the above set
        features.add(adv_adj)

    return features

## This function calculates the model parameters for the classifier
def train_multinomial_NB(features,train_data,index,most_freq_adv_adj):
    n = len(train_data)                         ## total number of documents
    B = len(index)                              ## total number of terms in the vocabulary

    prior = {}
    model_params = {}
    Tct_all= {}
##    rare_term_param = {}                        ## extracting model parameters for rare terms that can be used for test data that does not appear in train data
    
    for cat in ['0','1']:
        cnt = 0
        Tct_all[cat]= 0                             ## term count of all terms t in docs D belonging to class cat
        for docid in train_data.keys():             ## calculate the prior probability of the class cat
            if train_data[docid] == cat:
                cnt += 1
        prior[cat] = float(cnt)/float(n)
        
        for docid in train_data.keys():             ## calculating counts for all terms t in docs from class cat   
            if train_data[docid] == cat:                ## for each document in class cat
                for t in features:                          ## for each feature or term t
                    if index[t].__contains__(docid):
                        Tct_all[cat] = Tct_all[cat] + index[t][docid]   ## add term frequency in that doc to total term count of class cat
        
        for t in features:                          ## calculating the frequency of term t in docs D belonging to class cat
            Tct = 0
            for docid in train_data.keys():         
                if train_data[docid] == cat:        
                    if index[t].__contains__(docid):        ## documents D in class cat containing term t
                        if t in most_freq_adv_adj:                  ## give more weightage to adjectives and adverbs ending in 'ing' and 'y'
                            Tct = (index[t][docid]*1.5) + Tct         
                        else:
                            Tct = index[t][docid] + Tct             ## add term t's frequencies in docs D for docs D in class cat

            if t in model_params.keys():
                model_params[t][cat] = float(Tct + 1)/float(Tct_all[cat] + B)
            else:
                model_params[t] = {}
                model_params[t][cat] = float(Tct + 1)/float(Tct_all[cat] + B)

        
        if 'prior' in model_params.keys():                  ## adding the prior values to the model
            model_params['prior'][cat] = prior[cat]
        else:
            model_params['prior'] = {}
            model_params['prior'][cat] = prior[cat]


    for cat in ['0','1']:                                   ## writing the prior values to the model file
        print "**prior of class %s: %f" % (cat,model_params['prior'][cat])
        
    print "**term|class : P(t|c)"


    ## not used as reduction in performance is noticed
    rare_term_param = {}                                    ## extracting model parameters for rare terms that can be used for test data that does not appear in train data
    for cat in ['0','1']:
        min_param = float("inf")
        for t in model_params.keys():
            if model_params[t][cat] < min_param:
                min_param = model_params[t][cat]
        rare_term_param[cat] = min_param

    for cat in ['0','1']:    
        if 'rare_term' in model_params.keys():
            model_params['rare_term'][cat] = rare_term_param[cat]
        else:
            model_params['rare_term'] = {}
            model_params['rare_term'][cat] = rare_term_param[cat]
            
    for cat in ['0','1']:                                   ## writing the parameter values to the model file
        for t,val in sorted(model_params.iteritems(), key=lambda item: item[1],reverse = True):
            print "%s|%s :" % (t,cat),model_params[t][cat]

    return model_params               
            
    
def train_classifier():
    start = time.clock()
    index = {}
    train_data = {}
    
    index = pickle.load(open("index","rb"))
    train_data = pickle.load(open("train_data","rb"))
    most_freq_adv_adj = pickle.load(open("most_freq_adv_adj","rb"))
    
    MI = {}
    model_params = {}

    MI = feat_select(train_data,index)
    features = feat_extract(MI,most_freq_adv_adj)
    model_params = train_multinomial_NB(features,train_data,index,most_freq_adv_adj)
    
    ## pickle the MI dictionary for classfication                    
    pickle.dump(MI,open("MI","wb"))
    ## pickle the model parameters dictionary for classfication                    
    pickle.dump(model_params,open("model_params","wb"))
    
    elapsed = (time.clock() - start)
##    print elapsed
    
