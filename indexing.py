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
from nltk.stem.porter import PorterStemmer
import itertools

## This function creates the index for the train data.
## The index stores the [word][docid] pairs as keys and the frequencies of that word in the docid as values 

def create_index(train_file):

    pos_rev = {}
    neg_rev = {}
    train_data = {}                     ## contains the docid and category/class to which it belongs 
    train_sent = {}                     ## contains the docid and corresponding sentence for that docid
    ## stopwords from nltk.corpus
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'other', 'some', 'such','own','so', 'than', 'too','very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
    
##    print "creating index.."
    start = time.clock()
    index =  {}
    porter = nltk.PorterStemmer()

    docid = 0
    for line in train_file:                 
        docid +=1
        parts = line.strip().split(',')
        category = parts[0]
        sent = re.findall('(.*)',parts[1])[0]
        train_sent[docid] = sent                ## creating a dictionary that stores the documents and docids
        if category == '0':
            train_data[docid] = '0'             ## creating a dictionary that stores the docids and category to which it belongs
        elif category == '1':
            train_data[docid] = '1'
        
    for docid in train_sent.keys():             ## creating the index
        words = re.findall(r'\w+', train_sent[docid],flags = re.UNICODE | re.LOCALE)    ## extract words from the text
        for word in words:                          ## for each word
            word = word.lower().strip()             ## lower-case the word
            if index.__contains__(word):            ## check if index contains the word
                if index[word].__contains__(docid):     ## check if there is a doc containing this word
                    index[word][docid] += 1             ## if yes, increase the frequency
                else:                                   ## if word not in index, create an index entry for this word
                    index[word][docid] = 1
            else:                                   ## if word not in index, create an index entry for this word
                index[word]={}                      ## insert the position against the current docid
                index[word][docid] = 1
    
    for word in stopwords:                      ## removing stop-words from the index
        if word in index.keys():
            del index[word]

    words = []
    for word in index.keys():
        words.append(word)

    pos_list  = nltk.pos_tag(words)             ## get POS tags and store in a dictionary

    pos_dict = {}
    for pos_word in pos_list:
        word = pos_word[0]
        pos = pos_word[1]
        pos_dict[word] = pos

    adv_adj = []                                ## create a list of relevant adverbs (like ending with 'y' or 'ing') and adjectives
    for word in pos_dict.keys():
        if pos_dict[word] in ('JJ') or word.endswith('ing') or (pos_dict[word] == 'RB' and word.endswith('y')):
            adv_adj.append(word)

    adv_adj_freq = {}                           ## create a freq dict for the selected adverbs and adjectives
    for word in adv_adj:
        freq = 0
        for docid in index[word]:
            freq = freq + index[word][docid]
        adv_adj_freq[word] = freq

    k = 0                                       ## get top k frequent adverbs and adjectives
    most_freq_adv_adj = []
    for word,val in sorted(adv_adj_freq.iteritems(), key=lambda item: item[1],reverse = True):
        if k <=500:
            most_freq_adv_adj.append(word)
            k += 1
    
    ## pickle the index dictionaries                    
    pickle.dump(index,open("index","wb"))
    pickle.dump(train_data,open("train_data","wb"))
    pickle.dump(most_freq_adv_adj,open("most_freq_adv_adj","wb"))
    
    elapsed = (time.clock() - start)
##    print elapsed


